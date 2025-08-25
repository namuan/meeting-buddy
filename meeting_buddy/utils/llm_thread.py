"""LLM Thread for Meeting Buddy application.

This module contains the LLMThread class that handles
non-blocking LLM API calls to Ollama for processing transcription data.
"""

import json
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import requests

from ..models.llm_model import LLMRequest, LLMResponse


class LLMThread(threading.Thread):
    """Thread class for non-blocking LLM API calls to Ollama.

    This class processes LLM requests from a queue and generates
    responses using Ollama's streaming API.
    """

    def __init__(
        self,
        request_queue: queue.Queue,
        endpoint: str = "http://localhost:11434/api/generate",
        model: str = "llama3.2:latest",
        max_queue_size: int = 50,
        processing_timeout: float = 1.0,
        api_timeout: float = 30.0,
    ):
        """Initialize the LLMThread.

        Args:
            request_queue: Queue containing LLM requests to process
            endpoint: Ollama API endpoint URL
            model: Model name to use for generation
            max_queue_size: Maximum number of items to keep in queue
            processing_timeout: Timeout for queue operations in seconds
            api_timeout: Timeout for API calls in seconds
        """
        super().__init__(daemon=True)
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.request_queue = request_queue
        self.endpoint = endpoint
        self.model = model
        self.max_queue_size = max_queue_size
        self.processing_timeout = processing_timeout
        self.api_timeout = api_timeout
        self.max_retries = 3
        self.retry_delay = 1.0

        # Processing state
        self._processing = False
        self._stop_event = threading.Event()
        self._processing_lock = threading.RLock()  # Reentrant lock for thread safety
        self._cleanup_lock = threading.Lock()  # Lock for cleanup operations

        # Connection state
        self._is_connected = False
        self._last_connection_check: Optional[datetime] = None
        self._connection_check_interval = 60.0  # Check connection every 60 seconds

        # Statistics
        self._requests_processed = 0
        self._total_processing_time = 0.0
        self._last_processing_time = 0.0
        self._successful_requests = 0
        self._failed_requests = 0

        # Results storage
        self._llm_responses: list[LLMResponse] = []
        self._current_response = ""

        # Callbacks
        self._response_callback: Optional[Callable[[LLMResponse], None]] = None
        self._response_chunk_callback: Optional[Callable[[str], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._progress_callback: Optional[Callable[[str], None]] = None
        self._connection_status_callback: Optional[Callable[[bool], None]] = None

        # Check initial connection
        self._check_connection()

        self.logger.info(f"LLMThread initialized with endpoint: {endpoint}, model: {model}")
        self.logger.debug(
            f"LLM configuration: endpoint={endpoint}, model={model}, max_queue_size={max_queue_size}, "
            f"processing_timeout={processing_timeout}s, api_timeout={api_timeout}s"
        )

    def _check_connection(self) -> bool:
        """Check connection to Ollama API.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Simple health check - try to get model info
            health_url = self.endpoint.replace("/api/generate", "/api/tags")
            response = requests.get(health_url, timeout=5.0)
            response.raise_for_status()

            self._is_connected = True
            self._last_connection_check = datetime.now()

            if not self._processing:  # Only log during initial check
                self.logger.info("Successfully connected to Ollama API")

            if self._connection_status_callback:
                self._connection_status_callback(True)

            return True

        except Exception as e:
            self._is_connected = False
            error_msg = f"Failed to connect to Ollama API: {e}"

            if not self._processing:  # Only log during initial check
                self.logger.exception("Failed to connect to Ollama API")

            if self._connection_status_callback:
                self._connection_status_callback(False)
            if self._error_callback:
                self._error_callback(Exception(error_msg))

            return False

    def set_response_callback(self, callback: Callable[[LLMResponse], None]) -> None:
        """Set callback function for complete LLM responses.

        Args:
            callback: Function to call with LLMResponse when response is complete
        """
        self._response_callback = callback
        self.logger.debug("Response callback set")

    def set_response_chunk_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for streaming response chunks.

        Args:
            callback: Function to call with each response chunk
        """
        self._response_chunk_callback = callback
        self.logger.debug("Response chunk callback set")

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback function for error handling.

        Args:
            callback: Function to call when an error occurs
        """
        self._error_callback = callback
        self.logger.debug("Error callback set")

    def set_progress_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for progress updates.

        Args:
            callback: Function to call with progress messages
        """
        self._progress_callback = callback
        self.logger.debug("Progress callback set")

    def set_connection_status_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback function for connection status changes.

        Args:
            callback: Function to call when connection status changes
        """
        self._connection_status_callback = callback
        self.logger.debug("Connection status callback set")

    @property
    def is_processing(self) -> bool:
        """Check if currently processing requests."""
        return self._processing

    @property
    def is_connected(self) -> bool:
        """Check if connected to Ollama API."""
        return self._is_connected

    @property
    def requests_processed(self) -> int:
        """Get number of requests processed."""
        return self._requests_processed

    @property
    def successful_requests(self) -> int:
        """Get number of successful requests."""
        return self._successful_requests

    @property
    def failed_requests(self) -> int:
        """Get number of failed requests."""
        return self._failed_requests

    @property
    def average_processing_time(self) -> float:
        """Get average processing time per request in seconds."""
        if self._requests_processed == 0:
            return 0.0
        return self._total_processing_time / self._requests_processed

    @property
    def last_processing_time(self) -> float:
        """Get processing time of last request in seconds."""
        return self._last_processing_time

    def start_processing(self) -> bool:
        """Start the LLM processing.

        Returns:
            True if processing started successfully, False otherwise
        """
        with self._processing_lock:
            if not self._is_connected:
                self.logger.error("Cannot start processing: No connection to Ollama API")
                return False

            if self._processing:
                self.logger.warning("LLM processing already in progress")
                return True

            self._processing = True
            self._stop_event.clear()
            self.start()

            self.logger.info("LLM processing started")
            return True

    def stop_processing(self) -> None:
        """Stop the LLM processing."""
        with self._processing_lock:
            if not self._processing:
                self.logger.debug("LLM processing not in progress")
                return

            self.logger.info("Stopping LLM processing")
            self._stop_event.set()
            self._processing = False

        # Don't wait for thread to finish - let it cleanup in background
        # This prevents UI hanging when stopping processing
        self.logger.debug("LLM processing stop initiated (non-blocking)")

    def add_llm_request(self, request: LLMRequest) -> bool:
        """Add an LLM request to the processing queue.

        Args:
            request: LLMRequest to process

        Returns:
            True if request was added successfully, False if queue is full
        """
        if not self._processing:
            self.logger.debug("Not processing, ignoring LLM request")
            return False

        # Check queue size and remove old items if necessary
        while self.request_queue.qsize() >= self.max_queue_size:
            try:
                self.request_queue.get_nowait()
                self.logger.warning("Dropped LLM request due to full queue")
            except queue.Empty:
                break

        try:
            self.request_queue.put(request, timeout=0.1)
            self.logger.debug(
                f"Added LLM request to queue: prompt_len={len(request.prompt)}, queue size: {self.request_queue.qsize()}"
            )

            # Log queue statistics periodically
            if self.request_queue.qsize() % 5 == 0 and self.request_queue.qsize() > 0:
                self.logger.debug(f"LLM queue status: {self.request_queue.qsize()}/{self.max_queue_size} items")

            return True
        except queue.Full:
            self.logger.warning("Failed to add LLM request: queue full")
            return False

    def run(self) -> None:
        """Main thread execution method."""
        self.logger.debug("LLM processing thread started")

        while not self._stop_event.is_set():
            try:
                # Periodic connection check
                self._periodic_connection_check()

                # Get LLM request from queue
                queue_wait_start = time.time()
                request = self.request_queue.get(timeout=self.processing_timeout)
                queue_wait_time = time.time() - queue_wait_start

                self.logger.debug(
                    f"Retrieved LLM request from queue: prompt_len={len(request.prompt)}, wait time: {queue_wait_time:.3f}s"
                )

                # Process the request
                self._process_llm_request(request)

                # Mark task as done
                self.request_queue.task_done()

            except queue.Empty:
                # No requests to process, continue waiting
                continue
            except Exception as e:
                self.logger.exception("Error in LLM processing thread")
                if self._error_callback:
                    self._error_callback(e)

        # Process any remaining requests
        self._process_remaining_requests()

        self.logger.debug("LLM processing thread finished")

    def _periodic_connection_check(self) -> None:
        """Perform periodic connection checks."""
        if (
            self._last_connection_check is None
            or (datetime.now() - self._last_connection_check).total_seconds() > self._connection_check_interval
        ):
            self._check_connection()

    def _process_llm_request(self, request: LLMRequest) -> None:
        """Process a single LLM request.

        Args:
            request: LLMRequest to process
        """
        if request.processed:
            return

        start_time = time.time()
        retry_count = 0

        while retry_count <= self.max_retries and not self._stop_event.is_set():
            try:
                self._log_request_processing_start(request, retry_count)

                # Perform LLM API call
                response_text = self._make_llm_api_call(request)

                if response_text:
                    self._handle_successful_response(request, response_text, start_time)
                    return
                else:
                    self._handle_empty_response(request)
                    return

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count <= self.max_retries:
                    self.logger.warning(
                        f"LLM API request failed (attempt {retry_count}/{self.max_retries}): {e}. Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    self._handle_failed_request(request, e, start_time)
                    return
            except Exception as e:
                self._handle_failed_request(request, e, start_time)
                return

    def _log_request_processing_start(self, request: LLMRequest, retry_count: int) -> None:
        """Log the start of request processing."""
        retry_info = f" (retry {retry_count})" if retry_count > 0 else ""
        self.logger.debug(f"Processing LLM request{retry_info}: prompt_len={len(request.full_prompt)}")

    def _make_llm_api_call(self, request: LLMRequest) -> str:
        """Make the actual LLM API call.

        Args:
            request: LLMRequest to process

        Returns:
            Complete response text from the API
        """
        # Prepare API payload
        payload = {"model": self.model, "prompt": request.full_prompt, "stream": True}
        headers = {"Content-Type": "application/json"}

        self.logger.debug(f"Making LLM API call: model={self.model}, prompt_len={len(request.full_prompt)}")

        complete_response = ""

        # Make streaming API call
        with requests.post(
            self.endpoint, json=payload, headers=headers, stream=True, timeout=self.api_timeout
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line and not self._stop_event.is_set():
                    try:
                        decoded_line = line.decode("utf-8")
                        data = json.loads(decoded_line)
                        response_chunk = data.get("response", "")

                        if response_chunk:
                            complete_response += response_chunk

                            # Call chunk callback
                            if self._response_chunk_callback:
                                self._response_chunk_callback(response_chunk)

                        if data.get("done", False):
                            break

                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON response line: {e}")
                        continue

        return complete_response.strip()

    def _handle_successful_response(self, request: LLMRequest, response_text: str, start_time: float) -> None:
        """Handle successful LLM response.

        Args:
            request: The original LLMRequest
            response_text: The response text from the API
            start_time: When processing started
        """
        # Create response object
        llm_response = LLMResponse(text=response_text, timestamp=datetime.now(), is_complete=True)

        # Update request and add response
        request.processed = True
        request.response_text = response_text
        self._llm_responses.append(llm_response)
        self._current_response = response_text

        # Update statistics
        processing_time = time.time() - start_time
        self._requests_processed += 1
        self._successful_requests += 1
        self._total_processing_time += processing_time
        self._last_processing_time = processing_time

        self.logger.info(f"LLM response received: length={len(response_text)} chars, time={processing_time:.2f}s")

        self._log_performance_metrics(request, response_text, processing_time)
        self._call_response_callbacks(llm_response)

    def _handle_empty_response(self, request: LLMRequest) -> None:
        """Handle empty LLM response.

        Args:
            request: The original LLMRequest
        """
        self.logger.warning(f"Received empty response for LLM request: prompt_len={len(request.full_prompt)}")
        request.error_message = "Empty response received"
        self._requests_processed += 1
        self._failed_requests += 1

    def _handle_failed_request(self, request: LLMRequest, error: Exception, start_time: float) -> None:
        """Handle failed LLM request.

        Args:
            request: The original LLMRequest
            error: The exception that occurred
            start_time: When processing started
        """
        processing_time = time.time() - start_time
        error_msg = f"LLM request failed after {processing_time:.2f}s: {error}"

        self.logger.error(error_msg)
        request.error_message = str(error)

        # Update statistics
        self._requests_processed += 1
        self._failed_requests += 1
        self._total_processing_time += processing_time
        self._last_processing_time = processing_time

        if self._error_callback:
            self._error_callback(error)

    def _log_performance_metrics(self, request: LLMRequest, response_text: str, processing_time: float) -> None:
        """Log performance metrics.

        Args:
            request: The original LLMRequest
            response_text: The response text
            processing_time: Time taken to process
        """
        chars_per_sec = len(response_text) / processing_time if processing_time > 0 else 0
        tokens_estimate = len(response_text.split())  # Rough token estimate
        tokens_per_sec = tokens_estimate / processing_time if processing_time > 0 else 0

        self.logger.debug(
            f"LLM performance metrics - Processing: {processing_time:.2f}s, "
            f"Chars/sec: {chars_per_sec:.1f}, Tokens/sec: {tokens_per_sec:.1f}"
        )

    def _call_response_callbacks(self, llm_response: LLMResponse) -> None:
        """Call response callbacks.

        Args:
            llm_response: The LLMResponse to pass to callbacks
        """
        if self._response_callback:
            self._response_callback(llm_response)

    def _process_remaining_requests(self) -> None:
        """Process any remaining requests in the queue before stopping."""
        remaining_requests = 0

        while not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                self._process_llm_request(request)
                self.request_queue.task_done()
                remaining_requests += 1
            except queue.Empty:
                break
            except Exception:
                self.logger.exception("Error processing remaining LLM request")

        if remaining_requests > 0:
            self.logger.info(f"Processed {remaining_requests} remaining LLM requests during shutdown")
        else:
            self.logger.debug("No remaining LLM requests to process during shutdown")

    def clear_queue(self) -> int:
        """Clear all pending LLM requests from the queue.

        Returns:
            Number of requests that were cleared
        """
        cleared_count = 0

        with self._cleanup_lock:
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                    self.request_queue.task_done()
                    cleared_count += 1
                except queue.Empty:
                    break

            if cleared_count > 0:
                self.logger.info(f"Cleared {cleared_count} LLM requests from queue")

        return cleared_count

    def _cleanup_llm_resources(self) -> None:
        """Clean up LLM processing resources."""
        with self._cleanup_lock:
            try:
                # Clear any remaining queue items
                self.clear_queue()

                # Reset statistics
                self._requests_processed = 0
                self._successful_requests = 0
                self._failed_requests = 0
                self._total_processing_time = 0.0
                self._last_processing_time = 0.0

                self.logger.debug("LLM processing resources cleaned up")

            except Exception:
                self.logger.exception("Error during LLM processing cleanup")

    def get_processing_stats(self) -> dict:
        """Get current LLM processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            "is_processing": self._processing,
            "is_connected": self._is_connected,
            "endpoint": self.endpoint,
            "model": self.model,
            "requests_processed": self._requests_processed,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "queue_size": self.request_queue.qsize(),
            "average_processing_time": self.average_processing_time,
            "last_processing_time": self._last_processing_time,
            "total_processing_time": self._total_processing_time,
            "success_rate": self._successful_requests / max(self._requests_processed, 1),
            "last_connection_check": self._last_connection_check.isoformat() if self._last_connection_check else None,
        }

    def cleanup(self) -> None:
        """Clean up thread resources."""
        self.stop_processing()
        self._cleanup_llm_resources()
        self.logger.info("LLMThread cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
