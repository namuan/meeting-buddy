"""LLM Model for Meeting Buddy application.

This module contains the LLMModel class that handles
LLM API communication with Ollama for processing transcription data.
"""

import json
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import requests


class LLMRequest:
    """Data class representing an LLM request."""

    def __init__(self, prompt: str, transcription_text: str, timestamp: datetime):
        self.prompt = prompt
        self.transcription_text = transcription_text
        self.timestamp = timestamp
        self.processed = False
        self.response_text = ""
        self.error_message = ""

    def __str__(self) -> str:
        return f"LLMRequest({self.formatted_timestamp}, prompt_len={len(self.prompt)}, transcription_len={len(self.transcription_text)})"

    def __repr__(self) -> str:
        return f"LLMRequest(prompt='{self.prompt[:30]}...', transcription='{self.transcription_text[:30]}...')"

    @property
    def formatted_timestamp(self) -> str:
        """Get formatted timestamp string."""
        return self.timestamp.strftime("%H:%M:%S")

    @property
    def full_prompt(self) -> str:
        """Get the full prompt combining user prompt and transcription."""
        if self.transcription_text.strip():
            return f"{self.prompt}\n\nTranscription: {self.transcription_text}"
        return self.prompt


class LLMResponse:
    """Data class representing an LLM response."""

    def __init__(self, text: str, timestamp: datetime, request_id: Optional[str] = None, is_complete: bool = True):
        self.text = text
        self.timestamp = timestamp
        self.request_id = request_id
        self.is_complete = is_complete

    def __str__(self) -> str:
        return f"{self.formatted_timestamp}: {self.text[:100]}{'...' if len(self.text) > 100 else ''}"

    def __repr__(self) -> str:
        return f"LLMResponse(text='{self.text[:50]}...', complete={self.is_complete})"

    @property
    def formatted_timestamp(self) -> str:
        """Get formatted timestamp string."""
        return self.timestamp.strftime("%H:%M:%S")


class LLMModel:
    """Model class for managing LLM API communication with Ollama.

    This class handles LLM requests, streaming responses,
    and integration with transcription data.
    """

    def __init__(self, endpoint: str = "http://localhost:11434/api/generate", model: str = "llama3.2:latest"):
        """Initialize the LLMModel.

        Args:
            endpoint: Ollama API endpoint URL
            model: Model name to use for generation
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.endpoint = endpoint
        self.model = model
        self.timeout = 30.0
        self.max_retries = 3

        # Request management
        self._requests: list[LLMRequest] = []
        self._responses: list[LLMResponse] = []
        self._current_response: str = ""
        self._user_prompt: str = ""

        # Processing state
        self._is_processing: bool = False
        self._processing_thread: Optional[threading.Thread] = None
        self._request_queue: queue.Queue = queue.Queue()
        self._stop_processing: threading.Event = threading.Event()

        # Connection state
        self._is_connected: bool = False
        self._last_connection_check: Optional[datetime] = None

        # Callbacks
        self._response_callback: Optional[Callable[[str], None]] = None
        self._response_chunk_callback: Optional[Callable[[str], None]] = None
        self._error_callback: Optional[Callable[[str], None]] = None
        self._connection_status_callback: Optional[Callable[[bool], None]] = None

        self.logger.debug(
            f"LLM model initialized: endpoint={self.endpoint}, model={self.model}, timeout={self.timeout}s"
        )

    @property
    def requests(self) -> list[LLMRequest]:
        """Get list of all LLM requests."""
        return self._requests.copy()

    @property
    def responses(self) -> list[LLMResponse]:
        """Get list of all LLM responses."""
        return self._responses.copy()

    @property
    def current_response(self) -> str:
        """Get current accumulated response text."""
        return self._current_response

    @property
    def user_prompt(self) -> str:
        """Get current user prompt."""
        return self._user_prompt

    @property
    def is_processing(self) -> bool:
        """Check if currently processing requests."""
        return self._is_processing

    @property
    def is_connected(self) -> bool:
        """Check if connected to Ollama API."""
        return self._is_connected

    def set_user_prompt(self, prompt: str) -> None:
        """Set the user prompt for LLM processing.

        Args:
            prompt: User-defined prompt text
        """
        self._user_prompt = prompt.strip()
        self.logger.info(f"User prompt updated: length={len(self._user_prompt)} chars")
        self.logger.debug(
            f"User prompt content: '{self._user_prompt[:100]}{'...' if len(self._user_prompt) > 100 else ''}'"
        )

    def set_response_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for complete response updates.

        Args:
            callback: Function to call when response is updated
        """
        self._response_callback = callback
        self.logger.debug("Response callback set")

    def set_response_chunk_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for streaming response chunks.

        Args:
            callback: Function to call for each response chunk
        """
        self._response_chunk_callback = callback
        self.logger.debug("Response chunk callback set")

    def set_error_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for error notifications.

        Args:
            callback: Function to call when errors occur
        """
        self._error_callback = callback
        self.logger.debug("Error callback set")

    def set_connection_status_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback function for connection status changes.

        Args:
            callback: Function to call when connection status changes
        """
        self._connection_status_callback = callback
        self.logger.debug("Connection status callback set")

    def check_connection(self) -> bool:
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

            self.logger.info("Successfully connected to Ollama API")

            if self._connection_status_callback:
                self._connection_status_callback(True)

            return True

        except Exception as e:
            self._is_connected = False
            error_msg = f"Failed to connect to Ollama API: {e}"
            self.logger.exception("Failed to connect to Ollama API")

            if self._connection_status_callback:
                self._connection_status_callback(False)
            if self._error_callback:
                self._error_callback(error_msg)

            return False

    def process_transcription(self, transcription_text: str) -> bool:
        """Process transcription text with LLM.

        Args:
            transcription_text: Text from transcription to process

        Returns:
            True if request was queued successfully, False otherwise
        """
        if not self._user_prompt.strip():
            self.logger.warning("No user prompt set, skipping LLM processing")
            return False

        if not transcription_text.strip():
            self.logger.debug("Empty transcription text, skipping LLM processing")
            return False

        # Create LLM request
        request = LLMRequest(self._user_prompt, transcription_text, datetime.now())
        self._requests.append(request)

        self.logger.info(
            f"Created LLM request: prompt_len={len(self._user_prompt)}, transcription_len={len(transcription_text)}"
        )

        # Add to processing queue if processing is active
        if self._is_processing:
            try:
                self._request_queue.put(request)
                self.logger.debug(f"Added request to processing queue: queue size = {self._request_queue.qsize()}")
                return True
            except Exception:
                self.logger.exception("Failed to add request to processing queue")
                return False
        else:
            self.logger.debug("Processing not active, request stored but not queued")
            return False

    def start_processing(self) -> bool:
        """Start background processing of LLM requests.

        Returns:
            True if processing started successfully, False otherwise
        """
        if not self.check_connection():
            self.logger.error("Cannot start processing: No connection to Ollama API")
            return False

        if self._is_processing:
            self.logger.warning("Processing already active")
            return True

        self._is_processing = True
        self._stop_processing.clear()

        self._processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self._processing_thread.start()

        self.logger.info("Started LLM processing")
        self.logger.debug(
            f"Processing thread started: thread_id={self._processing_thread.ident}, daemon={self._processing_thread.daemon}"
        )
        return True

    def stop_processing(self) -> None:
        """Stop background processing of LLM requests."""
        if not self._is_processing:
            self.logger.debug("Processing not active")
            return

        self._stop_processing.set()
        self._is_processing = False

        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)

        self.logger.info("Stopped LLM processing")

    def _processing_worker(self) -> None:
        """Background worker thread for processing LLM requests."""
        self.logger.debug("LLM processing worker started")
        processed_count = 0
        start_time = time.time()

        while not self._stop_processing.is_set():
            try:
                # Get request from queue with timeout
                queue_wait_start = time.time()
                request = self._request_queue.get(timeout=1.0)
                queue_wait_time = time.time() - queue_wait_start

                self.logger.debug(
                    f"Retrieved request from queue: wait_time={queue_wait_time:.3f}s, queue_size={self._request_queue.qsize()}"
                )

                self._process_request(request)
                self._request_queue.task_done()
                processed_count += 1

                # Log processing statistics every 5 requests
                if processed_count % 5 == 0:
                    elapsed_time = time.time() - start_time
                    avg_processing_time = elapsed_time / processed_count if processed_count > 0 else 0
                    self.logger.debug(
                        f"LLM processing statistics: {processed_count} requests processed, avg_time={avg_processing_time:.3f}s/request"
                    )
            except queue.Empty:
                continue
            except Exception:
                self.logger.exception("Error in LLM processing worker")

        total_time = time.time() - start_time
        self.logger.info(f"LLM processing worker stopped: processed {processed_count} requests in {total_time:.2f}s")
        if processed_count > 0:
            self.logger.debug(f"Final LLM processing statistics: avg_time={total_time / processed_count:.3f}s/request")

    def _process_request(self, request: LLMRequest) -> None:
        """Process a single LLM request.

        Args:
            request: LLMRequest to process
        """
        if request.processed:
            return

        try:
            processing_start = time.time()
            complete_response = self._make_api_request(request)
            processing_time = time.time() - processing_start

            if complete_response.strip():
                self._handle_successful_request(request, complete_response, processing_time)
            else:
                self._handle_empty_request(request)

        except requests.exceptions.RequestException as e:
            self._handle_request_exception(request, e, "LLM API request failed")
        except Exception as e:
            self._handle_request_exception(request, e, "Error processing LLM request")

    def _make_api_request(self, request: LLMRequest) -> str:
        """Make the API request and return the complete response.

        Args:
            request: LLMRequest to process

        Returns:
            Complete response text
        """
        # Prepare API payload
        payload = {"model": self.model, "prompt": request.full_prompt, "stream": True}
        headers = {"Content-Type": "application/json"}

        self.logger.debug(f"Making LLM API call: model={self.model}, prompt_len={len(request.full_prompt)}")

        complete_response = ""

        # Make streaming API call
        with requests.post(self.endpoint, json=payload, headers=headers, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line and not self._stop_processing.is_set():
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

        return complete_response

    def _handle_successful_request(self, request: LLMRequest, complete_response: str, processing_time: float) -> None:
        """Handle a successful LLM request.

        Args:
            request: The original request
            complete_response: The complete response text
            processing_time: Time taken to process
        """
        # Create response object
        llm_response = LLMResponse(text=complete_response.strip(), timestamp=datetime.now(), is_complete=True)

        # Update request and add response
        request.processed = True
        request.response_text = complete_response.strip()
        self._responses.append(llm_response)

        # Update current response
        self._current_response = complete_response.strip()

        self.logger.info(f"LLM response received: length={len(complete_response)} chars, time={processing_time:.2f}s")

        # Log response statistics
        response_stats = {
            "processing_time_s": processing_time,
            "response_length": len(complete_response),
            "prompt_length": len(request.full_prompt),
            "total_responses": len(self._responses),
        }
        self.logger.debug(f"LLM response statistics: {response_stats}")

        # Call response callback
        if self._response_callback:
            self.logger.debug("Calling LLM response callback")
            self._response_callback(self._current_response)
        else:
            self.logger.debug("No LLM response callback set")

    def _handle_empty_request(self, request: LLMRequest) -> None:
        """Handle an empty LLM response.

        Args:
            request: The original request
        """
        self.logger.warning("Received empty response from LLM API")
        request.error_message = "Empty response received"

    def _handle_request_exception(self, request: LLMRequest, error: Exception, log_message: str) -> None:
        """Handle exceptions during request processing.

        Args:
            request: The original request
            error: The exception that occurred
            log_message: Message to log
        """
        error_msg = f"{log_message}: {error}"
        self.logger.exception(log_message)
        request.error_message = error_msg

        if self._error_callback:
            self._error_callback(error_msg)

    def clear_responses(self) -> None:
        """Clear all LLM responses and requests."""
        self._requests.clear()
        self._responses.clear()
        self._current_response = ""

        # Clear queue
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
                self._request_queue.task_done()
            except queue.Empty:
                break

        self.logger.info("Cleared all LLM data")

    def get_responses_by_timerange(self, start_time: datetime, end_time: datetime) -> list[LLMResponse]:
        """Get LLM responses within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of LLMResponse objects within the time range
        """
        results = []
        for response in self._responses:
            if start_time <= response.timestamp <= end_time:
                results.append(response)
        return results

    def export_conversation(self, format_type: str = "text") -> str:
        """Export conversation in specified format.

        Args:
            format_type: Export format ('text', 'timestamped', 'json')

        Returns:
            Formatted conversation string
        """
        if format_type == "text":
            lines = []
            for i, (request, response) in enumerate(zip(self._requests, self._responses)):
                lines.append(f"Request {i + 1}: {request.full_prompt}")
                lines.append(f"Response {i + 1}: {response.text}")
                lines.append("---")
            return "\n".join(lines)

        elif format_type == "timestamped":
            lines = []
            for request, response in zip(self._requests, self._responses):
                lines.append(f"[{request.formatted_timestamp}] Request: {request.full_prompt}")
                lines.append(f"[{response.formatted_timestamp}] Response: {response.text}")
                lines.append("")
            return "\n".join(lines)

        elif format_type == "json":
            data = {
                "conversation": [
                    {
                        "request": {"prompt": req.full_prompt, "timestamp": req.timestamp.isoformat()},
                        "response": {"text": resp.text, "timestamp": resp.timestamp.isoformat()},
                    }
                    for req, resp in zip(self._requests, self._responses)
                ],
                "current_response": self._current_response,
                "user_prompt": self._user_prompt,
            }
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_processing()
        self.clear_responses()
        self.logger.info("LLMModel cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
