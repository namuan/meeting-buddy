"""LLM Model for Meeting Buddy application.

This module contains the LLMModel class that handles
LLM API communication with Ollama for processing transcription data.
"""

import logging
from datetime import datetime
from typing import Callable, Optional


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
    """Model class for managing LLM data.

    This class handles LLM request and response data storage.
    Heavy processing (API calls) is handled by LLMThread in utils.
    """

    def __init__(self):
        """Initialize the LLMModel.

        Note: Heavy processing (API calls) is handled by LLMThread.
        This model only manages LLM data and state.
        """
        self.logger = logging.getLogger(__name__)

        # Request and response management
        self._requests: list[LLMRequest] = []
        self._responses: list[LLMResponse] = []
        self._current_response: str = ""
        self._user_prompt: str = ""

        # Callbacks for state change notifications
        self._response_callback: Optional[Callable[[str], None]] = None
        self._response_chunk_callback: Optional[Callable[[str], None]] = None
        self._error_callback: Optional[Callable[[str], None]] = None
        self._connection_status_callback: Optional[Callable[[bool], None]] = None

        self.logger.info("LLMModel initialized (data container only)")

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
    def request_count(self) -> int:
        """Get the number of requests stored."""
        return len(self._requests)

    @property
    def response_count(self) -> int:
        """Get the number of responses stored."""
        return len(self._responses)

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

    def add_request(self, request: LLMRequest) -> None:
        """Add an LLM request to storage.

        Args:
            request: LLMRequest to store
        """
        self._requests.append(request)
        self.logger.debug(
            f"LLM request added: prompt_len={len(request.prompt)}, transcription_len={len(request.transcription_text)}"
        )

    def add_response(self, response: LLMResponse) -> None:
        """Add an LLM response to storage.

        Args:
            response: LLMResponse to store
        """
        self._responses.append(response)

        # Update current response if this is a complete response
        if response.is_complete:
            self._current_response = response.text

        self.logger.debug(f"LLM response added: text_len={len(response.text)}, complete={response.is_complete}")

        # Notify observers
        if response.is_complete and self._response_callback:
            self._response_callback(response.text)
        if self._response_chunk_callback:
            self._response_chunk_callback(response.text)

    def update_current_response(self, response_text: str) -> None:
        """Update the current response text.

        Args:
            response_text: New response text
        """
        self._current_response = response_text
        self.logger.debug(f"Current response updated: length={len(response_text)}")

        # Notify observers
        if self._response_callback:
            self._response_callback(self._current_response)

    def append_response_chunk(self, chunk: str) -> None:
        """Append a chunk to the current response.

        Args:
            chunk: Response chunk to append
        """
        self._current_response += chunk
        self.logger.debug(f"Response chunk appended: chunk_len={len(chunk)}, total_len={len(self._current_response)}")

        # Notify observers
        if self._response_chunk_callback:
            self._response_chunk_callback(chunk)

    def clear_responses(self) -> None:
        """Clear all response data."""
        self._responses.clear()
        self._current_response = ""
        self.logger.info("Cleared all LLM response data")

    def clear_requests(self) -> None:
        """Clear all request data."""
        self._requests.clear()
        self.logger.info("Cleared all LLM request data")

    def clear_all(self) -> None:
        """Clear all LLM data."""
        self.clear_requests()
        self.clear_responses()
        self.logger.info("Cleared all LLM data")

    def get_recent_requests(self, count: int = 10) -> list[LLMRequest]:
        """Get the most recent requests.

        Args:
            count: Number of recent requests to return

        Returns:
            List of recent LLMRequest objects
        """
        return self._requests[-count:] if len(self._requests) >= count else self._requests.copy()

    def get_recent_responses(self, count: int = 10) -> list[LLMResponse]:
        """Get the most recent responses.

        Args:
            count: Number of recent responses to return

        Returns:
            List of recent LLMResponse objects
        """
        return self._responses[-count:] if len(self._responses) >= count else self._responses.copy()

    def notify_connection_status(self, is_connected: bool) -> None:
        """Notify observers of connection status change.

        Args:
            is_connected: Current connection status
        """
        self.logger.debug(f"Connection status updated: {is_connected}")
        if self._connection_status_callback:
            self._connection_status_callback(is_connected)

    def notify_error(self, error_message: str) -> None:
        """Notify observers of an error.

        Args:
            error_message: Error message to report
        """
        self.logger.debug(f"Error notification: {error_message}")
        if self._error_callback:
            self._error_callback(error_message)

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
            import json

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
        self.clear_all()
        self.logger.info("LLMModel cleanup completed")
