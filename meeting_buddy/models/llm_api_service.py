"""LLM API Service for Meeting Buddy application.

This module contains the LLMApiService class that centralizes
all LLM API communication logic with Ollama.
"""

import json
import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Optional

import requests

from .llm_model import LLMRequest, LLMResponse


class LLMApiService:
    """Service class for centralized LLM API communication.

    This class handles all direct API communication with Ollama,
    providing a clean interface for making LLM requests.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:11434/api/generate",
        model: str = "llama3.2:latest",
        api_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the LLMApiService.

        Args:
            endpoint: Ollama API endpoint URL
            model: Model name to use for generation
            api_timeout: Timeout for API calls in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.endpoint = endpoint
        self.model = model
        self.api_timeout = api_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Connection state
        self._is_connected = False
        self._last_connection_check: Optional[datetime] = None

        self.logger.info(f"LLMApiService initialized: endpoint={endpoint}, model={model}")

    def update_config(self, **kwargs) -> bool:
        """Update service configuration.

        Args:
            **kwargs: Configuration parameters to update
                     (endpoint, model, api_timeout, max_retries, retry_delay)

        Returns:
            True if configuration was updated successfully, False otherwise
        """
        config_changed = False

        for key, value in kwargs.items():
            if hasattr(self, key) and getattr(self, key) != value:
                setattr(self, key, value)
                config_changed = True
                self.logger.info(f"Updated {key}: {value}")

        if config_changed:
            # Reset connection state to force recheck with new config
            self._is_connected = False
            self._last_connection_check = None

        return config_changed

    def update_model(self, model: str) -> bool:
        """Update the Ollama model to use.

        Args:
            model: New model name to use

        Returns:
            True if model was updated, False if same model
        """
        if self.model != model:
            old_model = self.model
            self.model = model
            self.logger.info(f"Model updated from {old_model} to {model}")
            return True
        return False

    def update_endpoint(self, endpoint: str) -> bool:
        """Update the Ollama API endpoint.

        Args:
            endpoint: New endpoint URL

        Returns:
            True if endpoint was updated, False if same endpoint
        """
        if self.endpoint != endpoint:
            old_endpoint = self.endpoint
            self.endpoint = endpoint
            # Reset connection state to force recheck with new endpoint
            self._is_connected = False
            self._last_connection_check = None
            self.logger.info(f"Endpoint updated from {old_endpoint} to {endpoint}")
            return True
        return False

    def get_config(self) -> dict:
        """Get current service configuration.

        Returns:
            Dictionary containing current configuration
        """
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "api_timeout": self.api_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "is_connected": self._is_connected,
            "last_connection_check": self._last_connection_check.isoformat() if self._last_connection_check else None,
        }

    @property
    def is_connected(self) -> bool:
        """Check if connected to Ollama API."""
        return self._is_connected

    @property
    def last_connection_check(self) -> Optional[datetime]:
        """Get timestamp of last connection check."""
        return self._last_connection_check

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

            self.logger.debug("Successfully connected to Ollama API")
            return True

        except Exception as e:
            self._is_connected = False
            self._last_connection_check = datetime.now()
            self.logger.debug(f"Failed to connect to Ollama API: {e}")
            return False

    def make_request(self, request: LLMRequest) -> LLMResponse:
        """Make a complete LLM request and return the response.

        Args:
            request: LLMRequest to process

        Returns:
            LLMResponse with the complete response

        Raises:
            requests.RequestException: If the API request fails
            ValueError: If the response is invalid
        """
        self.logger.debug(f"Making LLM request: prompt_len={len(request.full_prompt)}")

        # Prepare API payload
        payload = {
            "model": self.model,
            "prompt": request.full_prompt,
            "stream": False,  # Non-streaming for complete response
        }
        headers = {"Content-Type": "application/json"}

        # Make API call
        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.api_timeout)
        response.raise_for_status()

        # Parse response
        response_data = response.json()
        response_text = response_data.get("response", "").strip()

        if not response_text:
            raise ValueError("Empty response received from API")

        # Create and return LLMResponse
        return LLMResponse(text=response_text, timestamp=datetime.now(), request_id=str(id(request)), is_complete=True)

    def make_streaming_request(self, request: LLMRequest) -> Iterator[str]:
        """Make a streaming LLM request and yield response chunks.

        Args:
            request: LLMRequest to process

        Yields:
            str: Response chunks as they arrive

        Raises:
            requests.RequestException: If the API request fails
        """
        self.logger.debug(f"Making streaming LLM request: prompt_len={len(request.full_prompt)}")

        # Prepare API payload
        payload = {
            "model": self.model,
            "prompt": request.full_prompt,
            "stream": True,  # Enable streaming
        }
        headers = {"Content-Type": "application/json"}

        # Make streaming API call
        with requests.post(
            self.endpoint, json=payload, headers=headers, stream=True, timeout=self.api_timeout
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode("utf-8")
                        data = json.loads(decoded_line)
                        response_chunk = data.get("response", "")

                        if response_chunk:
                            yield response_chunk

                        if data.get("done", False):
                            break

                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON response line: {e}")
                        continue

    def make_request_with_retries(self, request: LLMRequest) -> LLMResponse:
        """Make an LLM request with automatic retries.

        Args:
            request: LLMRequest to process

        Returns:
            LLMResponse with the complete response

        Raises:
            requests.RequestException: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.debug(f"Retrying LLM request (attempt {attempt + 1}/{self.max_retries + 1})")
                    import time

                    time.sleep(self.retry_delay)

                return self.make_request(request)

            except requests.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    self.logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
                else:
                    self.logger.exception(f"LLM request failed after {self.max_retries + 1} attempts")

        # If we get here, all retries failed
        raise last_exception

    def make_streaming_request_with_retries(self, request: LLMRequest) -> Iterator[str]:
        """Make a streaming LLM request with automatic retries.

        Args:
            request: LLMRequest to process

        Yields:
            str: Response chunks as they arrive

        Raises:
            requests.RequestException: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.debug(f"Retrying streaming LLM request (attempt {attempt + 1}/{self.max_retries + 1})")
                    import time

                    time.sleep(self.retry_delay)

                yield from self.make_streaming_request(request)
                return  # Success, exit retry loop

            except requests.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    self.logger.warning(f"Streaming LLM request failed (attempt {attempt + 1}): {e}")
                else:
                    self.logger.exception(f"Streaming LLM request failed after {self.max_retries + 1} attempts")

        # If we get here, all retries failed
        raise last_exception

    def get_available_models(self) -> list[str]:
        """Get list of available models from Ollama.

        Returns:
            List of available model names

        Raises:
            requests.RequestException: If the API request fails
        """
        try:
            tags_url = self.endpoint.replace("/api/generate", "/api/tags")
            response = requests.get(tags_url, timeout=5.0)
            response.raise_for_status()

            data = response.json()
            models = [model["name"] for model in data.get("models", [])]

            self.logger.debug(f"Available models: {models}")
            return models

        except Exception:
            self.logger.exception("Failed to get available models")
            raise

    def update_configuration(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        api_timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> None:
        """Update service configuration.

        Args:
            endpoint: New API endpoint URL
            model: New model name
            api_timeout: New API timeout
            max_retries: New max retries
            retry_delay: New retry delay
        """
        if endpoint is not None:
            self.endpoint = endpoint
            self.logger.info(f"Updated endpoint: {endpoint}")

        if model is not None:
            self.model = model
            self.logger.info(f"Updated model: {model}")

        if api_timeout is not None:
            self.api_timeout = api_timeout
            self.logger.debug(f"Updated API timeout: {api_timeout}s")

        if max_retries is not None:
            self.max_retries = max_retries
            self.logger.debug(f"Updated max retries: {max_retries}")

        if retry_delay is not None:
            self.retry_delay = retry_delay
            self.logger.debug(f"Updated retry delay: {retry_delay}s")

        # Reset connection status to force recheck
        self._is_connected = False
        self._last_connection_check = None
