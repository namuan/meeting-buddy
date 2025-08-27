"""Configuration Model for Meeting Buddy application.

This module contains the ConfigurationModel class that handles
app settings and model configurations for Whisper and Ollama models.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional


class ConfigurationData:
    """Data class representing configuration settings."""

    def __init__(
        self,
        whisper_model: str = "base",
        ollama_model: str = "llama3.2:latest",
        ollama_endpoint: str = "http://localhost:11434/api/generate",
        api_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        language: Optional[str] = None,
        auto_save: bool = True,
    ):
        """Initialize configuration data.

        Args:
            whisper_model: Whisper model name to use for transcription
            ollama_model: Ollama model name to use for LLM processing
            ollama_endpoint: Ollama API endpoint URL
            api_timeout: Timeout for API calls in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            language: Language code for transcription (None for auto-detect)
            auto_save: Whether to automatically save configuration changes
        """
        self.whisper_model = whisper_model
        self.ollama_model = ollama_model
        self.ollama_endpoint = ollama_endpoint
        self.api_timeout = api_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.language = language
        self.auto_save = auto_save
        self.last_modified = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "whisper_model": self.whisper_model,
            "ollama_model": self.ollama_model,
            "ollama_endpoint": self.ollama_endpoint,
            "api_timeout": self.api_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "language": self.language,
            "auto_save": self.auto_save,
            "last_modified": self.last_modified.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigurationData":
        """Create configuration from dictionary."""
        config = cls(
            whisper_model=data.get("whisper_model", "base"),
            ollama_model=data.get("ollama_model", "llama3.2:latest"),
            ollama_endpoint=data.get("ollama_endpoint", "http://localhost:11434/api/generate"),
            api_timeout=data.get("api_timeout", 30.0),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
            language=data.get("language"),
            auto_save=data.get("auto_save", True),
        )

        # Parse last_modified if present
        if "last_modified" in data:
            try:
                config.last_modified = datetime.fromisoformat(data["last_modified"])
            except (ValueError, TypeError):
                config.last_modified = datetime.now()

        return config

    def __str__(self) -> str:
        return f"ConfigurationData(whisper={self.whisper_model}, ollama={self.ollama_model})"

    def __repr__(self) -> str:
        return f"ConfigurationData(whisper_model='{self.whisper_model}', ollama_model='{self.ollama_model}')"


class ConfigurationModel:
    """Model class for managing application configuration.

    This class handles loading, saving, and managing configuration settings
    for Whisper and Ollama models, following the MVP architecture pattern.
    """

    # Available Whisper models
    AVAILABLE_WHISPER_MODELS: ClassVar[list[str]] = ["tiny", "base", "small", "medium", "large", "turbo"]

    # Default configuration file name
    DEFAULT_CONFIG_FILE = "meeting_buddy_config.json"

    def __init__(self, config_file_path: Optional[str] = None):
        """Initialize the ConfigurationModel.

        Args:
            config_file_path: Path to configuration file (defaults to user home directory)
        """
        self.logger = logging.getLogger(__name__)

        # Set up configuration file path
        if config_file_path:
            self.config_file_path = Path(config_file_path)
        else:
            self.config_file_path = Path.home() / self.DEFAULT_CONFIG_FILE

        # Initialize configuration data
        self._config_data = ConfigurationData()

        # Callbacks for configuration changes
        self._config_changed_callbacks: list[Callable[[ConfigurationData], None]] = []
        self._whisper_model_changed_callbacks: list[Callable[[str], None]] = []
        self._ollama_model_changed_callbacks: list[Callable[[str], None]] = []

        # Load existing configuration
        self.load_configuration()

        self.logger.info(
            "ConfigurationModel initialized",
            extra={
                "config_file": str(self.config_file_path),
                "whisper_model": self._config_data.whisper_model,
                "ollama_model": self._config_data.ollama_model,
            },
        )

    @property
    def config_data(self) -> ConfigurationData:
        """Get current configuration data."""
        return self._config_data

    @property
    def whisper_model(self) -> str:
        """Get current Whisper model."""
        return self._config_data.whisper_model

    @property
    def ollama_model(self) -> str:
        """Get current Ollama model."""
        return self._config_data.ollama_model

    @property
    def available_whisper_models(self) -> list[str]:
        """Get list of available Whisper models."""
        return self.AVAILABLE_WHISPER_MODELS.copy()

    def set_whisper_model(self, model_name: str) -> bool:
        """Set Whisper model.

        Args:
            model_name: Name of the Whisper model to use

        Returns:
            True if model was set successfully, False otherwise
        """
        if model_name not in self.AVAILABLE_WHISPER_MODELS:
            self.logger.error(
                "Invalid Whisper model",
                extra={
                    "requested_model": model_name,
                    "available_models": self.AVAILABLE_WHISPER_MODELS,
                },
            )
            return False

        old_model = self._config_data.whisper_model
        self._config_data.whisper_model = model_name
        self._config_data.last_modified = datetime.now()

        self.logger.info(
            "Whisper model changed",
            extra={
                "old_model": old_model,
                "new_model": model_name,
            },
        )

        # Notify callbacks
        for callback in self._whisper_model_changed_callbacks:
            try:
                callback(model_name)
            except Exception:
                self.logger.exception("Error in whisper model change callback", extra={"callback": str(callback)})

        # Auto-save if enabled
        if self._config_data.auto_save:
            self.save_configuration()

        # Notify general config change callbacks
        self._notify_config_changed()

        return True

    def set_ollama_model(self, model_name: str) -> bool:
        """Set Ollama model.

        Args:
            model_name: Name of the Ollama model to use

        Returns:
            True if model was set successfully, False otherwise
        """
        old_model = self._config_data.ollama_model
        self._config_data.ollama_model = model_name
        self._config_data.last_modified = datetime.now()

        self.logger.info(
            "Ollama model changed",
            extra={
                "old_model": old_model,
                "new_model": model_name,
            },
        )

        # Notify callbacks
        for callback in self._ollama_model_changed_callbacks:
            try:
                callback(model_name)
            except Exception:
                self.logger.exception("Error in ollama model change callback", extra={"callback": str(callback)})

        # Auto-save if enabled
        if self._config_data.auto_save:
            self.save_configuration()

        # Notify general config change callbacks
        self._notify_config_changed()

        return True

    def update_configuration(self, **kwargs) -> bool:
        """Update multiple configuration settings.

        Args:
            **kwargs: Configuration settings to update

        Returns:
            True if configuration was updated successfully, False otherwise
        """
        try:
            for key, value in kwargs.items():
                if hasattr(self._config_data, key):
                    setattr(self._config_data, key, value)
                else:
                    self.logger.warning("Unknown configuration key", extra={"key": key, "value": value})

            self._config_data.last_modified = datetime.now()

            # Auto-save if enabled
            if self._config_data.auto_save:
                self.save_configuration()

            # Notify callbacks
            self._notify_config_changed()

            self.logger.info("Configuration updated", extra={"updated_keys": list(kwargs.keys())})

            return True

        except Exception:
            self.logger.exception("Error updating configuration", extra={"kwargs": kwargs})
            return False

    def load_configuration(self) -> bool:
        """Load configuration from file.

        Returns:
            True if configuration was loaded successfully, False otherwise
        """
        try:
            if not self.config_file_path.exists():
                self.logger.info(
                    "Configuration file not found, using defaults", extra={"config_file": str(self.config_file_path)}
                )
                return True

            with open(self.config_file_path, encoding="utf-8") as f:
                data = json.load(f)

            self._config_data = ConfigurationData.from_dict(data)

            self.logger.info(
                "Configuration loaded successfully",
                extra={
                    "config_file": str(self.config_file_path),
                    "whisper_model": self._config_data.whisper_model,
                    "ollama_model": self._config_data.ollama_model,
                },
            )

            return True

        except Exception:
            self.logger.exception("Error loading configuration", extra={"config_file": str(self.config_file_path)})
            # Use default configuration on error
            self._config_data = ConfigurationData()
            return False

    def save_configuration(self) -> bool:
        """Save configuration to file.

        Returns:
            True if configuration was saved successfully, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save configuration
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                json.dump(self._config_data.to_dict(), f, indent=2, ensure_ascii=False)

            self.logger.info("Configuration saved successfully", extra={"config_file": str(self.config_file_path)})

            return True

        except Exception:
            self.logger.exception("Error saving configuration", extra={"config_file": str(self.config_file_path)})
            return False

    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values.

        Returns:
            True if configuration was reset successfully, False otherwise
        """
        try:
            self._config_data = ConfigurationData()

            # Auto-save if enabled
            if self._config_data.auto_save:
                self.save_configuration()

            # Notify callbacks
            self._notify_config_changed()

            self.logger.info("Configuration reset to defaults")

            return True

        except Exception:
            self.logger.exception("Error resetting configuration")
            return False

    def add_config_changed_callback(self, callback: Callable[[ConfigurationData], None]) -> None:
        """Add callback for configuration changes.

        Args:
            callback: Function to call when configuration changes
        """
        self._config_changed_callbacks.append(callback)
        self.logger.debug("Configuration change callback added", extra={"callback": str(callback)})

    def add_whisper_model_changed_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for Whisper model changes.

        Args:
            callback: Function to call when Whisper model changes
        """
        self._whisper_model_changed_callbacks.append(callback)
        self.logger.debug("Whisper model change callback added", extra={"callback": str(callback)})

    def add_ollama_model_changed_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for Ollama model changes.

        Args:
            callback: Function to call when Ollama model changes
        """
        self._ollama_model_changed_callbacks.append(callback)
        self.logger.debug("Ollama model change callback added", extra={"callback": str(callback)})

    def remove_config_changed_callback(self, callback: Callable[[ConfigurationData], None]) -> bool:
        """Remove configuration change callback.

        Args:
            callback: Callback function to remove

        Returns:
            True if callback was removed, False if not found
        """
        try:
            self._config_changed_callbacks.remove(callback)
            self.logger.debug("Configuration change callback removed", extra={"callback": str(callback)})
            return True
        except ValueError:
            return False

    def _notify_config_changed(self) -> None:
        """Notify all configuration change callbacks."""
        for callback in self._config_changed_callbacks:
            try:
                callback(self._config_data)
            except Exception:
                self.logger.exception("Error in configuration change callback", extra={"callback": str(callback)})

    def get_config_summary(self) -> dict[str, Any]:
        """Get a summary of current configuration.

        Returns:
            Dictionary containing configuration summary
        """
        return {
            "whisper_model": self._config_data.whisper_model,
            "ollama_model": self._config_data.ollama_model,
            "ollama_endpoint": self._config_data.ollama_endpoint,
            "language": self._config_data.language or "auto-detect",
            "config_file": str(self.config_file_path),
            "last_modified": self._config_data.last_modified.isoformat(),
            "auto_save": self._config_data.auto_save,
        }
