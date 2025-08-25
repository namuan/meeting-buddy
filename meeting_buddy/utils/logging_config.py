"""Logging configuration for Meeting Buddy application.

This module provides centralized logging configuration
for the entire application.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Global logging configuration storage
_logging_config: dict[str, Any] = {}


def _store_logging_config(config: dict[str, Any]) -> None:
    """Store logging configuration for global access."""
    global _logging_config
    _logging_config.update(config)


def get_logging_config() -> dict[str, Any]:
    """Get current logging configuration."""
    return _logging_config.copy()


class StructuredFormatter(logging.Formatter):
    """Enhanced formatter with data redaction capabilities."""

    def __init__(self, fmt=None, datefmt=None, redact_sensitive=True):
        super().__init__(fmt, datefmt)
        self.redact_sensitive = redact_sensitive
        self.sensitive_patterns = [
            (
                re.compile(r"(password|token|key|secret|api_key|auth|credential)=[^\s]+", re.IGNORECASE),
                r"\1=[REDACTED]",
            ),
            (re.compile(r"(prompt|transcription_text)=([^|]+)", re.IGNORECASE), self._redact_long_text),
        ]

    def _redact_long_text(self, match):
        """Redact long text fields."""
        key = match.group(1)
        value = match.group(2).strip()
        if len(value) > 50:
            return f"{key}={value[:20]}...{value[-10:]} (len={len(value)})"
        return match.group(0)

    def format(self, record):
        """Format log record with optional data redaction."""
        formatted = super().format(record)

        if self.redact_sensitive:
            for pattern, replacement in self.sensitive_patterns:
                if callable(replacement):
                    formatted = pattern.sub(replacement, formatted)
                else:
                    formatted = pattern.sub(replacement, formatted)

        return formatted


def _setup_trace_level() -> None:
    """Setup TRACE logging level if not already defined."""
    if not hasattr(logging, "TRACE"):
        logging.TRACE = 5
        logging.addLevelName(logging.TRACE, "TRACE")

        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(logging.TRACE):
                self._log(logging.TRACE, message, args, **kwargs)

        logging.Logger.trace = trace


def _get_logging_level(verbosity: int) -> int:
    """Get logging level based on verbosity."""
    if verbosity == 1:
        return logging.INFO
    elif verbosity == 2:
        return logging.DEBUG
    elif verbosity >= 3:
        return logging.TRACE
    else:
        return logging.WARNING


def _create_formatters(enable_structured: bool, redact_sensitive: bool) -> tuple:
    """Create logging formatters."""
    if enable_structured:
        detailed_formatter = StructuredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            redact_sensitive=redact_sensitive,
        )
        simple_formatter = StructuredFormatter(
            fmt="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            redact_sensitive=redact_sensitive,
        )
    else:
        detailed_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        simple_formatter = logging.Formatter(
            fmt="%(asctime)s - %(filename)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    return detailed_formatter, simple_formatter


def setup_logging(
    verbosity: int = 0,
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    enable_structured_logging: bool = True,
    redact_sensitive_data: bool = True,
) -> None:
    """Setup logging configuration for the application.

    Args:
        verbosity: Logging verbosity level (0=WARNING, 1=INFO, 2=DEBUG, 3=TRACE)
        log_file: Optional path to log file. If None, no file logging.
        log_to_console: Whether to log to console (default: True)
        enable_structured_logging: Whether to enable structured logging features
        redact_sensitive_data: Whether to redact sensitive data in logs
    """
    # Setup TRACE level and get logging level
    _setup_trace_level()
    logging_level = _get_logging_level(verbosity)

    # Store configuration for global access
    _store_logging_config({
        "verbosity": verbosity,
        "enable_structured_logging": enable_structured_logging,
        "redact_sensitive_data": redact_sensitive_data,
        "log_file": log_file,
        "log_to_console": log_to_console,
    })

    # Create formatters
    detailed_formatter, simple_formatter = _create_formatters(enable_structured_logging, redact_sensitive_data)

    # Create handlers list
    handlers = []

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging_level)
        console_handler.setFormatter(simple_formatter)
        handlers.append(console_handler)

    # File handler
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(detailed_formatter)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging to {log_file}: {e}", file=sys.stderr)

    # Configure root logger
    logging.basicConfig(
        level=logging_level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Capture warnings
    logging.captureWarnings(capture=True)

    # Set specific logger levels for third-party libraries
    _configure_third_party_loggers()

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {logging.getLevelName(logging_level)}")
    if log_file:
        logger.info(f"File logging enabled: {log_file}")
    if log_to_console:
        logger.info("Console logging enabled")


def _configure_third_party_loggers() -> None:
    """Configure logging levels for third-party libraries."""
    # Reduce noise from PyQt6
    logging.getLogger("PyQt6").setLevel(logging.WARNING)

    # Reduce noise from PyAudio (if it logs)
    logging.getLogger("pyaudio").setLevel(logging.WARNING)

    # Reduce noise from other common libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """Set the logging level for all loggers.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    try:
        numeric_level = getattr(logging, level.upper())
        logging.getLogger().setLevel(numeric_level)

        # Update all handlers
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(numeric_level)

        logger = logging.getLogger(__name__)
        logger.info(f"Log level changed to: {level.upper()}")
    except AttributeError:
        logger = logging.getLogger(__name__)
        logger.exception(f"Invalid log level: {level}")


def add_file_handler(log_file: str, level: str = "DEBUG") -> bool:
    """Add a file handler to the root logger.

    Args:
        log_file: Path to log file
        level: Logging level for the file handler

    Returns:
        True if handler was added successfully, False otherwise
    """
    try:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))

        # Use detailed formatter for file
        detailed_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(detailed_formatter)

        # Add to root logger
        logging.getLogger().addHandler(file_handler)

        logger = logging.getLogger(__name__)
        logger.info(f"File handler added: {log_file}")
        return True

    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception(f"Failed to add file handler {log_file}")
        return False


def remove_all_handlers() -> None:
    """Remove all handlers from the root logger."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get a logger instance for this class.

        Returns:
            Logger instance named after the class module and name
        """
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger


def get_verbosity_level() -> int:
    """Get current verbosity level from configuration."""
    return _logging_config.get("verbosity", 1)


def is_structured_logging_enabled() -> bool:
    """Check if structured logging is enabled."""
    return _logging_config.get("enable_structured_logging", True)


def is_sensitive_data_redaction_enabled() -> bool:
    """Check if sensitive data redaction is enabled."""
    return _logging_config.get("redact_sensitive_data", True)


# Default configuration for the application
def setup_default_logging() -> None:
    """Setup default logging configuration for the application."""
    setup_logging(
        verbosity=1,  # INFO level by default
        log_to_console=True,
        log_file=None,  # No file logging by default
        enable_structured_logging=True,
        redact_sensitive_data=True,
    )
