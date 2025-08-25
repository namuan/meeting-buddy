"""Logging configuration for Meeting Buddy application.

This module provides centralized logging configuration
for the entire application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(verbosity: int = 0, log_file: Optional[str] = None, log_to_console: bool = True) -> None:
    """Setup logging configuration for the application.

    Args:
        verbosity: Logging verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
        log_file: Optional path to log file. If None, no file logging.
        log_to_console: Whether to log to console (default: True)
    """
    # Determine logging level based on verbosity
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        fmt="%(asctime)s - %(filename)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

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


# Default configuration for the application
def setup_default_logging() -> None:
    """Setup default logging configuration for the application."""
    setup_logging(
        verbosity=1,  # INFO level by default
        log_to_console=True,
        log_file=None,  # No file logging by default
    )
