"""Enhanced structured logging utilities for Meeting Buddy application.

This module provides structured logging with contextual information,
data redaction, and performance metrics.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any

from .logging_config import LoggerMixin


class StructuredLogger(LoggerMixin):
    """Enhanced logger with structured logging capabilities."""

    def __init__(self, context: dict[str, Any] | None = None):
        """Initialize structured logger with optional context.

        Args:
            context: Default context to include in all log messages
        """
        self._context = context or {}
        self._performance_metrics = {}

    def _format_message(self, message: str, **kwargs) -> str:
        """Format log message with context and additional data.

        Args:
            message: Base log message
            **kwargs: Additional contextual data

        Returns:
            Formatted message with context
        """
        # Combine default context with provided kwargs
        full_context = {**self._context, **kwargs}

        # Redact sensitive data
        redacted_context = self._redact_sensitive_data(full_context)

        if redacted_context:
            context_str = " | ".join([f"{k}={v}" for k, v in redacted_context.items()])
            return f"{message} | {context_str}"
        return message

    def _redact_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive information from log data.

        Args:
            data: Dictionary containing log data

        Returns:
            Dictionary with sensitive data redacted
        """
        sensitive_keys = {
            "password",
            "token",
            "key",
            "secret",
            "api_key",
            "auth",
            "credential",
            "prompt",
            "transcription_text",
        }

        redacted = {}
        for key, value in data.items():
            key_lower = key.lower()

            # Check if key contains sensitive information
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 10:
                    redacted[key] = f"{value[:3]}...{value[-3:]} (len={len(value)})"
                else:
                    redacted[key] = "[REDACTED]"
            else:
                # Truncate very long strings
                if isinstance(value, str) and len(value) > 100:
                    redacted[key] = f"{value[:50]}...{value[-20:]} (len={len(value)})"
                else:
                    redacted[key] = value

        return redacted

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context."""
        self.logger.debug(self._format_message(message, **kwargs))

    def info(self, message: str, **kwargs) -> None:
        """Log info message with context."""
        self.logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context."""
        self.logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs) -> None:
        """Log error message with context."""
        self.logger.error(self._format_message(message, **kwargs))

    def exception(self, message: str, **kwargs) -> None:
        """Log exception message with context."""
        self.logger.exception(self._format_message(message, **kwargs))

    def update_context(self, **kwargs) -> None:
        """Update the default context for this logger.

        Args:
            **kwargs: Context data to update
        """
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear the default context."""
        self._context.clear()

    @contextmanager
    def context(self, **kwargs):
        """Temporary context manager for logging with additional context.

        Args:
            **kwargs: Temporary context data
        """
        original_context = self._context.copy()
        self._context.update(kwargs)
        try:
            yield self
        finally:
            self._context = original_context

    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            **kwargs: Additional performance context
        """
        # Update internal metrics
        if operation not in self._performance_metrics:
            self._performance_metrics[operation] = {
                "count": 0,
                "total_duration": 0.0,
                "min_duration": float("inf"),
                "max_duration": 0.0,
            }

        metrics = self._performance_metrics[operation]
        metrics["count"] += 1
        metrics["total_duration"] += duration
        metrics["min_duration"] = min(metrics["min_duration"], duration)
        metrics["max_duration"] = max(metrics["max_duration"], duration)

        avg_duration = metrics["total_duration"] / metrics["count"]

        self.debug(
            f"Performance: {operation}",
            duration_s=f"{duration:.3f}",
            avg_duration_s=f"{avg_duration:.3f}",
            count=metrics["count"],
            **kwargs,
        )

    def get_performance_stats(self) -> dict[str, dict[str, int | float]]:
        """Get performance statistics.

        Returns:
            Dictionary of performance metrics by operation
        """
        stats = {}
        for operation, metrics in self._performance_metrics.items():
            stats[operation] = {
                "count": metrics["count"],
                "total_duration": metrics["total_duration"],
                "avg_duration": metrics["total_duration"] / metrics["count"],
                "min_duration": metrics["min_duration"],
                "max_duration": metrics["max_duration"],
            }
        return stats

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self._performance_metrics.clear()


def timed_operation(operation_name: str, logger: StructuredLogger | None = None):
    """Decorator to automatically log operation timing.

    Args:
        operation_name: Name of the operation for logging
        logger: Optional logger instance. If None, creates a new one.

    Returns:
        Decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from first argument if it's a class with structured_logger
            actual_logger = logger
            if actual_logger is None and args and hasattr(args[0], "structured_logger"):
                actual_logger = args[0].structured_logger
            elif actual_logger is None:
                actual_logger = StructuredLogger()

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                actual_logger.log_performance(operation_name, duration, function=func.__name__, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                actual_logger.log_performance(
                    operation_name, duration, function=func.__name__, success=False, error=str(e)
                )
                raise

        return wrapper

    return decorator


class EnhancedLoggerMixin(LoggerMixin):
    """Enhanced logger mixin with structured logging capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._structured_logger = None

    @property
    def structured_logger(self) -> StructuredLogger:
        """Get structured logger instance for this class.

        Returns:
            StructuredLogger instance with class context
        """
        if self._structured_logger is None:
            context = {"class": self.__class__.__name__, "module": self.__class__.__module__}
            # Add instance ID if available
            if hasattr(self, "id"):
                context["instance_id"] = str(self.id)
            elif hasattr(self, "name"):
                context["instance_name"] = str(self.name)

            self._structured_logger = StructuredLogger(context)
        return self._structured_logger

    def log_method_call(self, method_name: str, **kwargs) -> None:
        """Log method call with parameters.

        Args:
            method_name: Name of the method being called
            **kwargs: Method parameters to log
        """
        self.structured_logger.debug(f"Method call: {method_name}", method=method_name, **kwargs)

    def log_state_change(self, old_state: Any, new_state: Any, **kwargs) -> None:
        """Log state changes.

        Args:
            old_state: Previous state
            new_state: New state
            **kwargs: Additional context
        """
        self.structured_logger.info("State change", old_state=str(old_state), new_state=str(new_state), **kwargs)

    def log_error_with_context(self, error: Exception, operation: str, **kwargs) -> None:
        """Log error with contextual information.

        Args:
            error: Exception that occurred
            operation: Operation that failed
            **kwargs: Additional context
        """
        self.structured_logger.error(
            f"Operation failed: {operation}",
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs,
        )


def create_contextual_logger(name: str, context: dict[str, Any] | None = None) -> StructuredLogger:
    """Create a contextual logger with the given name and context.

    Args:
        name: Logger name
        context: Default context for the logger

    Returns:
        StructuredLogger instance
    """
    logger = StructuredLogger(context)
    # Set the underlying logger name
    logger._logger = logging.getLogger(name)
    return logger


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance
    """
    return create_contextual_logger(name)
