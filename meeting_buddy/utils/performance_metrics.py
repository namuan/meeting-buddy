"""Performance metrics utilities for Meeting Buddy application.

This module provides performance tracking and metrics collection
for various operations throughout the application.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class PerformanceMetric:
    """Data class for storing performance metrics."""

    operation: str
    count: int = 0
    total_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    last_duration: float = 0.0
    error_count: int = 0
    success_count: int = 0
    recent_durations: list[float] = field(default_factory=list)

    @property
    def average_duration(self) -> float:
        """Calculate average duration."""
        return self.total_duration / self.count if self.count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.success_count / self.count * 100) if self.count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        return (self.error_count / self.count * 100) if self.count > 0 else 0.0

    def update(self, duration: float, success: bool = True, keep_recent: int = 100) -> None:
        """Update metrics with new measurement.

        Args:
            duration: Operation duration in seconds
            success: Whether the operation was successful
            keep_recent: Number of recent durations to keep for analysis
        """
        self.count += 1
        self.total_duration += duration
        self.last_duration = duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        # Keep recent durations for trend analysis
        self.recent_durations.append(duration)
        if len(self.recent_durations) > keep_recent:
            self.recent_durations.pop(0)

    def get_percentiles(self, percentiles: list[float] | None = None) -> dict[str, float]:
        """Calculate percentiles from recent durations.

        Args:
            percentiles: List of percentiles to calculate (0-100)

        Returns:
            Dictionary mapping percentile names to values
        """
        if percentiles is None:
            percentiles = [50, 90, 95, 99]

        if not self.recent_durations:
            return {f"p{p}": 0.0 for p in percentiles}

        sorted_durations = sorted(self.recent_durations)
        result = {}

        for p in percentiles:
            index = int((p / 100) * (len(sorted_durations) - 1))
            result[f"p{p}"] = sorted_durations[index]

        return result

    def to_dict(self) -> dict[str, str | int | float]:
        """Convert metrics to dictionary."""
        result = {
            "operation": self.operation,
            "count": self.count,
            "total_duration": self.total_duration,
            "average_duration": self.average_duration,
            "min_duration": self.min_duration if self.min_duration != float("inf") else 0.0,
            "max_duration": self.max_duration,
            "last_duration": self.last_duration,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
        }

        # Add percentiles
        result.update(self.get_percentiles())

        return result


class PerformanceTracker:
    """Thread-safe performance metrics tracker."""

    def __init__(self):
        self._metrics: dict[str, PerformanceMetric] = defaultdict(lambda: PerformanceMetric(""))
        self._lock = Lock()

    def record_operation(self, operation: str, duration: float, success: bool = True, **context) -> None:
        """Record a completed operation.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            success: Whether the operation was successful
            **context: Additional context (for logging)
        """
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = PerformanceMetric(operation)
            self._metrics[operation].update(duration, success)

    @contextmanager
    def measure_operation(self, operation: str, **context):
        """Context manager to measure operation duration.

        Args:
            operation: Name of the operation
            **context: Additional context

        Yields:
            Dictionary to store operation results
        """
        start_time = time.time()
        operation_result = {"success": True, "error": None}

        try:
            yield operation_result
        except Exception as e:
            operation_result["success"] = False
            operation_result["error"] = e
            raise
        finally:
            duration = time.time() - start_time
            self.record_operation(operation, duration, operation_result["success"], **context)

    def get_metrics(self, operation: str | None = None) -> dict[str, Any] | dict[str, dict[str, Any]]:
        """Get performance metrics.

        Args:
            operation: Specific operation name, or None for all operations

        Returns:
            Metrics dictionary for specific operation or all operations
        """
        with self._lock:
            if operation:
                return self._metrics.get(operation, PerformanceMetric(operation)).to_dict()
            else:
                return {op: metric.to_dict() for op, metric in self._metrics.items()}

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all performance metrics.

        Returns:
            Summary dictionary with aggregated statistics
        """
        with self._lock:
            if not self._metrics:
                return {
                    "total_operations": 0,
                    "total_duration": 0.0,
                    "operations_per_second": 0.0,
                    "average_duration": 0.0,
                    "success_rate": 0.0,
                }

            total_count = sum(m.count for m in self._metrics.values())
            total_duration = sum(m.total_duration for m in self._metrics.values())
            total_success = sum(m.success_count for m in self._metrics.values())

            return {
                "total_operations": total_count,
                "total_duration": total_duration,
                "operations_per_second": total_count / total_duration if total_duration > 0 else 0.0,
                "average_duration": total_duration / total_count if total_count > 0 else 0.0,
                "success_rate": (total_success / total_count * 100) if total_count > 0 else 0.0,
                "operation_count": len(self._metrics),
                "operations": list(self._metrics.keys()),
            }

    def reset_metrics(self, operation: str | None = None) -> None:
        """Reset performance metrics.

        Args:
            operation: Specific operation to reset, or None to reset all
        """
        with self._lock:
            if operation:
                if operation in self._metrics:
                    del self._metrics[operation]
            else:
                self._metrics.clear()

    def get_slow_operations(self, threshold: float = 1.0) -> list[dict[str, Any]]:
        """Get operations that are slower than threshold.

        Args:
            threshold: Duration threshold in seconds

        Returns:
            List of slow operations with their metrics
        """
        with self._lock:
            slow_ops = []
            for operation, metric in self._metrics.items():
                if metric.average_duration > threshold:
                    slow_ops.append({
                        "operation": operation,
                        "average_duration": metric.average_duration,
                        "max_duration": metric.max_duration,
                        "count": metric.count,
                    })

            # Sort by average duration (slowest first)
            return sorted(slow_ops, key=lambda x: x["average_duration"], reverse=True)

    def get_error_prone_operations(self, min_error_rate: float = 5.0) -> list[dict[str, Any]]:
        """Get operations with high error rates.

        Args:
            min_error_rate: Minimum error rate percentage

        Returns:
            List of error-prone operations with their metrics
        """
        with self._lock:
            error_ops = []
            for operation, metric in self._metrics.items():
                if metric.error_rate >= min_error_rate and metric.count >= 5:  # Minimum sample size
                    error_ops.append({
                        "operation": operation,
                        "error_rate": metric.error_rate,
                        "error_count": metric.error_count,
                        "total_count": metric.count,
                    })

            # Sort by error rate (highest first)
            return sorted(error_ops, key=lambda x: x["error_rate"], reverse=True)


# Global performance tracker instance
_global_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance.

    Returns:
        Global PerformanceTracker instance
    """
    return _global_tracker


def record_performance(operation: str, duration: float, success: bool = True, **context) -> None:
    """Record performance metrics using the global tracker.

    Args:
        operation: Name of the operation
        duration: Duration in seconds
        success: Whether the operation was successful
        **context: Additional context
    """
    _global_tracker.record_operation(operation, duration, success, **context)


def measure_performance(operation: str, **context):
    """Context manager to measure performance using the global tracker.

    Args:
        operation: Name of the operation
        **context: Additional context
    """
    return _global_tracker.measure_operation(operation, **context)


def get_performance_stats(operation: str | None = None) -> dict[str, Any] | dict[str, dict[str, Any]]:
    """Get performance statistics from the global tracker.

    Args:
        operation: Specific operation name, or None for all operations

    Returns:
        Performance statistics
    """
    return _global_tracker.get_metrics(operation)


def get_performance_summary() -> dict[str, Any]:
    """Get performance summary from the global tracker.

    Returns:
        Performance summary
    """
    return _global_tracker.get_summary()


def reset_performance_stats(operation: str | None = None) -> None:
    """Reset performance statistics in the global tracker.

    Args:
        operation: Specific operation to reset, or None to reset all
    """
    _global_tracker.reset_metrics(operation)
