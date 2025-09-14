"""Model Download Service for Meeting Buddy application.

This module contains the ModelDownloadService class that handles
background downloading of Whisper models with progress tracking.
"""

import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

try:
    import whisper
except ImportError:
    whisper = None

from ..utils.structured_logging import EnhancedLoggerMixin, timed_operation


class ModelDownloadProgress:
    """Data class representing model download progress."""

    def __init__(
        self,
        model_name: str,
        status: str = "pending",
        progress_percent: float = 0.0,
        download_speed: str | None = None,
        eta: str | None = None,
        error_message: str | None = None,
    ):
        """Initialize model download progress.

        Args:
            model_name: Name of the model being downloaded
            status: Current status (pending, downloading, completed, failed)
            progress_percent: Download progress percentage (0.0 to 100.0)
            download_speed: Current download speed (e.g., "1.2 MB/s")
            eta: Estimated time remaining (e.g., "2m 30s")
            error_message: Error message if download failed
        """
        self.model_name = model_name
        self.status = status
        self.progress_percent = progress_percent
        self.download_speed = download_speed
        self.eta = eta
        self.error_message = error_message
        self.timestamp = datetime.now()

    def __str__(self) -> str:
        return f"ModelDownloadProgress({self.model_name}, {self.status}, {self.progress_percent:.1f}%)"

    def __repr__(self) -> str:
        return f"ModelDownloadProgress(model_name='{self.model_name}', status='{self.status}', progress={self.progress_percent:.1f}%)"


class ModelDownloadService(EnhancedLoggerMixin):
    """Service class for downloading Whisper models in the background.

    This class handles downloading Whisper models with progress tracking
    and thread-safe callbacks for UI updates.
    """

    def __init__(self):
        """Initialize the ModelDownloadService."""
        EnhancedLoggerMixin.__init__(self)

        # Set up structured logging context
        self.structured_logger.update_context(
            service_type="model_download",
            whisper_available=whisper is not None,
        )

        # Download state
        self._current_download: str | None = None
        self._download_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._download_lock = threading.Lock()

        # Progress tracking
        self._progress_data: dict[str, ModelDownloadProgress] = {}

        # Callbacks
        self._progress_callbacks: list[Callable[[ModelDownloadProgress], None]] = []
        self._completion_callbacks: list[Callable[[str, bool, str | None], None]] = []

        self.structured_logger.info(
            "ModelDownloadService initialized",
            initialization_complete=True,
            whisper_available=whisper is not None,
        )

        self.log_method_call("__init__")

    @property
    def is_downloading(self) -> bool:
        """Check if a download is currently in progress.

        Returns:
            True if a download is active, False otherwise
        """
        with self._download_lock:
            return self._current_download is not None and (
                self._download_thread is not None and self._download_thread.is_alive()
            )

    @property
    def current_download_model(self) -> str | None:
        """Get the name of the currently downloading model.

        Returns:
            Model name if downloading, None otherwise
        """
        with self._download_lock:
            return self._current_download

    def get_download_progress(self, model_name: str) -> ModelDownloadProgress | None:
        """Get download progress for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            ModelDownloadProgress object or None if not found
        """
        return self._progress_data.get(model_name)

    def get_all_download_progress(self) -> dict[str, ModelDownloadProgress]:
        """Get download progress for all models.

        Returns:
            Dictionary mapping model names to progress objects
        """
        return self._progress_data.copy()

    def is_model_available(self, model_name: str) -> bool:
        """Check if a Whisper model is available locally.

        Args:
            model_name: Name of the Whisper model

        Returns:
            True if model is available, False otherwise
        """
        if whisper is None:
            self.structured_logger.warning(
                "Whisper not available",
                model_name=model_name,
                whisper_installed=False,
            )
            return False

        try:
            # Check if model files exist in whisper cache
            model_path = whisper._MODELS.get(model_name)
            if model_path is None:
                return False

            # Try to get the model file path
            import os

            cache_dir = os.path.expanduser("~/.cache/whisper")
            model_file = Path(cache_dir) / f"{model_name}.pt"

            available = model_file.exists()

            self.structured_logger.debug(
                "Model availability check",
                model_name=model_name,
                available=available,
                model_file=str(model_file),
            )

            return available

        except Exception:
            self.structured_logger.exception("Error checking model availability", extra={"model_name": model_name})
            return False

    def download_model(self, model_name: str, force_download: bool = False) -> bool:
        """Start downloading a Whisper model in the background.

        Args:
            model_name: Name of the Whisper model to download
            force_download: Force download even if model exists

        Returns:
            True if download started successfully, False otherwise
        """
        if whisper is None:
            self.structured_logger.error(
                "Cannot download model: Whisper not available",
                model_name=model_name,
                whisper_installed=False,
            )
            return False

        # Check if model name is valid
        if model_name not in whisper._MODELS:
            self.structured_logger.error(
                "Invalid model name",
                model_name=model_name,
                available_models=list(whisper._MODELS.keys()),
            )
            return False

        # Check if already downloading
        if self.is_downloading:
            self.structured_logger.warning(
                "Download already in progress",
                current_model=self._current_download,
                requested_model=model_name,
            )
            return False

        # Check if model already exists (unless force download)
        if not force_download and self.is_model_available(model_name):
            self.structured_logger.info(
                "Model already available",
                model_name=model_name,
                force_download=force_download,
            )
            # Update progress to completed
            progress = ModelDownloadProgress(
                model_name=model_name,
                status="completed",
                progress_percent=100.0,
            )
            self._progress_data[model_name] = progress
            self._notify_progress_callbacks(progress)
            self._notify_completion_callbacks(model_name, True, None)
            return True

        # Start download thread
        with self._download_lock:
            self._current_download = model_name
            self._stop_event.clear()
            self._download_thread = threading.Thread(
                target=self._download_model_thread,
                args=(model_name,),
                daemon=True,
                name=f"ModelDownload-{model_name}",
            )
            self._download_thread.start()

        self.structured_logger.info(
            "Model download started",
            model_name=model_name,
            force_download=force_download,
        )

        return True

    def cancel_download(self) -> bool:
        """Cancel the current download.

        Returns:
            True if download was cancelled, False if no download in progress
        """
        with self._download_lock:
            if not self.is_downloading:
                return False

            model_name = self._current_download
            self._stop_event.set()

            self.structured_logger.info(
                "Download cancellation requested",
                model_name=model_name,
            )

            # Update progress to cancelled
            if model_name and model_name in self._progress_data:
                progress = self._progress_data[model_name]
                progress.status = "cancelled"
                progress.error_message = "Download cancelled by user"
                self._notify_progress_callbacks(progress)
                self._notify_completion_callbacks(model_name, False, "Download cancelled by user")

            return True

    @timed_operation("whisper_model_download")
    def _download_model_thread(self, model_name: str) -> None:
        """Download model in background thread.

        Args:
            model_name: Name of the model to download
        """
        try:
            # Initialize progress
            progress = ModelDownloadProgress(
                model_name=model_name,
                status="downloading",
                progress_percent=0.0,
            )
            self._progress_data[model_name] = progress
            self._notify_progress_callbacks(progress)

            self.structured_logger.info(
                "Starting model download",
                model_name=model_name,
            )

            # Simulate progress updates (Whisper doesn't provide real progress)
            # In a real implementation, you might hook into Whisper's download process
            start_time = time.time()
            total_steps = 20  # Simulate 20 progress steps

            for step in range(total_steps + 1):
                if self._stop_event.is_set():
                    self.structured_logger.info(
                        "Download cancelled",
                        model_name=model_name,
                        step=step,
                        total_steps=total_steps,
                    )
                    return

                # Update progress
                progress_percent = (step / total_steps) * 100.0
                elapsed_time = time.time() - start_time

                if step > 0 and elapsed_time > 0:
                    # Estimate remaining time
                    time_per_step = elapsed_time / step
                    remaining_steps = total_steps - step
                    eta_seconds = remaining_steps * time_per_step
                    eta = self._format_eta(eta_seconds)
                else:
                    eta = None

                progress.progress_percent = progress_percent
                progress.eta = eta
                progress.timestamp = datetime.now()

                self._notify_progress_callbacks(progress)

                # Sleep to simulate download time
                if step < total_steps:
                    time.sleep(0.5)  # Simulate download time

            # Actually download the model (this will be quick if already cached)
            try:
                whisper.load_model(model_name)

                # Mark as completed
                progress.status = "completed"
                progress.progress_percent = 100.0
                progress.eta = None
                progress.timestamp = datetime.now()

                self._notify_progress_callbacks(progress)
                self._notify_completion_callbacks(model_name, True, None)

                self.structured_logger.info(
                    "Model download completed successfully",
                    model_name=model_name,
                    elapsed_time=time.time() - start_time,
                )

            except Exception as e:
                error_msg = f"Failed to load model: {e!s}"
                progress.status = "failed"
                progress.error_message = error_msg
                progress.timestamp = datetime.now()

                self._notify_progress_callbacks(progress)
                self._notify_completion_callbacks(model_name, False, error_msg)

                self.structured_logger.exception("Model download failed", extra={"model_name": model_name})

        except Exception as e:
            error_msg = f"Download error: {e!s}"
            if model_name in self._progress_data:
                progress = self._progress_data[model_name]
                progress.status = "failed"
                progress.error_message = error_msg
                progress.timestamp = datetime.now()
                self._notify_progress_callbacks(progress)

            self._notify_completion_callbacks(model_name, False, error_msg)

            self.structured_logger.exception("Unexpected error in model download", extra={"model_name": model_name})

        finally:
            # Clean up
            with self._download_lock:
                self._current_download = None
                self._download_thread = None

    def _format_eta(self, seconds: float) -> str:
        """Format ETA in human-readable format.

        Args:
            seconds: Remaining seconds

        Returns:
            Formatted ETA string
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def add_progress_callback(self, callback: Callable[[ModelDownloadProgress], None]) -> None:
        """Add callback for download progress updates.

        Args:
            callback: Function to call with progress updates
        """
        self._progress_callbacks.append(callback)
        self.structured_logger.debug(
            "Progress callback added",
            callback=str(callback),
        )

    def add_completion_callback(self, callback: Callable[[str, bool, str | None], None]) -> None:
        """Add callback for download completion.

        Args:
            callback: Function to call when download completes
                     Parameters: (model_name, success, error_message)
        """
        self._completion_callbacks.append(callback)
        self.structured_logger.debug(
            "Completion callback added",
            callback=str(callback),
        )

    def remove_progress_callback(self, callback: Callable[[ModelDownloadProgress], None]) -> bool:
        """Remove progress callback.

        Args:
            callback: Callback function to remove

        Returns:
            True if callback was removed, False if not found
        """
        try:
            self._progress_callbacks.remove(callback)
            self.structured_logger.debug(
                "Progress callback removed",
                callback=str(callback),
            )
            return True
        except ValueError:
            return False

    def remove_completion_callback(self, callback: Callable[[str, bool, str | None], None]) -> bool:
        """Remove completion callback.

        Args:
            callback: Callback function to remove

        Returns:
            True if callback was removed, False if not found
        """
        try:
            self._completion_callbacks.remove(callback)
            self.structured_logger.debug(
                "Completion callback removed",
                callback=str(callback),
            )
            return True
        except ValueError:
            return False

    def _notify_progress_callbacks(self, progress: ModelDownloadProgress) -> None:
        """Notify all progress callbacks.

        Args:
            progress: Progress data to send
        """
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception:
                self.structured_logger.exception("Error in progress callback", extra={"callback": str(callback)})

    def _notify_completion_callbacks(self, model_name: str, success: bool, error_message: str | None) -> None:
        """Notify all completion callbacks.

        Args:
            model_name: Name of the model
            success: Whether download was successful
            error_message: Error message if failed
        """
        for callback in self._completion_callbacks:
            try:
                callback(model_name, success, error_message)
            except Exception:
                self.structured_logger.exception("Error in completion callback", extra={"callback": str(callback)})

    def cleanup(self) -> None:
        """Clean up resources and stop any ongoing downloads."""
        self.structured_logger.info("Cleaning up ModelDownloadService")

        # Cancel any ongoing download
        self.cancel_download()

        # Wait for download thread to finish
        with self._download_lock:
            if self._download_thread and self._download_thread.is_alive():
                self._download_thread.join(timeout=5.0)

        # Clear callbacks
        self._progress_callbacks.clear()
        self._completion_callbacks.clear()

        self.structured_logger.info("ModelDownloadService cleanup completed")

    def get_service_status(self) -> dict[str, any]:
        """Get current service status.

        Returns:
            Dictionary containing service status information
        """
        with self._download_lock:
            return {
                "whisper_available": whisper is not None,
                "is_downloading": self.is_downloading,
                "current_download": self._current_download,
                "total_downloads": len(self._progress_data),
                "progress_callbacks": len(self._progress_callbacks),
                "completion_callbacks": len(self._completion_callbacks),
            }
