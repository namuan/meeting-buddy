"""Meeting Buddy Presenter for the application.

This module contains the MeetingBuddyPresenter class that coordinates
between models and views, handling business logic and user interactions.
"""

import logging
import os
import queue
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from ..models.audio_device_model import AudioDeviceModel
from ..models.configuration_model import ConfigurationModel
from ..models.llm_model import LLMModel, LLMRequest
from ..models.model_download_service import ModelDownloadProgress, ModelDownloadService
from ..models.recording_model import RecordingModel
from ..models.transcription_data import TranscriptionResult
from ..utils.audio_recorder_thread import AudioRecorderThread
from ..utils.audio_transcriber_thread import AudioTranscriberThread
from ..utils.llm_thread import LLMThread
from ..views.meeting_buddy_view import MeetingBuddyView


class MeetingBuddyPresenter(QObject):
    """Presenter class for Meeting Buddy application.

    This class coordinates between models and views, handling
    business logic and user interactions following the MVP
    (Model-View-Presenter) architecture pattern.
    """

    # Qt signals for thread-safe UI updates
    transcription_result_signal = pyqtSignal(str, float, str)  # text, confidence, timestamp
    transcription_status_signal = pyqtSignal(str)  # status message
    transcription_error_signal = pyqtSignal(str)  # error message

    # LLM-related signals
    llm_response_signal = pyqtSignal(str)  # complete LLM response
    llm_response_chunk_signal = pyqtSignal(str)  # streaming LLM response chunk
    llm_status_signal = pyqtSignal(str)  # LLM status message
    llm_error_signal = pyqtSignal(str)  # LLM error message
    llm_connection_status_signal = pyqtSignal(bool)  # LLM connection status

    # Configuration-related signals
    config_updated_signal = pyqtSignal()  # configuration was updated
    model_download_progress_signal = pyqtSignal(str, float, str)  # model_name, progress, status
    model_download_completed_signal = pyqtSignal(str, bool, str)  # model_name, success, message

    def __init__(self):
        """Initialize the MeetingBuddyPresenter."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Temporary folder management
        self._temp_dir: Optional[Path] = None
        self._temp_dir_created = False

        # Initialize temporary folder
        self._create_temp_folder()

        # Initialize models
        self.audio_model = AudioDeviceModel()
        self.recording_model = RecordingModel()
        self.llm_model = LLMModel()
        self.config_model = ConfigurationModel()
        self.download_service = ModelDownloadService()

        # Set up LLM model callbacks
        self.llm_model.set_response_callback(self._on_llm_model_response)
        self.llm_model.set_response_chunk_callback(self._on_llm_model_response_chunk)
        self.llm_model.set_error_callback(self._on_llm_model_error)
        self.llm_model.set_connection_status_callback(self._on_llm_model_connection_status)

        # Set up configuration callbacks
        self.config_model.add_config_changed_callback(self._on_config_changed)
        self.config_model.add_whisper_model_changed_callback(self._on_whisper_model_changed)
        self.config_model.add_ollama_model_changed_callback(self._on_ollama_model_changed)

        # Set up download service callbacks
        self.download_service.add_progress_callback(self._on_download_progress)
        self.download_service.add_completion_callback(self._on_download_completed)

        # Initialize view
        self.view = MeetingBuddyView()

        # Transcription threads
        self._audio_recorder_thread: Optional[AudioRecorderThread] = None
        self._audio_transcriber_thread: Optional[AudioTranscriberThread] = None
        self._transcription_queue: Optional[queue.Queue] = None
        self._transcription_active = False

        # LLM threads
        self._llm_thread: Optional[LLMThread] = None
        self._llm_request_queue: Optional[queue.Queue] = None
        self._llm_active = False

        # Connect view callbacks to presenter methods
        self._connect_view_callbacks()

        # Connect Qt signals to UI update methods
        self._connect_signals()

        # Set default prompt after callbacks are connected
        default_prompt = "Please summarize the key points and action items from this meeting transcription."
        self.view.set_prompt_text(default_prompt)
        self.llm_model.set_user_prompt(default_prompt)

        # Initialize configuration UI
        self._initialize_config_ui()

        self.logger.info("MeetingBuddyPresenter initialized")

    def _connect_view_callbacks(self) -> None:
        """Connect view callbacks to presenter methods."""
        self.view.on_input_device_changed = self._handle_input_device_changed
        self.view.on_start_recording = self._handle_start_recording
        self.view.on_stop_recording = self._handle_stop_recording
        self.view.on_prompt_changed = self._handle_prompt_changed
        self.view.on_whisper_model_changed = self._handle_whisper_model_changed
        self.view.on_ollama_model_changed = self._handle_ollama_model_changed

        self.logger.debug("View callbacks connected")

    def _connect_signals(self) -> None:
        """Connect Qt signals to UI update methods."""
        self.transcription_result_signal.connect(self._update_transcription_ui)
        self.transcription_status_signal.connect(self._update_transcription_status_ui)
        self.transcription_error_signal.connect(self._show_transcription_error_ui)

        # Connect LLM signals
        self.llm_response_signal.connect(self._update_llm_response_ui)
        self.llm_response_chunk_signal.connect(self._update_llm_response_chunk_ui)
        self.llm_status_signal.connect(self._update_llm_status_ui)
        self.llm_error_signal.connect(self._show_llm_error_ui)
        self.llm_connection_status_signal.connect(self._update_llm_connection_status_ui)

        # Connect configuration signals
        self.config_updated_signal.connect(self._update_config_ui)
        self.model_download_progress_signal.connect(self._update_download_progress_ui)
        self.model_download_completed_signal.connect(self._update_download_completed_ui)

        # Connect view signals to download service
        self.view.model_download_progress_updated.connect(self.view._on_download_progress_updated)
        self.view.model_download_completed.connect(self.view._on_download_completed)
        self.view.whisper_model_status_updated.connect(self.view._on_whisper_model_status_updated)
        self.view.ollama_model_status_updated.connect(self.view._on_ollama_model_status_updated)

        self.logger.debug("Qt signals connected")

    def _initialize_config_ui(self) -> None:
        """Initialize the configuration UI with current settings."""
        try:
            # Populate Whisper models
            available_models = self.config_model.available_whisper_models
            current_whisper = self.config_model.whisper_model
            self.view.populate_whisper_models(available_models, current_whisper)
            self.view.update_current_whisper_model(current_whisper)

            # Populate Ollama models (start with common models)
            common_ollama_models = [
                "llama3.2:latest",
                "llama3.1:latest",
                "llama3:latest",
                "mistral:latest",
                "codellama:latest",
                "phi3:latest",
            ]
            current_ollama = self.config_model.ollama_model
            self.view.populate_ollama_models(common_ollama_models, current_ollama)
            self.view.update_current_ollama_model(current_ollama)

            # Update LLM service with current configuration
            self._update_llm_service_config()

            self.logger.info("Configuration UI initialized successfully")

        except Exception as e:
            self.logger.exception("Error initializing configuration UI")
            self.view.set_download_status(f"Error initializing configuration: {e}")

    def _update_llm_service_config(self) -> None:
        """Update LLM service with current configuration."""
        try:
            config = self.config_model.config_data

            # Update LLM thread API service if it exists
            if self._llm_thread and hasattr(self._llm_thread, "api_service"):
                self._llm_thread.api_service.update_config(
                    model=config.ollama_model,
                    endpoint=config.ollama_endpoint,
                    api_timeout=config.api_timeout,
                    max_retries=config.max_retries,
                    retry_delay=config.retry_delay,
                )
                self.logger.debug("LLM service configuration updated")
            else:
                self.logger.debug("LLM thread not available, configuration will be applied when thread is created")

        except Exception:
            self.logger.exception("Error updating LLM service configuration")

    def _create_temp_folder(self) -> None:
        """Create a temporary folder for storing audio files and transcription data."""
        try:
            # Create a temporary directory with a meaningful prefix
            temp_dir = tempfile.mkdtemp(prefix="meeting_buddy_", suffix="_session")
            self._temp_dir = Path(temp_dir)
            self._temp_dir_created = True

            # Create subdirectories for different types of files
            (self._temp_dir / "audio_chunks").mkdir(exist_ok=True)
            (self._temp_dir / "recordings").mkdir(exist_ok=True)
            (self._temp_dir / "transcriptions").mkdir(exist_ok=True)

            self.logger.info(f"Created temporary folder: {self._temp_dir}")

        except Exception:
            self.logger.exception("Failed to create temporary folder")
            self._temp_dir = None
            self._temp_dir_created = False

    def _cleanup_temp_folder(self) -> None:
        """Clean up the temporary folder and all its contents."""
        if not self._temp_dir_created or not self._temp_dir:
            return

        try:
            if self._temp_dir.exists():
                shutil.rmtree(self._temp_dir)
                self.logger.info(f"Cleaned up temporary folder: {self._temp_dir}")
            else:
                self.logger.debug("Temporary folder already removed")

        except Exception:
            self.logger.exception(f"Failed to clean up temporary folder: {self._temp_dir}")
        finally:
            self._temp_dir = None
            self._temp_dir_created = False

    def get_temp_folder(self) -> Optional[Path]:
        """Get the path to the temporary folder.

        Returns:
            Path to temporary folder or None if not created
        """
        return self._temp_dir if self._temp_dir_created else None

    def get_audio_chunks_folder(self) -> Optional[Path]:
        """Get the path to the audio chunks subfolder.

        Returns:
            Path to audio chunks folder or None if temp folder not created
        """
        if self._temp_dir and self._temp_dir_created:
            return self._temp_dir / "audio_chunks"
        return None

    def get_recordings_folder(self) -> Optional[Path]:
        """Get the path to the recordings subfolder.

        Returns:
            Path to recordings folder or None if temp folder not created
        """
        if self._temp_dir and self._temp_dir_created:
            return self._temp_dir / "recordings"
        return None

    def get_transcriptions_folder(self) -> Optional[Path]:
        """Get the path to the transcriptions subfolder.

        Returns:
            Path to transcriptions folder or None if temp folder not created
        """
        if self._temp_dir and self._temp_dir_created:
            return self._temp_dir / "transcriptions"
        return None

    def create_temp_file(self, subfolder: str, prefix: str = "temp_", suffix: str = ".tmp") -> Optional[Path]:
        """Create a temporary file in the specified subfolder.

        Args:
            subfolder: Subfolder name (e.g., 'audio_chunks', 'recordings', 'transcriptions')
            prefix: File prefix
            suffix: File suffix/extension

        Returns:
            Path to created temporary file or None if failed
        """
        if not self._temp_dir or not self._temp_dir_created:
            self.logger.error("Temporary folder not available")
            return None

        try:
            subfolder_path = self._temp_dir / subfolder
            subfolder_path.mkdir(exist_ok=True)

            # Create temporary file
            fd, temp_file_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(subfolder_path))
            os.close(fd)  # Close the file descriptor

            temp_path = Path(temp_file_path)
            self.logger.debug(f"Created temporary file: {temp_path}")
            return temp_path

        except Exception:
            self.logger.exception(f"Failed to create temporary file in {subfolder}")
            return None

    def cleanup_temp_files(self, subfolder: Optional[str] = None, pattern: str = "*") -> int:
        """Clean up temporary files in a specific subfolder or all subfolders.

        Args:
            subfolder: Specific subfolder to clean (None for all)
            pattern: File pattern to match (default: all files)

        Returns:
            Number of files cleaned up
        """
        if not self._temp_dir or not self._temp_dir_created:
            return 0

        try:
            if subfolder:
                return self._cleanup_subfolder(subfolder, pattern)
            else:
                return self._cleanup_all_subfolders(pattern)
        except Exception:
            self.logger.exception("Failed to clean up temporary files")
            return 0

    def _cleanup_subfolder(self, subfolder: str, pattern: str) -> int:
        """Clean up files in a specific subfolder.

        Args:
            subfolder: Subfolder name to clean
            pattern: File pattern to match

        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        subfolder_path = self._temp_dir / subfolder

        if subfolder_path.exists():
            for file_path in subfolder_path.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    cleaned_count += 1

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} files from {subfolder}")

        return cleaned_count

    def _cleanup_all_subfolders(self, pattern: str) -> int:
        """Clean up files in all subfolders.

        Args:
            pattern: File pattern to match

        Returns:
            Number of files cleaned up
        """
        total_cleaned = 0
        subfolders = ["audio_chunks", "recordings", "transcriptions"]

        for subfolder_name in subfolders:
            total_cleaned += self._cleanup_subfolder(subfolder_name, pattern)

        return total_cleaned

    def get_temp_folder_size(self) -> int:
        """Get the total size of the temporary folder in bytes.

        Returns:
            Total size in bytes, or 0 if folder doesn't exist
        """
        if not self._temp_dir or not self._temp_dir_created:
            return 0

        try:
            total_size = 0
            for file_path in self._temp_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            self.logger.exception("Failed to calculate temp folder size")
            return 0

    def _initialize_ui(self) -> None:
        """Initialize UI with data from models."""
        # Populate device lists
        self._update_device_lists()

        # Set initial transcription
        self.view.set_transcription_text(self.recording_model.current_transcription)

        # Set initial recording state
        self.view.set_recording_state(self.recording_model.is_recording)

        # Initialize LLM processing and show status
        llm_started = self.start_llm_processing()
        if llm_started:
            self.view.set_llm_response_status("LLM ready - Enter a prompt and start recording")
        else:
            self.view.set_llm_response_status("LLM unavailable - Please ensure Ollama is running")

        self.logger.debug("UI initialized with model data")

    def _update_device_lists(self) -> None:
        """Update device combo boxes with current device lists."""
        # Update input devices
        input_devices = [str(device) for device in self.audio_model.input_devices]

        # Find the index of the currently selected device
        selected_index = 0  # Default to first device
        selected_device = self.audio_model.selected_input_device
        if selected_device:
            for i, device in enumerate(self.audio_model.input_devices):
                if device.index == selected_device.index:
                    selected_index = i
                    break

        self.view.populate_input_devices(input_devices, selected_index)
        self.logger.debug(f"Device lists updated, selected index: {selected_index}")

    # Event handlers
    def _handle_input_device_changed(self, index: int) -> None:
        """Handle input device selection change.

        Args:
            index: Index of selected device in the combo box
        """
        if self.audio_model.select_input_device(index):
            selected_device = self.audio_model.selected_input_device
            self.logger.info(f"Input device changed to: {selected_device}")
            self.view.show_info_message("Device Selected", f"Input device: {selected_device.name}")
        else:
            self.logger.error(f"Failed to select input device at index {index}")
            self.view.show_error_message("Device Selection Error", "Failed to select input device")

    def _handle_start_recording(self) -> None:
        """Handle start recording button click."""
        try:
            if self.recording_model.is_recording:
                self.logger.warning("Already recording, ignoring start request")
                return

            # Start recording
            recording = self.recording_model.start_recording()
            self.view.set_recording_state(True)
            self.view.clear_transcription_text()

            # Also start transcription if available
            if self.recording_model.transcription_enabled:
                self.logger.info("Starting transcription along with recording")
                transcription_started = self.start_transcription()
                if transcription_started:
                    self.logger.info("Transcription started successfully with recording")
                else:
                    self.logger.warning("Failed to start transcription, but recording will continue")
            else:
                self.logger.info("Transcription not enabled, recording without transcription")

            # Ensure LLM processing is active for real-time analysis
            if not self._llm_active:
                self.logger.info("Starting LLM processing with recording")
                self.start_llm_processing()

            self.logger.info(f"Started recording: {recording.name}")
            self.view.show_info_message("Recording Started", f"Recording '{recording.name}' started")

        except Exception as e:
            self.logger.exception("Failed to start recording")
            self.view.show_error_message("Recording Error", f"Failed to start recording: {e!s}")

    def _handle_stop_recording(self) -> None:
        """Handle stop recording button click."""
        try:
            if not self.recording_model.is_recording:
                self.logger.warning("Not recording, ignoring stop request")
                return

            # Stop transcription if it's active
            if self._transcription_active:
                self.logger.info("Stopping transcription along with recording")
                self.stop_transcription()

            # Stop recording
            recording = self.recording_model.stop_recording()
            self.view.set_recording_state(False)

            if recording:
                self.logger.info(f"Stopped recording: {recording.name}")
                self.view.show_info_message("Recording Stopped", f"Recording '{recording.name}' saved")

        except Exception as e:
            self.logger.exception("Failed to stop recording")
            self.view.show_error_message("Recording Error", f"Failed to stop recording: {e!s}")

    def _handle_prompt_changed(self, prompt_text: str) -> None:
        """Handle LLM prompt text change.

        Args:
            prompt_text: New prompt text
        """
        self.llm_model.set_user_prompt(prompt_text)
        self.logger.debug(f"LLM prompt updated: {len(prompt_text)} characters")

    # Public methods for external control
    def refresh_devices(self) -> None:
        """Refresh the audio device lists."""
        try:
            self.audio_model.refresh_devices()
            self._update_device_lists()
            self.logger.info("Audio devices refreshed")
            self.view.show_info_message("Devices Refreshed", "Audio device lists updated")
        except Exception as e:
            self.logger.exception("Failed to refresh devices")
            self.view.show_error_message("Refresh Error", f"Failed to refresh devices: {e!s}")

    def update_transcription(self, text: str) -> None:
        """Update the current transcription text.

        Args:
            text: New transcription text
        """
        self.recording_model.update_transcription(text)
        self.view.set_transcription_text(text)
        self.logger.debug(f"Transcription updated: {len(text)} characters")

    def append_transcription(self, text: str) -> None:
        """Append text to the current transcription.

        Args:
            text: Text to append
        """
        self.recording_model.append_transcription(text)
        self.view.set_transcription_text(self.recording_model.current_transcription)
        self.logger.debug(f"Transcription appended: {len(text)} characters")

    def clear_transcription(self) -> None:
        """Clear the current transcription."""
        self.recording_model.clear_transcription()
        self.view.clear_transcription_text()
        self.logger.debug("Transcription cleared")

    def is_recording(self) -> bool:
        """Check if currently recording.

        Returns:
            True if recording, False otherwise
        """
        return self.recording_model.is_recording

    def get_current_recording_name(self) -> Optional[str]:
        """Get the name of the current recording.

        Returns:
            Current recording name or None
        """
        recording = self.recording_model.current_recording
        return recording.name if recording else None

    def show_view(self) -> None:
        """Show the main view window."""
        self.view.show()

        # Initialize UI with data from models after view is shown
        self._initialize_ui()

        self.logger.info("View displayed and initialized")

    # Transcription control methods
    def start_transcription(self) -> bool:
        """Start real-time transcription.

        Returns:
            True if transcription started successfully, False otherwise
        """
        if self._transcription_active:
            self.logger.warning("Transcription already active")
            return True

        if not self.recording_model.transcription_enabled:
            self.logger.error("Transcription not enabled in recording model")
            self.view.show_error_message("Transcription Error", "Transcription is not available")
            return False

        try:
            self.logger.debug("Starting transcription initialization process")

            # Get selected input device for recording
            selected_device = self.audio_model.selected_input_device
            if not selected_device:
                self.logger.error("No input device selected for recording")
                self.view.show_error_message("Device Error", "Please select an input device for recording")
                return False

            self.logger.debug(
                f"Selected input device for recording: {selected_device.name} (index: {selected_device.index})"
            )

            # Log transcription model status
            model_available = self.recording_model.transcription_model is not None
            # TranscriptionModel no longer loads Whisper directly - that's handled by AudioTranscriberThread
            model_loaded = model_available  # Model is available if transcription model exists
            transcription_stats = {
                "model_available": model_available,
                "model_loaded": model_loaded,
                "transcription_enabled": self.recording_model.transcription_enabled,
            }
            self.logger.debug(f"Transcription model status: {transcription_stats}")

            # Create transcription queue
            self._transcription_queue = queue.Queue(maxsize=50)
            self.logger.debug("Created transcription queue with maxsize=50")

            # Create and configure audio transcriber thread
            transcriber_config = {
                "audio_queue": self._transcription_queue,
                "model_name": "base",  # Use base model for good balance of speed/accuracy
                "language": None,  # Auto-detect language
            }
            self.logger.debug(f"Creating audio transcriber thread with config: {transcriber_config}")

            self._audio_transcriber_thread = AudioTranscriberThread(**transcriber_config)

            # Set up transcriber callbacks
            self._audio_transcriber_thread.set_transcription_callback(self._on_transcription_result)
            self._audio_transcriber_thread.set_error_callback(self._on_transcription_error)
            self._audio_transcriber_thread.set_progress_callback(self._on_transcription_progress)

            # Create and configure audio recorder thread
            recorder_config = {
                "device": selected_device,
                "transcription_model": self.recording_model.transcription_model,
                "chunk_duration_seconds": 2.0,  # 2-second chunks for good responsiveness
                "sample_rate": 16000,  # Standard rate for Whisper
                "channels": 1,  # Mono audio
            }
            self.logger.debug(f"Creating audio recorder thread with config: {recorder_config}")

            self._audio_recorder_thread = AudioRecorderThread(**recorder_config)

            # Set up recorder callbacks
            self._audio_recorder_thread.set_chunk_ready_callback(self._on_audio_chunk_ready)
            self._audio_recorder_thread.set_error_callback(self._on_recording_error)

            # Start transcriber thread first
            self.logger.debug("Starting transcriber thread...")
            transcriber_start_time = time.time()
            if not self._audio_transcriber_thread.start_transcribing():
                self.logger.error("Failed to start transcriber thread")
                self._cleanup_transcription_threads()
                return False
            transcriber_start_duration = time.time() - transcriber_start_time
            self.logger.debug(f"Transcriber thread started in {transcriber_start_duration:.3f}s")

            # Start recorder thread
            self.logger.debug("Starting recorder thread...")
            recorder_start_time = time.time()
            if not self._audio_recorder_thread.start_recording():
                self.logger.error("Failed to start recorder thread")
                self._cleanup_transcription_threads()
                return False
            recorder_start_duration = time.time() - recorder_start_time
            self.logger.debug(f"Recorder thread started in {recorder_start_duration:.3f}s")

            self._transcription_active = True

            # Update UI for live transcription
            self.transcription_status_signal.emit("Listening for audio...")

            # Log final transcription setup
            setup_summary = {
                "device": selected_device.name,
                "sample_rate": 16000,
                "chunk_duration": 2.0,
                "model": "base",
                "queue_maxsize": 50,
                "transcriber_active": self._audio_transcriber_thread.is_transcribing
                if self._audio_transcriber_thread
                else False,
                "recorder_active": self._audio_recorder_thread.is_recording if self._audio_recorder_thread else False,
            }
            self.logger.info(f"Transcription started successfully: {setup_summary}")
            self.view.show_info_message("Transcription Started", "Real-time transcription is now active")
            return True

        except Exception as e:
            self.logger.exception("Failed to start transcription")
            self.view.show_error_message("Transcription Error", f"Failed to start transcription: {e}")
            self._cleanup_transcription_threads()
            return False

    def stop_transcription(self) -> None:
        """Stop real-time transcription."""
        if not self._transcription_active:
            self.logger.debug("Transcription not active")
            return

        try:
            stop_start_time = time.time()
            self.logger.info("Stopping transcription")

            # Log current transcription statistics before stopping
            if self._audio_transcriber_thread:
                transcriber_stats = self._audio_transcriber_thread.get_transcription_stats()
                self.logger.debug(f"Transcriber statistics before stop: {transcriber_stats}")

            if self._audio_recorder_thread:
                recorder_stats = self._audio_recorder_thread.get_audio_stats()
                self.logger.debug(f"Recorder statistics before stop: {recorder_stats}")

            self._transcription_active = False

            # Stop threads
            self._cleanup_transcription_threads()

            stop_duration = time.time() - stop_start_time

            # Update UI status
            self.transcription_status_signal.emit("Transcription stopped")

            self.logger.info(f"Transcription stopped successfully in {stop_duration:.3f}s")
            self.view.show_info_message("Transcription Stopped", "Real-time transcription has been stopped")

        except Exception as e:
            self.logger.exception("Error stopping transcription")
            self.view.show_error_message("Transcription Error", f"Error stopping transcription: {e}")

    def _cleanup_transcription_threads(self) -> None:
        """Clean up transcription threads and resources."""
        try:
            # Stop recorder thread first (producer) - non-blocking
            if self._audio_recorder_thread:
                self.logger.debug("Stopping audio recorder thread")
                self._audio_recorder_thread._stop_event.set()
                self._audio_recorder_thread._recording = False
                # Don't wait for thread to finish - let it cleanup in background
                self._audio_recorder_thread = None

            # Stop transcriber thread (consumer) - non-blocking
            if self._audio_transcriber_thread:
                self.logger.debug("Stopping audio transcriber thread")
                self._audio_transcriber_thread._stop_event.set()
                self._audio_transcriber_thread._transcribing = False
                # Don't wait for thread to finish - let it cleanup in background
                self._audio_transcriber_thread = None

            # Clear queue with proper synchronization
            if self._transcription_queue:
                self.logger.debug("Clearing transcription queue")
                cleared_count = 0
                while not self._transcription_queue.empty():
                    try:
                        self._transcription_queue.get_nowait()
                        self._transcription_queue.task_done()
                        cleared_count += 1
                    except queue.Empty:
                        break
                    except Exception:
                        # Ignore individual item errors during cleanup
                        break

                if cleared_count > 0:
                    self.logger.debug(f"Cleared {cleared_count} items from transcription queue")

                self._transcription_queue = None

            self.logger.debug("Transcription threads cleanup initiated (non-blocking)")

        except Exception:
            self.logger.exception("Error cleaning up transcription threads")

    def _on_audio_chunk_ready(self, audio_data, timestamp) -> None:
        """Handle audio chunk ready for transcription.

        Args:
            audio_data: Audio data as numpy array
            timestamp: Timestamp when chunk was recorded
        """
        if not self._transcription_active or not self._audio_transcriber_thread:
            return

        try:
            # Log audio chunk characteristics
            chunk_info = {
                "samples": len(audio_data),
                "timestamp": timestamp.isoformat(),
                "rms_level": float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0,
                "peak_level": float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0,
            }
            self.logger.debug(f"Processing audio chunk: {chunk_info}")

            # Add chunk to transcription queue
            success = self._audio_transcriber_thread.add_audio_chunk(audio_data, timestamp)
            if success:
                self.logger.debug(f"Successfully added audio chunk for transcription: {len(audio_data)} samples")
            else:
                self.logger.warning(f"Failed to add audio chunk for transcription: {len(audio_data)} samples")
        except Exception:
            self.logger.exception("Error processing audio chunk")

    def _on_transcription_result(self, result: TranscriptionResult) -> None:
        """Handle transcription result.

        Args:
            result: Transcription result from Whisper
        """
        try:
            # Log detailed transcription result
            result_info = {
                "text": result.text,
                "confidence": result.confidence,
                "timestamp": result.timestamp.isoformat(),
                "chunk_id": getattr(result, "chunk_id", "unknown"),
                "text_length": len(result.text),
            }
            self.logger.debug(f"Received transcription result: {result_info}")

            # Store result in the recording model
            self.recording_model.add_transcription_result(result)

            # Emit signal for thread-safe UI update
            self.transcription_result_signal.emit(result.text, result.confidence, result.timestamp.isoformat())

            self.logger.info(f"Transcription result: '{result.text}' (confidence: {result.confidence:.2f})")

        except Exception:
            self.logger.exception("Error handling transcription result")

    def _update_transcription_ui(self, text: str, confidence: float, timestamp: str) -> None:
        """Update transcription UI (called via Qt signal on main thread).

        Args:
            text: Transcription text
            confidence: Confidence score
            timestamp: Timestamp string
        """
        try:
            self.logger.debug(f"Updating transcription UI: '{text}' (confidence: {confidence:.2f})")

            # Update the view with live transcription
            self.view.update_transcription_live(text, is_final=True)

            # Also update the recording model
            self.recording_model.append_transcription(text)

            # Highlight the new text briefly
            self.view.highlight_recent_transcription(text)

            # Update status
            word_count = self.view.get_transcription_word_count()
            char_count = self.view.get_transcription_character_count()
            status_message = f"Transcribing... ({word_count} words, {char_count} chars)"
            self.transcription_status_signal.emit(status_message)

            # Process with LLM if active and prompt is set - send full transcription for context
            full_transcription = self.recording_model.current_transcription
            self._process_transcription_with_llm(full_transcription)

            self.logger.debug(f"Transcription UI update completed for '{text}'")

        except Exception:
            self.logger.exception("Error updating transcription UI")

    def _update_transcription_status_ui(self, status: str) -> None:
        """Update transcription status UI (called via Qt signal on main thread).

        Args:
            status: Status message
        """
        try:
            self.view.set_transcription_status(status)
        except Exception:
            self.logger.exception("Error updating transcription status UI")

    def _show_transcription_error_ui(self, error_message: str) -> None:
        """Show transcription error UI (called via Qt signal on main thread).

        Args:
            error_message: Error message to display
        """
        try:
            self.view.show_error_message("Transcription Error", error_message)
        except Exception:
            self.logger.exception("Error showing transcription error UI")

    def _on_transcription_error(self, error: Exception) -> None:
        """Handle transcription error.

        Args:
            error: Exception that occurred during transcription
        """
        self.logger.error(f"Transcription error: {error}")
        # Emit signal for thread-safe UI update
        self.transcription_error_signal.emit(f"Transcription error: {error}")

    def _on_transcription_progress(self, message: str) -> None:
        """Handle transcription progress update.

        Args:
            message: Progress message
        """
        self.logger.debug(f"Transcription progress: {message}")
        # Could update a progress indicator in the view if needed

    def _on_recording_error(self, error: Exception) -> None:
        """Handle recording error.

        Args:
            error: Exception that occurred during recording
        """
        self.logger.error(f"Recording error: {error}")
        self.view.show_error_message("Recording Error", f"Audio recording error: {error}")

        # Stop transcription on recording error
        if self._transcription_active:
            self.stop_transcription()

    # LLM-related methods
    def _process_transcription_with_llm(self, transcription_text: str) -> None:
        """Process transcription text with LLM if conditions are met.

        Args:
            transcription_text: Transcription text to process
        """
        if not self._llm_active or not self.llm_model.user_prompt.strip():
            return

        try:
            # Send transcription to LLM thread for processing
            if self._llm_thread and self._llm_request_queue:
                from datetime import datetime

                request = LLMRequest(
                    prompt=self.llm_model.user_prompt, transcription_text=transcription_text, timestamp=datetime.now()
                )
                try:
                    self._llm_request_queue.put_nowait(request)
                    self.logger.debug(f"Sent transcription to LLM: {len(transcription_text)} chars")
                except queue.Full:
                    self.logger.warning("LLM request queue is full, dropping transcription")
        except Exception:
            self.logger.exception("Error processing transcription with LLM")

    def start_llm_processing(self) -> bool:
        """Start LLM processing.

        Returns:
            True if LLM processing started successfully, False otherwise
        """
        if self._llm_active:
            self.logger.warning("LLM processing already active")
            return True

        try:
            self.logger.info("Starting LLM processing")

            # LLM connection checking is now handled by LLMThread
            # The model only manages data

            # Create LLM request queue
            self._llm_request_queue = queue.Queue(maxsize=20)

            # Create and configure LLM thread with current configuration
            config = self.config_model.config_data
            self._llm_thread = LLMThread(
                request_queue=self._llm_request_queue,
                endpoint=config.ollama_endpoint,
                model=config.ollama_model,
                api_timeout=config.api_timeout,
            )

            # Update API service with additional configuration
            self._llm_thread.api_service.update_config(max_retries=config.max_retries, retry_delay=config.retry_delay)

            # Set up LLM thread callbacks
            self._llm_thread.set_response_callback(self._on_llm_response)
            self._llm_thread.set_response_chunk_callback(self._on_llm_response_chunk)
            self._llm_thread.set_error_callback(self._on_llm_error)
            self._llm_thread.set_connection_status_callback(self._on_llm_connection_status)

            # Start LLM thread
            if not self._llm_thread.start_processing():
                self.logger.error("Failed to start LLM thread")
                self._cleanup_llm_thread()
                return False

            # LLM model no longer has start_processing method
            # Processing is handled by the LLMThread

            self._llm_active = True
            self.llm_status_signal.emit("LLM processing started")
            self.logger.info("LLM processing started successfully")
            return True

        except Exception as e:
            self.logger.exception("Failed to start LLM processing")
            self.llm_error_signal.emit(f"Failed to start LLM processing: {e}")
            self._cleanup_llm_thread()
            return False

    def stop_llm_processing(self) -> None:
        """Stop LLM processing."""
        if not self._llm_active:
            self.logger.debug("LLM processing not active")
            return

        try:
            self.logger.info("Stopping LLM processing")
            self._llm_active = False

            # LLM model no longer has stop_processing method
            # Processing is handled by the LLMThread

            # Clean up LLM thread
            self._cleanup_llm_thread()

            self.llm_status_signal.emit("LLM processing stopped")
            self.logger.info("LLM processing stopped successfully")

        except Exception as e:
            self.logger.exception("Error stopping LLM processing")
            self.llm_error_signal.emit(f"Error stopping LLM processing: {e}")

    def _cleanup_llm_thread(self) -> None:
        """Clean up LLM thread and resources."""
        try:
            # Stop LLM thread
            if self._llm_thread:
                self.logger.debug("Stopping LLM thread")
                self._llm_thread._stop_event.set()
                self._llm_thread._processing = False
                self._llm_thread = None

            # Clear LLM request queue
            if self._llm_request_queue:
                self.logger.debug("Clearing LLM request queue")
                cleared_count = 0
                while not self._llm_request_queue.empty():
                    try:
                        self._llm_request_queue.get_nowait()
                        self._llm_request_queue.task_done()
                        cleared_count += 1
                    except queue.Empty:
                        break

                if cleared_count > 0:
                    self.logger.debug(f"Cleared {cleared_count} items from LLM request queue")

                self._llm_request_queue = None

            self.logger.debug("LLM thread cleanup completed")

        except Exception:
            self.logger.exception("Error cleaning up LLM thread")

    def _on_llm_response(self, response) -> None:
        """Handle complete LLM response.

        Args:
            response: LLMResponse object
        """
        try:
            self.logger.info(f"Received LLM response: {len(response.text)} characters")
            self.llm_response_signal.emit(response.text)
        except Exception:
            self.logger.exception("Error handling LLM response")

    def _on_llm_response_chunk(self, chunk: str) -> None:
        """Handle streaming LLM response chunk.

        Args:
            chunk: Response chunk text
        """
        try:
            self.llm_response_chunk_signal.emit(chunk)
        except Exception:
            self.logger.exception("Error handling LLM response chunk")

    def _on_llm_error(self, error: Exception) -> None:
        """Handle LLM error.

        Args:
            error: Exception that occurred during LLM processing
        """
        self.logger.error(f"LLM error: {error}")
        self.llm_error_signal.emit(f"LLM error: {error}")

    def _on_llm_connection_status(self, connected: bool) -> None:
        """Handle LLM connection status change.

        Args:
            connected: True if connected, False otherwise
        """
        self.logger.debug(f"LLM connection status: {connected}")
        self.llm_connection_status_signal.emit(connected)

    # LLM Model callback methods (called directly from LLM model)
    def _on_llm_model_response(self, response_text: str) -> None:
        """Handle LLM model response callback.

        Args:
            response_text: Complete response text from LLM model
        """
        try:
            self.logger.info(f"Received LLM model response: {len(response_text)} characters")
            self.llm_response_signal.emit(response_text)
        except Exception:
            self.logger.exception("Error handling LLM model response")

    def _on_llm_model_response_chunk(self, chunk: str) -> None:
        """Handle LLM model response chunk callback.

        Args:
            chunk: Response chunk from LLM model
        """
        try:
            self.llm_response_chunk_signal.emit(chunk)
        except Exception:
            self.logger.exception("Error handling LLM model response chunk")

    def _on_llm_model_error(self, error_msg: str) -> None:
        """Handle LLM model error callback.

        Args:
            error_msg: Error message from LLM model
        """
        self.logger.error(f"LLM model error: {error_msg}")
        self.llm_error_signal.emit(error_msg)

    def _on_llm_model_connection_status(self, connected: bool) -> None:
        """Handle LLM model connection status callback.

        Args:
            connected: True if connected, False otherwise
        """
        self.logger.debug(f"LLM model connection status: {connected}")
        self.llm_connection_status_signal.emit(connected)

    # LLM UI update methods (called via Qt signals)
    def _update_llm_response_ui(self, response_text: str) -> None:
        """Update LLM response UI (called via Qt signal on main thread).

        Args:
            response_text: Complete LLM response text
        """
        try:
            self.view.set_llm_response_text(response_text)
            self.logger.debug(f"Updated LLM response UI: {len(response_text)} characters")
        except Exception:
            self.logger.exception("Error updating LLM response UI")

    def _update_llm_response_chunk_ui(self, chunk: str) -> None:
        """Update LLM response UI with streaming chunk (called via Qt signal on main thread).

        Args:
            chunk: Response chunk text
        """
        try:
            self.view.append_llm_response_text(chunk)
        except Exception:
            self.logger.exception("Error updating LLM response chunk UI")

    def _update_llm_status_ui(self, status: str) -> None:
        """Update LLM status UI (called via Qt signal on main thread).

        Args:
            status: Status message
        """
        try:
            self.view.set_llm_response_status(status)
        except Exception:
            self.logger.exception("Error updating LLM status UI")

    def _show_llm_error_ui(self, error_message: str) -> None:
        """Show LLM error UI (called via Qt signal on main thread).

        Args:
            error_message: Error message to display
        """
        try:
            self.view.show_error_message("LLM Error", error_message)
        except Exception:
            self.logger.exception("Error showing LLM error UI")

    def _update_llm_connection_status_ui(self, connected: bool) -> None:
        """Update LLM connection status UI (called via Qt signal on main thread).

        Args:
            connected: True if connected, False otherwise
        """
        try:
            status = "Connected to Ollama" if connected else "Disconnected from Ollama"
            self.view.set_llm_response_status(status)
        except Exception:
            self.logger.exception("Error updating LLM connection status UI")

    def is_transcription_active(self) -> bool:
        """Check if transcription is currently active.

        Returns:
            True if transcription is active, False otherwise
        """
        return self._transcription_active

    def get_transcription_stats(self) -> dict:
        """Get current transcription statistics.

        Returns:
            Dictionary containing transcription statistics
        """
        stats = {
            "transcription_active": self._transcription_active,
            "transcription_enabled": self.recording_model.transcription_enabled,
        }

        if self._audio_recorder_thread:
            stats.update(self._audio_recorder_thread.get_audio_stats())

        if self._audio_transcriber_thread:
            stats.update(self._audio_transcriber_thread.get_transcription_stats())

        return stats

    # Configuration callback methods
    def _on_config_changed(self, config_data) -> None:
        """Handle configuration changes.

        Args:
            config_data: Updated configuration data
        """
        try:
            self.logger.info("Configuration changed, updating services")

            # Update LLM service configuration
            self._update_llm_service_config()

            # Emit signal for UI updates
            self.config_updated_signal.emit()

        except Exception:
            self.logger.exception("Error handling configuration change")

    def _on_whisper_model_changed(self, model_name: str) -> None:
        """Handle Whisper model changes.

        Args:
            model_name: New Whisper model name
        """
        try:
            self.logger.info(f"Whisper model changed to: {model_name}")

            # Update transcriber thread if it exists and can be changed
            if self._audio_transcriber_thread and self._audio_transcriber_thread.can_change_model():
                success = self._audio_transcriber_thread.update_model_config(
                    model_name=model_name, language=self.config_model.config_data.language
                )
                if success:
                    self.logger.info(f"Transcriber updated to use model: {model_name}")
                else:
                    self.logger.warning(f"Failed to update transcriber model to: {model_name}")

            # Update UI
            self.view.emit_whisper_model_status_update(model_name)

        except Exception:
            self.logger.exception(f"Error handling Whisper model change to {model_name}")

    def _on_ollama_model_changed(self, model_name: str) -> None:
        """Handle Ollama model changes.

        Args:
            model_name: New Ollama model name
        """
        try:
            self.logger.info(f"Ollama model changed to: {model_name}")

            # Update LLM service if thread exists
            if self._llm_thread and hasattr(self._llm_thread, "api_service"):
                self._llm_thread.api_service.update_model(model_name)
                self.logger.debug(f"LLM service model updated to: {model_name}")
            else:
                self.logger.debug(
                    f"LLM thread not available, model {model_name} will be applied when thread is created"
                )

            # Update UI
            self.view.emit_ollama_model_status_update(model_name)

        except Exception:
            self.logger.exception(f"Error handling Ollama model change to {model_name}")

    def _on_download_progress(self, progress: ModelDownloadProgress) -> None:
        """Handle model download progress updates.

        Args:
            progress: Download progress data
        """
        try:
            # Emit signal for thread-safe UI update
            self.model_download_progress_signal.emit(progress.model_name, progress.progress_percent, progress.status)

        except Exception:
            self.logger.exception("Error handling download progress update")

    def _on_download_completed(self, model_name: str, success: bool, error_message: str) -> None:
        """Handle model download completion.

        Args:
            model_name: Name of the downloaded model
            success: Whether download was successful
            error_message: Error message if failed
        """
        try:
            # Emit signal for thread-safe UI update
            self.model_download_completed_signal.emit(model_name, success, error_message or "")

            if success:
                self.logger.info(f"Model {model_name} downloaded successfully")
            else:
                self.logger.error(f"Model {model_name} download failed: {error_message}")

        except Exception:
            self.logger.exception("Error handling download completion")

    # Configuration UI callback handlers
    def _handle_whisper_model_changed(self, model_name: str) -> None:
        """Handle Whisper model selection from UI.

        Args:
            model_name: Selected Whisper model name
        """
        try:
            if model_name and model_name != self.config_model.whisper_model:
                # Check if model is available locally
                if not self.download_service.is_model_available(model_name):
                    # Start download
                    self.logger.info(f"Starting download for Whisper model: {model_name}")
                    success = self.download_service.download_model(model_name)
                    if not success:
                        self.view.set_download_status(f"Failed to start download for {model_name}")
                        return

                # Update configuration
                success = self.config_model.set_whisper_model(model_name)
                if success:
                    self.logger.info(f"Whisper model updated to: {model_name}")
                else:
                    self.logger.error(f"Failed to update Whisper model to: {model_name}")

        except Exception:
            self.logger.exception(f"Error handling Whisper model change from UI: {model_name}")

    def _handle_ollama_model_changed(self, model_name: str) -> None:
        """Handle Ollama model selection from UI.

        Args:
            model_name: Selected Ollama model name
        """
        try:
            if model_name and model_name != self.config_model.ollama_model:
                # Update configuration
                success = self.config_model.set_ollama_model(model_name)
                if success:
                    self.logger.info(f"Ollama model updated to: {model_name}")
                else:
                    self.logger.error(f"Failed to update Ollama model to: {model_name}")

        except Exception:
            self.logger.exception(f"Error handling Ollama model change from UI: {model_name}")

    # Configuration UI update methods (Qt slots)
    def _update_config_ui(self) -> None:
        """Update configuration UI elements."""
        try:
            config = self.config_model.config_data
            self.view.update_current_whisper_model(config.whisper_model)
            self.view.update_current_ollama_model(config.ollama_model)
            self.logger.debug("Configuration UI updated")
        except Exception:
            self.logger.exception("Error updating configuration UI")

    def _update_download_progress_ui(self, model_name: str, progress: float, status: str) -> None:
        """Update download progress in UI.

        Args:
            model_name: Name of the model being downloaded
            progress: Progress percentage
            status: Download status
        """
        try:
            self.view.emit_download_progress_update(model_name, progress, status)
        except Exception:
            self.logger.exception("Error updating download progress UI")

    def _update_download_completed_ui(self, model_name: str, success: bool, message: str) -> None:
        """Update UI when download completes.

        Args:
            model_name: Name of the downloaded model
            success: Whether download was successful
            message: Success or error message
        """
        try:
            self.view.emit_download_completed(model_name, success, message)
        except Exception:
            self.logger.exception("Error updating download completion UI")

    # Public LLM control methods
    def is_llm_active(self) -> bool:
        """Check if LLM processing is currently active.

        Returns:
            True if LLM processing is active, False otherwise
        """
        return self._llm_active

    def get_llm_stats(self) -> dict:
        """Get current LLM processing statistics.

        Returns:
            Dictionary containing LLM processing statistics
        """
        stats = {
            "llm_active": self._llm_active,
            "user_prompt": self.llm_model.user_prompt,
        }

        if self._llm_thread:
            stats.update(self._llm_thread.get_processing_stats())

        return stats

    def set_llm_prompt(self, prompt: str) -> None:
        """Set the LLM prompt text.

        Args:
            prompt: Prompt text to set
        """
        self.llm_model.set_user_prompt(prompt)
        self.view.set_prompt_text(prompt)
        self.logger.info(f"LLM prompt set: {len(prompt)} characters")

    def clear_llm_response(self) -> None:
        """Clear the LLM response text."""
        self.llm_model.clear_responses()
        self.view.clear_llm_response_text()
        self.logger.debug("LLM response cleared")

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop LLM processing if active
            if self._llm_active:
                self.stop_llm_processing()

            # Stop transcription if active
            if self._transcription_active:
                self.stop_transcription()

            # Stop any active recording
            if self.recording_model.is_recording:
                self.recording_model.stop_recording()

            # Clean up models
            self.audio_model.cleanup()
            self.recording_model.cleanup()
            self.llm_model.cleanup()

            # Clean up configuration services
            if hasattr(self, "download_service"):
                self.download_service.cleanup()
            if hasattr(self, "config_model"):
                self.config_model.save_configuration()

            # Clean up threads
            self._cleanup_transcription_threads()
            self._cleanup_llm_thread()

            # Clean up temporary folder
            self._cleanup_temp_folder()

            self.logger.info("Presenter cleanup completed")
        except Exception:
            self.logger.exception("Error during cleanup")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
