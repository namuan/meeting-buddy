"""Meeting Buddy Presenter for the application.

This module contains the MeetingBuddyPresenter class that coordinates
between models and views, handling business logic and user interactions.
"""

import logging
from typing import Optional

from ..models.audio_device_model import AudioDeviceModel
from ..models.recording_model import RecordingModel
from ..views.meeting_buddy_view import MeetingBuddyView


class MeetingBuddyPresenter:
    """Presenter class for Meeting Buddy application.

    This class coordinates between models and views, handling
    business logic and user interactions following the MVP
    (Model-View-Presenter) architecture pattern.
    """

    def __init__(self):
        """Initialize the MeetingBuddyPresenter."""
        self.logger = logging.getLogger(__name__)

        # Initialize models
        self.audio_model = AudioDeviceModel()
        self.recording_model = RecordingModel()

        # Initialize view
        self.view = MeetingBuddyView()

        # Connect view callbacks to presenter methods
        self._connect_view_callbacks()

        self.logger.info("MeetingBuddyPresenter initialized")

    def _connect_view_callbacks(self) -> None:
        """Connect view callbacks to presenter methods."""
        self.view.on_output_device_changed = self._handle_output_device_changed
        self.view.on_start_recording = self._handle_start_recording
        self.view.on_stop_recording = self._handle_stop_recording
        self.view.on_progress_changed = self._handle_progress_changed
        self.view.on_recording_selected = self._handle_recording_selected

        self.logger.debug("View callbacks connected")

    def _initialize_ui(self) -> None:
        """Initialize UI with data from models."""
        # Populate device lists
        self._update_device_lists()

        # Populate recordings list
        self._update_recordings_list()

        # Set initial transcription
        self.view.set_transcription_text(self.recording_model.current_transcription)

        # Set initial recording state
        self.view.set_recording_state(self.recording_model.is_recording)

        self.logger.debug("UI initialized with model data")

    def _update_device_lists(self) -> None:
        """Update device combo boxes with current device lists."""
        # Update output devices
        output_devices = [str(device) for device in self.audio_model.output_devices]
        self.view.populate_output_devices(output_devices)

        self.logger.debug("Device lists updated")

    def _update_recordings_list(self) -> None:
        """Update recordings list with current recordings."""
        recordings = self.recording_model.get_recordings_display_list()
        self.view.populate_recordings(recordings)
        self.logger.debug("Recordings list updated")

    # Event handlers
    def _handle_output_device_changed(self, index: int) -> None:
        """Handle output device selection change.

        Args:
            index: Index of selected device in the combo box
        """
        if self.audio_model.select_output_device(index):
            selected_device = self.audio_model.selected_output_device
            self.logger.info(f"Output device changed to: {selected_device}")
            self.view.show_info_message("Device Selected", f"Output device: {selected_device.name}")
        else:
            self.logger.error(f"Failed to select output device at index {index}")
            self.view.show_error_message("Device Selection Error", "Failed to select output device")

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

            # Stop recording
            recording = self.recording_model.stop_recording()
            self.view.set_recording_state(False)

            if recording:
                self.logger.info(f"Stopped recording: {recording.name}")
                self.view.show_info_message("Recording Stopped", f"Recording '{recording.name}' saved")

                # Update recordings list
                self._update_recordings_list()

        except Exception as e:
            self.logger.exception("Failed to stop recording")
            self.view.show_error_message("Recording Error", f"Failed to stop recording: {e!s}")

    def _handle_progress_changed(self, value: int) -> None:
        """Handle progress slider value change.

        Args:
            value: New progress value (0-100)
        """
        self.logger.debug(f"Progress changed to: {value}%")
        # This could be used for seeking in playback or other progress-related functionality
        # For now, just log the change

    def _handle_recording_selected(self, index: int) -> None:
        """Handle recording list item selection.

        Args:
            index: Index of selected recording
        """
        recording = self.recording_model.get_recording_by_index(index)
        if recording:
            self.logger.info(f"Selected recording: {recording.name}")

            # Display the recording's transcription
            if recording.transcription:
                self.view.set_transcription_text(recording.transcription)
            else:
                self.view.set_transcription_text("No transcription available for this recording.")

            self.view.show_info_message("Recording Selected", f"Loaded: {recording.name}")
        else:
            self.logger.error(f"Invalid recording index: {index}")
            self.view.show_error_message("Selection Error", "Invalid recording selection")

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

    def get_selected_output_device(self) -> Optional[str]:
        """Get the currently selected output device name.

        Returns:
            Selected output device name or None
        """
        device = self.audio_model.selected_output_device
        return device.name if device else None

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

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop any active recording
            if self.recording_model.is_recording:
                self.recording_model.stop_recording()

            # Clean up audio model
            self.audio_model.cleanup()

            self.logger.info("Presenter cleanup completed")
        except Exception:
            self.logger.exception("Error during cleanup")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
