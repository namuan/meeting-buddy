"""Recording Model for Meeting Buddy application.

This module contains the RecordingModel class that handles
recording data, transcription content, and recording metadata.
"""

import logging
from datetime import datetime
from typing import Optional


class RecordingInfo:
    """Data class representing a recording."""

    def __init__(self, name: str, timestamp: datetime, file_path: Optional[str] = None):
        self.name = name
        self.timestamp = timestamp
        self.file_path = file_path
        self.transcription: str = ""
        self.duration: float = 0.0  # Duration in seconds
        self.is_active: bool = False

    def __str__(self) -> str:
        return f"{self.name} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    def __repr__(self) -> str:
        return f"RecordingInfo(name='{self.name}', timestamp={self.timestamp})"

    @property
    def formatted_timestamp(self) -> str:
        """Get formatted timestamp string."""
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def formatted_duration(self) -> str:
        """Get formatted duration string."""
        if self.duration < 60:
            return f"{self.duration:.1f}s"
        elif self.duration < 3600:
            minutes = int(self.duration // 60)
            seconds = int(self.duration % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(self.duration // 3600)
            minutes = int((self.duration % 3600) // 60)
            return f"{hours}h {minutes}m"


class RecordingModel:
    """Model class for managing recordings and transcriptions.

    This class handles recording metadata, transcription content,
    and recording session management.
    """

    def __init__(self):
        """Initialize the RecordingModel."""
        self.logger = logging.getLogger(__name__)
        self._recordings: list[RecordingInfo] = []
        self._current_recording: Optional[RecordingInfo] = None
        self._current_transcription: str = ""
        self._is_recording: bool = False

        # Initialize with some sample recordings for demo purposes
        self._initialize_sample_recordings()

    def _initialize_sample_recordings(self) -> None:
        """Initialize with sample recordings for demonstration."""
        sample_recordings = [
            RecordingInfo(name="Recording 1", timestamp=datetime(2025, 7, 25, 7, 54, 53)),
            RecordingInfo(name="Recording 2", timestamp=datetime(2025, 8, 25, 7, 54, 53)),
        ]

        for recording in sample_recordings:
            self._recordings.append(recording)
            self.logger.debug(f"Added sample recording: {recording}")

    @property
    def recordings(self) -> list[RecordingInfo]:
        """Get list of all recordings."""
        return self._recordings.copy()

    @property
    def current_recording(self) -> Optional[RecordingInfo]:
        """Get currently active recording."""
        return self._current_recording

    @property
    def current_transcription(self) -> str:
        """Get current transcription content."""
        return self._current_transcription

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    def start_recording(self, recording_name: Optional[str] = None) -> RecordingInfo:
        """Start a new recording session.

        Args:
            recording_name: Optional name for the recording. If None, auto-generates.

        Returns:
            The created RecordingInfo object

        Raises:
            RuntimeError: If already recording
        """
        if self._is_recording:
            raise RuntimeError("Already recording. Stop current recording first.")

        # Generate recording name if not provided
        if not recording_name:
            recording_count = len(self._recordings) + 1
            recording_name = f"Recording {recording_count}"

        # Create new recording
        self._current_recording = RecordingInfo(name=recording_name, timestamp=datetime.now())
        self._current_recording.is_active = True
        self._is_recording = True
        self._current_transcription = ""

        self.logger.info(f"Started recording: {self._current_recording}")
        return self._current_recording

    def stop_recording(self) -> Optional[RecordingInfo]:
        """Stop the current recording session.

        Returns:
            The stopped RecordingInfo object, or None if not recording
        """
        if not self._is_recording or not self._current_recording:
            self.logger.warning("No active recording to stop")
            return None

        # Finalize recording
        self._current_recording.is_active = False
        self._current_recording.transcription = self._current_transcription
        self._is_recording = False

        # Add to recordings list
        self._recordings.append(self._current_recording)

        stopped_recording = self._current_recording
        self.logger.info(f"Stopped recording: {stopped_recording}")

        # Clear current recording
        self._current_recording = None

        return stopped_recording

    def update_transcription(self, transcription_text: str) -> None:
        """Update the current transcription content.

        Args:
            transcription_text: The transcription text to set
        """
        self._current_transcription = transcription_text

        # Also update current recording if active
        if self._current_recording:
            self._current_recording.transcription = transcription_text

        self.logger.debug(f"Updated transcription: {len(transcription_text)} characters")

    def append_transcription(self, additional_text: str) -> None:
        """Append text to the current transcription.

        Args:
            additional_text: Text to append to current transcription
        """
        if self._current_transcription:
            self._current_transcription += " " + additional_text
        else:
            self._current_transcription = additional_text

        # Also update current recording if active
        if self._current_recording:
            self._current_recording.transcription = self._current_transcription

        self.logger.debug(f"Appended to transcription: {len(additional_text)} characters")

    def clear_transcription(self) -> None:
        """Clear the current transcription content."""
        self._current_transcription = ""

        # Also clear current recording transcription if active
        if self._current_recording:
            self._current_recording.transcription = ""

        self.logger.debug("Cleared transcription content")

    def get_recording_by_index(self, index: int) -> Optional[RecordingInfo]:
        """Get a recording by its index in the recordings list.

        Args:
            index: Index of the recording

        Returns:
            RecordingInfo object or None if index is invalid
        """
        if 0 <= index < len(self._recordings):
            return self._recordings[index]
        return None

    def delete_recording(self, index: int) -> bool:
        """Delete a recording by its index.

        Args:
            index: Index of the recording to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        if 0 <= index < len(self._recordings):
            deleted_recording = self._recordings.pop(index)
            self.logger.info(f"Deleted recording: {deleted_recording}")
            return True

        self.logger.warning(f"Invalid recording index for deletion: {index}")
        return False

    def update_recording_duration(self, duration: float) -> None:
        """Update the duration of the current recording.

        Args:
            duration: Duration in seconds
        """
        if self._current_recording:
            self._current_recording.duration = duration
            self.logger.debug(f"Updated recording duration: {duration:.1f}s")

    def get_recordings_display_list(self) -> list[str]:
        """Get a list of recording display strings for UI.

        Returns:
            List of formatted recording strings
        """
        return [str(recording) for recording in self._recordings]
