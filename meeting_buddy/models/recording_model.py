"""Recording Model for Meeting Buddy application.

This module contains the RecordingModel class that handles
recording data, transcription content, and recording metadata.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from .transcription_model import AudioChunk, TranscriptionModel, TranscriptionResult


class RecordingInfo:
    """Data class representing a recording."""

    def __init__(self, name: str, timestamp: datetime, file_path: Optional[str] = None):
        self.name = name
        self.timestamp = timestamp
        self.file_path = file_path
        self.transcription: str = ""
        self.duration: float = 0.0  # Duration in seconds
        self.is_active: bool = False
        self.audio_chunks: list[AudioChunk] = []
        self.transcription_results: list[TranscriptionResult] = []
        self.has_transcription: bool = False

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
        self._current_recording: Optional[RecordingInfo] = None
        self._current_transcription: str = ""
        self._is_recording: bool = False

        # Transcription integration
        self._transcription_model: Optional[TranscriptionModel] = None
        self._transcription_enabled: bool = False

        # Initialize transcription model
        self._initialize_transcription_model()

    def _initialize_transcription_model(self) -> None:
        """Initialize the transcription model."""
        try:
            # Create a transcription model that acts as a data container; real-time transcription is handled
            # by AudioTranscriberThread to avoid duplicate Whisper model loading.
            self._transcription_model = TranscriptionModel(use_whisper=False)
            # Enable transcription if the container is available
            self._transcription_enabled = self._transcription_model is not None

            # Set up callbacks (kept for compatibility; the model won't process Whisper itself)
            if self._transcription_model:
                self._transcription_model.set_transcription_callback(self._on_transcription_updated)
                self._transcription_model.set_chunk_processed_callback(self._on_chunk_processed)

            self.logger.info(f"Transcription model initialized (container mode): enabled={self._transcription_enabled}")
        except Exception:
            self.logger.exception("Failed to initialize transcription model")
            self._transcription_enabled = False

    def _on_transcription_updated(self, transcription: str) -> None:
        """Callback for when transcription is updated.

        Args:
            transcription: Updated transcription text
        """
        self._current_transcription = transcription
        if self._current_recording:
            self._current_recording.transcription = transcription
            self._current_recording.has_transcription = True
        self.logger.debug(f"Transcription updated: {len(transcription)} characters")

    def _on_chunk_processed(self, result: TranscriptionResult) -> None:
        """Callback for when an audio chunk is processed.

        Args:
            result: Transcription result for the processed chunk
        """
        if self._current_recording:
            self._current_recording.transcription_results.append(result)
        self.logger.debug(f"Chunk processed: {result.text}")

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

    @property
    def transcription_enabled(self) -> bool:
        """Check if transcription is enabled and available."""
        return self._transcription_enabled

    @property
    def transcription_model(self) -> Optional[TranscriptionModel]:
        """Get the transcription model instance."""
        return self._transcription_model

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recording_name = f"Recording_{timestamp}"

        # Create new recording
        self._current_recording = RecordingInfo(name=recording_name, timestamp=datetime.now())
        self._current_recording.is_active = True
        self._is_recording = True
        self._current_transcription = ""

        # Start transcription processing if enabled
        if self._transcription_enabled and self._transcription_model:
            self._transcription_model.start_processing()
            self.logger.debug("Started transcription processing")

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

        # Stop transcription processing if enabled
        if self._transcription_enabled and self._transcription_model:
            self._transcription_model.stop_processing()
            self.logger.debug("Stopped transcription processing")

        # Finalize recording
        self._current_recording.is_active = False
        self._current_recording.transcription = self._current_transcription

        # Copy transcription data from transcription model
        if self._transcription_model:
            self._current_recording.audio_chunks = self._transcription_model.audio_chunks.copy()
            self._current_recording.transcription_results = self._transcription_model.transcription_results.copy()

        self._is_recording = False

        stopped_recording = self._current_recording
        self.logger.info(f"Stopped recording: {stopped_recording}")

        # Clear current recording and transcription model data
        self._current_recording = None
        if self._transcription_model:
            self._transcription_model.clear_transcription()

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

    def update_recording_duration(self, duration: float) -> None:
        """Update the duration of the current recording.

        Args:
            duration: Duration in seconds
        """
        if self._current_recording:
            self._current_recording.duration = duration
            self.logger.debug(f"Updated recording duration: {duration:.1f}s")

    def add_audio_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[AudioChunk]:
        """Add an audio chunk for transcription during recording.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data

        Returns:
            The created AudioChunk object, or None if not recording or transcription disabled
        """
        if not self._is_recording or not self._transcription_enabled or not self._transcription_model:
            return None

        chunk = self._transcription_model.add_audio_chunk(audio_data, sample_rate)

        # Also add to current recording for immediate access
        if self._current_recording:
            self._current_recording.audio_chunks.append(chunk)

        self.logger.debug(f"Added audio chunk: {chunk}")
        return chunk

    def get_transcription_results(self, recording_index: Optional[int] = None) -> list[TranscriptionResult]:
        """Get transcription results for a recording or current session.

        Args:
            recording_index: Index of recording to get results for, or None for current session

        Returns:
            List of TranscriptionResult objects
        """
        if recording_index is not None:
            # Get results for specific recording
            recording = self.get_recording_by_index(recording_index)
            return recording.transcription_results if recording else []
        else:
            # Get results for current session
            if self._transcription_model:
                return self._transcription_model.transcription_results
            return []

    def add_transcription_result(self, result: TranscriptionResult) -> None:
        """Add a transcription result from the transcriber thread and update state."""
        # Update current transcription text
        self.append_transcription(result.text)
        # Store structured result
        if self._current_recording:
            self._current_recording.transcription_results.append(result)
        # Also reflect in transcription model container if available
        if self._transcription_model:
            # Private access to maintain existing pattern in model; safe minimal change
            self._transcription_model._transcription_results.append(result)
        self.logger.debug(f"Stored transcription result: {result}")

    def get_audio_chunks(self, recording_index: Optional[int] = None) -> list[AudioChunk]:
        """Get audio chunks for a recording or current session.

        Args:
            recording_index: Index of recording to get chunks for, or None for current session

        Returns:
            List of AudioChunk objects
        """
        if recording_index is not None:
            # Get chunks for specific recording
            recording = self.get_recording_by_index(recording_index)
            return recording.audio_chunks if recording else []
        else:
            # Get chunks for current session
            if self._transcription_model:
                return self._transcription_model.audio_chunks
            return []

    def export_transcription(self, recording_index: int, format_type: str = "text") -> str:
        """Export transcription for a specific recording.

        Args:
            recording_index: Index of the recording
            format_type: Export format ('text', 'timestamped', 'json')

        Returns:
            Formatted transcription string

        Raises:
            ValueError: If recording index is invalid or format type is unsupported
        """
        recording = self.get_recording_by_index(recording_index)
        if not recording:
            raise ValueError(f"Invalid recording index: {recording_index}")

        if format_type == "text":
            return recording.transcription
        elif format_type == "timestamped":
            lines = []
            for result in recording.transcription_results:
                lines.append(f"[{result.formatted_timestamp}] {result.text}")
            return "\n".join(lines)
        elif format_type == "json":
            import json

            data = {
                "recording_name": recording.name,
                "timestamp": recording.timestamp.isoformat(),
                "duration": recording.duration,
                "transcription": recording.transcription,
                "results": [
                    {"text": result.text, "timestamp": result.timestamp.isoformat(), "confidence": result.confidence}
                    for result in recording.transcription_results
                ],
            }
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def enable_transcription(self) -> bool:
        """Enable transcription if model is available.

        Returns:
            True if transcription was enabled, False otherwise
        """
        if self._transcription_model and self._transcription_model.model_loaded:
            self._transcription_enabled = True
            self.logger.info("Transcription enabled")
            return True
        else:
            self.logger.warning("Cannot enable transcription: model not available")
            return False

    def disable_transcription(self) -> None:
        """Disable transcription."""
        self._transcription_enabled = False
        if self._transcription_model:
            self._transcription_model.stop_processing()
        self.logger.info("Transcription disabled")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._transcription_model:
            self._transcription_model.cleanup()
        self.logger.info("RecordingModel cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
