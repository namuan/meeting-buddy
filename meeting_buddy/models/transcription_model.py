"""Transcription Model for Meeting Buddy application.

This module contains the TranscriptionModel class that handles
transcription data storage and management.
"""

import logging
from datetime import datetime
from typing import Callable, Optional

from .transcription_data import AudioChunk, TranscriptionResult


class TranscriptionModel:
    """Model class for managing transcription data.

    This class handles transcription data storage and management.
    Heavy processing (Whisper) is handled by AudioTranscriberThread in utils.
    """

    def __init__(self):
        """Initialize the TranscriptionModel.

        Note: Heavy processing (Whisper) is handled by AudioTranscriberThread.
        This model only manages transcription data and state.
        """
        self.logger = logging.getLogger(__name__)

        # Audio chunk management
        self._audio_chunks: list[AudioChunk] = []
        self._transcription_results: list[TranscriptionResult] = []
        self._current_transcription: str = ""

        # Callbacks for state change notifications
        self._transcription_callback: Optional[Callable[[str], None]] = None
        self._chunk_processed_callback: Optional[Callable[[TranscriptionResult], None]] = None

        self.logger.info("TranscriptionModel initialized (data container only)")

    # Data management methods

    @property
    def audio_chunks(self) -> list[AudioChunk]:
        """Get list of all audio chunks."""
        return self._audio_chunks.copy()

    @property
    def transcription_results(self) -> list[TranscriptionResult]:
        """Get list of all transcription results."""
        return self._transcription_results.copy()

    @property
    def current_transcription(self) -> str:
        """Get current combined transcription text."""
        return self._current_transcription

    @property
    def chunk_count(self) -> int:
        """Get the number of audio chunks stored."""
        return len(self._audio_chunks)

    @property
    def result_count(self) -> int:
        """Get the number of transcription results stored."""
        return len(self._transcription_results)

    def set_transcription_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for transcription updates.

        Args:
            callback: Function to call when transcription is updated
        """
        self._transcription_callback = callback
        self.logger.debug("Transcription callback set")

    def set_chunk_processed_callback(self, callback: Callable[[TranscriptionResult], None]) -> None:
        """Set callback function for when a chunk is processed.

        Args:
            callback: Function to call when a chunk is processed
        """
        self._chunk_processed_callback = callback
        self.logger.debug("Chunk processed callback set")

    def add_audio_chunk(self, chunk: AudioChunk) -> None:
        """Add an audio chunk to storage.

        Note: This method is primarily for data storage. Audio processing
        should be handled by AudioTranscriberThread.

        Args:
            chunk: AudioChunk object to store
        """
        self._audio_chunks.append(chunk)
        self.logger.debug(f"Audio chunk stored: {len(chunk.audio_data)} samples, {chunk.duration_seconds:.2f}s")

    def add_transcription_result(self, result: TranscriptionResult) -> None:
        """Add a transcription result to storage.

        Args:
            result: TranscriptionResult to store
        """
        self._transcription_results.append(result)

        # Update current transcription
        if self._current_transcription:
            self._current_transcription += " " + result.text
        else:
            self._current_transcription = result.text

        self.logger.debug(
            f"Transcription result added: '{result.text}' (total length: {len(self._current_transcription)})"
        )

        # Notify observers
        if self._transcription_callback:
            self._transcription_callback(self._current_transcription)
        if self._chunk_processed_callback:
            self._chunk_processed_callback(result)

    def update_transcription(self, transcription: str) -> None:
        """Update the current transcription text.

        Args:
            transcription: New transcription text
        """
        self._current_transcription = transcription
        self.logger.debug(f"Transcription updated: length={len(transcription)}")

        # Notify observers
        if self._transcription_callback:
            self._transcription_callback(self._current_transcription)

    def clear_transcription(self) -> None:
        """Clear all transcription data."""
        self._audio_chunks.clear()
        self._transcription_results.clear()
        self._current_transcription = ""
        self.logger.info("Cleared all transcription data")

    def get_transcription_by_timerange(self, start_time: datetime, end_time: datetime) -> list[TranscriptionResult]:
        """Get transcription results within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of TranscriptionResult objects within the time range
        """
        results = []
        for result in self._transcription_results:
            if start_time <= result.timestamp <= end_time:
                results.append(result)
        return results

    def export_transcription(self, format_type: str = "text") -> str:
        """Export transcription in specified format.

        Args:
            format_type: Export format ('text', 'timestamped', 'json')

        Returns:
            Formatted transcription string
        """
        if format_type == "text":
            return self._current_transcription
        elif format_type == "timestamped":
            lines = []
            for result in self._transcription_results:
                lines.append(f"[{result.formatted_timestamp}] {result.text}")
            return "\n".join(lines)
        elif format_type == "json":
            import json

            data = {
                "transcription": self._current_transcription,
                "results": [
                    {"text": result.text, "timestamp": result.timestamp.isoformat(), "confidence": result.confidence}
                    for result in self._transcription_results
                ],
            }
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_transcription()
        self.logger.info("TranscriptionModel cleanup completed")
