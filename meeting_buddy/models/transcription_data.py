"""Transcription data structures for Meeting Buddy application.

This module contains shared data classes used by both the TranscriptionModel
and AudioTranscriberThread to avoid duplication.
"""

from datetime import datetime
from typing import Optional

import numpy as np


class AudioChunk:
    """Data class representing an audio chunk for transcription."""

    def __init__(self, audio_data: np.ndarray, timestamp: datetime, sample_rate: int = 16000):
        self.audio_data = audio_data
        self.timestamp = timestamp
        self.sample_rate = sample_rate
        self.transcribed = False
        self.transcription_text = ""

    def __str__(self) -> str:
        return f"AudioChunk({len(self.audio_data)} samples, {self.timestamp.strftime('%H:%M:%S')})"

    def __repr__(self) -> str:
        return f"AudioChunk(samples={len(self.audio_data)}, timestamp={self.timestamp})"

    @property
    def duration_seconds(self) -> float:
        """Get duration of audio chunk in seconds."""
        return len(self.audio_data) / self.sample_rate

    @property
    def formatted_timestamp(self) -> str:
        """Get formatted timestamp string."""
        return self.timestamp.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds


class TranscriptionResult:
    """Data class representing a transcription result."""

    def __init__(self, text: str, confidence: float, timestamp: datetime, chunk_id: Optional[str] = None):
        self.text = text
        self.confidence = confidence
        self.timestamp = timestamp
        self.chunk_id = chunk_id

    def __str__(self) -> str:
        return f"{self.formatted_timestamp}: {self.text} (confidence: {self.confidence:.2f})"

    def __repr__(self) -> str:
        return f"TranscriptionResult(text='{self.text[:50]}...', confidence={self.confidence})"

    @property
    def formatted_timestamp(self) -> str:
        """Get formatted timestamp string."""
        return self.timestamp.strftime("%H:%M:%S")


class TranscriptionStats:
    """Data class for transcription statistics."""

    def __init__(self):
        self.chunks_processed = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        self.successful_transcriptions = 0
        self.empty_transcriptions = 0
        self.error_count = 0

    @property
    def average_processing_time(self) -> float:
        """Get average processing time per chunk in seconds."""
        if self.chunks_processed == 0:
            return 0.0
        return self.total_processing_time / self.chunks_processed

    @property
    def success_rate(self) -> float:
        """Get success rate as a percentage."""
        if self.chunks_processed == 0:
            return 0.0
        return (self.successful_transcriptions / self.chunks_processed) * 100.0

    def update_processing_time(self, processing_time: float) -> None:
        """Update processing time statistics."""
        self.chunks_processed += 1
        self.total_processing_time += processing_time
        self.last_processing_time = processing_time

    def record_successful_transcription(self) -> None:
        """Record a successful transcription."""
        self.successful_transcriptions += 1

    def record_empty_transcription(self) -> None:
        """Record an empty transcription."""
        self.empty_transcriptions += 1

    def record_error(self) -> None:
        """Record a transcription error."""
        self.error_count += 1

    def reset(self) -> None:
        """Reset all statistics."""
        self.chunks_processed = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        self.successful_transcriptions = 0
        self.empty_transcriptions = 0
        self.error_count = 0

    def to_dict(self) -> dict:
        """Convert statistics to dictionary."""
        return {
            "chunks_processed": self.chunks_processed,
            "total_processing_time": self.total_processing_time,
            "last_processing_time": self.last_processing_time,
            "average_processing_time": self.average_processing_time,
            "successful_transcriptions": self.successful_transcriptions,
            "empty_transcriptions": self.empty_transcriptions,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
        }
