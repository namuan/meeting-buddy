"""Transcription Model for Meeting Buddy application.

This module contains the TranscriptionModel class that handles
transcription data, audio chunks, and speech-to-text processing.
"""

import logging
import queue
import threading
from datetime import datetime
from typing import Callable, Optional

import numpy as np

try:
    import whisper
except ImportError:
    whisper = None


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


class TranscriptionModel:
    """Model class for managing transcription data and audio processing.

    This class handles audio chunks, transcription processing,
    and speech-to-text conversion using Whisper.
    """

    def __init__(self, model_name: str = "base", use_whisper: bool = True, language: Optional[str] = None):
        """Initialize the TranscriptionModel.

        Args:
            model_name: Whisper model name ('tiny', 'base', 'small', 'medium', 'large')
            use_whisper: If False, skip loading Whisper and only act as a data container
            language: Optional language code for transcription (unused for now)
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.model_name = model_name
        self.use_whisper = use_whisper
        self.language = language

        # Audio chunk management
        self._audio_chunks: list[AudioChunk] = []
        self._transcription_results: list[TranscriptionResult] = []
        self._current_transcription: str = ""

        # Processing state
        self._is_processing: bool = False
        self._processing_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._stop_processing: threading.Event = threading.Event()

        # Whisper model
        self._whisper_model: Optional[object] = None
        self._model_loaded: bool = False

        # Callbacks
        self._transcription_callback: Optional[Callable[[str], None]] = None
        self._chunk_processed_callback: Optional[Callable[[TranscriptionResult], None]] = None

        self._initialize_whisper()
        self.logger.debug(
            f"Whisper model configuration: model_loaded={self._model_loaded}, callbacks_set={bool(self._transcription_callback and self._chunk_processed_callback)}"
        )

    def _initialize_whisper(self) -> None:
        """Initialize Whisper model for transcription."""
        if not self.use_whisper:
            self.logger.info("Whisper usage disabled for TranscriptionModel (acting as data container only)")
            self._model_loaded = False
            return

        if whisper is None:
            self.logger.error("Whisper not available. Install openai-whisper package.")
            return

        try:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            self._whisper_model = whisper.load_model(self.model_name)
            self._model_loaded = True
            self.logger.info("Whisper model loaded successfully")
        except Exception:
            self.logger.exception("Failed to load Whisper model")
            self._model_loaded = False

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
    def is_processing(self) -> bool:
        """Check if currently processing audio chunks."""
        return self._is_processing

    @property
    def model_loaded(self) -> bool:
        """Check if Whisper model is loaded and ready."""
        return self._model_loaded

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

    def add_audio_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000) -> AudioChunk:
        """Add an audio chunk for transcription.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data

        Returns:
            The created AudioChunk object
        """
        chunk = AudioChunk(audio_data, datetime.now(), sample_rate)
        self._audio_chunks.append(chunk)

        # Log chunk statistics
        chunk_stats = {
            "chunk_id": len(self._audio_chunks),
            "samples": len(audio_data),
            "duration_s": chunk.duration_seconds,
            "sample_rate": sample_rate,
            "rms_level": float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0,
        }
        self.logger.debug(f"Audio chunk created: {chunk_stats}")

        # Add to processing queue if processing is active
        if self._is_processing:
            try:
                self._audio_queue.put(chunk)
                self.logger.debug(f"Added chunk to processing queue: queue size = {self._audio_queue.qsize()}")
            except Exception:
                self.logger.exception("Failed to add chunk to processing queue")
        else:
            self.logger.debug("Processing not active, chunk stored but not queued for processing")

        return chunk

    def start_processing(self) -> bool:
        """Start background processing of audio chunks.

        Returns:
            True if processing started successfully, False otherwise
        """
        if not self._model_loaded:
            self.logger.error("Cannot start processing: Whisper model not loaded")
            return False

        if self._is_processing:
            self.logger.warning("Processing already active")
            return True

        self._is_processing = True
        self._stop_processing.clear()

        self._processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self._processing_thread.start()

        self.logger.info("Started transcription processing")
        self.logger.debug(
            f"Processing thread started: thread_id={self._processing_thread.ident}, daemon={self._processing_thread.daemon}"
        )
        return True

    def stop_processing(self) -> None:
        """Stop background processing of audio chunks."""
        if not self._is_processing:
            self.logger.debug("Processing not active")
            return

        self._stop_processing.set()
        self._is_processing = False

        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)

        self.logger.info("Stopped transcription processing")

    def _processing_worker(self) -> None:
        """Background worker thread for processing audio chunks."""
        import time

        self.logger.debug("Processing worker started")
        processed_count = 0
        start_time = time.time()

        while not self._stop_processing.is_set():
            try:
                # Get chunk from queue with timeout
                queue_wait_start = time.time()
                chunk = self._audio_queue.get(timeout=1.0)
                queue_wait_time = time.time() - queue_wait_start

                self.logger.debug(
                    f"Retrieved chunk from queue: wait_time={queue_wait_time:.3f}s, queue_size={self._audio_queue.qsize()}"
                )

                self._process_chunk(chunk)
                self._audio_queue.task_done()
                processed_count += 1

                # Log processing statistics every 10 chunks
                if processed_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    avg_processing_time = elapsed_time / processed_count if processed_count > 0 else 0
                    self.logger.debug(
                        f"Processing statistics: {processed_count} chunks processed, avg_time={avg_processing_time:.3f}s/chunk"
                    )
            except queue.Empty:
                continue
            except Exception:
                self.logger.exception("Error in processing worker")

        total_time = time.time() - start_time
        self.logger.info(f"Processing worker stopped: processed {processed_count} chunks in {total_time:.2f}s")
        if processed_count > 0:
            self.logger.debug(f"Final processing statistics: avg_time={total_time / processed_count:.3f}s/chunk")

    def _process_chunk(self, chunk: AudioChunk) -> None:
        """Process a single audio chunk for transcription.

        Args:
            chunk: AudioChunk to process
        """
        if not self._model_loaded or chunk.transcribed:
            return

        try:
            import time

            processing_start = time.time()

            # Convert audio data to format expected by Whisper
            audio_float32 = chunk.audio_data.astype(np.float32)

            # Normalize audio if needed
            if audio_float32.max() > 1.0:
                audio_float32 = audio_float32 / np.max(np.abs(audio_float32))

            # Transcribe using Whisper
            result = self._whisper_model.transcribe(audio_float32)
            transcription_text = result["text"].strip()
            processing_time = time.time() - processing_start

            if transcription_text:
                # Create transcription result
                confidence = 1.0  # Whisper doesn't provide confidence scores
                transcription_result = TranscriptionResult(
                    text=transcription_text, confidence=confidence, timestamp=chunk.timestamp
                )

                # Update chunk and add result
                chunk.transcribed = True
                chunk.transcription_text = transcription_text
                self._transcription_results.append(transcription_result)

                # Update current transcription
                if self._current_transcription:
                    self._current_transcription += " " + transcription_text
                else:
                    self._current_transcription = transcription_text

                self.logger.info(f"Transcribed chunk: '{transcription_text}' (length: {len(transcription_text)} chars)")

                # Log transcription statistics
                transcription_stats = {
                    "chunk_duration_s": len(audio_float32) / chunk.sample_rate
                    if hasattr(audio_float32, "__len__")
                    else 0,
                    "processing_time_s": processing_time,
                    "text_length": len(transcription_text),
                    "total_transcription_length": len(self._current_transcription),
                    "chunks_processed": len(self._transcription_results),
                }
                self.logger.debug(f"Transcription statistics: {transcription_stats}")

                # Call callbacks
                if self._transcription_callback:
                    self.logger.debug("Calling transcription callback")
                    self._transcription_callback(self._current_transcription)
                else:
                    self.logger.debug("No transcription callback set")

                if self._chunk_processed_callback:
                    self.logger.debug("Calling chunk processed callback")
                    self._chunk_processed_callback(transcription_result)
                else:
                    self.logger.debug("No chunk processed callback set")

        except Exception:
            self.logger.exception("Error processing audio chunk")

    def clear_transcription(self) -> None:
        """Clear all transcription data."""
        self._audio_chunks.clear()
        self._transcription_results.clear()
        self._current_transcription = ""

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
            except queue.Empty:
                break

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
        self.stop_processing()
        self.clear_transcription()
        self.logger.info("TranscriptionModel cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
