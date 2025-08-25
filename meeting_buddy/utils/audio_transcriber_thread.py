"""Audio Transcriber Thread for Meeting Buddy application.

This module contains the AudioTranscriberThread class that handles
real-time transcription processing using Whisper.
"""

import logging
import queue
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import numpy as np

try:
    import whisper
except ImportError:
    whisper = None

from ..models.transcription_data import TranscriptionResult, TranscriptionStats


class AudioTranscriberThread(threading.Thread):
    """Thread class for real-time audio transcription using Whisper.

    This class processes audio chunks from a queue and generates
    transcription results using OpenAI's Whisper model.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        model_name: str = "base",
        language: Optional[str] = None,
        max_queue_size: int = 100,
        processing_timeout: float = 1.0,
    ):
        """Initialize the AudioTranscriberThread.

        Args:
            audio_queue: Queue containing audio chunks to transcribe
            model_name: Whisper model name ('tiny', 'base', 'small', 'medium', 'large')
            language: Language code for transcription (None for auto-detect)
            max_queue_size: Maximum number of items to keep in queue
            processing_timeout: Timeout for queue operations in seconds
        """
        super().__init__(daemon=True)
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.audio_queue = audio_queue
        self.model_name = model_name
        self.language = language
        self.max_queue_size = max_queue_size
        self.processing_timeout = processing_timeout

        # Whisper model
        self._whisper_model: Optional[object] = None
        self._model_loaded = False

        # Processing state
        self._transcribing = False
        self._stop_event = threading.Event()
        self._transcribing_lock = threading.RLock()  # Reentrant lock for thread safety
        self._cleanup_lock = threading.Lock()  # Lock for cleanup operations

        # Statistics and results storage
        self._stats = TranscriptionStats()
        self._transcription_results: list[TranscriptionResult] = []

        # Callbacks
        self._transcription_callback: Optional[Callable[[TranscriptionResult], None]] = None
        self._chunk_processed_callback: Optional[Callable[[TranscriptionResult], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._progress_callback: Optional[Callable[[str], None]] = None

        # Initialize Whisper model
        self._initialize_whisper()

        self.logger.info(f"AudioTranscriberThread initialized with model: {model_name}")
        self.logger.debug(
            f"Transcriber configuration: model={model_name}, language={language}, max_queue_size={max_queue_size}, timeout={processing_timeout}s"
        )

    def _initialize_whisper(self) -> None:
        """Initialize Whisper model for transcription."""
        if whisper is None:
            self.logger.error("Whisper not available. Install openai-whisper package.")
            return

        try:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            if self._progress_callback:
                self._progress_callback(f"Loading {self.model_name} model...")

            self._whisper_model = whisper.load_model(self.model_name)
            self._model_loaded = True

            self.logger.info(f"Whisper model '{self.model_name}' loaded successfully")
            self.logger.debug("Model loading completed in background thread")
            if self._progress_callback:
                self._progress_callback("Model loaded successfully")

        except Exception as e:
            self.logger.exception("Failed to load Whisper model")
            self._model_loaded = False
            if self._error_callback:
                self._error_callback(e)

    def set_transcription_callback(self, callback: Callable[[TranscriptionResult], None]) -> None:
        """Set callback function for transcription results.

        Args:
            callback: Function to call with TranscriptionResult when transcription is complete
        """
        self._transcription_callback = callback
        self.logger.debug("Transcription callback set")

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback function for error handling.

        Args:
            callback: Function to call when an error occurs
        """
        self._error_callback = callback
        self.logger.debug("Error callback set")

    def set_progress_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for progress updates.

        Args:
            callback: Function to call with progress messages
        """
        self._progress_callback = callback
        self.logger.debug("Progress callback set")

    def set_chunk_processed_callback(self, callback: Callable[[TranscriptionResult], None]) -> None:
        """Set callback function for chunk processing completion.

        Args:
            callback: Function to call when a chunk is processed
        """
        self._chunk_processed_callback = callback
        self.logger.debug("Chunk processed callback set")

    @property
    def is_transcribing(self) -> bool:
        """Check if currently transcribing."""
        return self._transcribing

    @property
    def model_loaded(self) -> bool:
        """Check if Whisper model is loaded and ready."""
        return self._model_loaded

    @property
    def chunks_processed(self) -> int:
        """Get number of chunks processed."""
        return self._stats.chunks_processed

    @property
    def average_processing_time(self) -> float:
        """Get average processing time per chunk in seconds."""
        return self._stats.average_processing_time

    @property
    def last_processing_time(self) -> float:
        """Get processing time of last chunk in seconds."""
        return self._stats.last_processing_time

    def start_transcribing(self) -> bool:
        """Start the transcription process.

        Returns:
            True if transcription started successfully, False otherwise
        """
        with self._transcribing_lock:
            if not self._model_loaded:
                self.logger.error("Cannot start transcribing: Whisper model not loaded")
                return False

            if self._transcribing:
                self.logger.warning("Transcription already in progress")
                return True

            self._transcribing = True
            self._stop_event.clear()
            self.start()

            self.logger.info("Audio transcription started")
            return True

    def stop_transcribing(self) -> None:
        """Stop the transcription process."""
        with self._transcribing_lock:
            if not self._transcribing:
                self.logger.debug("Transcription not in progress")
                return

            self.logger.info("Stopping audio transcription")
            self._stop_event.set()
            self._transcribing = False

        # Don't wait for thread to finish - let it cleanup in background
        # This prevents UI hanging when stopping transcription
        self.logger.debug("Transcription stop initiated (non-blocking)")

    def add_audio_chunk(self, audio_data: np.ndarray, timestamp: datetime, chunk_id: Optional[str] = None) -> bool:
        """Add an audio chunk to the transcription queue.

        Args:
            audio_data: Audio data as numpy array
            timestamp: Timestamp when the chunk was recorded
            chunk_id: Optional identifier for the chunk

        Returns:
            True if chunk was added successfully, False if queue is full
        """
        if not self._transcribing:
            self.logger.debug("Not transcribing, ignoring audio chunk")
            return False

        # Check queue size and remove old items if necessary
        while self.audio_queue.qsize() >= self.max_queue_size:
            try:
                self.audio_queue.get_nowait()
                self.logger.warning("Dropped audio chunk due to full queue")
            except queue.Empty:
                break

        try:
            chunk_data = {
                "audio_data": audio_data,
                "timestamp": timestamp,
                "chunk_id": chunk_id or f"chunk_{timestamp.timestamp()}",
            }
            self.audio_queue.put(chunk_data, timeout=0.1)
            self.logger.debug(
                f"Added audio chunk to queue: {chunk_data['chunk_id']}, queue size: {self.audio_queue.qsize()}"
            )

            # Log queue statistics periodically
            if self.audio_queue.qsize() % 10 == 0 and self.audio_queue.qsize() > 0:
                self.logger.debug(f"Queue status: {self.audio_queue.qsize()}/{self.max_queue_size} items")

            return True
        except queue.Full:
            self.logger.warning("Failed to add audio chunk: queue full")
            return False

    def run(self) -> None:
        """Main thread execution method."""
        self.logger.debug("Audio transcription thread started")

        while not self._stop_event.is_set():
            try:
                # Get audio chunk from queue
                queue_wait_start = time.time()
                chunk_data = self.audio_queue.get(timeout=self.processing_timeout)
                queue_wait_time = time.time() - queue_wait_start

                self.logger.debug(
                    f"Retrieved chunk from queue: {chunk_data.get('chunk_id', 'unknown')}, wait time: {queue_wait_time:.3f}s"
                )

                # Process the chunk
                self._process_audio_chunk(chunk_data)

                # Mark task as done
                self.audio_queue.task_done()

            except queue.Empty:
                # No chunks to process, continue waiting
                continue
            except Exception as e:
                self.logger.exception("Error in transcription thread")
                if self._error_callback:
                    self._error_callback(e)

        # Process any remaining chunks
        self._process_remaining_chunks()

        self.logger.debug("Audio transcription thread finished")

    def _process_audio_chunk(self, chunk_data: dict) -> None:
        """Process a single audio chunk for transcription.

        Args:
            chunk_data: Dictionary containing audio data, timestamp, and chunk_id
        """
        if not self._model_loaded:
            return

        start_time = time.time()

        try:
            audio_data = chunk_data["audio_data"]
            timestamp = chunk_data["timestamp"]
            chunk_id = chunk_data["chunk_id"]

            self._log_chunk_processing_start(chunk_id, audio_data)

            # Prepare audio data for Whisper
            processed_audio = self._prepare_audio_for_whisper(audio_data, chunk_id)
            if processed_audio is None:
                return

            # Perform transcription
            transcription_result = self._transcribe_with_whisper(processed_audio, chunk_id, timestamp)

            if transcription_result:
                self._handle_successful_transcription(transcription_result, start_time, processed_audio)
            else:
                self._handle_empty_transcription(chunk_id, processed_audio)

        except Exception as e:
            self._stats.record_error()
            self.logger.exception(f"Error processing audio chunk: {chunk_data.get('chunk_id', 'unknown')}")
            if self._error_callback:
                self._error_callback(e)

    def _log_chunk_processing_start(self, chunk_id: str, audio_data) -> None:
        """Log the start of chunk processing."""
        self.logger.debug(f"Processing audio chunk: {chunk_id}")
        self.logger.debug(
            f"Audio data shape: {audio_data.shape if hasattr(audio_data, 'shape') else len(audio_data)}, "
            f"dtype: {audio_data.dtype if hasattr(audio_data, 'dtype') else type(audio_data)}"
        )

    def _prepare_audio_for_whisper(self, audio_data, chunk_id: str):
        """Prepare audio data for Whisper transcription."""
        if not isinstance(audio_data, np.ndarray):
            self.logger.error(f"Invalid audio data type: {type(audio_data)}")
            return None

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize if needed
        if audio_data.max() > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        return audio_data

    def _transcribe_with_whisper(self, audio_data, chunk_id: str, timestamp):
        """Perform Whisper transcription."""
        transcribe_options = {
            "fp16": False,  # Use fp32 for better compatibility
            "language": self.language,
        }

        self.logger.debug(f"Starting Whisper transcription for chunk {chunk_id} with options: {transcribe_options}")
        transcribe_start_time = time.time()

        result = self._whisper_model.transcribe(audio_data, **transcribe_options)
        transcribe_duration = time.time() - transcribe_start_time

        transcription_text = result["text"].strip()
        self.logger.debug(f"Whisper transcription completed in {transcribe_duration:.2f}s for chunk {chunk_id}")

        self._log_whisper_result_details(result)

        if transcription_text:
            return TranscriptionResult(
                text=transcription_text,
                confidence=1.0,  # Default confidence
                timestamp=timestamp,
                chunk_id=chunk_id,
            )
        return None

    def _log_whisper_result_details(self, result: dict) -> None:
        """Log additional Whisper result information."""
        if "segments" in result:
            segment_count = len(result["segments"])
            self.logger.debug(f"Transcription segments: {segment_count} segments detected")

        if "language" in result:
            detected_language = result["language"]
            self.logger.debug(f"Detected language: {detected_language}")

    def _handle_successful_transcription(self, transcription_result, start_time: float, audio_data) -> None:
        """Handle successful transcription result."""
        # Update transcription data
        self._transcription_results.append(transcription_result)

        # Update statistics
        processing_time = time.time() - start_time
        self._stats.update_processing_time(processing_time)
        self._stats.record_successful_transcription()

        self.logger.info(
            f"Transcribed chunk {transcription_result.chunk_id}: '{transcription_result.text}' "
            f"(processed in {processing_time:.2f}s)"
        )

        self._log_performance_metrics(audio_data, processing_time, transcription_result.text)
        self._call_transcription_callbacks(transcription_result)

    def _call_transcription_callbacks(self, transcription_result) -> None:
        """Call transcription callbacks."""
        if self._transcription_callback:
            self._transcription_callback(transcription_result)

        if self._chunk_processed_callback:
            self._chunk_processed_callback(transcription_result)

    def _log_performance_metrics(self, audio_data, processing_time: float, transcription_text: str) -> None:
        """Log performance metrics."""
        # Note: sample_rate is not defined in this class, using a default value
        sample_rate = getattr(self, "sample_rate", 16000)  # Default to 16kHz
        audio_duration = len(audio_data) / sample_rate
        real_time_factor = audio_duration / processing_time if processing_time > 0 else 0
        chars_per_sec = len(transcription_text) / processing_time if processing_time > 0 else 0

        self.logger.debug(
            f"Performance metrics - Audio: {audio_duration:.2f}s, Processing: {processing_time:.2f}s, "
            f"RTF: {real_time_factor:.2f}x, Chars/sec: {chars_per_sec:.1f}"
        )

    def _handle_empty_transcription(self, chunk_id: str, audio_data) -> None:
        """Handle empty transcription result."""
        # Record empty transcription in stats
        self._stats.record_empty_transcription()

        # Note: sample_rate is not defined in this class, using a default value
        sample_rate = getattr(self, "sample_rate", 16000)  # Default to 16kHz
        self.logger.debug(
            f"Empty transcription for chunk: {chunk_id} (audio duration: {len(audio_data) / sample_rate:.2f}s)"
        )

        # Log audio characteristics for empty transcriptions
        audio_stats = {
            "samples": len(audio_data),
            "duration_s": len(audio_data) / sample_rate,
            "rms_level": np.sqrt(np.mean(audio_data**2)),
            "peak_level": np.max(np.abs(audio_data)),
            "zero_crossings": np.sum(np.diff(np.signbit(audio_data))),
        }
        self.logger.debug(f"Audio characteristics for empty transcription: {audio_stats}")

    def _process_remaining_chunks(self) -> None:
        """Process any remaining chunks in the queue before stopping."""
        remaining_chunks = 0

        while not self.audio_queue.empty():
            try:
                chunk_data = self.audio_queue.get_nowait()
                self._process_audio_chunk(chunk_data)
                self.audio_queue.task_done()
                remaining_chunks += 1
            except queue.Empty:
                break
            except Exception:
                self.logger.exception("Error processing remaining chunk")

        if remaining_chunks > 0:
            self.logger.info(f"Processed {remaining_chunks} remaining chunks during shutdown")
        else:
            self.logger.debug("No remaining chunks to process during shutdown")

    def clear_queue(self) -> int:
        """Clear all pending audio chunks from the queue.

        Returns:
            Number of chunks that were cleared
        """
        cleared_count = 0

        with self._cleanup_lock:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                    cleared_count += 1
                except queue.Empty:
                    break

            if cleared_count > 0:
                self.logger.info(f"Cleared {cleared_count} chunks from transcription queue")

        return cleared_count

    def _cleanup_transcription_resources(self) -> None:
        """Clean up transcription resources."""
        with self._cleanup_lock:
            try:
                # Clear any remaining queue items
                self.clear_queue()

                # Reset statistics
                self._stats.reset()

                self.logger.debug("Transcription resources cleaned up")

            except Exception:
                self.logger.exception("Error during transcription cleanup")

    def get_transcription_stats(self) -> dict:
        """Get current transcription statistics.

        Returns:
            Dictionary containing transcription statistics
        """
        stats_dict = self._stats.to_dict()
        stats_dict.update({
            "is_transcribing": self._transcribing,
            "model_loaded": self._model_loaded,
            "model_name": self.model_name,
            "language": self.language,
            "queue_size": self.audio_queue.qsize(),
        })
        return stats_dict

    def reload_model(self, model_name: Optional[str] = None) -> bool:
        """Reload the Whisper model.

        Args:
            model_name: Optional new model name to load

        Returns:
            True if model was reloaded successfully, False otherwise
        """
        if model_name:
            self.model_name = model_name

        self.logger.info(f"Reloading Whisper model: {self.model_name}")

        # Clear existing model
        self._whisper_model = None
        self._model_loaded = False

        # Reload model
        self._initialize_whisper()

        return self._model_loaded

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if self._transcribing:
                self.stop_transcribing()
            self._cleanup_transcription_resources()
        except Exception:
            # Log exceptions in destructor but don't raise them
            import logging

            logging.getLogger(__name__).debug("Exception during destructor cleanup", exc_info=True)
