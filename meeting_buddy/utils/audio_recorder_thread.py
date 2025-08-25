"""Audio Recorder Thread for Meeting Buddy application.

This module contains the AudioRecorderThread class that handles
chunked audio recording and integration with transcription processing.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pyaudio

from ..models.audio_device_model import AudioDeviceInfo
from ..models.transcription_data import AudioChunk
from ..models.transcription_model import TranscriptionModel


class AudioRecorderThread(threading.Thread):
    """Thread class for recording audio in chunks and feeding to transcription.

    This class handles continuous audio recording from a selected device,
    processes audio data into chunks, and feeds them to a transcription model
    for real-time speech-to-text conversion.
    """

    def __init__(
        self,
        device: AudioDeviceInfo,
        transcription_model: Optional[TranscriptionModel] = None,
        chunk_duration_seconds: float = 2.0,
        sample_rate: int = 16000,
        channels: int = 1,
        format_type: int = pyaudio.paInt16,
        frames_per_buffer: int = 1024,
    ):
        """Initialize the AudioRecorderThread.

        Args:
            device: Audio device to record from
            transcription_model: Optional transcription model for processing chunks
            chunk_duration_seconds: Duration of each audio chunk in seconds
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            format_type: PyAudio format type
            frames_per_buffer: Number of frames per buffer
        """
        super().__init__(daemon=True)
        self.logger = logging.getLogger(__name__)

        # Audio configuration
        self.device = device
        self.chunk_duration_seconds = chunk_duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self.format_type = format_type
        self.frames_per_buffer = frames_per_buffer

        # Transcription integration
        self.transcription_model = transcription_model

        # Recording state
        self._recording = False
        self._stop_event = threading.Event()
        self._recording_lock = threading.RLock()  # Reentrant lock for thread safety
        self._cleanup_lock = threading.Lock()  # Lock for cleanup operations
        self._pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

        # Audio data management
        self._current_chunk_frames: list[bytes] = []
        self._chunk_start_time: Optional[datetime] = None
        self._total_frames_recorded = 0

        # Callbacks
        self._chunk_ready_callback: Optional[Callable[[np.ndarray, datetime], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None

        self.logger.info(f"AudioRecorderThread initialized for device: {device.name}")
        self.logger.debug(
            f"Recording configuration: {chunk_duration_seconds}s chunks, {sample_rate}Hz, {channels}ch, {frames_per_buffer} frames/buffer"
        )

    def set_chunk_ready_callback(self, callback: Callable[[np.ndarray, datetime], None]) -> None:
        """Set callback function for when an audio chunk is ready.

        Args:
            callback: Function to call with (audio_data, timestamp) when chunk is ready
        """
        self._chunk_ready_callback = callback
        self.logger.debug("Chunk ready callback set")

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback function for error handling.

        Args:
            callback: Function to call when an error occurs
        """
        self._error_callback = callback
        self.logger.debug("Error callback set")

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @property
    def total_frames_recorded(self) -> int:
        """Get total number of frames recorded."""
        return self._total_frames_recorded

    @property
    def recording_duration_seconds(self) -> float:
        """Get total recording duration in seconds."""
        return self._total_frames_recorded / (self.sample_rate * self.channels)

    def start_recording(self) -> bool:
        """Start the recording process.

        Returns:
            True if recording started successfully, False otherwise
        """
        with self._recording_lock:
            if self._recording:
                self.logger.warning("Recording already in progress")
                return True

            try:
                self._initialize_audio()
                self._recording = True
                self._stop_event.clear()
                self.start()
                self.logger.info("Audio recording started")
                return True
            except Exception as e:
                self.logger.exception("Failed to start recording")
                if self._error_callback:
                    self._error_callback(e)
                return False

    def stop_recording(self) -> None:
        """Stop the recording process."""
        with self._recording_lock:
            if not self._recording:
                self.logger.debug("Recording not in progress")
                return

            self.logger.info("Stopping audio recording")
            self._stop_event.set()
            self._recording = False

        # Don't wait for thread to finish - let it cleanup in background
        # This prevents UI hanging when stopping recording
        self.logger.debug("Recording stop initiated (non-blocking)")

    def _initialize_audio(self) -> None:
        """Initialize PyAudio and audio stream."""
        try:
            self._pyaudio_instance = pyaudio.PyAudio()

            # Validate device
            device_info = self._pyaudio_instance.get_device_info_by_index(self.device.index)
            if device_info["maxInputChannels"] < self.channels:
                raise ValueError(f"Device does not support {self.channels} input channels")

            # Create audio stream
            self._stream = self._pyaudio_instance.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device.index,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self._audio_callback,
            )

            self.logger.debug(f"Audio stream initialized: {self.sample_rate}Hz, {self.channels}ch")
            self.logger.debug(
                f"Device validation passed: {device_info['name']} supports {device_info['maxInputChannels']} input channels"
            )

        except Exception as e:
            self._cleanup_audio()
            raise RuntimeError(f"Failed to initialize audio: {e}") from e

    def _audio_callback(self, in_data: bytes, frame_count: int, time_info: dict, status: int) -> tuple:
        """PyAudio callback for processing audio data.

        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Timing information
            status: Stream status

        Returns:
            Tuple of (output_data, continue_flag)
        """
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        if self._recording and not self._stop_event.is_set():
            # Initialize chunk if needed
            if not self._current_chunk_frames:
                self._chunk_start_time = datetime.now()

            self._current_chunk_frames.append(in_data)
            self._total_frames_recorded += frame_count

            # Log audio level for debugging (every 100 frames to avoid spam)
            if self._total_frames_recorded % (self.frames_per_buffer * 100) == 0:
                audio_array = np.frombuffer(in_data, dtype=np.int16)
                audio_level = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
                self.logger.debug(
                    f"Audio level check: max amplitude = {audio_level}, frames recorded = {self._total_frames_recorded}"
                )

            # Check if chunk is complete
            chunk_duration = len(self._current_chunk_frames) * self.frames_per_buffer / self.sample_rate
            if chunk_duration >= self.chunk_duration_seconds:
                self.logger.debug(
                    f"Chunk complete: {chunk_duration:.2f}s duration, {len(self._current_chunk_frames)} frames"
                )
                self._process_chunk()

        return (in_data, pyaudio.paContinue)

    def _process_chunk(self) -> None:
        """Process the current audio chunk."""
        if not self._current_chunk_frames or not self._chunk_start_time:
            return

        try:
            # Convert frames to numpy array
            audio_data = np.frombuffer(b"".join(self._current_chunk_frames), dtype=np.int16)

            # Convert to float32 and normalize
            audio_float32 = audio_data.astype(np.float32)
            if audio_float32.max() > 0:
                audio_float32 = audio_float32 / 32768.0  # Normalize int16 to float32

            # Handle multi-channel audio
            if self.channels > 1:
                # Reshape and take first channel or average channels
                audio_float32 = audio_float32.reshape(-1, self.channels)
                audio_float32 = np.mean(audio_float32, axis=1)  # Average channels to mono

            self.logger.debug(
                f"Processed audio chunk: {len(audio_float32)} samples, RMS level: {np.sqrt(np.mean(audio_float32** 2)):.4f}"
            )

            # Log chunk statistics
            chunk_stats = {
                "samples": len(audio_float32),
                "duration_ms": len(audio_float32) / self.sample_rate * 1000,
                "rms_level": np.sqrt(np.mean(audio_float32**2)),
                "peak_level": np.max(np.abs(audio_float32)),
                "timestamp": self._chunk_start_time.isoformat() if self._chunk_start_time else "unknown",
            }
            self.logger.debug(f"Chunk statistics: {chunk_stats}")

            # Send to transcription model if available
            if self.transcription_model:
                self.logger.debug(f"Sending chunk to transcription model: {len(audio_float32)} samples")
                audio_chunk = AudioChunk(audio_float32, self._chunk_start_time, self.sample_rate)
                self.transcription_model.add_audio_chunk(audio_chunk)
            else:
                self.logger.debug("No transcription model available, skipping transcription")

            # Call chunk ready callback
            if self._chunk_ready_callback:
                self._chunk_ready_callback(audio_float32, self._chunk_start_time)

        except Exception as e:
            self.logger.exception("Error processing audio chunk")
            if self._error_callback:
                self._error_callback(e)
        finally:
            # Reset chunk data
            self._current_chunk_frames.clear()
            self._chunk_start_time = None

    def run(self) -> None:
        """Main thread execution method."""
        self.logger.debug("Audio recording thread started")

        try:
            if self._stream:
                self._stream.start_stream()
                self.logger.debug("Audio stream started")

            # Keep thread alive while recording
            loop_count = 0
            while not self._stop_event.is_set():
                time.sleep(0.1)
                loop_count += 1

                # Log recording status every 10 seconds
                if loop_count % 100 == 0:  # 100 * 0.1s = 10s
                    self.logger.debug(
                        f"Recording status: {self.recording_duration_seconds:.1f}s recorded, {self._total_frames_recorded} frames"
                    )

            # Process any remaining chunk data
            if self._current_chunk_frames:
                self._process_chunk()

        except Exception as e:
            self.logger.exception("Error in recording thread")
            if self._error_callback:
                self._error_callback(e)
        finally:
            self._cleanup_audio()
            self.logger.debug("Audio recording thread finished")

    def _cleanup_audio(self) -> None:
        """Clean up audio resources."""
        with self._cleanup_lock:
            try:
                if self._stream:
                    try:
                        if self._stream.is_active():
                            self._stream.stop_stream()
                    except Exception:
                        self.logger.debug("Stream was already stopped")

                    try:
                        self._stream.close()
                    except Exception:
                        self.logger.debug("Stream was already closed")

                    self._stream = None
                    self.logger.debug("Audio stream closed")

                if self._pyaudio_instance:
                    try:
                        self._pyaudio_instance.terminate()
                    except Exception:
                        self.logger.debug("PyAudio was already terminated")

                    self._pyaudio_instance = None
                    self.logger.debug("PyAudio terminated")

            except Exception:
                self.logger.exception("Error during audio cleanup")

    def get_audio_stats(self) -> dict:
        """Get current audio recording statistics.

        Returns:
            Dictionary containing recording statistics
        """
        return {
            "is_recording": self._recording,
            "total_frames": self._total_frames_recorded,
            "duration_seconds": self.recording_duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "device_name": self.device.name,
            "chunk_duration": self.chunk_duration_seconds,
        }

    def __del__(self):
        """Destructor to ensure cleanup."""
        if self._recording:
            self.stop_recording()
        self._cleanup_audio()
