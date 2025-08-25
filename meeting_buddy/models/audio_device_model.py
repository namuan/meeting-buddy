"""Audio Device Model for Meeting Buddy application.

This module contains the AudioDeviceModel class that handles
audio device detection, enumeration, and management.
"""

import logging
from typing import Optional

import pyaudio


class AudioDeviceInfo:
    """Data class representing an audio device."""

    def __init__(self, name: str, index: int, channels: int, sample_rate: int, device_type: str):
        self.name = name
        self.index = index
        self.channels = channels
        self.sample_rate = sample_rate
        self.device_type = device_type  # 'input' or 'output'

    def __str__(self) -> str:
        return f"{self.name} ({self.channels} ch)"

    def __repr__(self) -> str:
        return f"AudioDeviceInfo(name='{self.name}', index={self.index}, type='{self.device_type}')"


class AudioDeviceModel:
    """Model class for managing audio devices.

    This class handles the detection, enumeration, and management
    of audio input devices using PyAudio.
    """

    def __init__(self):
        """Initialize the AudioDeviceModel."""
        self.logger = logging.getLogger(__name__)
        self._pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self._input_devices: list[AudioDeviceInfo] = []
        self._selected_input_device: Optional[AudioDeviceInfo] = None

        self._initialize_pyaudio()
        self.refresh_devices()

    def _initialize_pyaudio(self) -> None:
        """Initialize PyAudio instance."""
        try:
            self._pyaudio_instance = pyaudio.PyAudio()
            self.logger.info("PyAudio initialized successfully")
        except Exception:
            self.logger.exception("Failed to initialize PyAudio")
            raise

    def refresh_devices(self) -> None:
        """Refresh the list of available audio devices."""
        if not self._pyaudio_instance:
            self.logger.error("PyAudio not initialized")
            return

        self.logger.debug("Refreshing audio devices")
        self._input_devices.clear()

        device_count = self._pyaudio_instance.get_device_count()
        self.logger.debug(f"Found {device_count} total audio devices")

        for i in range(device_count):
            try:
                device_info = self._pyaudio_instance.get_device_info_by_index(i)
                self._process_device_info(device_info, i)
            except Exception as e:
                self.logger.warning(f"Error getting device info for index {i}: {e}")
                continue

        self.logger.info(f"Device refresh complete: {len(self._input_devices)} input devices")

        # Auto-select first available device if none selected
        if not self._selected_input_device and self._input_devices:
            self.select_input_device(0)

    def _process_device_info(self, device_info: dict, index: int) -> None:
        """Process a single device info dictionary."""
        device_name = device_info["name"]
        input_channels = device_info["maxInputChannels"]
        sample_rate = int(device_info["defaultSampleRate"])

        self.logger.debug(f"Processing device: {device_name} (index: {index}, input: {input_channels})")

        # Add as input device if it has input channels
        if input_channels > 0:
            input_device = AudioDeviceInfo(
                name=device_name,
                index=index,
                channels=input_channels,
                sample_rate=sample_rate,
                device_type="input",
            )
            self._input_devices.append(input_device)
            self.logger.debug(f"Added input device: {input_device}")

    @property
    def input_devices(self) -> list[AudioDeviceInfo]:
        """Get list of available input devices."""
        return self._input_devices.copy()

    @property
    def selected_input_device(self) -> Optional[AudioDeviceInfo]:
        """Get currently selected input device."""
        return self._selected_input_device

    def select_input_device(self, device_index: int) -> bool:
        """Select an input device by its index in the input devices list.

        Args:
            device_index: Index in the input_devices list (not PyAudio device index)

        Returns:
            True if selection was successful, False otherwise
        """
        if not (0 <= device_index < len(self._input_devices)):
            self.logger.error(f"Invalid input device index: {device_index}")
            return False

        self._selected_input_device = self._input_devices[device_index]
        self.logger.info(f"Selected input device: {self._selected_input_device}")
        return True

    def get_pyaudio_instance(self) -> Optional[pyaudio.PyAudio]:
        """Get the PyAudio instance for direct use.

        Returns:
            PyAudio instance or None if not initialized
        """
        return self._pyaudio_instance

    def cleanup(self) -> None:
        """Clean up PyAudio resources."""
        if self._pyaudio_instance:
            try:
                self._pyaudio_instance.terminate()
                self.logger.info("PyAudio terminated successfully")
            except Exception:
                self.logger.exception("Error terminating PyAudio")
            finally:
                self._pyaudio_instance = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
