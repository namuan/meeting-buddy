#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pyqt6",
#   "pyaudio",
#   "numpy"
# ]
# ///
"""A PyQt6 application to record system audio from applications on macOS using BlackHole.

Usage:
./audio_recorder.py -h

./audio_recorder.py -v  # To log INFO messages
./audio_recorder.py -vv # To log DEBUG messages
"""

import logging
import sys
import threading
import wave
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np
import pyaudio
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget


def setup_logging(verbosity):
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(
        handlers=[logging.StreamHandler()],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    logging.captureWarnings(capture=True)


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    return parser.parse_args()


class AudioRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.p = pyaudio.PyAudio()
        self.frames_sys = []
        self.stream_sys = None
        self.sys_index = None
        self.find_devices()
        self.initUI()
        logging.info("AudioRecorder initialized")

    def find_devices(self):
        logging.debug("Searching for audio devices")
        desktop_audio_index = None
        blackhole_index = None

        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            logging.debug(f"Found device: {dev['name']}, index: {i}")
            # Prioritize Desktop Audio (multi-output device) over BlackHole
            if "Desktop Audio" in dev["name"] and dev["maxInputChannels"] > 0:
                desktop_audio_index = i
                logging.info(f"Desktop Audio device found at index {i}")
            elif "BlackHole" in dev["name"] and dev["maxInputChannels"] > 0:
                blackhole_index = i
                logging.info(f"BlackHole device found at index {i}")

        # Use Desktop Audio if available, otherwise fall back to BlackHole
        if desktop_audio_index is not None:
            self.sys_index = desktop_audio_index
            logging.info(f"Using Desktop Audio device at index {desktop_audio_index}")
        elif blackhole_index is not None:
            self.sys_index = blackhole_index
            logging.info(f"Using BlackHole device at index {blackhole_index}")
        else:
            logging.error("Could not find BlackHole or Desktop Audio device for system audio capture")
            sys.exit(1)

    def initUI(self):
        self.setWindowTitle("System Audio Recorder")
        central = QWidget()
        layout = QVBoxLayout()
        self.start_btn = QPushButton("Start Recording")
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn = QPushButton("Stop Recording")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.setGeometry(300, 300, 300, 150)
        logging.debug("UI initialized")

    def callback_sys(self, in_data, frame_count, time_info, status):
        if self.recording:
            self.frames_sys.append(in_data)
            logging.debug("Captured system audio frame")
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.frames_sys = []
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            logging.info("Starting system audio recording")
            # Get device info to determine proper channel count
            sys_dev = self.p.get_device_info_by_index(self.sys_index)

            self.sys_channels = min(2, sys_dev["maxInputChannels"])  # Use up to 2 channels for system

            logging.info(f"Using {self.sys_channels} channels for system audio")

            self.stream_sys = self.p.open(
                format=pyaudio.paInt16,
                channels=self.sys_channels,
                rate=44100,
                input=True,
                input_device_index=self.sys_index,
                frames_per_buffer=1024,
                stream_callback=self.callback_sys,
            )
            threading.Thread(target=self.record, daemon=True).start()
            logging.info("Recording thread started")

    def record(self):
        self.stream_sys.start_stream()
        logging.debug("System audio stream started")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream_sys.stop_stream()
            self.stream_sys.close()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            logging.info("Recording stopped, saving audio")
            self.save_audio()

    def save_audio(self):
        logging.debug("Processing system audio for saving")
        # Convert frames to numpy arrays
        sys_data = np.frombuffer(b"".join(self.frames_sys), dtype=np.int16)

        # Reshape data based on channel count
        if self.sys_channels == 1:
            # System data is mono
            output_channels = 1
            output_data = sys_data
        else:
            # System data is stereo, keep as stereo
            output_channels = self.sys_channels
            output_data = sys_data

        logging.debug(f"System audio data length: {len(sys_data)} samples, {output_channels} channels")

        # Save to WAV file
        wf = wave.open("system_audio.wav", "wb")
        wf.setnchannels(output_channels)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(output_data.tobytes())
        wf.close()
        self.p.terminate()
        logging.info("System audio saved to system_audio.wav")


def main(args):
    logging.debug(f"Starting application with verbosity: {args.verbose}")
    app = QApplication(sys.argv)
    ex = AudioRecorder()
    ex.show()
    logging.info("Application window displayed")
    sys.exit(app.exec())


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
