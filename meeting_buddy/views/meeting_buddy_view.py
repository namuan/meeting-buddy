"""Meeting Buddy View for the application.

This module contains the MeetingBuddyView class that handles
all UI components and user interface logic.
"""

import logging
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MeetingBuddyView(QMainWindow):
    """Main view class for Meeting Buddy application.

    This class handles all UI components and user interface logic
    following the MVP (Model-View-Presenter) architecture pattern.
    """

    def __init__(self):
        """Initialize the MeetingBuddyView."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Callback functions (to be set by presenter)
        self.on_output_device_changed: Optional[Callable[[int], None]] = None
        self.on_start_recording: Optional[Callable[[], None]] = None
        self.on_stop_recording: Optional[Callable[[], None]] = None
        self.on_progress_changed: Optional[Callable[[int], None]] = None
        self.on_recording_selected: Optional[Callable[[int], None]] = None

        # UI components
        self.output_device_combo: Optional[QComboBox] = None
        self.progress_slider: Optional[QSlider] = None
        self.start_button: Optional[QPushButton] = None
        self.stop_button: Optional[QPushButton] = None
        self.transcription_text: Optional[QTextEdit] = None
        self.recordings_list: Optional[QListWidget] = None

        self._setup_ui()
        self.logger.info("MeetingBuddyView initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Meeting Buddy - MVP Architecture")
        self.setGeometry(100, 100, 500, 450)  # Increased height for new layout

        # Set the main background color
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f0f0f0"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Create UI sections
        self._create_device_selection_section(main_layout)
        self._create_controls_section(main_layout)
        self._create_transcription_section(main_layout)
        self._create_recordings_section(main_layout)

        self._apply_styles()
        self.logger.debug("UI setup completed")

    def _create_device_selection_section(self, main_layout: QVBoxLayout) -> None:
        """Create the audio device selection section."""
        # Output device selection
        output_layout = QHBoxLayout()
        output_layout.setSpacing(10)

        select_output_label = QLabel("Select Output Device")
        select_output_label.setFont(QFont("System", 13))

        self.output_device_combo = QComboBox()
        self.output_device_combo.setFont(QFont("System", 13))
        self.output_device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.output_device_combo.currentIndexChanged.connect(self._on_output_device_changed)

        output_layout.addWidget(select_output_label)
        output_layout.addWidget(self.output_device_combo)
        main_layout.addLayout(output_layout)

    def _create_controls_section(self, main_layout: QVBoxLayout) -> None:
        """Create the recording controls section."""
        # Middle layout for controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setValue(25)
        self.progress_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.progress_slider.valueChanged.connect(self._on_progress_changed)

        self.start_button = QPushButton("Start")
        self.start_button.setFont(QFont("System", 13))
        self.start_button.clicked.connect(self._on_start_recording)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setFont(QFont("System", 13))
        self.stop_button.clicked.connect(self._on_stop_recording)
        self.stop_button.setEnabled(False)  # Initially disabled

        controls_layout.addWidget(self.progress_slider)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        main_layout.addLayout(controls_layout)

    def _create_transcription_section(self, main_layout: QVBoxLayout) -> None:
        """Create the transcription display section."""
        # Transcription section
        transcription_label = QLabel("Transcribed Content")
        transcription_label.setFont(QFont("System", 13))
        transcription_label.setContentsMargins(0, 10, 0, 5)

        self.transcription_text = QTextEdit()
        self.transcription_text.setPlaceholderText("Transcribed content will appear here...")
        self.transcription_text.setMaximumHeight(120)
        self.transcription_text.setFont(QFont("System", 12))

        main_layout.addWidget(transcription_label)
        main_layout.addWidget(self.transcription_text)

    def _create_recordings_section(self, main_layout: QVBoxLayout) -> None:
        """Create the recordings list section."""
        # Recordings section
        recordings_label = QLabel("Recordings")
        recordings_label.setFont(QFont("System", 13))
        recordings_label.setContentsMargins(0, 10, 0, 5)

        self.recordings_list = QListWidget()
        # Use a monospaced font for the recordings list
        recordings_font = QFont("Menlo")
        if recordings_font.styleHint() != QFont.StyleHint.Monospace:
            recordings_font = QFont("Courier")  # Fallback
        recordings_font.setPointSize(13)
        self.recordings_list.setFont(recordings_font)
        self.recordings_list.itemClicked.connect(self._on_recording_selected)

        main_layout.addWidget(recordings_label)
        main_layout.addWidget(self.recordings_list)

    def _apply_styles(self) -> None:
        """Apply custom styles to the UI components."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #000000;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QPushButton {
                background-color: white;
                border: 1px solid #bbbbbb;
                border-radius: 5px;
                padding: 5px 20px;
                min-width: 50px;
            }
            QPushButton:pressed {
                background-color: #e6e6e6;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #999999;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbbbbb;
                background: #e0e0e0;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #007aff;
                border: 1px solid #bbbbbb;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #bbbbbb;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 8px;
            }
        """)

    # Callback wrapper methods
    def _on_output_device_changed(self, index: int) -> None:
        """Handle output device selection change."""
        if self.on_output_device_changed:
            self.on_output_device_changed(index)

    def _on_start_recording(self) -> None:
        """Handle start recording button click."""
        if self.on_start_recording:
            self.on_start_recording()

    def _on_stop_recording(self) -> None:
        """Handle stop recording button click."""
        if self.on_stop_recording:
            self.on_stop_recording()

    def _on_progress_changed(self, value: int) -> None:
        """Handle progress slider value change."""
        if self.on_progress_changed:
            self.on_progress_changed(value)

    def _on_recording_selected(self, item: QListWidgetItem) -> None:
        """Handle recording list item selection."""
        if self.on_recording_selected and self.recordings_list:
            index = self.recordings_list.row(item)
            self.on_recording_selected(index)

    # Public methods for updating UI from presenter
    def populate_output_devices(self, devices: list[str]) -> None:
        """Populate the output device combo box.

        Args:
            devices: List of device display names
        """
        if self.output_device_combo is None:
            self.logger.error("output_device_combo is None!")
            return

        self.output_device_combo.clear()
        if not devices:
            self.output_device_combo.addItem("No output devices found")
            self.logger.warning("No output devices to populate")
            return

        for device in devices:
            self.output_device_combo.addItem(device)

        self.logger.debug(f"Populated {len(devices)} output devices")

    def populate_recordings(self, recordings: list[str]) -> None:
        """Populate the recordings list.

        Args:
            recordings: List of recording display names
        """
        if self.recordings_list is None:
            return

        self.recordings_list.clear()
        for recording in recordings:
            item = QListWidgetItem(recording)
            self.recordings_list.addItem(item)

        self.logger.debug(f"Populated {len(recordings)} recordings")

    def set_transcription_text(self, text: str) -> None:
        """Set the transcription text content.

        Args:
            text: Transcription text to display
        """
        if self.transcription_text is not None:
            self.transcription_text.setPlainText(text)
            self.logger.debug(f"Set transcription text: {len(text)} characters")

    def append_transcription_text(self, text: str) -> None:
        """Append text to the transcription content.

        Args:
            text: Text to append
        """
        if self.transcription_text is not None:
            current_text = self.transcription_text.toPlainText()
            new_text = current_text + " " + text if current_text else text
            self.transcription_text.setPlainText(new_text)
            self.logger.debug(f"Appended transcription text: {len(text)} characters")

    def clear_transcription_text(self) -> None:
        """Clear the transcription text content."""
        if self.transcription_text is not None:
            self.transcription_text.clear()
            self.logger.debug("Cleared transcription text")

    def set_recording_state(self, is_recording: bool) -> None:
        """Update UI to reflect recording state.

        Args:
            is_recording: True if currently recording, False otherwise
        """
        if self.start_button is not None and self.stop_button is not None:
            self.start_button.setEnabled(not is_recording)
            self.stop_button.setEnabled(is_recording)

            if is_recording:
                self.start_button.setText("Recording...")
            else:
                self.start_button.setText("Start")

            self.logger.debug(f"Set recording state: {is_recording}")

    def set_progress_value(self, value: int) -> None:
        """Set the progress slider value.

        Args:
            value: Progress value (0-100)
        """
        if self.progress_slider is not None:
            self.progress_slider.setValue(value)

    def get_progress_value(self) -> int:
        """Get the current progress slider value.

        Returns:
            Current progress value (0-100)
        """
        if self.progress_slider is not None:
            return self.progress_slider.value()
        return 0

    def show_error_message(self, title: str, message: str) -> None:
        """Show an error message to the user.

        Args:
            title: Error dialog title
            message: Error message content
        """
        # For now, just log the error. Could be extended with QMessageBox
        self.logger.error(f"{title}: {message}")

    def show_info_message(self, title: str, message: str) -> None:
        """Show an info message to the user.

        Args:
            title: Info dialog title
            message: Info message content
        """
        # For now, just log the info. Could be extended with QMessageBox
        self.logger.info(f"{title}: {message}")
