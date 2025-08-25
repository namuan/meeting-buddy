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
        self.on_input_device_changed: Optional[Callable[[int], None]] = None
        self.on_start_recording: Optional[Callable[[], None]] = None
        self.on_stop_recording: Optional[Callable[[], None]] = None
        self.on_progress_changed: Optional[Callable[[int], None]] = None
        self.on_prompt_changed: Optional[Callable[[str], None]] = None
        self.on_test_llm: Optional[Callable[[], None]] = None

        # UI components
        self.input_device_combo: Optional[QComboBox] = None
        self.progress_slider: Optional[QSlider] = None
        self.start_button: Optional[QPushButton] = None
        self.stop_button: Optional[QPushButton] = None
        self.prompt_input: Optional[QTextEdit] = None
        self.transcription_text: Optional[QTextEdit] = None
        self.llm_response_text: Optional[QTextEdit] = None

        self._setup_ui()
        self.logger.info("MeetingBuddyView initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Meeting Buddy - MVP Architecture")
        self.setGeometry(100, 100, 800, 700)  # Increased size for LLM integration

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
        self._create_prompt_section(main_layout)
        self._create_transcription_section(main_layout)
        self._create_llm_response_section(main_layout)

        self._apply_styles()
        self.logger.debug("UI setup completed")

    def _create_device_selection_section(self, main_layout: QVBoxLayout) -> None:
        """Create the device selection section."""
        # Input device selection (for recording)
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        select_input_label = QLabel("Select Input Device (Recording)")
        select_input_label.setFont(QFont("System", 13))

        self.input_device_combo = QComboBox()
        self.input_device_combo.setFont(QFont("System", 13))
        self.input_device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.input_device_combo.currentIndexChanged.connect(self._on_input_device_changed)

        input_layout.addWidget(select_input_label)
        input_layout.addWidget(self.input_device_combo)
        main_layout.addLayout(input_layout)

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

    def _create_prompt_section(self, main_layout: QVBoxLayout) -> None:
        """Create the LLM prompt input section."""
        # LLM Prompt section
        prompt_label = QLabel("LLM Prompt (Multi-line)")
        prompt_label.setFont(QFont("System", 13))
        prompt_label.setContentsMargins(0, 10, 0, 5)
        prompt_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Enter your prompt for the LLM here...\n\nExample: 'Summarize the key points from this meeting transcription.'"
        )
        # Default prompt will be set by presenter after callbacks are connected
        self.prompt_input.setFont(QFont("System", 12))
        self.prompt_input.setMaximumHeight(120)  # Limit height to keep it compact
        self.prompt_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.prompt_input.textChanged.connect(self._on_prompt_changed)

        # Add test LLM button
        test_llm_button = QPushButton("Test LLM")
        test_llm_button.setFont(QFont("System", 11))
        test_llm_button.clicked.connect(self._on_test_llm_clicked)
        test_llm_button.setMaximumWidth(100)

        prompt_button_layout = QHBoxLayout()
        prompt_button_layout.addWidget(self.prompt_input)
        prompt_button_layout.addWidget(test_llm_button)

        main_layout.addWidget(prompt_label)
        main_layout.addLayout(prompt_button_layout)

    def _create_transcription_section(self, main_layout: QVBoxLayout) -> None:
        """Create the transcription display section."""
        # Transcription section
        transcription_label = QLabel("Transcribed Content")
        transcription_label.setFont(QFont("System", 13))
        transcription_label.setContentsMargins(0, 10, 0, 5)
        # Set fixed height for the label
        transcription_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self.transcription_text = QTextEdit()
        self.transcription_text.setPlaceholderText("Transcribed content will appear here...")
        self.transcription_text.setFont(QFont("System", 12))
        # Set expanding size policy to fill available space
        self.transcription_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        main_layout.addWidget(transcription_label)
        main_layout.addWidget(self.transcription_text)

    def _create_llm_response_section(self, main_layout: QVBoxLayout) -> None:
        """Create the LLM response display section."""
        # LLM Response section
        llm_response_label = QLabel("LLM Response")
        llm_response_label.setFont(QFont("System", 13))
        llm_response_label.setContentsMargins(0, 10, 0, 5)
        llm_response_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self.llm_response_text = QTextEdit()
        self.llm_response_text.setPlaceholderText("LLM responses will appear here...")
        self.llm_response_text.setFont(QFont("System", 12))
        self.llm_response_text.setReadOnly(True)  # Read-only for responses
        self.llm_response_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        main_layout.addWidget(llm_response_label)
        main_layout.addWidget(self.llm_response_text)

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
    def _on_input_device_changed(self, index: int) -> None:
        """Handle input device selection change."""
        if self.on_input_device_changed:
            self.on_input_device_changed(index)

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

    def _on_prompt_changed(self) -> None:
        """Handle prompt text change."""
        if self.on_prompt_changed and self.prompt_input:
            prompt_text = self.prompt_input.toPlainText()
            self.on_prompt_changed(prompt_text)

    def _on_test_llm_clicked(self) -> None:
        """Handle test LLM button click."""
        if self.on_test_llm:
            self.on_test_llm()

    # Public methods for updating UI from presenter
    def populate_input_devices(self, devices: list[str]) -> None:
        """Populate the input device combo box.

        Args:
            devices: List of device display names
        """
        if self.input_device_combo is None:
            self.logger.error("input_device_combo is None!")
            return

        self.input_device_combo.clear()
        if not devices:
            self.input_device_combo.addItem("No input devices found")
            self.logger.warning("No input devices to populate")
            return

        for device in devices:
            self.input_device_combo.addItem(device)

        self.logger.debug(f"Populated {len(devices)} input devices")

    def set_transcription_text(self, text: str) -> None:
        """Set the transcription text content.

        Args:
            text: Transcription text to display
        """
        if self.transcription_text is not None:
            self.transcription_text.setPlainText(text)

            # Auto-scroll to bottom for live updates
            self._scroll_transcription_to_bottom()

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

            # Auto-scroll to bottom for live updates
            self._scroll_transcription_to_bottom()

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

    def _scroll_transcription_to_bottom(self) -> None:
        """Scroll the transcription text area to the bottom for live updates."""
        if self.transcription_text is not None:
            # Move cursor to end and ensure it's visible
            cursor = self.transcription_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.transcription_text.setTextCursor(cursor)
            self.transcription_text.ensureCursorVisible()

    def update_transcription_live(self, new_text: str, is_final: bool = False) -> None:
        """Update transcription with live text, handling partial and final results.

        Args:
            new_text: New transcription text to add
            is_final: Whether this is a final transcription result or partial
        """
        if self.transcription_text is not None:
            if is_final:
                # For final results, append with proper spacing
                self.append_transcription_text(new_text)
            else:
                # For partial results, show in a different style or just update
                current_text = self.transcription_text.toPlainText()
                # Add partial text with visual indicator
                display_text = current_text + " [" + new_text + "...]"
                self.transcription_text.setPlainText(display_text)
                self._scroll_transcription_to_bottom()

            self.logger.debug(f"Live transcription update: {len(new_text)} chars, final={is_final}")

    def set_transcription_status(self, status: str) -> None:
        """Set transcription status message.

        Args:
            status: Status message to display
        """
        # Could be used to show status like "Listening...", "Processing...", etc.
        # For now, just update the placeholder text
        if self.transcription_text is not None and not self.transcription_text.toPlainText().strip():
            self.transcription_text.setPlaceholderText(status)

        self.logger.debug(f"Transcription status: {status}")

    def highlight_recent_transcription(self, text: str) -> None:
        """Highlight recently added transcription text.

        Args:
            text: Text to highlight
        """
        if self.transcription_text is not None:
            # Get current cursor position
            cursor = self.transcription_text.textCursor()

            # Find the recently added text and highlight it briefly
            full_text = self.transcription_text.toPlainText()
            if text in full_text:
                start_pos = full_text.rfind(text)
                if start_pos >= 0:
                    # Select the new text
                    cursor.setPosition(start_pos)
                    cursor.setPosition(start_pos + len(text), cursor.MoveMode.KeepAnchor)
                    self.transcription_text.setTextCursor(cursor)

                    # Could add temporary highlighting here if needed
                    # For now, just ensure it's visible
                    self.transcription_text.ensureCursorVisible()

        self.logger.debug(f"Highlighted recent transcription: {len(text)} characters")

    def get_transcription_word_count(self) -> int:
        """Get the current word count of transcription text.

        Returns:
            Number of words in transcription
        """
        if self.transcription_text is not None:
            text = self.transcription_text.toPlainText().strip()
            if text:
                return len(text.split())
        return 0

    def get_transcription_character_count(self) -> int:
        """Get the current character count of transcription text.

        Returns:
            Number of characters in transcription
        """
        if self.transcription_text is not None:
            return len(self.transcription_text.toPlainText())
        return 0

    # LLM Prompt and Response methods
    def get_prompt_text(self) -> str:
        """Get the current prompt text.

        Returns:
            Current prompt text
        """
        if self.prompt_input is not None:
            return self.prompt_input.toPlainText().strip()
        return ""

    def set_prompt_text(self, text: str) -> None:
        """Set the prompt text content.

        Args:
            text: Prompt text to set
        """
        if self.prompt_input is not None:
            self.prompt_input.setPlainText(text)
            self.logger.debug(f"Set prompt text: {len(text)} characters")

    def clear_prompt_text(self) -> None:
        """Clear the prompt text content."""
        if self.prompt_input is not None:
            self.prompt_input.clear()
            self.logger.debug("Cleared prompt text")

    def set_llm_response_text(self, text: str) -> None:
        """Set the LLM response text content.

        Args:
            text: LLM response text to display
        """
        if self.llm_response_text is not None:
            self.llm_response_text.setPlainText(text)

            # Auto-scroll to bottom for live updates
            self._scroll_llm_response_to_bottom()

            self.logger.debug(f"Set LLM response text: {len(text)} characters")

    def append_llm_response_text(self, text: str) -> None:
        """Append text to the LLM response content.

        Args:
            text: Text to append
        """
        if self.llm_response_text is not None:
            current_text = self.llm_response_text.toPlainText()
            new_text = current_text + text if current_text else text
            self.llm_response_text.setPlainText(new_text)

            # Auto-scroll to bottom for live updates
            self._scroll_llm_response_to_bottom()

            self.logger.debug(f"Appended LLM response text: {len(text)} characters")

    def clear_llm_response_text(self) -> None:
        """Clear the LLM response text content."""
        if self.llm_response_text is not None:
            self.llm_response_text.clear()
            self.logger.debug("Cleared LLM response text")

    def set_llm_response_status(self, status: str) -> None:
        """Set LLM response status message.

        Args:
            status: Status message to display
        """
        if self.llm_response_text is not None and not self.llm_response_text.toPlainText().strip():
            self.llm_response_text.setPlaceholderText(status)

        self.logger.debug(f"LLM response status: {status}")

    def _scroll_llm_response_to_bottom(self) -> None:
        """Scroll the LLM response text area to the bottom for live updates."""
        if self.llm_response_text is not None:
            # Move cursor to end and ensure it's visible
            cursor = self.llm_response_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.llm_response_text.setTextCursor(cursor)
            self.llm_response_text.ensureCursorVisible()

    def update_llm_response_live(self, new_text: str, is_complete: bool = False) -> None:
        """Update LLM response with live text, handling streaming and complete results.

        Args:
            new_text: New response text to add
            is_complete: Whether this is a complete response or streaming chunk
        """
        if self.llm_response_text is not None:
            if is_complete:
                # For complete responses, set the full text
                self.set_llm_response_text(new_text)
            else:
                # For streaming responses, append the chunk
                self.append_llm_response_text(new_text)

            self.logger.debug(f"Live LLM response update: {len(new_text)} chars, complete={is_complete}")

    def get_llm_response_character_count(self) -> int:
        """Get the current character count of LLM response text.

        Returns:
            Number of characters in LLM response
        """
        if self.llm_response_text is not None:
            return len(self.llm_response_text.toPlainText())
        return 0
