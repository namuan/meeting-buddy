"""Meeting Buddy View for the application.

This module contains the MeetingBuddyView class that handles
all UI components and user interface logic.
"""

import logging
from collections.abc import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MeetingBuddyView(QMainWindow):
    """Main view class for Meeting Buddy application.

    This class handles all UI components and user interface logic
    following the MVP (Model-View-Presenter) architecture pattern.
    """

    # Qt signals for thread-safe UI updates
    model_download_progress_updated = pyqtSignal(str, float, str)  # model_name, progress_percent, status
    model_download_completed = pyqtSignal(str, bool, str)  # model_name, success, message
    whisper_model_status_updated = pyqtSignal(str)  # current_model_name
    ollama_model_status_updated = pyqtSignal(str)  # current_model_name

    def __init__(self):
        """Initialize the MeetingBuddyView."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Callback functions (to be set by presenter)
        self.on_input_device_changed: Callable[[int], None] | None = None
        self.on_start_recording: Callable[[], None] | None = None
        self.on_stop_recording: Callable[[], None] | None = None
        self.on_whisper_model_changed: Callable[[str], None] | None = None
        self.on_ollama_model_changed: Callable[[str], None] | None = None
        self.on_prompt_changed: Callable[[str], None] | None = None
        # UI components
        self.input_device_combo: QComboBox | None = None
        self.start_button: QPushButton | None = None
        self.stop_button: QPushButton | None = None
        self.whisper_model_combo: QComboBox | None = None
        self.ollama_model_combo: QComboBox | None = None
        self.current_whisper_label: QLabel | None = None
        self.current_ollama_label: QLabel | None = None
        self.download_progress_bar: QProgressBar | None = None
        self.download_status_label: QLabel | None = None
        self.prompt_input: QTextEdit | None = None
        self.transcription_text: QTextEdit | None = None
        self.llm_response_text: QTextEdit | None = None

        self._setup_ui()
        self._connect_signals()
        self.logger.info("MeetingBuddyView initialized")

    def _connect_signals(self) -> None:
        """Connect Qt signals to their respective slots."""
        self.model_download_progress_updated.connect(self._on_download_progress_updated)
        self.model_download_completed.connect(self._on_download_completed)
        self.whisper_model_status_updated.connect(self._on_whisper_model_status_updated)
        self.ollama_model_status_updated.connect(self._on_ollama_model_status_updated)
        self.logger.debug("Qt signals connected")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Meeting Buddy")
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
        self._create_configuration_section(main_layout)
        self._create_prompt_section(main_layout)
        self._create_transcription_section(main_layout)
        self._create_llm_response_section(main_layout)

        self._apply_styles()
        self.logger.debug("UI setup completed")

    def _create_device_selection_section(self, main_layout: QVBoxLayout) -> None:
        """Create the device selection section with Start/Stop buttons."""
        # Input device selection with Start/Stop buttons
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        select_input_label = QLabel("Select Input Device (Recording)")
        select_input_label.setFont(QFont("System", 13))

        self.input_device_combo = QComboBox()
        self.input_device_combo.setFont(QFont("System", 13))
        self.input_device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.input_device_combo.currentIndexChanged.connect(self._on_input_device_changed)

        self.start_button = QPushButton("Start")
        self.start_button.setFont(QFont("System", 13))
        self.start_button.clicked.connect(self._on_start_recording)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setFont(QFont("System", 13))
        self.stop_button.clicked.connect(self._on_stop_recording)
        self.stop_button.setEnabled(False)  # Initially disabled

        input_layout.addWidget(select_input_label)
        input_layout.addWidget(self.input_device_combo)
        input_layout.addWidget(self.start_button)
        input_layout.addWidget(self.stop_button)
        main_layout.addLayout(input_layout)

    def _create_configuration_section(self, main_layout: QVBoxLayout) -> None:
        """Create the configuration section for model selection and status display."""
        # Configuration section label
        config_label = QLabel("Model Configuration")
        config_label.setFont(QFont("System", 13))
        config_label.setContentsMargins(0, 10, 0, 5)
        config_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(config_label)

        # Whisper model configuration
        whisper_layout = QHBoxLayout()
        whisper_layout.setSpacing(10)

        whisper_label = QLabel("Whisper Model:")
        whisper_label.setFont(QFont("System", 12))
        whisper_label.setMinimumWidth(120)

        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.setFont(QFont("System", 12))
        self.whisper_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.whisper_model_combo.currentTextChanged.connect(self._on_whisper_model_changed)

        self.current_whisper_label = QLabel("Current: base")
        self.current_whisper_label.setFont(QFont("System", 11))
        self.current_whisper_label.setStyleSheet("color: #666666;")
        self.current_whisper_label.setMinimumWidth(100)

        whisper_layout.addWidget(whisper_label)
        whisper_layout.addWidget(self.whisper_model_combo)
        whisper_layout.addWidget(self.current_whisper_label)
        main_layout.addLayout(whisper_layout)

        # Ollama model configuration
        ollama_layout = QHBoxLayout()
        ollama_layout.setSpacing(10)

        ollama_label = QLabel("Ollama Model:")
        ollama_label.setFont(QFont("System", 12))
        ollama_label.setMinimumWidth(120)

        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.setFont(QFont("System", 12))
        self.ollama_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ollama_model_combo.setEditable(True)  # Allow custom model names
        self.ollama_model_combo.currentTextChanged.connect(self._on_ollama_model_changed)

        self.current_ollama_label = QLabel("Current: llama3.2:latest")
        self.current_ollama_label.setFont(QFont("System", 11))
        self.current_ollama_label.setStyleSheet("color: #666666;")
        self.current_ollama_label.setMinimumWidth(150)

        ollama_layout.addWidget(ollama_label)
        ollama_layout.addWidget(self.ollama_model_combo)
        ollama_layout.addWidget(self.current_ollama_label)
        main_layout.addLayout(ollama_layout)

        # Download progress section
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(5)

        self.download_status_label = QLabel("Ready")
        self.download_status_label.setFont(QFont("System", 11))
        self.download_status_label.setStyleSheet("color: #666666;")

        self.download_progress_bar = QProgressBar()
        self.download_progress_bar.setFont(QFont("System", 10))
        self.download_progress_bar.setVisible(False)  # Initially hidden
        self.download_progress_bar.setMinimum(0)
        self.download_progress_bar.setMaximum(100)

        progress_layout.addWidget(self.download_status_label)
        progress_layout.addWidget(self.download_progress_bar)
        main_layout.addLayout(progress_layout)

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

        main_layout.addWidget(prompt_label)
        main_layout.addWidget(self.prompt_input)

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
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #cccccc;
                color: #000000;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 12px;
                color: #000000;
                background-color: transparent;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #0078d4 !important;
                color: white !important;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #106ebe !important;
                color: white !important;
            }
            QComboBox::item {
                color: #000000;
                background-color: white;
            }
            QComboBox::item:selected {
                background-color: #0078d4;
                color: white;
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

    def _on_prompt_changed(self) -> None:
        """Handle prompt text change."""
        if self.on_prompt_changed and self.prompt_input:
            prompt_text = self.prompt_input.toPlainText()
            self.on_prompt_changed(prompt_text)

    def _on_whisper_model_changed(self, model_name: str) -> None:
        """Handle Whisper model selection change."""
        if self.on_whisper_model_changed and model_name:
            self.on_whisper_model_changed(model_name)

    def _on_ollama_model_changed(self, model_name: str) -> None:
        """Handle Ollama model selection change."""
        if self.on_ollama_model_changed and model_name:
            self.on_ollama_model_changed(model_name)

    # Qt signal slots for thread-safe UI updates
    def _on_download_progress_updated(self, model_name: str, progress_percent: float, status: str) -> None:
        """Handle download progress updates (Qt slot).

        Args:
            model_name: Name of the model being downloaded
            progress_percent: Progress percentage (0.0 to 100.0)
            status: Current download status
        """
        if self.download_progress_bar is not None:
            self.download_progress_bar.setValue(int(progress_percent))
            self.download_progress_bar.setVisible(True)

        if self.download_status_label is not None:
            self.download_status_label.setText(f"Downloading {model_name}: {status}")

        self.logger.debug(f"Download progress updated: {model_name} - {progress_percent:.1f}% - {status}")

    def _on_download_completed(self, model_name: str, success: bool, message: str) -> None:
        """Handle download completion (Qt slot).

        Args:
            model_name: Name of the model that was downloaded
            success: Whether the download was successful
            message: Success or error message
        """
        if self.download_progress_bar is not None:
            if success:
                self.download_progress_bar.setValue(100)
            self.download_progress_bar.setVisible(False)

        if self.download_status_label is not None:
            if success:
                self.download_status_label.setText(f"✓ {model_name} downloaded successfully")
            else:
                self.download_status_label.setText(f"✗ Failed to download {model_name}: {message}")

        self.logger.info(f"Download completed: {model_name} - Success: {success} - {message}")

    def _on_whisper_model_status_updated(self, model_name: str) -> None:
        """Handle Whisper model status updates (Qt slot).

        Args:
            model_name: Current Whisper model name
        """
        if self.current_whisper_label is not None:
            self.current_whisper_label.setText(f"Current: {model_name}")

        self.logger.debug(f"Whisper model status updated: {model_name}")

    def _on_ollama_model_status_updated(self, model_name: str) -> None:
        """Handle Ollama model status updates (Qt slot).

        Args:
            model_name: Current Ollama model name
        """
        if self.current_ollama_label is not None:
            self.current_ollama_label.setText(f"Current: {model_name}")

        self.logger.debug(f"Ollama model status updated: {model_name}")

    # Public methods for updating UI from presenter
    def populate_input_devices(self, devices: list[str], selected_index: int = 0) -> None:
        """Populate the input device combo box.

        Args:
            devices: List of device display names
            selected_index: Index of the device to select (default: 0)
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

        # Set the selected device
        if 0 <= selected_index < len(devices):
            self.input_device_combo.setCurrentIndex(selected_index)
            self.logger.debug(f"Set selected device index to: {selected_index}")

        self.logger.debug(f"Populated {len(devices)} input devices")

    def populate_whisper_models(self, models: list[str], selected_model: str = "base") -> None:
        """Populate the Whisper model combo box.

        Args:
            models: List of available Whisper model names
            selected_model: Model name to select (default: "base")
        """
        if self.whisper_model_combo is None:
            self.logger.error("whisper_model_combo is None!")
            return

        self.whisper_model_combo.clear()
        if not models:
            self.whisper_model_combo.addItem("No models available")
            self.logger.warning("No Whisper models to populate")
            return

        for model in models:
            self.whisper_model_combo.addItem(model)

        # Set the selected model
        if selected_model in models:
            self.whisper_model_combo.setCurrentText(selected_model)
            self.logger.debug(f"Set selected Whisper model to: {selected_model}")
        elif models:
            self.whisper_model_combo.setCurrentIndex(0)
            self.logger.debug(f"Set Whisper model to first available: {models[0]}")

        self.logger.debug(f"Populated {len(models)} Whisper models")

    def populate_ollama_models(self, models: list[str], selected_model: str = "llama3.2:latest") -> None:
        """Populate the Ollama model combo box.

        Args:
            models: List of available Ollama model names
            selected_model: Model name to select (default: "llama3.2:latest")
        """
        if self.ollama_model_combo is None:
            self.logger.error("ollama_model_combo is None!")
            return

        self.ollama_model_combo.clear()
        if not models:
            self.ollama_model_combo.addItem("No models available")
            self.logger.warning("No Ollama models to populate")
            return

        for model in models:
            self.ollama_model_combo.addItem(model)

        # Set the selected model
        if selected_model in models:
            self.ollama_model_combo.setCurrentText(selected_model)
            self.logger.debug(f"Set selected Ollama model to: {selected_model}")
        elif models:
            self.ollama_model_combo.setCurrentIndex(0)
            self.logger.debug(f"Set Ollama model to first available: {models[0]}")
        else:
            # Allow custom model name even if not in list
            self.ollama_model_combo.setCurrentText(selected_model)
            self.logger.debug(f"Set custom Ollama model: {selected_model}")

        self.logger.debug(f"Populated {len(models)} Ollama models")

    def update_current_whisper_model(self, model_name: str) -> None:
        """Update the current Whisper model status display.

        Args:
            model_name: Current Whisper model name
        """
        if self.current_whisper_label is not None:
            self.current_whisper_label.setText(f"Current: {model_name}")
            self.logger.debug(f"Updated current Whisper model display: {model_name}")

    def update_current_ollama_model(self, model_name: str) -> None:
        """Update the current Ollama model status display.

        Args:
            model_name: Current Ollama model name
        """
        if self.current_ollama_label is not None:
            self.current_ollama_label.setText(f"Current: {model_name}")
            self.logger.debug(f"Updated current Ollama model display: {model_name}")

    def set_whisper_model_selection(self, model_name: str) -> None:
        """Set the selected Whisper model in the combo box.

        Args:
            model_name: Model name to select
        """
        if self.whisper_model_combo is not None:
            self.whisper_model_combo.setCurrentText(model_name)
            self.logger.debug(f"Set Whisper model selection: {model_name}")

    def set_ollama_model_selection(self, model_name: str) -> None:
        """Set the selected Ollama model in the combo box.

        Args:
            model_name: Model name to select
        """
        if self.ollama_model_combo is not None:
            self.ollama_model_combo.setCurrentText(model_name)
            self.logger.debug(f"Set Ollama model selection: {model_name}")

    def get_selected_whisper_model(self) -> str:
        """Get the currently selected Whisper model.

        Returns:
            Selected Whisper model name
        """
        if self.whisper_model_combo is not None:
            return self.whisper_model_combo.currentText()
        return ""

    def get_selected_ollama_model(self) -> str:
        """Get the currently selected Ollama model.

        Returns:
            Selected Ollama model name
        """
        if self.ollama_model_combo is not None:
            return self.ollama_model_combo.currentText()
        return ""

    # Thread-safe signal emission methods
    def emit_download_progress_update(self, model_name: str, progress_percent: float, status: str) -> None:
        """Emit download progress update signal (thread-safe).

        Args:
            model_name: Name of the model being downloaded
            progress_percent: Progress percentage (0.0 to 100.0)
            status: Current download status
        """
        self.model_download_progress_updated.emit(model_name, progress_percent, status)

    def emit_download_completed(self, model_name: str, success: bool, message: str) -> None:
        """Emit download completion signal (thread-safe).

        Args:
            model_name: Name of the model that was downloaded
            success: Whether the download was successful
            message: Success or error message
        """
        self.model_download_completed.emit(model_name, success, message)

    def emit_whisper_model_status_update(self, model_name: str) -> None:
        """Emit Whisper model status update signal (thread-safe).

        Args:
            model_name: Current Whisper model name
        """
        self.whisper_model_status_updated.emit(model_name)

    def emit_ollama_model_status_update(self, model_name: str) -> None:
        """Emit Ollama model status update signal (thread-safe).

        Args:
            model_name: Current Ollama model name
        """
        self.ollama_model_status_updated.emit(model_name)

    def show_download_progress(self, show: bool = True) -> None:
        """Show or hide the download progress bar.

        Args:
            show: Whether to show the progress bar
        """
        if self.download_progress_bar is not None:
            self.download_progress_bar.setVisible(show)

    def set_download_status(self, status: str) -> None:
        """Set the download status message.

        Args:
            status: Status message to display
        """
        if self.download_status_label is not None:
            self.download_status_label.setText(status)

    def reset_download_progress(self) -> None:
        """Reset the download progress display."""
        if self.download_progress_bar is not None:
            self.download_progress_bar.setValue(0)
            self.download_progress_bar.setVisible(False)

        if self.download_status_label is not None:
            self.download_status_label.setText("Ready")

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
