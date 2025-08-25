import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSlider,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QSizePolicy
)
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt

class MeetingBuddyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Meeting Buddy - metting-buddy.ui")
        self.setGeometry(100, 100, 500, 350)

        # Set the main background color to approximate the screenshot
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f0f0f0"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Top layout for audio device selection
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)

        select_audio_label = QLabel("Select Audio Device")
        select_audio_label.setFont(QFont("System", 13))

        self.audio_device_combo = QComboBox()
        self.audio_device_combo.addItem("AirPod")
        self.audio_device_combo.setFont(QFont("System", 13))
        self.audio_device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        top_layout.addWidget(select_audio_label)
        top_layout.addWidget(self.audio_device_combo)

        # Middle layout for controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setValue(25)
        self.progress_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        start_button = QPushButton("Start")
        start_button.setFont(QFont("System", 13))
        
        stop_button = QPushButton("Stop")
        stop_button.setFont(QFont("System", 13))

        controls_layout.addWidget(self.progress_slider)
        controls_layout.addWidget(start_button)
        controls_layout.addWidget(stop_button)

        # Recordings section
        recordings_label = QLabel("Recordings")
        recordings_label.setFont(QFont("System", 13))
        recordings_label.setContentsMargins(0, 10, 0, 5)

        self.recordings_list = QListWidget()
        # Use a monospaced font for the recordings list
        recordings_font = QFont("Menlo")
        if recordings_font.styleHint() != QFont.StyleHint.Monospace:
             recordings_font = QFont("Courier") # Fallback
        recordings_font.setPointSize(13)
        self.recordings_list.setFont(recordings_font)
        
        # Add items to the list
        item1 = QListWidgetItem("Recording 1 - 2025-07-25 07:54:53")
        self.recordings_list.addItem(item1)
        item2 = QListWidgetItem("Recording 2 - 2025-08-25 07:54:53")
        self.recordings_list.addItem(item2)

        # Add layouts to main layout
        main_layout.addLayout(top_layout)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(recordings_label)
        main_layout.addWidget(self.recordings_list)

        self.apply_styles()

    def apply_styles(self):
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
            QComboBox::down-arrow {
                /* This is a placeholder for a system-like arrow.
                   It might be hard to replicate the exact macOS double arrow with QSS.
                   Qt usually handles this based on the OS style. */
                image: url(down_arrow.png); /* You'd need an image file */
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
        """)

def main():
    app = QApplication(sys.argv)
    window = MeetingBuddyApp()
    window.show()
    sys.exit(app.exec())