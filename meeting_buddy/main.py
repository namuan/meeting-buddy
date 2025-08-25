# meeting_buddy/main.py
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow


class MeetingBuddyWindow(QMainWindow):
    """Main window for Meeting Buddy application."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Meeting Buddy")
        self.setGeometry(100, 100, 800, 600)


def main():
    """Main entry point for the Meeting Buddy application."""
    app = QApplication(sys.argv)
    
    window = MeetingBuddyWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
