# focused_reminder/main.py
import json  # =============================================================================
import logging
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtCore import QUrl as QtQUrl
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QFontMetrics,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPen,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtWidgets import QPushButton as DialogButton
from pyremindkit import RemindKit

# =============================================================================
# THEME AND COLOR CONFIGURATION
# =============================================================================
# Default theme configuration
DEFAULT_THEME = {
    # Border gradient colors (RGB + Alpha)
    "border_color_start": (0, 120, 255, 200),  # Blue start
    "border_color_end": (0, 255, 255, 200),  # Cyan end
    "border_width": 10,
    # Notification bar colors (RGB + Alpha)
    "notification_bg_start": (0, 120, 255, 240),  # Blue start (more opaque)
    "notification_bg_end": (0, 255, 255, 240),  # Cyan end (more opaque)
    "notification_text_color": (0, 0, 0),  # Black text
    "notification_corner_radius": 10,
    # Button styling
    "button_bg_normal": "rgba(255,255,255,0)",  # Transparent background
    "button_bg_hover": "rgba(255,255,255,80)",  # Semi-transparent white on hover
    "button_text_color": "black",
    "button_corner_radius": 8,
    "button_width": 36,
    "button_height": 32,
    # Button icons
    "icon_pause": "⏸",
    "icon_play": "▶",
    "icon_reset": "⏮",
    "icon_done": "✓",
    "icon_snooze": "💤",
    # Fonts
    "main_font_family": "Arial",
    "main_font_size": 14,
    "icon_font_family": "Segoe UI Symbol",
    "icon_font_size": 16,
    # Layout dimensions
    "notification_padding": 20,
    "notification_offset": 5,
    "button_spacing": 6,
    "button_container_padding": 10,
    "reminder_time_window_minutes": 15,
    # Full-screen gradient settings
    "fullscreen_gradient_opacity": 30,  # Alpha value for gradient background (0-255)
    "fullscreen_gradient_enabled": True,  # Whether to show gradient when timer stops
}

# Predefined themes
PREDEFINED_THEMES = {
    "Default (Blue-Cyan)": DEFAULT_THEME,
    "Dark Mode": {
        **DEFAULT_THEME,
        "border_color_start": (50, 50, 50, 200),
        "border_color_end": (100, 100, 100, 200),
        "notification_bg_start": (40, 40, 40, 240),
        "notification_bg_end": (60, 60, 60, 240),
        "notification_text_color": (255, 255, 255),
        "button_text_color": "white",
        "button_bg_hover": "rgba(255,255,255,30)",
    },
    "Warm (Orange-Red)": {
        **DEFAULT_THEME,
        "border_color_start": (255, 140, 0, 200),
        "border_color_end": (255, 69, 0, 200),
        "notification_bg_start": (255, 140, 0, 240),
        "notification_bg_end": (255, 165, 0, 240),
    },
    "Nature (Green)": {
        **DEFAULT_THEME,
        "border_color_start": (0, 255, 0, 200),
        "border_color_end": (34, 139, 34, 200),
        "notification_bg_start": (144, 238, 144, 240),
        "notification_bg_end": (152, 251, 152, 240),
    },
    "Purple Dreams": {
        **DEFAULT_THEME,
        "border_color_start": (138, 43, 226, 200),
        "border_color_end": (75, 0, 130, 200),
        "notification_bg_start": (147, 112, 219, 240),
        "notification_bg_end": (138, 43, 226, 240),
        "notification_text_color": (255, 255, 255),
        "button_text_color": "white",
        "button_bg_hover": "rgba(255,255,255,40)",
    },
    "Ocean Sunset": {
        **DEFAULT_THEME,
        "border_color_start": (255, 94, 77, 200),
        "border_color_end": (255, 154, 0, 200),
        "notification_bg_start": (255, 94, 77, 240),
        "notification_bg_end": (255, 206, 84, 240),
    },
}

# Current active theme
current_theme = DEFAULT_THEME.copy()
# Track the currently selected predefined theme name
current_theme_name = "Default (Blue-Cyan)"


# Determine OS-specific config directory
def get_config_dir():
    if sys.platform == "win32":
        config_dir = Path.home() / "AppData" / "Roaming"
    elif sys.platform == "darwin":
        config_dir = Path.home() / "Library" / "Application Support"
    else:  # Linux and other Unix-like
        config_dir = Path.home() / ".config"
    config_dir = config_dir / "screen_borders"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


CONFIG_FILE = get_config_dir() / "config.json"


def load_config():
    """Load configuration from the OS-standard config location."""
    global current_theme, current_theme_name
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                saved_config = json.load(f)
            # Load the selected theme name if available
            if "_selected_theme_name" in saved_config:
                current_theme_name = saved_config.pop("_selected_theme_name")
            # Update only the keys that exist in the saved config
            current_theme.update(saved_config)
            logging.info(f"Configuration loaded from {CONFIG_FILE}")
        except Exception as e:
            logging.warning(f"Failed to load config from {CONFIG_FILE}: {e}")


def save_config():
    """Save current configuration to the OS-standard config location."""
    try:
        # Create a copy of current_theme and add the selected theme name
        config_to_save = current_theme.copy()
        config_to_save["_selected_theme_name"] = current_theme_name
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, indent=2)
        logging.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception:
        logging.exception(f"Failed to save config to {CONFIG_FILE}")


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


class ColorButton(QPushButton):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(60, 30)
        self.update_color()
        self.clicked.connect(self.choose_color)

    def update_color(self):
        r, g, b = self.color[:3]
        self.setStyleSheet(f"QPushButton {{ background-color: rgb({r},{g},{b}); border: 2px solid #ccc; }}")

    def choose_color(self):
        color = QColorDialog.getColor(QColor(*self.color[:3]))
        if color.isValid():
            self.color = (color.red(), color.green(), color.blue(), self.color[3] if len(self.color) > 3 else 255)
            self.update_color()


def themes_match(theme1, theme2):
    """Compare two themes to see if they match on key visual properties."""
    # Key properties that define a theme's visual appearance
    key_properties = [
        "border_color_start",
        "border_color_end",
        "notification_bg_start",
        "notification_bg_end",
        "notification_text_color",
        "button_text_color",
        "button_bg_hover",
        "icon_pause",
        "icon_play",
        "icon_reset",
        "icon_done",
        "icon_snooze",
        "main_font_family",
        "main_font_size",
    ]

    return all(theme1.get(prop) == theme2.get(prop) for prop in key_properties)


def find_matching_theme_name(current_config):
    """Find the predefined theme name that matches the current configuration."""
    for theme_name, theme_config in PREDEFINED_THEMES.items():
        if themes_match(current_config, theme_config):
            return theme_name
    return None


class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Screen Border Configuration")
        self.setModal(True)
        self.setFixedSize(650, 750)
        # Store original theme for cancel functionality
        self.original_theme = current_theme.copy()
        # Store original theme name for cancel functionality
        self.original_theme_name = current_theme_name
        # Store original timer for cancel functionality
        self.original_timer_seconds = getattr(parent, "countdown_seconds", 25 * 60)
        self.init_ui()

    def _separator(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.HLine)
        frame.setFrameShadow(QFrame.Shadow.Sunken)
        return frame

    def init_ui(self):
        layout = QVBoxLayout()
        # Predefined themes section
        theme_group = QGroupBox("Predefined Themes")
        theme_layout = QVBoxLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(PREDEFINED_THEMES.keys())

        # Set the currently selected theme name
        if current_theme_name in PREDEFINED_THEMES:
            index = list(PREDEFINED_THEMES.keys()).index(current_theme_name)
            self.theme_combo.setCurrentIndex(index)
        else:
            # Fallback to matching theme if current_theme_name is not valid
            matching_theme = find_matching_theme_name(current_theme)
            if matching_theme:
                index = list(PREDEFINED_THEMES.keys()).index(matching_theme)
                self.theme_combo.setCurrentIndex(index)

        self.theme_combo.currentTextChanged.connect(self.load_predefined_theme)
        theme_layout.addWidget(self.theme_combo)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        # Visual Appearance section
        visual_group = QGroupBox("Visual Appearance")
        visual_layout = QFormLayout()

        # Border colors
        border_label = QLabel("Border Colors:")
        visual_layout.addRow(border_label)
        border_layout = QHBoxLayout()
        self.border_start_btn = ColorButton(current_theme["border_color_start"])
        self.border_end_btn = ColorButton(current_theme["border_color_end"])
        border_layout.addWidget(QLabel("Start:"))
        border_layout.addWidget(self.border_start_btn)
        border_layout.addWidget(QLabel("End:"))
        border_layout.addWidget(self.border_end_btn)

        # Border width
        self.border_width_spin = QSpinBox()
        self.border_width_spin.setRange(1, 50)
        self.border_width_spin.setValue(current_theme["border_width"])
        border_layout.addWidget(QLabel("Border Width:"))
        border_layout.addWidget(self.border_width_spin)
        border_layout.addStretch()
        visual_layout.addRow(border_layout)
        visual_layout.addRow(self._separator())

        # Notification background colors
        notif_label = QLabel("Notification Background:")
        visual_layout.addRow(notif_label)
        notif_layout = QHBoxLayout()
        self.notif_start_btn = ColorButton(current_theme["notification_bg_start"])
        self.notif_end_btn = ColorButton(current_theme["notification_bg_end"])
        self.text_color_btn = ColorButton(current_theme["notification_text_color"])
        notif_layout.addWidget(QLabel("Start:"))
        notif_layout.addWidget(self.notif_start_btn)
        notif_layout.addWidget(QLabel("End:"))
        notif_layout.addWidget(self.notif_end_btn)
        notif_layout.addWidget(QLabel("Text Color:"))
        notif_layout.addWidget(self.text_color_btn)
        notif_layout.addStretch()
        visual_layout.addRow(notif_layout)
        visual_layout.addRow(self._separator())

        # Full-screen gradient configuration
        self.gradient_opacity_spin = QSpinBox()
        self.gradient_opacity_spin.setRange(0, 255)
        self.gradient_opacity_spin.setValue(current_theme.get("fullscreen_gradient_opacity", 30))
        visual_layout.addRow("Gradient Opacity (0-255):", self.gradient_opacity_spin)

        from PyQt6.QtWidgets import QCheckBox

        self.gradient_enabled_check = QCheckBox()
        self.gradient_enabled_check.setChecked(current_theme.get("fullscreen_gradient_enabled", True))
        visual_layout.addRow("Enable Full-screen Gradient:", self.gradient_enabled_check)
        visual_layout.addRow(self._separator())

        # Font settings
        self.font_family_edit = QLineEdit(current_theme["main_font_family"])
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 72)
        self.font_size_spin.setValue(current_theme["main_font_size"])

        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Family:"))
        font_layout.addWidget(self.font_family_edit)
        font_layout.addWidget(QLabel("Size:"))
        font_layout.addWidget(self.font_size_spin)
        font_layout.addStretch()

        main_font_label = QLabel("Main Font:")
        visual_layout.addRow(main_font_label)
        visual_layout.addRow("", font_layout)

        visual_group.setLayout(visual_layout)
        layout.addWidget(visual_group)

        # Button Icons section
        icons_group = QGroupBox("Button Icons")
        icons_layout = QFormLayout()

        self.icon_pause_edit = QLineEdit(current_theme["icon_pause"])
        self.icon_play_edit = QLineEdit(current_theme["icon_play"])
        self.icon_reset_edit = QLineEdit(current_theme["icon_reset"])
        self.icon_done_edit = QLineEdit(current_theme["icon_done"])
        self.icon_snooze_edit = QLineEdit(current_theme["icon_snooze"])

        # Create horizontal layouts for each icon pair to maintain alignment
        pause_play_layout = QHBoxLayout()
        pause_play_layout.addWidget(QLabel("Pause:"))
        pause_play_layout.addWidget(self.icon_pause_edit)
        pause_play_layout.addWidget(QLabel("Play:"))
        pause_play_layout.addWidget(self.icon_play_edit)
        pause_play_layout.addStretch()

        reset_done_layout = QHBoxLayout()
        reset_done_layout.addWidget(QLabel("Reset:"))
        reset_done_layout.addWidget(self.icon_reset_edit)
        reset_done_layout.addWidget(QLabel("Done:"))
        reset_done_layout.addWidget(self.icon_done_edit)
        reset_done_layout.addStretch()

        snooze_layout = QHBoxLayout()
        snooze_layout.addWidget(QLabel("Snooze:"))
        snooze_layout.addWidget(self.icon_snooze_edit)
        snooze_layout.addStretch()

        icons_layout.addRow("", pause_play_layout)
        icons_layout.addRow("", reset_done_layout)
        icons_layout.addRow("", snooze_layout)

        icons_group.setLayout(icons_layout)
        layout.addWidget(icons_group)

        # Behavior Settings section
        behavior_group = QGroupBox("Behavior Settings")
        behavior_layout = QFormLayout()

        # Timer configuration (minutes)
        self.timer_minutes_spin = QSpinBox()
        self.timer_minutes_spin.setRange(1, 24 * 60)  # 1 minute to 24 hours
        parent_widget = self.parent()
        current_timer = getattr(parent_widget, "countdown_seconds", 25 * 60)
        self.timer_minutes_spin.setValue(max(1, current_timer // 60))
        behavior_layout.addRow("Timer (minutes):", self.timer_minutes_spin)

        # Reminder time window configuration (minutes)
        self.reminder_time_window_spin = QSpinBox()
        self.reminder_time_window_spin.setRange(1, 120)
        self.reminder_time_window_spin.setValue(current_theme["reminder_time_window_minutes"])
        behavior_layout.addRow("Reminder Time Window (minutes):", self.reminder_time_window_spin)

        behavior_group.setLayout(behavior_layout)
        layout.addWidget(behavior_group)

        # Buttons
        button_layout = QHBoxLayout()
        apply_btn = DialogButton("Apply")
        apply_btn.clicked.connect(self.apply_changes)
        ok_btn = DialogButton("OK")
        ok_btn.clicked.connect(self.accept_changes)
        cancel_btn = DialogButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(apply_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_predefined_theme(self, theme_name):
        if theme_name in PREDEFINED_THEMES:
            theme = PREDEFINED_THEMES[theme_name]
            self.update_ui_from_theme(theme)

    def update_ui_from_theme(self, theme):
        self.border_start_btn.color = theme["border_color_start"]
        self.border_start_btn.update_color()
        self.border_end_btn.color = theme["border_color_end"]
        self.border_end_btn.update_color()
        self.border_width_spin.setValue(theme["border_width"])

        self.notif_start_btn.color = theme["notification_bg_start"]
        self.notif_start_btn.update_color()
        self.notif_end_btn.color = theme["notification_bg_end"]
        self.notif_end_btn.update_color()
        self.text_color_btn.color = theme["notification_text_color"]
        self.text_color_btn.update_color()

        self.icon_pause_edit.setText(theme["icon_pause"])
        self.icon_play_edit.setText(theme["icon_play"])
        self.icon_reset_edit.setText(theme["icon_reset"])
        self.icon_done_edit.setText(theme["icon_done"])
        self.icon_snooze_edit.setText(theme["icon_snooze"])

        self.font_family_edit.setText(theme["main_font_family"])
        self.font_size_spin.setValue(theme["main_font_size"])

        # Update reminder time window if it exists in the theme
        if "reminder_time_window_minutes" in theme:
            self.reminder_time_window_spin.setValue(theme["reminder_time_window_minutes"])

        # Update gradient configuration if it exists in the theme
        if "fullscreen_gradient_opacity" in theme:
            self.gradient_opacity_spin.setValue(theme["fullscreen_gradient_opacity"])
        if "fullscreen_gradient_enabled" in theme:
            self.gradient_enabled_check.setChecked(theme["fullscreen_gradient_enabled"])

    def get_current_config(self):
        return {
            "border_color_start": self.border_start_btn.color,
            "border_color_end": self.border_end_btn.color,
            "border_width": self.border_width_spin.value(),
            "notification_bg_start": self.notif_start_btn.color,
            "notification_bg_end": self.notif_end_btn.color,
            "notification_text_color": self.text_color_btn.color,
            "notification_corner_radius": current_theme["notification_corner_radius"],
            "button_bg_normal": current_theme["button_bg_normal"],
            "button_bg_hover": current_theme["button_bg_hover"],
            "button_text_color": current_theme["button_text_color"],
            "button_corner_radius": current_theme["button_corner_radius"],
            "button_width": current_theme["button_width"],
            "button_height": current_theme["button_height"],
            "icon_pause": self.icon_pause_edit.text(),
            "icon_play": self.icon_play_edit.text(),
            "icon_reset": self.icon_reset_edit.text(),
            "icon_done": self.icon_done_edit.text(),
            "icon_snooze": self.icon_snooze_edit.text(),
            "main_font_family": self.font_family_edit.text(),
            "main_font_size": self.font_size_spin.value(),
            "icon_font_family": current_theme["icon_font_family"],
            "icon_font_size": current_theme["icon_font_size"],
            "notification_padding": current_theme["notification_padding"],
            "notification_offset": current_theme["notification_offset"],
            "button_spacing": current_theme["button_spacing"],
            "button_container_padding": current_theme["button_container_padding"],
            "reminder_time_window_minutes": self.reminder_time_window_spin.value(),
            "fullscreen_gradient_opacity": self.gradient_opacity_spin.value(),
            "fullscreen_gradient_enabled": self.gradient_enabled_check.isChecked(),
        }

    def apply_changes(self):
        global current_theme, current_theme_name
        current_theme.update(self.get_current_config())
        # Update the selected theme name
        current_theme_name = self.theme_combo.currentText()
        if self.parent():
            # Apply theme
            self.parent().apply_theme()
            # Apply timer value (entered in minutes) and restart timer similar to reset
            new_timer = self.timer_minutes_spin.value() * 60
            self.parent().countdown_seconds = new_timer
            self.parent().remaining = new_timer
            self.parent().running = True
            if hasattr(self.parent(), "timer"):
                self.parent().timer.start(1000)
            # Ensure buttons reflect running state after reset
            if self.parent().pause_resume_btn:
                self.parent().pause_resume_btn.setText(current_theme["icon_pause"])
            self.parent().update()
        # Save config after apply
        save_config()

    def accept_changes(self):
        self.apply_changes()
        self.accept()

    def reject(self):
        global current_theme, current_theme_name
        current_theme.clear()
        current_theme.update(self.original_theme)
        # Restore original theme name
        current_theme_name = self.original_theme_name
        if self.parent():
            # Re-apply theme
            self.parent().apply_theme()
            # Restore original timer value
            self.parent().countdown_seconds = self.original_timer_seconds
            self.parent().remaining = self.original_timer_seconds
            if self.parent().pause_resume_btn:
                self.parent().pause_resume_btn.setText(current_theme["icon_pause"])
        super().reject()


class BorderWidget(QWidget):
    def __init__(self, countdown_seconds=25 * 60, screen=None, manager=None):
        super().__init__()
        self.countdown_seconds = countdown_seconds
        self.remaining = countdown_seconds
        self.running = True  # auto-start
        self.control_widget = None
        self.complete_reminder_btn = None
        self.pause_resume_btn = None
        self.reset_btn = None
        self.remind_kit = None
        self.screen = screen  # Store the specific screen for this widget
        self.manager = manager  # Reference to the MultiScreenManager

        logging.debug("Initializing BorderWidget")
        self.init_remindkit()

        # Track the reminder currently shown in the notification bar
        self.current_reminder_id = None
        self.current_reminder_title = None

        # Visual feedback state for click detection
        self.click_highlight_active = False
        self.click_highlight_timer = QTimer(self)
        self.click_highlight_timer.timeout.connect(self.clear_click_highlight)
        self.click_highlight_timer.setSingleShot(True)

        self.init_ui()
        self.init_timer()
        self.setup_shortcuts()

    def init_remindkit(self):
        """Initialize RemindKit instance and test basic connection.

        If RemindKit cannot access reminders due to lack of permission, prompt the user
        with instructions and (on macOS) a button that opens Privacy & Security -> Reminders.
        """
        try:
            self.remind_kit = RemindKit()
            # Test basic connection by getting default calendar
            default_calendar = self.remind_kit.calendars.get_default()
            logging.info(f"RemindKit initialized successfully. Default calendar: {default_calendar.name}")
        except Exception as e:
            # Log the exception for debugging
            logging.exception("Failed to initialize RemindKit")
            self.remind_kit = None
            # Inspect the error message to see if it looks like a permissions/authorization issue.
            err_str = str(e).lower()
            if "unauthor" in err_str or "access" in err_str or "permission" in err_str:
                # Ask the user to grant permission
                self.prompt_for_reminders_permission(error_message=str(e))
            else:
                # Unknown failure - show info but don't aggressively prompt
                logging.warning("RemindKit init failed for a non-permission reason. Exception: %s", e)

    def prompt_for_reminders_permission(self, error_message: str = ""):
        """Show a dialog explaining the permission problem and offer to open Settings.

        On macOS we attempt to open the System Settings privacy Reminders pane using the
        x-apple.systempreferences URL scheme. On other platforms we give instructions.
        """
        logging.info("Prompting user to grant Reminders permission.")
        title = "Reminders Permission Required"
        body = (
            "This app needs permission to access your Reminders so it can show and complete reminders.\n\n"
            "Please allow Reminders access in your system settings.\n\n"
            "If you'd like, click 'Open Settings' to be taken to the privacy settings page (macOS).\n\n"
            "Error details:\n" + (error_message or "<no details>")
        )

        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(body)
        msg.setIcon(QMessageBox.Icon.Warning)
        open_button = msg.addButton("Open Settings", QMessageBox.ButtonRole.AcceptRole)
        msg.setDefaultButton(open_button)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == open_button:
            # Try to take the user to the correct OS settings pane.
            try:
                if sys.platform == "darwin":
                    # For macOS attempt to open Privacy & Security -> Reminders (anchor may vary by macOS version)
                    # This uses the x-apple.systempreferences URL scheme which works on many macOS releases.
                    url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Reminders"
                    # Fall back to general security pane if the reminders anchor isn't supported
                    if not QDesktopServices.openUrl(QtQUrl(url)):
                        QDesktopServices.openUrl(QtQUrl("x-apple.systempreferences:com.apple.preference.security"))
                elif sys.platform == "win32":
                    # Windows doesn't have a single "Reminders" privacy page equivalent.
                    # Open the general Settings app to Privacy & app permissions, and instruct the user.
                    # This uses the ms-settings: URI scheme available on modern Windows.
                    try:
                        os.system("start ms-settings:privacy")  # noqa: S605, S607
                    except Exception:
                        # fallback: open Settings app generically if possible
                        logging.warning("Could not open Windows settings programmatically.")
                        QDesktopServices.openUrl(QtQUrl("about:blank"))
                else:
                    # Linux / other: can't reliably open system privacy settings; open Reminders app if present
                    # Try to open the Reminders app (macOS style) or show nothing; user must open settings manually.
                    logging.info("Non-macOS platform: instruct the user to open system settings manually.")
            except Exception:
                logging.exception("Failed to open system settings programmatically.")
        else:
            logging.info("User dismissed the permission prompt.")

    def get_incomplete_reminders_due_today(self, filter_by_time_window=False):
        """Get incomplete reminders due today or in the past

        Args:
            filter_by_time_window (bool): If True, only return reminders within the configured time window
        """
        if self.remind_kit:
            try:
                from datetime import datetime, timedelta

                tomorrow = datetime.now() + timedelta(days=1)
                tomorrow = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
                reminders = list(self.remind_kit.get_reminders(due_before=tomorrow, is_completed=False))
                logging.info(f"Found {len(reminders)} incomplete reminders due before tomorrow")

                # Optionally filter by time window
                if filter_by_time_window:
                    filtered_reminders = []
                    for reminder in reminders:
                        if self.is_reminder_due_soon(reminder):
                            filtered_reminders.append(reminder)
                    logging.info(f"Filtered to {len(filtered_reminders)} reminders within time window")
                    return filtered_reminders

                return reminders
            except Exception as e:
                # If we receive an authorization-like error while fetching, prompt once for permission.
                logging.exception("Failed to get incomplete reminders")
                err_str = str(e).lower()
                if "unauthor" in err_str or "access" in err_str or "permission" in err_str:
                    # Prompt user for permission and then return empty list for now.
                    self.prompt_for_reminders_permission(error_message=str(e))
                return []
        return []

    def is_reminder_due_soon(self, reminder):
        """Check if a reminder is due within the configured time window

        Handles edge cases:
        - Reminders without due dates (always considered due soon)
        - Past due reminders (always considered due soon)
        - Timezone-aware comparisons
        - Invalid or malformed due dates
        """
        import logging
        from datetime import datetime, timedelta

        # Edge case 1: If no due date, consider it always due soon
        if not reminder.due_date:
            logging.debug(f"Reminder '{getattr(reminder, 'title', 'Unknown')}' has no due date, including")
            return True

        try:
            # Get current time and time window
            now = datetime.now()
            time_window_minutes = current_theme.get("reminder_time_window_minutes", 15)
            time_window = timedelta(minutes=time_window_minutes)

            # Edge case 2: Handle timezone-aware vs naive datetime comparison
            reminder_due = reminder.due_date
            if reminder_due.tzinfo is not None and now.tzinfo is None:
                # Convert timezone-aware reminder to local time
                reminder_due = reminder_due.replace(tzinfo=None)
                logging.debug("Converted timezone-aware due date to naive for comparison")
            elif reminder_due.tzinfo is None and now.tzinfo is not None:
                # Convert naive reminder to timezone-aware
                now = now.replace(tzinfo=None)
                logging.debug("Converted timezone-aware current time to naive for comparison")

            # Calculate time until due
            time_until_due = reminder_due - now

            # Edge case 3: Past due reminders are always considered due soon
            if time_until_due.total_seconds() < 0:
                logging.debug(f"Reminder '{getattr(reminder, 'title', 'Unknown')}' is past due, including")
                return True

            # Check if due date is within the time window
            is_due_soon = time_until_due <= time_window
            if is_due_soon:
                logging.debug(
                    f"Reminder '{getattr(reminder, 'title', 'Unknown')}' is within time window: {time_until_due}"
                )

            return is_due_soon

        except Exception as e:
            # Edge case 4: Handle any unexpected errors with due date processing
            logging.warning(f"Error processing due date for reminder '{getattr(reminder, 'title', 'Unknown')}': {e}")
            # When in doubt, include the reminder to avoid missing important items
            return True

    def get_next_reminder_text(self):
        """Get text for the next due reminder and cache its identity for completion"""
        # Use the optimized filtering at the data retrieval level
        filtered_reminders = self.get_incomplete_reminders_due_today(filter_by_time_window=True)

        if filtered_reminders:
            # Sort filtered reminders by due date (earliest first)
            filtered_reminders.sort(key=lambda r: r.due_date if r.due_date else float("inf"))
            for reminder in filtered_reminders:
                logging.debug(f"Found {reminder} next due reminder")
            # Get the first reminder (most urgent by due date)
            next_reminder = filtered_reminders[0]
            # Cache the currently displayed reminder id/title so we can complete exactly this one
            try:
                self.current_reminder_id = next_reminder.id
            except Exception:
                # Fallback to None if structure differs
                self.current_reminder_id = None
            self.current_reminder_title = getattr(next_reminder, "title", None)
            return f"📋 {next_reminder.title}"

        # No reminders to show; clear cache
        self.current_reminder_id = None
        self.current_reminder_title = None
        return ""

    def setup_shortcuts(self):
        # Create Cmd+, shortcut for preferences
        self.config_shortcut = QShortcut(QKeySequence("Ctrl+,"), self)
        self.config_shortcut.activated.connect(self.open_config)
        # Also support Cmd+, on macOS
        self.config_shortcut_mac = QShortcut(QKeySequence("Meta+,"), self)
        self.config_shortcut_mac.activated.connect(self.open_config)

    def open_config(self):
        dialog = ConfigDialog(self)
        dialog.exec()

    def init_ui(self):
        # Get the screen geometry - use specific screen if provided, otherwise primary
        if self.screen:
            screen_geometry = self.screen.geometry()
            logging.info(f"Using specific screen geometry: {screen_geometry}")
        else:
            screen_geometry = QApplication.primaryScreen().geometry()
            logging.info(f"Using primary screen geometry: {screen_geometry}")
        self.setGeometry(screen_geometry)

        # Make the window frameless and transparent
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Set window to stay on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        logging.debug("Window flags set: Frameless and transparent")
        self.create_buttons()

    def create_buttons(self):
        # Remove existing control widget if it exists
        if self.control_widget is not None:
            self.control_widget.deleteLater()

        # Create new control widget for buttons
        self.control_widget = QWidget(self)
        self.control_widget.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.control_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(current_theme["button_spacing"])

        # Complete Reminder button
        self.complete_reminder_btn = QPushButton(current_theme["icon_done"], self.control_widget)
        self.complete_reminder_btn.setFixedSize(current_theme["button_width"], current_theme["button_height"])
        self.complete_reminder_btn.setFlat(True)
        icon_font = QFont(current_theme["icon_font_family"], current_theme["icon_font_size"])
        self.complete_reminder_btn.setFont(icon_font)
        self.complete_reminder_btn.setStyleSheet(
            f"QPushButton {{ background-color: {current_theme['button_bg_normal']}; color: {current_theme['button_text_color']}; border: none; padding: 0px; }}"
            f"QPushButton:hover {{ background-color: {current_theme['button_bg_hover']}; border-radius: {current_theme['button_corner_radius']}px; }}"
        )
        self.complete_reminder_btn.clicked.connect(self.complete_current_reminder)

        # Snooze button
        self.snooze_btn = QPushButton(current_theme["icon_snooze"], self.control_widget)
        self.snooze_btn.setFixedSize(current_theme["button_width"], current_theme["button_height"])
        self.snooze_btn.setFlat(True)
        self.snooze_btn.setFont(icon_font)
        self.snooze_btn.setStyleSheet(
            f"QPushButton {{ background-color: {current_theme['button_bg_normal']}; color: {current_theme['button_text_color']}; border: none; padding: 0px; }}"
            f"QPushButton:hover {{ background-color: {current_theme['button_bg_hover']}; border-radius: {current_theme['button_corner_radius']}px; }}"
        )
        self.snooze_btn.clicked.connect(self.snooze_current_reminder)

        # Pause/Resume button
        self.pause_resume_btn = QPushButton(current_theme["icon_pause"], self.control_widget)
        self.pause_resume_btn.setFixedSize(current_theme["button_width"], current_theme["button_height"])
        self.pause_resume_btn.setFlat(True)
        icon_font = QFont(current_theme["icon_font_family"], current_theme["icon_font_size"])
        self.pause_resume_btn.setFont(icon_font)
        self.pause_resume_btn.setStyleSheet(
            f"QPushButton {{ background-color: {current_theme['button_bg_normal']}; color: {current_theme['button_text_color']}; border: none; padding: 0px; }}"
            f"QPushButton:hover {{ background-color: {current_theme['button_bg_hover']}; border-radius: {current_theme['button_corner_radius']}px; }}"
        )
        self.pause_resume_btn.clicked.connect(self.toggle_pause_resume)

        # Reset button
        self.reset_btn = QPushButton(current_theme["icon_reset"], self.control_widget)
        self.reset_btn.setFixedSize(current_theme["button_width"], current_theme["button_height"])
        self.reset_btn.setFlat(True)
        self.reset_btn.setFont(icon_font)
        self.reset_btn.setStyleSheet(
            f"QPushButton {{ background-color: {current_theme['button_bg_normal']}; color: {current_theme['button_text_color']}; border: none; padding: 0px; }}"
            f"QPushButton:hover {{ background-color: {current_theme['button_bg_hover']}; border-radius: {current_theme['button_corner_radius']}px; }}"
        )
        self.reset_btn.clicked.connect(self.reset_timer)

        button_layout.addWidget(self.complete_reminder_btn)
        button_layout.addWidget(self.snooze_btn)
        button_layout.addWidget(self.pause_resume_btn)
        button_layout.addWidget(self.reset_btn)

        self.control_widget.setLayout(button_layout)

    def should_show_fullscreen_gradient(self):
        """Determine if full-screen gradient should be displayed.

        Returns True when timer is not running (paused or finished) and
        gradient is enabled in theme configuration.
        """
        return not self.running and current_theme.get("fullscreen_gradient_enabled", True)

    def apply_theme(self):
        """Apply the current theme to the widget"""
        self.create_buttons()
        self.update()

    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

    def toggle_pause_resume(self):
        if self.running:
            self.timer.stop()
            self.pause_resume_btn.setText(current_theme["icon_play"])
            self.running = False
        else:
            self.timer.start(1000)
            self.pause_resume_btn.setText(current_theme["icon_pause"])
            self.running = True
        self.update()

        # Synchronize with other screens
        if self.manager:
            self.manager.sync_timer_state(self)

    def reset_timer(self):
        self.remaining = self.countdown_seconds
        self.running = True
        self.timer.start(1000)
        self.pause_resume_btn.setText(current_theme["icon_pause"])
        self.update()

        # Synchronize with other screens
        if self.manager:
            self.manager.sync_timer_state(self)

    def get_configured_duration(self):
        """Get the current duration from config.json"""
        load_config()  # Refresh config in case it changed
        return self.countdown_seconds

    def tick(self):
        if self.remaining > 0:
            self.remaining -= 1
        else:
            self.timer.stop()
            self.running = False
            self.pause_resume_btn.setText(current_theme["icon_play"])
            self.remaining = self.countdown_seconds  # Reset to configured duration
        self.update()

    def complete_current_reminder(self):
        """Mark exactly the reminder currently displayed as completed, without re-fetching."""
        if not self.remind_kit:
            logging.warning("RemindKit is not initialized; cannot complete reminder.")
            return
        if not self.current_reminder_id:
            logging.info("No current reminder cached/displayed to complete.")
            return
        try:
            self.remind_kit.update_reminder(self.current_reminder_id, is_completed=True)
            title = self.current_reminder_title or "<untitled>"
            logging.info(f"Marked reminder '{title}' as completed")
            # Clear cached reminder since it is now completed; UI will fetch next on repaint
            self.current_reminder_id = None
            self.current_reminder_title = None
            self.update()

            # Synchronize reminder completion across multiple screens if manager is available
            if self.manager:
                self.manager.sync_reminder_action(self, "complete")
        except Exception:
            logging.exception(f"Failed to complete reminder id={self.current_reminder_id}")

    def snooze_current_reminder(self):
        """Move the currently displayed reminder 6 hours into the future from current time."""
        if not self.remind_kit:
            logging.warning("RemindKit is not initialized; cannot snooze reminder.")
            return
        if not self.current_reminder_id:
            logging.info("No current reminder cached/displayed to snooze.")
            return
        try:
            from datetime import datetime, timedelta

            # Always set reminder to exactly 6 hours from current time, regardless of original due date
            current_time = datetime.now()
            new_due_date = current_time + timedelta(hours=6)

            # Get the current reminder details for debugging
            current_reminder = self._get_current_reminder_details()
            if not current_reminder:
                return

            # Check if this is a past event (due date is before current time)
            is_past_event = self._is_past_event(current_reminder, current_time)

            # Handle the snooze operation
            updated_reminder = self._handle_snooze_operation(current_reminder, new_due_date, is_past_event)

            # Log success
            self._log_snooze_success(current_time, new_due_date, updated_reminder)

            # Verify the snooze operation
            self._verify_snooze_operation(is_past_event, updated_reminder)

            # Clear cached reminder since it's been moved; UI will fetch next on repaint
            self.current_reminder_id = None
            self.current_reminder_title = None
            self.update()

            # Synchronize reminder snooze across multiple screens if manager is available
            if self.manager:
                self.manager.sync_reminder_action(self, "snooze")
        except Exception:
            logging.exception(f"Failed to snooze reminder id={self.current_reminder_id}")
            # Don't clear the cached reminder if snooze failed, so user can try again

    def _get_current_reminder_details(self):
        """Get current reminder details for debugging."""
        try:
            current_reminder = self.remind_kit.get_reminder_by_id(self.current_reminder_id)
            logging.info(
                f"Current reminder details - Title: {current_reminder.title}, Original due: {current_reminder.due_date}, Is completed: {getattr(current_reminder, 'is_completed', 'unknown')}"
            )
            return current_reminder
        except Exception as e:
            logging.warning(f"Could not fetch current reminder details: {e}")
            return None

    def _is_past_event(self, current_reminder, current_time):
        """Check if this is a past event."""
        is_past_event = False
        if current_reminder and current_reminder.due_date:
            is_past_event = current_reminder.due_date < current_time
            logging.info(f"Event is {'past' if is_past_event else 'future'} event")
        return is_past_event

    def _handle_snooze_operation(self, current_reminder, new_due_date, is_past_event):
        """Handle the actual snooze operation."""
        if is_past_event:
            return self._handle_past_event_snooze(current_reminder, new_due_date)
        else:
            # For future events, use direct update
            return self.remind_kit.update_reminder(self.current_reminder_id, due_date=new_due_date)

    def _handle_past_event_snooze(self, current_reminder, new_due_date):
        """Handle snoozing of past events by creating new reminder and completing old one."""
        logging.info("Handling past event by creating new reminder and completing old one")
        try:
            # Create new reminder with snoozed time
            new_reminder = self.remind_kit.create_reminder(
                title=current_reminder.title,
                due_date=new_due_date,
                notes=getattr(current_reminder, "notes", None),
                priority=getattr(current_reminder, "priority", None),
                calendar_id=getattr(current_reminder, "calendar_id", None),
            )
            logging.info(f"Created new reminder with ID: {new_reminder.id}")

            # Complete the original reminder
            self.remind_kit.update_reminder(self.current_reminder_id, is_completed=True)
            logging.info(f"Completed original reminder ID: {self.current_reminder_id}")

            return new_reminder
        except Exception:
            logging.exception("Failed to create new reminder for past event")
            # Fall back to trying direct update
            return self.remind_kit.update_reminder(self.current_reminder_id, due_date=new_due_date)

    def _log_snooze_success(self, current_time, new_due_date, updated_reminder):
        """Log successful snooze operation."""
        title = self.current_reminder_title or "<untitled>"
        logging.info(
            f"Successfully snoozed reminder '{title}' from {current_time} to {new_due_date} (6 hours from now)"
        )
        logging.info(
            f"Updated reminder due date: {updated_reminder.due_date if updated_reminder else 'No return value'}"
        )

    def _verify_snooze_operation(self, is_past_event, updated_reminder):
        """Verify the snooze operation was successful."""
        try:
            logging.info("=== Post-snooze reminder verification ===")
            # Check if the snoozed reminder still appears in today's list
            current_reminders = self.get_incomplete_reminders_due_today()
            logging.info(f"Found {len(current_reminders)} incomplete reminders after snooze:")

            # For past events, we created a new reminder, so check that ID instead
            verification_id = updated_reminder.id if (is_past_event and updated_reminder) else self.current_reminder_id

            self._log_current_reminders(current_reminders, verification_id)
            self._verify_snoozed_reminder(verification_id)

            if is_past_event:
                self._verify_original_reminder_completed()

            logging.info("=== End post-snooze verification ===")
        except Exception as e:
            logging.warning(f"Post-snooze verification failed: {e}")

    def _log_current_reminders(self, current_reminders, verification_id):
        """Log current reminders for verification."""
        for i, reminder in enumerate(current_reminders):
            is_snoozed = reminder.id == verification_id
            logging.info(
                f"  {i + 1}. '{reminder.title}' (ID: {reminder.id}) - Due: {reminder.due_date} {'<-- SNOOZED' if is_snoozed else ''}"
            )

    def _verify_snoozed_reminder(self, verification_id):
        """Verify the snoozed reminder by ID."""
        try:
            snoozed_reminder = self.remind_kit.get_reminder_by_id(verification_id)
            logging.info(
                f"Snoozed reminder verification - Title: {snoozed_reminder.title}, New due: {snoozed_reminder.due_date}, Completed: {getattr(snoozed_reminder, 'is_completed', 'unknown')}"
            )
        except Exception as e:
            logging.warning(f"Could not fetch snoozed reminder by ID: {e}")

    def _verify_original_reminder_completed(self):
        """Verify the original reminder was completed for past events."""
        try:
            old_reminder = self.remind_kit.get_reminder_by_id(self.current_reminder_id)
            logging.info(
                f"Original reminder status - Title: {old_reminder.title}, Completed: {getattr(old_reminder, 'is_completed', 'unknown')}"
            )
        except Exception as e:
            logging.warning(f"Could not fetch original reminder by ID: {e}")
            return

    def calculate_reminder_text_area_bounds(self):
        """Calculate the bounds of the reminder text area.

        This method extracts and reuses the text area calculation logic
        from the paintEvent method, allowing other methods to determine
        the position and size of the notification bar area.

        Returns:
            tuple: (text_area_x, text_area_y, text_area_width, text_area_height)
                  representing the bounds of the reminder text area
        """
        width = self.width()

        # Font settings (reuse from paintEvent)
        font = QFont(current_theme["main_font_family"], current_theme["main_font_size"])
        font_metrics = QFontMetrics(font)

        # Build text string (reuse from paintEvent)
        mm, ss = divmod(self.remaining, 60)
        timer_str = f"{mm:02}:{ss:02}"

        # Get reminder text if available
        reminder_text = self.get_next_reminder_text()

        display_text = f"{timer_str}   {reminder_text}" if reminder_text else timer_str
        text_width = font_metrics.horizontalAdvance(display_text)
        text_height = font_metrics.height()

        # Calculate button width based on visible buttons (reuse from paintEvent)
        visible_buttons = [self.pause_resume_btn, self.reset_btn]  # Always visible
        if reminder_text:  # Add reminder-specific buttons when reminder is available
            visible_buttons.extend([self.complete_reminder_btn, self.snooze_btn])

        btn_total_width = sum(btn.width() for btn in visible_buttons if btn is not None)
        btn_total_width += (len(visible_buttons) - 1) * current_theme["button_spacing"]  # Spacings between buttons

        # Calculate text area dimensions (reuse from paintEvent)
        text_area_width = (
            text_width
            + current_theme["notification_padding"]
            + btn_total_width
            + current_theme["button_container_padding"]
        )
        text_area_height = text_height + current_theme["notification_padding"]
        text_area_x = (width - text_area_width) // 2
        text_area_y = current_theme["notification_offset"]

        return (text_area_x, text_area_y, text_area_width, text_area_height)

    def paintEvent(self, event):
        logging.debug("Painting border and notification bar")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        logging.debug(f"Widget dimensions: width={width}, height={height}")

        # Draw full-screen gradient background when timer is not running
        if self.should_show_fullscreen_gradient():
            fullscreen_gradient = QLinearGradient(QPointF(0, 0), QPointF(width, height))
            # Use theme colors with configurable opacity for background effect
            start_color = QColor(*current_theme["border_color_start"])
            end_color = QColor(*current_theme["border_color_end"])
            gradient_opacity = current_theme.get("fullscreen_gradient_opacity", 30)
            start_color.setAlpha(gradient_opacity)
            end_color.setAlpha(gradient_opacity)
            fullscreen_gradient.setColorAt(0.0, start_color)
            fullscreen_gradient.setColorAt(1.0, end_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(fullscreen_gradient))
            painter.drawRect(0, 0, width, height)

        # Gradient for border
        gradient = QLinearGradient(QPointF(0, 0), QPointF(width, height))
        gradient.setColorAt(0.0, QColor(*current_theme["border_color_start"]))
        gradient.setColorAt(1.0, QColor(*current_theme["border_color_end"]))
        pen = QPen(gradient, current_theme["border_width"])
        painter.setPen(pen)

        # Font settings
        font = QFont(current_theme["main_font_family"], current_theme["main_font_size"])
        painter.setFont(font)
        font_metrics = QFontMetrics(font)

        # Build text string
        mm, ss = divmod(self.remaining, 60)
        timer_str = f"{mm:02}:{ss:02}"

        # Get reminder text if available
        reminder_text = self.get_next_reminder_text()

        # Enable the 'complete' and 'snooze' buttons only when a reminder is available
        if getattr(self, "complete_reminder_btn", None) is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self.complete_reminder_btn.setVisible(bool(reminder_text))

        if getattr(self, "snooze_btn", None) is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self.snooze_btn.setVisible(bool(reminder_text))

        display_text = f"{timer_str}   {reminder_text}" if reminder_text else timer_str
        text_width = font_metrics.horizontalAdvance(display_text)
        text_height = font_metrics.height()

        # Notification bar geometry
        # Calculate button width based on visible buttons
        visible_buttons = [self.pause_resume_btn, self.reset_btn]  # Always visible
        if reminder_text:  # Add reminder-specific buttons when reminder is available
            visible_buttons.extend([self.complete_reminder_btn, self.snooze_btn])

        btn_total_width = sum(btn.width() for btn in visible_buttons)
        btn_total_width += (len(visible_buttons) - 1) * current_theme["button_spacing"]  # Spacings between buttons
        text_area_width = (
            text_width
            + current_theme["notification_padding"]
            + btn_total_width
            + current_theme["button_container_padding"]
        )
        text_area_height = text_height + current_theme["notification_padding"]
        text_area_x = (width - text_area_width) // 2
        text_area_y = current_theme["notification_offset"]

        # Draw notification bar background
        notification_gradient = QLinearGradient(
            QPointF(text_area_x, text_area_y), QPointF(text_area_x + text_area_width, text_area_y)
        )
        notification_gradient.setColorAt(0.0, QColor(*current_theme["notification_bg_start"]))
        notification_gradient.setColorAt(1.0, QColor(*current_theme["notification_bg_end"]))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(notification_gradient))
        text_area = QRectF(
            text_area_x, text_area_y + current_theme["notification_offset"], text_area_width, text_area_height
        )
        painter.drawRoundedRect(
            text_area, current_theme["notification_corner_radius"], current_theme["notification_corner_radius"]
        )

        # Draw click highlight effect if active
        if self.click_highlight_active and reminder_text:
            # Create a subtle white overlay for the highlight effect
            highlight_color = QColor(255, 255, 255, 60)  # Semi-transparent white
            painter.setBrush(QBrush(highlight_color))
            painter.drawRoundedRect(
                text_area, current_theme["notification_corner_radius"], current_theme["notification_corner_radius"]
            )
            logging.debug("Drawing click highlight effect")

        # Draw text
        painter.setPen(QColor(*current_theme["notification_text_color"]))
        text_rect = text_area.adjusted(
            current_theme["notification_padding"] / 2,
            current_theme["notification_padding"] / 2,
            -(current_theme["notification_padding"] / 2 + btn_total_width + current_theme["button_container_padding"]),
            -current_theme["notification_padding"] / 2,
        )
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            display_text,
        )

        # Draw borders
        painter.setPen(pen)
        painter.drawLine(0, height - 1, width - 1, height - 1)  # Bottom
        painter.drawLine(0, 0, 0, height - 1)  # Left
        painter.drawLine(width - 1, 0, width - 1, height - 1)  # Right
        painter.drawLine(
            0, current_theme["notification_offset"], width - 1, current_theme["notification_offset"]
        )  # Top

        # Position controls with proper z-ordering
        if self.control_widget:
            # Force the control widget to resize based on visible buttons only
            self.control_widget.resize(btn_total_width, self.control_widget.height())

            control_x = int(
                text_area_x + text_area_width - btn_total_width - (current_theme["button_container_padding"] // 2)
            )
            control_y = int(
                text_area_y
                + current_theme["notification_offset"]
                + (text_area_height - self.control_widget.height()) // 2
            )
            self.control_widget.move(control_x, control_y)
            # Ensure control widget stays on top with proper z-ordering
            self.control_widget.raise_()
            self.control_widget.show()

        logging.info("Gradient border and notification bar drawn successfully")

    def mousePressEvent(self, event):
        """Handle mouse press events on the BorderWidget.

        This method captures mouse clicks anywhere on the border widget,
        with special handling for clicks within the reminder text area.
        Handles edge case when no reminder is displayed.
        """
        click_pos = event.position()
        logging.info(f"[CLICK EVENT] Mouse press detected at position: ({click_pos.x():.1f}, {click_pos.y():.1f})")

        # Check if there's a reminder currently displayed
        reminder_text = self.get_next_reminder_text()

        if not reminder_text:
            # Edge case: No reminder is displayed, ignore clicks in text area
            logging.info("[CLICK EVENT] No reminder displayed, handling as general click")
            self.handle_general_click(click_pos)
        else:
            # Get the reminder text area bounds
            text_area_x, text_area_y, text_area_width, text_area_height = self.calculate_reminder_text_area_bounds()

            logging.debug(
                f"[CLICK EVENT] Reminder text area bounds: x={text_area_x}, y={text_area_y}, width={text_area_width}, height={text_area_height}"
            )
            logging.debug(
                f"[CLICK EVENT] Current reminder: '{reminder_text[:50]}{'...' if len(reminder_text) > 50 else ''}'"
            )

            # Check if click occurred within the reminder text area
            if (
                text_area_x <= click_pos.x() <= text_area_x + text_area_width
                and text_area_y <= click_pos.y() <= text_area_y + text_area_height
            ):
                logging.info("[CLICK EVENT] Click detected WITHIN reminder text area - triggering app opening")
                self.handle_reminder_area_click(click_pos)
            else:
                logging.info("[CLICK EVENT] Click detected OUTSIDE reminder text area - general handling")
                self.handle_general_click(click_pos)

        # Call the parent implementation to ensure proper event handling
        super().mousePressEvent(event)

    def handle_reminder_area_click(self, click_pos):
        """Handle clicks that occur within the reminder text area.

        Args:
            click_pos: The position where the click occurred
        """
        logging.info(
            f"[APP OPENING] Handling reminder area click at position: ({click_pos.x():.1f}, {click_pos.y():.1f})"
        )

        # Log current reminder context
        if self.current_reminder_title:
            logging.info(
                f"[APP OPENING] Current reminder context: '{self.current_reminder_title}' (ID: {self.current_reminder_id})"
            )
        else:
            logging.info("[APP OPENING] No specific reminder context available")

        # Trigger visual feedback for successful click detection
        self.show_click_highlight()

        # Open the Reminders app when clicking on the reminder text area
        logging.info("[APP OPENING] Attempting to open Reminders app...")
        success = self.open_reminders_app()

        if success:
            logging.info("[APP OPENING] ✓ Successfully opened Reminders app from click")
            # Synchronize the app opening action across multiple screens if manager is available
            if self.manager:
                logging.debug(
                    f"[APP OPENING] Synchronizing app opening across {len(self.manager.border_widgets)} screens"
                )
                self.manager.sync_reminder_action(self, "open_app")
        else:
            logging.error("[APP OPENING] ✗ Failed to open Reminders app from click")
            # Handle error case with user feedback
            self.handle_reminders_app_error()

        # Additional functionality could be added here:
        # - Show reminder context menu
        # - Toggle reminder completion
        # - Open reminder details dialog
        # - etc.

    def show_click_highlight(self):
        """Show visual feedback for successful click detection.

        Activates a brief highlight effect that will be cleared automatically
        after a short duration. Also synchronizes across multiple screens.
        """
        self.click_highlight_active = True
        self.update()  # Trigger repaint to show highlight

        # Start timer to clear highlight after 200ms
        self.click_highlight_timer.start(200)
        logging.debug("[VISUAL FEEDBACK] Click highlight activated for 200ms")

        # Synchronize highlight across multiple screens if manager is available
        if self.manager:
            logging.debug(
                f"[VISUAL FEEDBACK] Synchronizing highlight across {len(self.manager.border_widgets)} screens"
            )
            self.manager.sync_click_highlight(self)

    def clear_click_highlight(self):
        """Clear the visual feedback highlight.

        This method is called automatically by the timer to remove
        the highlight effect after the specified duration.
        """
        self.click_highlight_active = False
        self.update()  # Trigger repaint to clear highlight
        logging.debug("[VISUAL FEEDBACK] Click highlight cleared automatically")

    def handle_general_click(self, click_pos):
        """Handle clicks that occur outside the reminder text area.

        Args:
            click_pos: The position where the click occurred
        """
        logging.info(f"[GENERAL CLICK] Handling general click at position: ({click_pos.x():.1f}, {click_pos.y():.1f})")
        logging.debug("[GENERAL CLICK] Click occurred outside reminder text area - no app opening triggered")

        # Placeholder for general click functionality
        # This could be used to:
        # - Toggle control visibility
        # - Open configuration dialog
        # - Show general context menu
        # - etc.
        pass

    def open_reminders_app(self):
        """Open the macOS Reminders app using QDesktopServices.

        This method attempts to open the native Reminders application
        on macOS using the system's default URL handler.

        Based on research:
        - x-apple-reminderkit:// is the current working scheme for iOS 13+ and macOS 13+
        - reminders:// was deprecated and doesn't work on macOS 13 (Ventura) and later
        - x-apple-reminder:// was an older scheme that no longer works

        Returns:
            bool: True if the app was successfully opened, False otherwise
        """
        try:
            import sys

            platform_info = "macOS" if sys.platform == "darwin" else sys.platform
            logging.info(f"[URL SCHEME] Starting Reminders app opening sequence on {platform_info}")

            # Primary scheme: x-apple-reminderkit:// (works on modern macOS/iOS)
            logging.debug("[URL SCHEME] Attempting primary scheme: x-apple-reminderkit://")
            reminders_url = QtQUrl("x-apple-reminderkit://")
            success = QDesktopServices.openUrl(reminders_url)

            if success:
                logging.info("[URL SCHEME] ✓ Successfully opened Reminders app using x-apple-reminderkit://")
                return True
            else:
                # Fallback 1: try the older reminders:// scheme (for older macOS versions)
                logging.warning("[URL SCHEME] ✗ Primary scheme 'x-apple-reminderkit://' failed")
                logging.debug("[URL SCHEME] Attempting fallback 1: reminders://")
                fallback_url = QtQUrl("reminders://")
                success = QDesktopServices.openUrl(fallback_url)

                if success:
                    logging.info("[URL SCHEME] ✓ Successfully opened Reminders app using reminders:// fallback")
                    return True
                else:
                    # Fallback 2: try the even older x-apple-reminder:// scheme
                    logging.warning("[URL SCHEME] ✗ Fallback 1 'reminders://' failed")
                    logging.debug("[URL SCHEME] Attempting fallback 2: x-apple-reminder://")
                    old_fallback_url = QtQUrl("x-apple-reminder://")
                    success = QDesktopServices.openUrl(old_fallback_url)

                    if success:
                        logging.info(
                            "[URL SCHEME] ✓ Successfully opened Reminders app using x-apple-reminder:// fallback"
                        )
                        return True
                    else:
                        # All URL schemes failed
                        logging.error("[URL SCHEME] ✗ All URL schemes failed to open Reminders app:")
                        logging.error("[URL SCHEME]   1. x-apple-reminderkit:// (primary) - FAILED")
                        logging.error("[URL SCHEME]   2. reminders:// (fallback 1) - FAILED")
                        logging.error("[URL SCHEME]   3. x-apple-reminder:// (fallback 2) - FAILED")
                        logging.error(
                            "[URL SCHEME] Possible causes: app not installed, security restrictions, or unsupported system"
                        )
                        return False

        except Exception:
            logging.exception("Exception occurred while trying to open Reminders app")
            return False

    def handle_reminders_app_error(self):
        """Handle errors when the Reminders app cannot be opened.

        Provides user feedback and potential solutions when the system
        fails to open the Reminders application.
        """
        logging.error("Handling Reminders app opening error")

        try:
            # Show a user-friendly error message
            from PyQt6.QtWidgets import QMessageBox

            msg = QMessageBox(self)
            msg.setWindowTitle("Unable to Open Reminders")
            msg.setIcon(QMessageBox.Icon.Warning)

            error_text = (
                "Could not open the Reminders app automatically.\n\n"
                "This might happen if:\n"
                "• The Reminders app is not installed\n"
                "• URL schemes are blocked by system security\n"
                "• The app is not available on this system\n\n"
                "You can try:\n"
                "• Opening Reminders manually from Applications\n"
                "• Checking system security settings\n"
                "• Restarting the application"
            )

            msg.setText(error_text)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)

            # Show the dialog non-blocking to avoid interfering with the main app
            msg.show()

            # Auto-close the dialog after 5 seconds
            from PyQt6.QtCore import QTimer

            close_timer = QTimer(self)
            close_timer.timeout.connect(msg.close)
            close_timer.setSingleShot(True)
            close_timer.start(5000)  # 5 seconds

            logging.info("Error dialog shown to user")

        except Exception:
            logging.exception("Reminders app could not be opened. Please open it manually.")


class MultiScreenManager:
    """Manages BorderWidget instances across multiple screens."""

    def __init__(self, countdown_seconds=25 * 60):
        self.countdown_seconds = countdown_seconds
        self.border_widgets = []
        self.create_widgets_for_all_screens()

    def create_widgets_for_all_screens(self):
        """Create a BorderWidget for each available screen."""
        app = QApplication.instance()
        screens = app.screens()

        logging.info(f"Found {len(screens)} screen(s)")

        for i, screen in enumerate(screens):
            logging.info(f"Creating BorderWidget for screen {i}: {screen.name()}")
            widget = BorderWidget(countdown_seconds=self.countdown_seconds, screen=screen, manager=self)
            self.border_widgets.append(widget)

    def show_all(self):
        """Show all BorderWidget instances."""
        for widget in self.border_widgets:
            widget.showMaximized()
            logging.info(f"BorderWidget shown on screen: {widget.screen.name() if widget.screen else 'primary'}")

    def sync_timer_state(self, source_widget):
        """Synchronize timer state across all widgets when one changes."""
        for widget in self.border_widgets:
            if widget != source_widget:
                widget.remaining = source_widget.remaining
                widget.running = source_widget.running

                # Synchronize timer objects
                if source_widget.running:
                    widget.timer.start(1000)
                    if widget.pause_resume_btn:
                        widget.pause_resume_btn.setText(current_theme["icon_pause"])
                else:
                    widget.timer.stop()
                    if widget.pause_resume_btn:
                        widget.pause_resume_btn.setText(current_theme["icon_play"])

                widget.update()  # Trigger repaint

    def sync_click_highlight(self, source_widget):
        """Synchronize click highlight across all screens.

        When a click is detected on one screen, show the highlight effect
        on all screens to provide consistent visual feedback.

        Args:
            source_widget: The widget where the click originated
        """
        logging.debug(f"Synchronizing click highlight across {len(self.border_widgets)} screens")
        for widget in self.border_widgets:
            if widget != source_widget:
                widget.click_highlight_active = True
                widget.update()  # Trigger repaint to show highlight
                # Start timer to clear highlight after 200ms
                widget.click_highlight_timer.start(200)

    def sync_reminder_action(self, source_widget, action_type, *args):
        """Synchronize reminder actions across all screens.

        When a reminder action (complete, snooze) is performed on one screen,
        update the reminder state on all screens.

        Args:
            source_widget: The widget where the action originated
            action_type: Type of action ('complete', 'snooze', 'open_app')
            *args: Additional arguments for the action
        """
        logging.debug(f"Synchronizing reminder action '{action_type}' across {len(self.border_widgets)} screens")

        for widget in self.border_widgets:
            if widget != source_widget:
                if action_type == "complete":
                    # Clear the current reminder on all screens since it's completed
                    widget.current_reminder_id = None
                    widget.current_reminder_title = None
                elif action_type == "snooze":
                    # Clear the current reminder on all screens since it's snoozed
                    widget.current_reminder_id = None
                    widget.current_reminder_title = None
                elif action_type == "open_app":
                    # No state change needed for opening app
                    pass

                widget.update()  # Trigger repaint to reflect changes

    def get_widgets(self):
        """Get all BorderWidget instances."""
        return self.border_widgets


def main():
    """Main entry point for the focused reminder application."""
    args = parse_args()
    setup_logging(args.verbose)
    load_config()  # Load config at startup

    logging.debug(f"Starting main with verbosity: {args.verbose}")
    app = QApplication([])
    logging.info("QApplication initialized")

    # Create MultiScreenManager to handle all screens
    screen_manager = MultiScreenManager()
    screen_manager.show_all()
    logging.info("All BorderWidgets shown across multiple screens")

    app.exec()


if __name__ == "__main__":
    main()
