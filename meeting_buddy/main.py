"""Main entry point for Meeting Buddy application.

This module provides the main entry point for the Meeting Buddy application
using the MVP (Model-View-Presenter) architecture pattern.
"""

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from PyQt6.QtWidgets import QApplication

from .presenters.meeting_buddy_presenter import MeetingBuddyPresenter
from .utils.logging_config import setup_logging


def create_argument_parser() -> ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = ArgumentParser(
        description="Meeting Buddy - Audio Recording and Transcription Tool",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run with default settings
  %(prog)s -v                 # Run with verbose logging
  %(prog)s -vv                # Run with debug logging
""",
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity (use -v for info, -vv for debug)"
    )

    return parser


def main() -> None:
    """Main entry point for the Meeting Buddy application."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging based on verbosity
    setup_logging(verbosity=args.verbose)

    # Create Qt application
    app = QApplication(sys.argv)

    # Create and show the main window using MVP pattern
    presenter = MeetingBuddyPresenter()
    presenter.show_view()

    # Start the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
