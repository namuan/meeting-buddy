"""Main entry point for Meeting Buddy application.

This module provides the main entry point for the Meeting Buddy application
using the MVP (Model-View-Presenter) architecture pattern.
"""

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from PyQt6.QtWidgets import QApplication

from .presenters.meeting_buddy_presenter import MeetingBuddyPresenter
from .utils.logging_config import setup_logging
from .utils.performance_metrics import get_performance_summary, reset_performance_stats
from .utils.structured_logging import get_structured_logger


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
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -v for info, -vv for debug, -vvv for trace)",
    )

    parser.add_argument("--log-file", type=str, help="Log to file (in addition to console)")

    parser.add_argument("--no-structured-logging", action="store_true", help="Disable structured logging features")

    parser.add_argument("--no-redaction", action="store_true", help="Disable sensitive data redaction in logs")

    parser.add_argument("--performance-stats", action="store_true", help="Show performance statistics on exit")

    parser.add_argument("--reset-performance", action="store_true", help="Reset performance statistics on startup")

    return parser


def main() -> None:
    """Main entry point for the Meeting Buddy application."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup enhanced logging
    setup_logging(
        verbosity=args.verbose,
        log_file=args.log_file,
        log_to_console=True,
        enable_structured_logging=not args.no_structured_logging,
        redact_sensitive_data=not args.no_redaction,
    )

    # Get structured logger for main
    logger = get_structured_logger(__name__)

    # Reset performance stats if requested
    if args.reset_performance:
        reset_performance_stats()
        logger.info("Performance statistics reset", reset_requested=True)

    logger.info(
        "Meeting Buddy application starting",
        verbosity_level=args.verbose,
        structured_logging=not args.no_structured_logging,
        data_redaction=not args.no_redaction,
        log_file=args.log_file or "console_only",
    )

    try:
        # Create Qt application
        app = QApplication(sys.argv)

        logger.info("Qt application created", qt_version=app.applicationVersion())

        # Create and show the main window using MVP pattern
        with logger.context(component="presenter_initialization"):
            presenter = MeetingBuddyPresenter()
            presenter.show_view()
            logger.info("Main window initialized and displayed")

        # Start the event loop
        logger.info("Starting Qt event loop")
        exit_code = app.exec()

        logger.info("Application shutting down", exit_code=exit_code)

        # Show performance statistics if requested
        if args.performance_stats:
            _show_performance_statistics(logger)

        sys.exit(exit_code)

    except Exception as e:
        logger.exception("Application startup failed", error_type=type(e).__name__, error_message=str(e))
        raise


def _show_performance_statistics(logger) -> None:
    """Display performance statistics.

    Args:
        logger: Logger instance for output
    """
    try:
        summary = get_performance_summary()

        if summary["total_operations"] > 0:
            logger.info(
                "Performance Statistics Summary",
                total_operations=summary["total_operations"],
                total_duration=f"{summary['total_duration']:.2f}s",
                avg_duration=f"{summary['average_duration']:.3f}s",
                ops_per_second=f"{summary['operations_per_second']:.1f}",
                success_rate=f"{summary['success_rate']:.1f}%",
                operation_types=summary["operation_count"],
            )
        else:
            logger.info("No performance data collected")

    except Exception as e:
        logger.exception("Failed to display performance statistics", error=str(e))


if __name__ == "__main__":
    main()
