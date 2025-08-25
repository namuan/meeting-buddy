#!/usr/bin/env python3
"""Entry point script for Meeting Buddy application.

This script serves as the main entry point for PyInstaller builds,
avoiding relative import issues by importing from the package properly.
"""

from meeting_buddy.main import main

if __name__ == "__main__":
    main()
