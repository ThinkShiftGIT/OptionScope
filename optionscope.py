"""
OptionScope - Entry point script for launching the application.

This script provides a clean entry point for starting the application
using the Poetry script defined in pyproject.toml.
"""

import sys
import traceback

def run_application():
    """Run the OptionScope application."""
    try:
        from app.app import main
        main()
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all dependencies are installed:")
        print("  poetry install")
        sys.exit(1)
    except Exception as e:
        print(f"Error running application: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_application()
