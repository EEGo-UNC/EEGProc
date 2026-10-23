"""Compatibility entry point for the reusable training tracker."""
from pathlib import Path
import sys

# Keep direct execution from an uninstalled checkout working.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from training_tracker.core import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
