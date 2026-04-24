"""
Pytest configuration.

This file ensures that the project root is available on Python's import path
so tests can import modules from the src package.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))