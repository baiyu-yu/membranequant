"""Ensure the package parent is on sys.path so imports work from any cwd."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent


def ensure_import_path() -> Path:
    """Add repo root (parent of this package) to sys.path if missing."""
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _REPO_ROOT
