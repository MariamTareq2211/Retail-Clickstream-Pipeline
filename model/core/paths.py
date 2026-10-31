"""Filesystem layout of the project.

Every module resolves paths through this file so that no script hardcodes an
absolute location. Relative paths coming from ``config.yaml`` are interpreted
against the project root, which keeps the repository portable across machines.
"""

from __future__ import annotations

from pathlib import Path

# ``model/core/paths.py`` -> parents[2] is the project root (the folder that
# contains ``data/`` and ``model/``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "model"
CONFIG_DIR = MODEL_DIR / "config"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_ROOT = MODEL_DIR / "artifacts"
REPORTS_ROOT = MODEL_DIR / "reports"


def resolve_path(path: str | Path) -> Path:
    """Resolve a configured path against the project root.

    Args:
        path: Absolute path, or a path relative to the project root.

    Returns:
        An absolute ``Path`` (not necessarily existing).
    """
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory (and its parents) if needed and return it."""
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
