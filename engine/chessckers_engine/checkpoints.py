"""Helpers for the `engine/weights/` checkpoint directory.

`DEFAULT_WEIGHTS_DIR` resolves to `<engine project root>/weights/` regardless of
where the package is invoked from, so both training (`train.py`) and inference
(`__main__.py`) agree on where to look.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def latest_checkpoint(weights_dir: Path | None = None) -> Path | None:
    """Most-recently-modified `*.pt` under `weights_dir`, or None if none exist."""
    d = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
    if not d.exists():
        return None
    candidates = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def default_checkpoint_path(weights_dir: Path | None = None) -> Path:
    """Fresh timestamped `.pt` path under `weights_dir`. Creates the dir if missing."""
    d = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt"
