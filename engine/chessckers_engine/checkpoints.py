"""Helpers for the `engine/weights/` checkpoint directory.

`DEFAULT_WEIGHTS_DIR` resolves to `<engine project root>/weights/` regardless of
where the package is invoked from, so both training (`train.py`) and inference
(`__main__.py`) agree on where to look.

`load_checkpoint(model, path)` loads a state dict with `strict=False`, which
makes M4-phase-1 checkpoints (no value_head keys) load cleanly into the
post-AlphaZero model — the value head simply stays at its random init until
self-play training fills it in. Logs any missing or unexpected keys.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

log = logging.getLogger("chessckers_engine.checkpoints")

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


def load_checkpoint(model: nn.Module, path: str | Path) -> tuple[list[str], list[str]]:
    """Load weights with strict=False so old checkpoints (without value_head)
    load gracefully. Returns (missing_keys, unexpected_keys) and logs any.
    Loads onto the model's current device so it works regardless of where
    the model lives (cpu/cuda/mps)."""
    target_device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=target_device, weights_only=True)
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing:
        log.info("checkpoint %s missing keys (kept at random init): %s", path, missing)
    if unexpected:
        log.warning("checkpoint %s has unexpected keys (ignored): %s", path, unexpected)
    return missing, unexpected
