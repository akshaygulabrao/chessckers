"""Device selection helper for GPU-ready training/inference.

Single entry point: `pick_device(name)`. Pass "auto" to get the best available
(cuda > mps > cpu); pass an explicit name to force it. Use the returned
`torch.device` everywhere — module forward calls, encoded tensors, etc.

Usage:
    device = pick_device("auto")
    model = ChesskersScorer().to(device)
    pos = encode_position(fen).to(device)
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("chessckers_engine.device")


def pick_device(name: str = "auto") -> torch.device:
    """Resolve a device name to a torch.device.

    "auto" picks cuda if available, else mps, else cpu. Explicit names
    ("cuda", "mps", "cpu") are honored as-is and will raise if unavailable."""
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            chosen = torch.device("cuda")
        elif torch.backends.mps.is_available():
            chosen = torch.device("mps")
        else:
            chosen = torch.device("cpu")
        log.info("device=auto resolved to %s", chosen)
        return chosen
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS not available")
    return torch.device(name)


def model_device(model: torch.nn.Module) -> torch.device:
    """Get the device a module's parameters live on. Cheaper than tracking
    device separately — just queries the first parameter."""
    return next(model.parameters()).device
