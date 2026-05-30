"""Tests for the checkpoint-loading gate in probe_drift.

The renderer and eval paths are covered elsewhere; what's worth pinning here
is detect_arch — it decides which checkpoints load vs. get skipped, and a
wrong decision either crashes the probe or silently mis-loads weights.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from chessckers_engine.encoding import MOVE_D
from chessckers_engine.model import ChesskersScorer

# probe_drift.py lives at the engine root, not inside the package.
_SPEC = importlib.util.spec_from_file_location(
    "probe_drift", Path(__file__).resolve().parent.parent / "probe_drift.py"
)
probe_drift = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe_drift)


def test_detect_arch_reads_current_model():
    arch = probe_drift.detect_arch(ChesskersScorer().state_dict())
    assert arch == {"c_filters": 96, "d_hidden": 256, "n_blocks": 4}


def test_detect_arch_honors_nondefault_hyperparams():
    arch = probe_drift.detect_arch(
        ChesskersScorer(d_hidden=128, c_filters=64, n_blocks=2).state_dict()
    )
    assert arch == {"c_filters": 64, "d_hidden": 128, "n_blocks": 2}


def test_detect_arch_rejects_pre_residual_topology():
    # an old checkpoint: a trunk conv + a move encoder but no residual blocks
    fake = {
        "position_trunk.0.weight": torch.zeros(32, 14, 3, 3),
        "move_encoder.0.weight": torch.zeros(128, MOVE_D),
    }
    with pytest.raises(ValueError, match="residual"):
        probe_drift.detect_arch(fake)


def test_detect_arch_rejects_old_move_encoding():
    fake = {
        "position_trunk.0.weight": torch.zeros(96, 14, 3, 3),
        "position_trunk.3.conv1.weight": torch.zeros(96, 96, 3, 3),
        "move_encoder.0.weight": torch.zeros(256, 140),  # old 140-dim encoding
    }
    with pytest.raises(ValueError, match="move encoding"):
        probe_drift.detect_arch(fake)


def test_load_model_returns_reason_for_incompatible(tmp_path):
    p = tmp_path / "old.pt"
    torch.save({"position_trunk.0.weight": torch.zeros(32, 14, 3, 3),
                "move_encoder.0.weight": torch.zeros(128, 140)}, p)
    model, reason = probe_drift.load_model(p, "cpu")
    assert model is None and "residual" in reason


def test_load_model_loads_current_checkpoint(tmp_path):
    p = tmp_path / "cur.pt"
    torch.save(ChesskersScorer().state_dict(), p)
    model, arch = probe_drift.load_model(p, "cpu")
    assert model is not None and arch["n_blocks"] == 4
