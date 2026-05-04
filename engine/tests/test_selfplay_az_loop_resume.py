"""Tests for the resume mechanism in selfplay_az_loop.

The mechanism saves per-iter state so a kill (e.g., spot-instance preemption)
can be resumed from the last completed iter without losing the model, the
best checkpoint, or the replay buffer."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az import AZExample
from chessckers_engine.selfplay_az_loop import (
    _atomic_write,
    _load_resume_state,
    _save_resume_state,
)
from chessckers_engine.train_az import save_checkpoint


def _example(visit_dist: list[float], value: float) -> AZExample:
    """Minimal AZExample for round-trip testing."""
    return AZExample(
        fen="dummy", legal_moves=[{"uci": f"M{i}"} for i in range(len(visit_dist))],
        visit_distribution=visit_dist, value_target=value,
    )


def test_atomic_write_no_partial_file_on_failure(tmp_path: Path):
    """If the write_fn raises mid-write, the destination file must NOT exist
    (only the .tmp may, and that's overwritten on next attempt)."""
    target = tmp_path / "x.json"

    def writer_that_fails(p: Path) -> None:
        p.write_text("partial")
        raise RuntimeError("simulated kill mid-write")

    try:
        _atomic_write(target, writer_that_fails)
    except RuntimeError:
        pass
    assert not target.exists(), "destination must not exist after failed write"


def test_atomic_write_rename_is_visible(tmp_path: Path):
    target = tmp_path / "y.json"
    _atomic_write(target, lambda p: p.write_text("ok"))
    assert target.read_text() == "ok"
    # tmp file should be cleaned up by the rename
    assert not target.with_suffix(".json.tmp").exists()


def test_save_resume_state_writes_state_and_buffer(tmp_path: Path):
    rb: deque = deque(maxlen=5)
    rb.append([_example([0.5, 0.5], 0.0), _example([1.0], 1.0)])
    rb.append([_example([0.3, 0.7], -1.0)])

    _save_resume_state(tmp_path, completed_iter=2, total_iters=10, seed=42, replay_buffer=rb)

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["completed_iters"] == 2
    assert state["total_iters"] == 10
    assert state["seed"] == 42
    assert "updated_at" in state

    # Replay buffer should round-trip: same lengths and same example fields.
    rb_loaded = torch.load(tmp_path / "replay_buffer.pt", map_location="cpu", weights_only=False)
    assert len(rb_loaded) == 2
    assert len(rb_loaded[0]) == 2 and len(rb_loaded[1]) == 1
    assert rb_loaded[0][0].value_target == 0.0
    assert rb_loaded[1][0].value_target == -1.0


def test_load_resume_state_restores_model_buffer_and_iter(tmp_path: Path):
    """End-to-end: save → reset model weights → load → verify model and buffer match."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    # Mark the model: stash one tensor's data so we can verify it's restored.
    original_param = next(iter(model.parameters())).data.clone()

    # Pretend iter 3 completed: save model checkpoint + best.pt + state + buffer.
    save_checkpoint(model, tmp_path / "iter-az-003.pt")
    save_checkpoint(model, tmp_path / "best.pt")
    rb = deque([[_example([1.0], 0.5)]] * 3, maxlen=10)
    _save_resume_state(tmp_path, completed_iter=3, total_iters=15, seed=1, replay_buffer=rb)

    # Now perturb the in-memory weights to simulate a fresh restart with random init.
    for p in model.parameters():
        p.data.zero_()
    best = ChesskersScorer()
    for p in best.parameters():
        p.data.zero_()

    start_iter, rb_loaded = _load_resume_state(tmp_path, model, best, buffer_iters=10)

    assert start_iter == 3, "start_iter should equal completed_iters (next iter to run)"
    # Weights restored from iter-az-003.pt
    restored_param = next(iter(model.parameters())).data
    assert torch.allclose(restored_param, original_param), "model weights not restored"
    # best.pt also restored
    restored_best = next(iter(best.parameters())).data
    assert torch.allclose(restored_best, original_param), "best weights not restored"
    # Replay buffer restored
    assert len(rb_loaded) == 3
    assert rb_loaded[0][0].value_target == 0.5


def test_load_resume_state_handles_missing_replay_buffer(tmp_path: Path):
    """If state.json + iter checkpoint exist but replay buffer was lost (e.g.,
    killed between iter ckpt and buffer save), we should warn but not crash."""
    torch.manual_seed(0)
    model = ChesskersScorer()
    save_checkpoint(model, tmp_path / "iter-az-001.pt")
    save_checkpoint(model, tmp_path / "best.pt")
    # Write state.json by hand WITHOUT a replay_buffer.pt
    (tmp_path / "state.json").write_text(json.dumps({
        "completed_iters": 1, "total_iters": 5, "seed": 0, "updated_at": "x"
    }))

    start_iter, rb = _load_resume_state(tmp_path, model, None, buffer_iters=10)
    assert start_iter == 1
    assert len(rb) == 0, "missing buffer file should yield empty buffer, not crash"
