"""End-to-end smoke tests for the async coordinator.

These spin up real subprocesses (mp spawn context) so they run slower than
the per-component tests. They exist to catch wiring bugs that only show up
when workers, trainer, and buffer are all running simultaneously."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chessckers_engine.selfplay_az_async import run_async_training


TINY_ARCH = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}


@pytest.mark.slow
def test_async_run_produces_games_and_trainer_steps(tmp_path: Path):
    """Smoke test: 2 workers + trainer for ~25s. Buffer should fill, trainer
    should take steps, weights file should exist, summary should write."""
    run_dir = tmp_path / "run"
    summary = run_async_training(
        run_dir=run_dir,
        model_arch=TINY_ARCH,
        device="cpu",
        n_workers=2,
        n_sims=4,
        mcts_batch_size=1,
        max_plies=40,
        trainer_batch_size=4,
        weight_save_every=2,
        checkpoint_every=10,
        min_buffer_games=2,
        eval_every_seconds=1e9,    # disable eval for the smoke
        eval_games=2,              # cheap defaults in case eval fires anyway
        eval_sims=4,
        eval_workers=2,
        run_seconds=25.0,
        main_loop_poll_seconds=0.5,
        seed=42,
    )

    assert (run_dir / "weights.pt").exists()
    assert (run_dir / "run_summary.json").exists()
    assert summary["buffer_games"] >= 2, summary
    assert summary["trainer_steps"] > 0, summary

    # Summary file matches return value.
    on_disk = json.loads((run_dir / "run_summary.json").read_text())
    assert on_disk["trainer_steps"] == summary["trainer_steps"]


@pytest.mark.slow
def test_async_run_shared_inference_mode(tmp_path: Path):
    """Coordinator with shared_inference=True: workers share one inference
    server. Verify games + trainer steps land and the server saw requests."""
    run_dir = tmp_path / "run"
    summary = run_async_training(
        run_dir=run_dir,
        model_arch=TINY_ARCH,
        device="cpu",
        n_workers=2,
        n_sims=4,
        max_plies=30,
        trainer_batch_size=4,
        weight_save_every=2,
        checkpoint_every=10,
        min_buffer_games=2,
        eval_every_seconds=1e9,
        eval_games=2, eval_sims=4, eval_workers=2,
        run_seconds=20.0,
        main_loop_poll_seconds=0.5,
        seed=42,
        shared_inference=True,
        shared_max_batch_size=4,
        shared_timeout_ms=5.0,
    )
    assert (run_dir / "weights.pt").exists()
    assert summary["buffer_games"] >= 2, summary
    assert summary["trainer_steps"] > 0, summary


@pytest.mark.slow
def test_async_run_writes_eval_log_when_enabled(tmp_path: Path):
    """With eval_every_seconds=1, the coordinator should produce at least one
    eval line during a 20s run (assuming the trainer takes >=1 step)."""
    run_dir = tmp_path / "run"
    run_async_training(
        run_dir=run_dir,
        model_arch=TINY_ARCH,
        device="cpu",
        n_workers=1,
        n_sims=4,
        mcts_batch_size=1,
        max_plies=30,
        trainer_batch_size=4,
        weight_save_every=2,
        checkpoint_every=10,
        min_buffer_games=1,
        eval_every_seconds=2.0,
        eval_games=2,
        eval_sims=4,
        eval_workers=2,
        run_seconds=60.0,  # pure-Python move-gen (no Rust accel): more wall-clock to reach an eval cycle
        main_loop_poll_seconds=0.5,
        seed=7,
    )
    eval_log = run_dir / "eval.jsonl"
    assert eval_log.exists()
    lines = eval_log.read_text().strip().splitlines()
    assert len(lines) >= 1
    parsed = json.loads(lines[0])
    assert "as_white_vs_random" in parsed
    assert "as_black_vs_random" in parsed
