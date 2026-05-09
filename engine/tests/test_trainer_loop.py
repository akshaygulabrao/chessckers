"""Tests for the continuous TrainerLoop used in async training."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.replay_buffer import ReplayBuffer
from chessckers_engine.selfplay_az import AZExample
from chessckers_engine.trainer_loop import TrainerLoop


TINY_ARCH = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _ex(value: float = 0.0) -> AZExample:
    return AZExample(
        fen=START_FEN,
        legal_moves=[
            {"uci": "e2e4", "from": "e2", "to": "e4", "piece": "P"},
            {"uci": "d2d4", "from": "d2", "to": "d4", "piece": "P"},
        ],
        visit_distribution=[0.6, 0.4],
        value_target=value,
    )


def _seed_buffer(buf: ReplayBuffer, n_games: int, examples_per_game: int = 4) -> None:
    for gid in range(1, n_games + 1):
        examples = [_ex(value=(0.5 if i % 2 == 0 else -0.5)) for i in range(examples_per_game)]
        buf.append_game(worker_id=0, game_id=gid, examples=examples)


def test_trainer_takes_steps_and_saves_weights(tmp_path: Path):
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH)
    buffer = ReplayBuffer(tmp_path / "buf")
    _seed_buffer(buffer, n_games=4)

    weights_path = tmp_path / "weights.pt"
    loop = TrainerLoop(
        model=model,
        buffer=buffer,
        weights_path=weights_path,
        checkpoint_dir=tmp_path / "ckpt",
        batch_size=4,
        min_buffer_games=2,
        weight_save_every=2,
        checkpoint_every=10,
        max_steps=5,
        log_every=0,
    )
    loop.run()

    assert loop.step == 5
    assert weights_path.exists()
    # No half-written tmp file leftover.
    assert not list(weights_path.parent.glob("*.tmp"))


def test_trainer_writes_checkpoints_at_interval(tmp_path: Path):
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH)
    buffer = ReplayBuffer(tmp_path / "buf")
    _seed_buffer(buffer, n_games=2)

    ckpt_dir = tmp_path / "ckpt"
    loop = TrainerLoop(
        model=model,
        buffer=buffer,
        weights_path=tmp_path / "weights.pt",
        checkpoint_dir=ckpt_dir,
        batch_size=2,
        min_buffer_games=1,
        weight_save_every=100,
        checkpoint_every=3,
        max_steps=7,
        log_every=0,
    )
    loop.run()

    # Steps 3, 6 trigger interval checkpoints; final flush adds one more.
    interval_checkpoints = sorted(ckpt_dir.glob("step_0000000[36].pt"))
    final_checkpoints = list(ckpt_dir.glob("*_final.pt"))
    assert len(interval_checkpoints) == 2
    assert len(final_checkpoints) == 1


def test_trainer_blocks_until_buffer_has_min_games(tmp_path: Path):
    """Workers fill the buffer at start. Trainer should sit in _wait_for_buffer
    until min_buffer_games is met, then start stepping."""
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH)
    buffer = ReplayBuffer(tmp_path / "buf")

    loop = TrainerLoop(
        model=model,
        buffer=buffer,
        weights_path=tmp_path / "weights.pt",
        checkpoint_dir=tmp_path / "ckpt",
        batch_size=2,
        min_buffer_games=3,
        max_steps=2,
        log_every=0,
        wait_poll_seconds=0.05,
    )

    # Run trainer in a thread so we can race a writer against it.
    result = {}

    def _run():
        result["steps"] = loop.run()

    t = threading.Thread(target=_run)
    t.start()
    # Give the trainer a moment to enter _wait_for_buffer.
    time.sleep(0.15)
    assert loop.step == 0, "trainer must not step before buffer is ready"

    # Now seed the buffer and let the trainer proceed.
    _seed_buffer(buffer, n_games=3)
    t.join(timeout=10.0)
    assert not t.is_alive(), "trainer did not finish after buffer filled"
    assert result["steps"] == 2


def test_trainer_stop_event_halts_loop(tmp_path: Path):
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH)
    buffer = ReplayBuffer(tmp_path / "buf")
    _seed_buffer(buffer, n_games=2)

    stop = threading.Event()
    loop = TrainerLoop(
        model=model,
        buffer=buffer,
        weights_path=tmp_path / "weights.pt",
        checkpoint_dir=tmp_path / "ckpt",
        batch_size=2,
        min_buffer_games=1,
        log_every=0,
        wait_poll_seconds=0.05,
        # No max_steps — would run forever without the stop.
        stop_event=stop,
    )

    def _run():
        loop.run()

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.3)  # let it take some steps
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive(), "trainer did not honor stop_event"
    assert loop.step > 0


def test_trainer_checkpoint_includes_optimizer_state(tmp_path: Path):
    """Durable checkpoints should be a dict with model + optimizer + step,
    not the bare state_dict format. Critical for spot-preemption resume:
    Adam moments take time to build up and starting fresh costs LR adaptation."""
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH)
    buffer = ReplayBuffer(tmp_path / "buf")
    _seed_buffer(buffer, n_games=2)

    ckpt_dir = tmp_path / "ckpt"
    loop = TrainerLoop(
        model=model,
        buffer=buffer,
        weights_path=tmp_path / "weights.pt",
        checkpoint_dir=ckpt_dir,
        batch_size=2,
        min_buffer_games=1,
        weight_save_every=100,
        checkpoint_every=3,
        max_steps=4,
        log_every=0,
    )
    loop.run()

    # Find the durable checkpoint at step 3.
    [ckpt] = list(ckpt_dir.glob("step_00000003.pt"))
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict), f"checkpoint should be a dict, got {type(payload)}"
    assert "model" in payload
    assert "optimizer" in payload
    assert "step" in payload
    assert payload["step"] == 3
    # Adam optimizer state has per-parameter "exp_avg" / "exp_avg_sq" tensors.
    opt_state = payload["optimizer"]
    assert "state" in opt_state
    # At least one parameter has accumulated moments.
    assert any("exp_avg" in v for v in opt_state["state"].values()), opt_state


def test_trainer_resumes_optimizer_step_from_checkpoint(tmp_path: Path):
    """After saving a checkpoint, a fresh TrainerLoop pointing resume_from at
    that file should pick up at the saved step number (not 0)."""
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH)
    buffer = ReplayBuffer(tmp_path / "buf")
    _seed_buffer(buffer, n_games=2)

    ckpt_dir = tmp_path / "ckpt"
    weights_path = tmp_path / "weights.pt"

    # Phase 1: train for 6 steps, write a checkpoint at step 3.
    loop1 = TrainerLoop(
        model=model, buffer=buffer,
        weights_path=weights_path, checkpoint_dir=ckpt_dir,
        batch_size=2, min_buffer_games=1, weight_save_every=100,
        checkpoint_every=3, max_steps=6, log_every=0,
    )
    loop1.run()
    [ckpt] = list(ckpt_dir.glob("step_00000003.pt"))

    # Phase 2: fresh trainer, fresh model, resume from step-3 checkpoint.
    # max_steps is *absolute* (matches "total steps reached"), so capping at 8
    # means 5 additional steps after resuming at step=3.
    fresh_model = ChesskersScorer(**TINY_ARCH)
    loop2 = TrainerLoop(
        model=fresh_model, buffer=buffer,
        weights_path=tmp_path / "weights2.pt",
        checkpoint_dir=tmp_path / "ckpt2",
        batch_size=2, min_buffer_games=1, weight_save_every=100,
        checkpoint_every=100, max_steps=8, log_every=0,
        resume_from=ckpt,
    )
    loop2.run()
    assert loop2.step == 8, f"resumed trainer should reach step 8, got {loop2.step}"


def test_trainer_resume_from_legacy_state_dict_is_tolerated(tmp_path: Path):
    """An old-format checkpoint (raw state_dict, no optimizer) should load
    without crashing — model weights apply, optimizer + step start fresh."""
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH)
    buffer = ReplayBuffer(tmp_path / "buf")
    _seed_buffer(buffer, n_games=2)
    legacy = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), legacy)

    fresh = ChesskersScorer(**TINY_ARCH)
    loop = TrainerLoop(
        model=fresh, buffer=buffer,
        weights_path=tmp_path / "weights.pt",
        checkpoint_dir=tmp_path / "ckpt",
        batch_size=2, min_buffer_games=1, weight_save_every=100,
        checkpoint_every=100, max_steps=3, log_every=0,
        resume_from=legacy,
    )
    loop.run()
    # 3 fresh steps starting from step=0 (legacy doesn't carry step).
    assert loop.step == 3


def test_trainer_loss_is_finite(tmp_path: Path):
    """Sanity: training a few steps on synthetic data should yield finite losses
    (not NaN/inf), which would indicate gradient or encoding bugs."""
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH)
    buffer = ReplayBuffer(tmp_path / "buf")
    _seed_buffer(buffer, n_games=3, examples_per_game=4)

    loop = TrainerLoop(
        model=model,
        buffer=buffer,
        weights_path=tmp_path / "weights.pt",
        checkpoint_dir=tmp_path / "ckpt",
        batch_size=4,
        min_buffer_games=1,
        max_steps=10,
        log_every=0,
    )
    loop.run()

    # All parameters should be finite after training.
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite parameter after training: {name}"
