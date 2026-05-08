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
