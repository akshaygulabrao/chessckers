"""Tests for the async self-play worker.

Runs the worker in-process (not via mp.spawn) so the assertions are easy
to express. The pickle/spawn-compatibility path is exercised end-to-end
in the Phase 5 smoke run, not here."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.replay_buffer import ReplayBuffer
from chessckers_engine.selfplay_worker_async import play_forever
from chessckers_engine.train_az import save_checkpoint


TINY_ARCH = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}


def _payload(buffer_root: Path, weights_path: Path, **overrides) -> dict:
    base = {
        "worker_id": 0,
        "device": "cpu",
        "model_arch": TINY_ARCH,
        "weights_path": str(weights_path),
        "buffer_root": str(buffer_root),
        "n_sims": 4,
        "c_puct": 1.5,
        "temperature": 1.0,
        "dirichlet_alpha": None,
        "dirichlet_eps": 0.25,
        "mcts_batch_size": 1,
        "vloss_batch": 1,
        "max_plies": 60,
        "seed": 0,
        "stop_path": None,
        "max_games": 1,
        "weights_poll_seconds": 0.05,
    }
    base.update(overrides)
    return base


def test_worker_plays_max_games_and_appends_to_buffer(tmp_path: Path):
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH)
    weights_path = tmp_path / "weights.pt"
    save_checkpoint(model, weights_path)

    buffer_root = tmp_path / "buf"
    payload = _payload(buffer_root, weights_path, max_games=2)

    n_played = play_forever(payload)
    assert n_played == 2

    buffer = ReplayBuffer(buffer_root)
    assert buffer.count_games() == 2
    # Each game should produce >= 1 example (unless the game ended at ply 0,
    # which shouldn't happen from the standard start position).
    assert buffer.count_examples() > 0


def test_worker_waits_for_weights_file_before_starting(tmp_path: Path):
    torch.manual_seed(0)
    weights_path = tmp_path / "weights.pt"  # NOT created yet
    buffer_root = tmp_path / "buf"
    stop_path = tmp_path / "STOP"

    payload = _payload(
        buffer_root, weights_path,
        max_games=1, stop_path=str(stop_path),
    )

    result = {}

    def _run():
        result["n"] = play_forever(payload)

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.2)  # Give worker time to enter the wait loop.
    # Worker is blocked on missing weights — buffer should still be empty.
    buffer = ReplayBuffer(buffer_root)
    assert buffer.count_games() == 0

    # Now trip the stop signal so the test doesn't hang.
    stop_path.touch()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert result["n"] == 0


def test_worker_hot_reloads_weights_on_mtime_change(tmp_path: Path):
    """Worker should call load_checkpoint when weights file mtime advances.

    We can't easily probe internal state, so we instead verify the worker
    runs successfully across two games while we mutate the file mid-flight."""
    torch.manual_seed(0)
    model_a = ChesskersScorer(**TINY_ARCH)
    weights_path = tmp_path / "weights.pt"
    save_checkpoint(model_a, weights_path)

    buffer_root = tmp_path / "buf"
    stop_path = tmp_path / "STOP"
    payload = _payload(
        buffer_root, weights_path,
        max_games=3, stop_path=str(stop_path),
    )

    result = {}

    def _run():
        result["n"] = play_forever(payload)

    t = threading.Thread(target=_run)
    t.start()
    # Let one game kick off, then write new weights.
    time.sleep(0.05)
    torch.manual_seed(99)
    model_b = ChesskersScorer(**TINY_ARCH)
    save_checkpoint(model_b, weights_path)
    # Bump mtime explicitly in case the OS gives us same-second writes.
    import os
    os.utime(weights_path, (time.time() + 10, time.time() + 10))

    t.join(timeout=90.0)  # pure-Python move-gen (no Rust accel): 3 games need headroom
    assert not t.is_alive()
    assert result["n"] == 3

    # All 3 games landed in the buffer.
    buffer = ReplayBuffer(buffer_root)
    assert buffer.count_games() == 3


def test_worker_shared_inference_mode_plays_games(tmp_path: Path):
    """Worker in shared-inference mode (request_q + response_q in payload)
    skips local model loading and routes leaf evals through CrossInferenceClient
    talking to a CrossInferenceServer in this test process."""
    import multiprocessing as mp

    from chessckers_engine.cross_inference import CrossInferenceServer

    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH).eval()
    ctx = mp.get_context("spawn")
    request_q = ctx.Queue()
    response_q = ctx.Queue()
    server = CrossInferenceServer(model, request_q, [response_q],
                                   max_batch_size=4, timeout_ms=5.0)
    server.start()

    buffer_root = tmp_path / "buf"
    payload = _payload(buffer_root, weights_path=tmp_path / "unused.pt", max_games=2)
    # Replace per-worker keys with shared-mode keys.
    payload.pop("weights_path", None)
    payload.pop("device", None)
    payload.pop("model_arch", None)
    payload.pop("mcts_batch_size", None)
    payload["request_q"] = request_q
    payload["response_q"] = response_q

    try:
        n_played = play_forever(payload)
    finally:
        server.shutdown()

    assert n_played == 2
    buffer = ReplayBuffer(buffer_root)
    assert buffer.count_games() == 2
    assert buffer.count_examples() > 0
    # Server should have processed >= one batch.
    stats = server.stats()
    assert stats["n_requests"] > 0


def test_worker_honors_stop_file_between_games(tmp_path: Path):
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH)
    weights_path = tmp_path / "weights.pt"
    save_checkpoint(model, weights_path)

    buffer_root = tmp_path / "buf"
    stop_path = tmp_path / "STOP"
    payload = _payload(
        buffer_root, weights_path,
        max_games=None, stop_path=str(stop_path),
    )

    result = {}

    def _run():
        result["n"] = play_forever(payload)

    t = threading.Thread(target=_run)
    t.start()
    # Let it complete at least one game, then ask it to stop.
    time.sleep(2.0)
    stop_path.touch()
    t.join(timeout=20.0)
    assert not t.is_alive(), "worker did not honor stop signal"
    assert result["n"] >= 1
