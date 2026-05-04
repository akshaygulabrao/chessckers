"""Tests for the multiprocess worker path in selfplay_az_loop.

The cloud-run-001 attempt confirmed thread-based workers can't drive a 4090
(GIL contention pins the GPU at 1% util). The fix is to spawn subprocesses,
each with its own model copy + InferenceServer + CUDA context.

These tests cover the worker function on CPU. CUDA-specific behavior is
verified in production runs since the test box has no GPU."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az_loop import (
    _eval_game_subprocess,
    _play_game_subprocess,
    _run_eval_parallel,
)
from chessckers_engine.train_az import save_checkpoint


def _payload(state_path: Path, model_arch: dict, **overrides) -> dict:
    base = {
        "state_path": str(state_path),
        "model_arch": model_arch,
        "device": "cpu",
        "mcts_batch_size": 1,
        "n_sims": 4,
        "c_puct": 1.5,
        "temperature": 1.0,
        "seed": 0,
        "dirichlet_alpha": None,
        "dirichlet_eps": 0.25,
        "vloss_batch": 1,
    }
    base.update(overrides)
    return base


def test_subprocess_worker_returns_az_game(tmp_path: Path):
    """Smoke test: the worker function plays one game end-to-end and returns
    a populated AZGame. Tests the 'no inference server' path (mcts_batch_size=1)."""
    torch.manual_seed(0)
    arch = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}  # tiny — fast smoke
    model = ChesskersScorer(**arch)
    state_path = tmp_path / "weights.pt"
    save_checkpoint(model, state_path)

    payload = _payload(state_path, arch, n_sims=4)
    game = _play_game_subprocess(payload)

    assert game.outcome in {"white", "black", "draw"}
    # Records are populated whenever the game advanced past ply 0.
    # (Even immediate-end games would still produce one record.)
    assert game.records is not None


def test_subprocess_worker_with_inference_server(tmp_path: Path):
    """Same smoke test but exercising the InferenceServer path (mcts_batch_size>1)
    inside the worker. The server runs in-process within the worker."""
    torch.manual_seed(0)
    arch = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}
    model = ChesskersScorer(**arch)
    state_path = tmp_path / "weights.pt"
    save_checkpoint(model, state_path)

    payload = _payload(state_path, arch, n_sims=4, mcts_batch_size=4, vloss_batch=2)
    game = _play_game_subprocess(payload)

    assert game.outcome in {"white", "black", "draw"}


def test_subprocess_worker_seeds_are_deterministic(tmp_path: Path):
    """Same payload → same game outcome. Different seed → may differ.
    This guards against accidental nondeterminism in worker init."""
    torch.manual_seed(0)
    arch = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}
    model = ChesskersScorer(**arch)
    state_path = tmp_path / "weights.pt"
    save_checkpoint(model, state_path)

    p1 = _payload(state_path, arch, n_sims=8, seed=42, temperature=0.0)
    g1 = _play_game_subprocess(p1)
    g2 = _play_game_subprocess(p1)
    # Same seed + temperature=0 (greedy) → same number of records.
    # (Full move-by-move equality would require deterministic torch ops.)
    assert len(g1.records) == len(g2.records), \
        f"same seed+greedy should yield same length: {len(g1.records)} vs {len(g2.records)}"


def test_eval_game_subprocess_random_vs_random(tmp_path: Path):
    """Both sides random → outcome is decisive (no model to load)."""
    arch = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}
    payload = {
        "device": "cpu",
        "model_arch": arch,
        "n_sims": 4,
        "white_model_path": None,
        "black_model_path": None,
    }
    outcome = _eval_game_subprocess(payload)
    assert outcome in {"white", "black", "draw"}


def test_eval_game_subprocess_model_vs_random(tmp_path: Path):
    """Model as white vs random as black runs and returns outcome."""
    arch = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}
    model = ChesskersScorer(**arch)
    state_path = tmp_path / "weights.pt"
    save_checkpoint(model, state_path)
    payload = {
        "device": "cpu",
        "model_arch": arch,
        "n_sims": 4,
        "white_model_path": str(state_path),
        "black_model_path": None,
    }
    outcome = _eval_game_subprocess(payload)
    assert outcome in {"white", "black", "draw"}


def test_run_eval_parallel_returns_correct_count(tmp_path: Path):
    """The parallel pool path should run all N games and return W+B+D == N."""
    arch = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}
    model = ChesskersScorer(**arch)
    state_path = tmp_path / "weights.pt"
    save_checkpoint(model, state_path)
    n_games = 6
    counts = _run_eval_parallel(
        white_model_path=str(state_path),
        black_model_path=None,
        n_games=n_games,
        n_sims=4,
        model_arch=arch,
        device="cpu",
        workers=3,
    )
    assert counts["games"] == n_games
    assert counts["white"] + counts["black"] + counts["draw"] == n_games


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only test")
def test_subprocess_worker_loads_to_cuda(tmp_path: Path):
    """If CUDA is available, worker loads model to cuda:0."""
    arch = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}
    model = ChesskersScorer(**arch)
    state_path = tmp_path / "weights.pt"
    save_checkpoint(model, state_path)
    payload = _payload(state_path, arch, device="cuda", n_sims=4)
    game = _play_game_subprocess(payload)
    assert game.outcome in {"white", "black", "draw"}
