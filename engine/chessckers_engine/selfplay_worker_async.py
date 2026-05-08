"""Async self-play worker: runs games forever, appends to ReplayBuffer.

Spawned as a subprocess by the coordinator (`selfplay_az_async`). Each
worker holds its own model copy, plays games end-to-end, and writes each
finished game to the shared file-backed buffer. Between games it
mtime-polls the trainer's weights file and hot-reloads when it changes.

Stop signal: presence of `stop_path` (a sentinel file). Coordinator
creates this file to ask all workers to wind down cleanly after their
in-flight game finishes — no SIGTERM, no half-written buffer entries.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger("chessckers_engine.selfplay_worker_async")


def _stop_requested(stop_path: Path | None) -> bool:
    return stop_path is not None and stop_path.exists()


def play_forever(payload: dict) -> int:
    """Run self-play games until stop file appears (or max_games hit).

    Returns the number of games played. Top-level so it pickles for
    spawn-context multiprocessing.

    payload keys:
      worker_id, device, model_arch, weights_path, buffer_root,
      n_sims, c_puct, temperature, dirichlet_alpha, dirichlet_eps,
      mcts_batch_size, vloss_batch, max_plies, seed, stop_path,
      max_games (None = forever), weights_poll_seconds.
    """
    import torch as _torch

    from chessckers_engine.checkpoints import load_checkpoint
    from chessckers_engine.inference_server import InferenceServer as _IS
    from chessckers_engine.model import ChesskersScorer as _Scorer
    from chessckers_engine.replay_buffer import ReplayBuffer
    from chessckers_engine.selfplay_az import (
        az_game_to_examples,
        play_az_game,
    )
    from chessckers_engine.variant_py import PyVariantClient as _PVC

    worker_id = int(payload["worker_id"])
    weights_path = Path(payload["weights_path"])
    stop_path = Path(payload["stop_path"]) if payload.get("stop_path") else None
    buffer = ReplayBuffer(payload["buffer_root"])
    poll_s = float(payload.get("weights_poll_seconds", 2.0))
    max_games = payload.get("max_games")
    max_plies = int(payload.get("max_plies", 400))

    device = _torch.device(payload["device"])
    model = _Scorer(**payload["model_arch"]).to(device).eval()

    # Wait for initial weights to land before starting any games — otherwise
    # this worker's randomly-init'd net would generate worthless first games.
    while not weights_path.exists():
        if _stop_requested(stop_path):
            return 0
        time.sleep(poll_s)
    last_mtime = -1.0

    use_server = int(payload["mcts_batch_size"]) > 1
    server = _IS(model, max_batch_size=int(payload["mcts_batch_size"])) if use_server else None
    evaluator = server if server is not None else model

    client = _PVC()
    rng = _torch.Generator().manual_seed(int(payload["seed"]))
    games_played = 0

    try:
        while not _stop_requested(stop_path):
            if max_games is not None and games_played >= int(max_games):
                break
            try:
                cur_mtime = weights_path.stat().st_mtime
            except FileNotFoundError:
                time.sleep(poll_s)
                continue
            if cur_mtime > last_mtime:
                try:
                    load_checkpoint(model, weights_path)
                    model.eval()
                    last_mtime = cur_mtime
                except (EOFError, RuntimeError, OSError) as e:
                    # Trainer may be mid-write; skip and try next iteration.
                    log.debug("worker %d weight reload failed: %s", worker_id, e)
                    time.sleep(poll_s)
                    continue

            game = play_az_game(
                evaluator, client,
                n_sims=int(payload["n_sims"]),
                c_puct=float(payload["c_puct"]),
                temperature=float(payload["temperature"]),
                max_plies=max_plies,
                rng=rng,
                dirichlet_alpha=payload.get("dirichlet_alpha"),
                dirichlet_eps=float(payload.get("dirichlet_eps", 0.25)),
                vloss_batch=int(payload.get("vloss_batch", 1)),
            )
            games_played += 1
            examples = az_game_to_examples(game)
            buffer.append_game(
                worker_id=worker_id, game_id=games_played, examples=examples
            )
    finally:
        client.close()
        if server is not None:
            server.shutdown()
    return games_played
