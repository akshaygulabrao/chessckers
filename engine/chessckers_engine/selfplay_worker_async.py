"""Async self-play worker: runs games forever, appends to ReplayBuffer.

Spawned as a subprocess by the coordinator (`selfplay_az_async`). Two
inference modes, picked by the payload:

  * **Shared** (`request_q`+`response_q` present): pure-CPU worker. All
    leaf evals go to the coordinator's `CrossInferenceServer`, which
    batches across all workers and shares the trainer's live model.
    No GPU model in the worker, no weights file polling.

  * **Per-worker** (no queues in payload, falls back to weights-file mode):
    each worker holds its own model copy on `device` and mtime-polls
    `weights_path` to hot-reload after trainer broadcasts. Kept for
    fallback / when the trainer's GPU is preferred to stay solo.

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


def play_forever_subprocess(payload: dict) -> None:
    """Subprocess entry point: wraps `play_forever` and forces `os._exit`.

    A bare `play_forever` returns cleanly, but the multiprocessing.Queue's
    `QueueFeederThread` is a (technically daemon) background thread that
    holds the process open until its buffered items are flushed to the
    underlying pipe. In our shutdown sequence the parent has already stopped
    reading by then, so the feeder blocks → process never exits → parent's
    `p.join()` waits the full 300s timeout. `os._exit(0)` skips Python's
    normal interpreter shutdown (which is what waits on those threads),
    sidestepping the deadlock entirely.
    """
    import os as _os
    import sys as _sys
    rc = 0
    try:
        play_forever(payload)
    except SystemExit as e:
        rc = int(e.code or 0)
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        try:
            _sys.stdout.flush()
            _sys.stderr.flush()
        except Exception:
            pass
        _os._exit(rc)


def _pin_to_cpu(cpu_id: int) -> None:
    """Pin the current process to a specific CPU core (Linux only).

    Without pinning, the kernel scheduler bounces processes between cores —
    each migration invalidates L1/L2 caches and hurts MCTS branch prediction.
    On macOS `sched_setaffinity` doesn't exist; this is a silent no-op there.
    """
    import os as _os
    if hasattr(_os, "sched_setaffinity"):
        try:
            _os.sched_setaffinity(0, {cpu_id})
        except OSError as e:
            log.debug("sched_setaffinity(cpu=%d) failed: %s", cpu_id, e)


def play_forever(payload: dict) -> int:
    """Run self-play games until stop file appears (or max_games hit).

    Returns the number of games played. Top-level so it pickles for
    spawn-context multiprocessing.

    payload keys (common):
      worker_id, buffer_root, n_sims, c_puct, temperature,
      dirichlet_alpha, dirichlet_eps, vloss_batch, max_plies, seed,
      stop_path, max_games (None = forever).
    payload keys (shared-inference mode):
      request_q, response_q.
    payload keys (per-worker mode):
      device, model_arch, weights_path, mcts_batch_size,
      weights_poll_seconds.
    """
    import torch as _torch

    from chessckers_engine.replay_buffer import ReplayBuffer
    from chessckers_engine.selfplay_az import (
        az_game_to_examples,
        play_az_game,
    )
    from chessckers_engine.variant_py import PyVariantClient as _PVC

    worker_id = int(payload["worker_id"])
    if "pin_cpu" in payload and payload["pin_cpu"] is not None:
        _pin_to_cpu(int(payload["pin_cpu"]))
    stop_path = Path(payload["stop_path"]) if payload.get("stop_path") else None
    buffer = ReplayBuffer(payload["buffer_root"])
    max_games = payload.get("max_games")
    max_plies = int(payload.get("max_plies", 400))

    # Heartbeat setup. run_dir is derived from buffer_root's parent;
    # heartbeats land at run_dir/heartbeats/<machine>_<worker_id>.json.
    # `machine` is a string tag passed by the launcher (local / leena /
    # vast / ...); defaults to "unknown" if not set so heartbeats still
    # work for legacy callers.
    from chessckers_engine import heartbeat as _hb
    machine = str(payload.get("machine", "unknown"))
    run_dir = Path(payload["buffer_root"]).parent
    incarnation_id = time.time()
    # Emit a 0-games heartbeat at startup so the coord knows we're alive
    # even before the first game finishes.
    try:
        _hb.write(run_dir, machine=machine, worker_id=worker_id,
                  role="worker", games_played=0, incarnation_id=incarnation_id)
    except OSError as e:
        log.warning("worker %d initial heartbeat failed: %s", worker_id, e)

    use_shared = "request_q" in payload and "response_q" in payload

    # ---- Inference setup ----
    server = None
    if use_shared:
        from chessckers_engine.cross_inference import CrossInferenceClient
        evaluator = CrossInferenceClient(
            worker_id=worker_id,
            request_q=payload["request_q"],
            response_q=payload["response_q"],
            q_index=payload.get("q_index", worker_id),
        )
        last_mtime = None  # unused
        weights_path = None
    else:
        from chessckers_engine.checkpoints import load_checkpoint
        from chessckers_engine.inference_server import InferenceServer as _IS
        from chessckers_engine.model import ChesskersScorer as _Scorer

        weights_path = Path(payload["weights_path"])
        poll_s = float(payload.get("weights_poll_seconds", 2.0))
        device = _torch.device(payload["device"])
        model = _Scorer(**payload["model_arch"]).to(device).eval()
        # Wait for initial weights to land — otherwise we'd self-play with random init.
        while not weights_path.exists():
            if _stop_requested(stop_path):
                return 0
            time.sleep(poll_s)
        last_mtime = -1.0
        use_in_proc_server = int(payload["mcts_batch_size"]) > 1
        if use_in_proc_server:
            server = _IS(model, max_batch_size=int(payload["mcts_batch_size"]))
            evaluator = server
        else:
            evaluator = model

    client = _PVC()
    rng = _torch.Generator().manual_seed(int(payload["seed"]))
    games_played = 0

    try:
        while not _stop_requested(stop_path):
            if max_games is not None and games_played >= int(max_games):
                break

            # Per-worker mode: refresh local model from weights file when newer.
            if not use_shared:
                try:
                    cur_mtime = weights_path.stat().st_mtime
                except FileNotFoundError:
                    time.sleep(float(payload.get("weights_poll_seconds", 2.0)))
                    continue
                if cur_mtime > last_mtime:
                    try:
                        load_checkpoint(model, weights_path)
                        model.eval()
                        last_mtime = cur_mtime
                    except (EOFError, RuntimeError, OSError) as e:
                        log.debug("worker %d weight reload failed: %s", worker_id, e)
                        time.sleep(float(payload.get("weights_poll_seconds", 2.0)))
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
            _game_path = buffer.append_game(
                worker_id=worker_id, game_id=games_played, examples=examples
            )
            # Sidecar metadata so the trainer can fold this game into its
            # per-machine + per-seed + W/B/D dashboards (examples alone carry no
            # outcome/machine). Synced alongside the pkl; consumed on ingest.
            try:
                import json as _json
                Path(str(_game_path) + ".meta").write_text(_json.dumps({
                    "worker_id": worker_id, "machine": machine,
                    "outcome": game.outcome, "plies": len(game.records),
                    "seed_fen": game.records[0].fen if game.records else None,
                }))
            except (OSError, AttributeError, TypeError):
                pass
            try:
                _hb.write(run_dir, machine=machine, worker_id=worker_id,
                          role="worker", games_played=games_played,
                          incarnation_id=incarnation_id)
            except OSError as e:
                log.debug("worker %d heartbeat write failed: %s", worker_id, e)
    finally:
        client.close()
        if server is not None:
            server.shutdown()
        # Multiprocessing-Queue cleanup: by default each producer process
        # has a feeder thread that blocks the process from terminating until
        # buffered items are flushed to the underlying pipe. Tell the queue
        # NOT to join its feeder thread — any in-flight final put may be
        # lost, which is fine since we're shutting down. Combined with the
        # `os._exit` in `play_forever_subprocess`, this lets the worker
        # process die promptly so the coordinator's `is_alive()` poll can
        # detect it.
        if use_shared:
            try:
                payload["request_q"].cancel_join_thread()
                payload["response_q"].cancel_join_thread()
            except Exception:
                pass
    return games_played
