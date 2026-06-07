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

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("chessckers_engine.selfplay_worker_async")


def _stop_requested(stop_path: Path | None) -> bool:
    return stop_path is not None and stop_path.exists()


# Self-play knobs that may be live-overridden mid-run via run_dir/selfplay.json
# (server-published, mirrored onto each box by fleet_client, or written directly by
# the operator on the trainer host). JSON key -> type. A missing file or a missing
# key means the launch-time payload value stands, so a standalone worker — and any
# box before its first server pull — behaves exactly as before.
_LIVE_KEYS = {
    "sims": int, "c_puct": float, "temperature": float,
    "dirichlet_alpha": float, "dirichlet_eps": float, "max_plies": int,
}


def _read_live_overrides(params_path: Path | None, cache: dict) -> dict:
    """Current override dict from `params_path`, re-parsed only when the file's
    mtime changes (cache holds the last mtime + parsed values). A missing file or a
    malformed/partial write keeps the last good values, so a half-written update
    landing mid-game never crashes a worker — it just applies one game later."""
    if params_path is None:
        return cache.get("vals", {})
    try:
        mtime = params_path.stat().st_mtime
    except OSError:
        return cache.get("vals", {})
    if mtime != cache.get("mtime"):
        try:
            raw = json.loads(params_path.read_text())
            if isinstance(raw, dict):
                cache["vals"] = {k: conv(raw[k]) for k, conv in _LIVE_KEYS.items()
                                 if k in raw}
                cache["mtime"] = mtime
        except (OSError, ValueError, TypeError):
            pass  # keep last good
    return cache.get("vals", {})


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
    # Live self-play params: re-read run_dir/selfplay.json each game (mtime-gated)
    # so a server-published change anneals this worker at the next game boundary —
    # no restart, no lost in-flight game. `live_cache` carries the parsed values.
    params_path = Path(payload["params_path"]) if payload.get("params_path") else None
    live_cache: dict = {}

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
        from chessckers_engine.model import build_model as _build_model

        weights_path = Path(payload["weights_path"])
        poll_s = float(payload.get("weights_poll_seconds", 2.0))
        device = _torch.device(payload["device"])
        model = _build_model(**payload["model_arch"]).to(device).eval()
        # Wait for initial weights to land — otherwise we'd self-play with random init.
        while not weights_path.exists():
            if _stop_requested(stop_path):
                return 0
            time.sleep(poll_s)
        last_mtime = -1.0
        # Native C++ search mode: drive play_az_game through cpp.run_mcts_native
        # (~4.8x). The PyTorch model is loaded only to produce the canonical
        # state_dict, which is exported to a flat .bin the C++ net loads/reloads.
        use_native = bool(payload.get("use_native", False))
        native_search = None
        net_box = [None]
        native_bin = None
        if use_native:
            import chessckers_cpp  # noqa: F401  — fail fast if the extension is missing
            from chessckers_engine.native_search import make_native_search_fn
            native_bin = run_dir / f"native_{worker_id}.bin"
            native_search = make_native_search_fn(net_box)
        use_in_proc_server = int(payload["mcts_batch_size"]) > 1 and not use_native
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
                        if use_native:
                            import chessckers_cpp
                            from chessckers_engine.native_net import export_state_dict
                            _tmp = str(native_bin) + ".tmp"
                            export_state_dict(model.state_dict(), _tmp)
                            Path(_tmp).replace(native_bin)
                            net_box[0] = chessckers_cpp.ChesskersNet(str(native_bin))
                        last_mtime = cur_mtime
                    except (EOFError, RuntimeError, OSError) as e:
                        log.debug("worker %d weight reload failed: %s", worker_id, e)
                        time.sleep(float(payload.get("weights_poll_seconds", 2.0)))
                        continue

            ov = _read_live_overrides(params_path, live_cache)
            game = play_az_game(
                evaluator, client,
                n_sims=int(ov.get("sims", payload["n_sims"])),
                c_puct=float(ov.get("c_puct", payload["c_puct"])),
                temperature=float(ov.get("temperature", payload["temperature"])),
                max_plies=int(ov.get("max_plies", max_plies)),
                rng=rng,
                dirichlet_alpha=ov.get("dirichlet_alpha", payload.get("dirichlet_alpha")),
                dirichlet_eps=float(ov.get("dirichlet_eps", payload.get("dirichlet_eps", 0.25))),
                vloss_batch=int(payload.get("vloss_batch", 1)),
                search_fn=native_search if not use_shared else None,
                resign_threshold=float(payload.get("resign_threshold", 0.0)),
                resign_no_resign_frac=float(payload.get("resign_no_resign_frac", 0.1)),
                resign_consecutive=int(payload.get("resign_consecutive", 2)),
                resign_min_ply=int(payload.get("resign_min_ply", 8)),
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


def _claim_job(jobs_dir: Path, worker_id: int) -> tuple | None:
    """Atomically claim one queued job for this worker (the lc0 client-drives-each-game model).

    fleet_client mints jobs as `<seq>.json` in run_dir/jobs/ (one per POST /next_game). A worker
    claims one by renaming it to `<seq>.json.c<wid>` — POSIX rename is atomic, so of N workers
    racing for the same file exactly one wins and the losers fall through to the next candidate.
    (Claimed files end in `.c<wid>`, not `.json`, so they drop out of the unclaimed glob the
    client tops up against.) Returns (seq, claimed_path, job_dict), or None when nothing is
    claimable right now. A malformed file is claimed-then-dropped so it can't wedge the queue."""
    try:
        candidates = sorted(jobs_dir.glob("*.json"))
    except OSError:
        return None
    for src in candidates:
        dst = src.with_name(src.name + f".c{worker_id}")
        try:
            os.rename(src, dst)            # atomic; the losing races get FileNotFoundError
        except OSError:
            continue                        # another worker took it (or it vanished) — next one
        try:
            job = json.loads(dst.read_text())
        except (OSError, ValueError):
            try:
                dst.unlink()
            except OSError:
                pass
            continue
        return src.stem, dst, job
    return None


def play_jobs_forever(payload: dict) -> int:
    """Job-driven self-play / gate executor — the lc0 *client-drives-each-game* model.

    Replaces the autonomous `play_forever` loop for FLEET workers: instead of self-playing
    continuously, the worker claims ONE server-assigned job at a time from run_dir/jobs/
    (minted by fleet_client via POST /next_game) and plays exactly that —
      • train -> one self-play game with the job's params, appended to buffer/ (the client
                 uploads it via /upload_game, exactly as the autonomous path did);
      • match -> one keep-best gate game between the two nets the client already fetched by
                 content address (paths carried in the job), the outcome written to
                 run_dir/match_out/ for the client to POST to /match_result.
    Per-worker inference only (a job-driven worker holds its own model + optional native search).
    The net setup + hot-reload below MIRRORS play_forever's per-worker branch — keep the two in
    sync (the native .bin export is training-critical). Returns the count of self-play games."""
    import torch as _torch

    from chessckers_engine import heartbeat as _hb
    from chessckers_engine.checkpoints import load_checkpoint
    from chessckers_engine.inference_server import InferenceServer as _IS
    from chessckers_engine.model import build_model as _build_model
    from chessckers_engine.replay_buffer import ReplayBuffer
    from chessckers_engine.selfplay_az import az_game_to_examples, play_az_game
    from chessckers_engine.variant_py import PyVariantClient as _PVC

    worker_id = int(payload["worker_id"])
    if payload.get("pin_cpu") is not None:
        _pin_to_cpu(int(payload["pin_cpu"]))
    stop_path = Path(payload["stop_path"]) if payload.get("stop_path") else None
    buffer = ReplayBuffer(payload["buffer_root"])
    run_dir = Path(payload["buffer_root"]).parent
    jobs_dir = run_dir / "jobs"
    match_out = run_dir / "match_out"
    gate_dir = run_dir / "_gate"
    machine = str(payload.get("machine", "unknown"))
    max_plies_default = int(payload.get("max_plies", 400))
    incarnation_id = time.time()

    # ---- per-worker inference net (mirror of play_forever's per-worker branch; keep in sync) ----
    weights_path = Path(payload["weights_path"])
    poll_s = float(payload.get("weights_poll_seconds", 2.0))
    device = _torch.device(payload["device"])
    model = _build_model(**payload["model_arch"]).to(device).eval()
    use_native = bool(payload.get("use_native", False))
    native_search = None
    net_box = [None]
    native_bin = None
    if use_native:
        import chessckers_cpp  # noqa: F401 — fail fast if the ext is missing
        from chessckers_engine.native_search import make_native_search_fn
        native_bin = run_dir / f"native_{worker_id}.bin"
        native_search = make_native_search_fn(net_box)
    use_in_proc_server = int(payload["mcts_batch_size"]) > 1 and not use_native
    server = _IS(model, max_batch_size=int(payload["mcts_batch_size"])) if use_in_proc_server else None
    evaluator = server if server is not None else model
    last_mtime = -1.0
    loaded = False

    client = _PVC()
    rng = _torch.Generator().manual_seed(int(payload["seed"]))
    match_runner = None  # lazily built on the first match job (heavy: torch + the native ext)
    games_played = 0

    try:
        _hb.write(run_dir, machine=machine, worker_id=worker_id, role="worker",
                  games_played=0, incarnation_id=incarnation_id)
    except OSError as e:
        log.warning("worker %d initial heartbeat failed: %s", worker_id, e)

    # Never play on a random-init net: wait for the client's first weights.pt.
    while not weights_path.exists():
        if _stop_requested(stop_path):
            return 0
        time.sleep(poll_s)

    try:
        while not _stop_requested(stop_path):
            # Hot-reload the net when the client syncs a fresher weights.pt (mtime poll). Same
            # native re-export as play_forever; a mid-write reload just skips this tick.
            try:
                cur_mtime = weights_path.stat().st_mtime
            except FileNotFoundError:
                time.sleep(poll_s)
                continue
            if cur_mtime > last_mtime:
                try:
                    load_checkpoint(model, weights_path)
                    model.eval()
                    if use_native:
                        import chessckers_cpp
                        from chessckers_engine.native_net import export_state_dict
                        _tmp = str(native_bin) + ".tmp"
                        export_state_dict(model.state_dict(), _tmp)
                        Path(_tmp).replace(native_bin)
                        net_box[0] = chessckers_cpp.ChesskersNet(str(native_bin))
                    last_mtime = cur_mtime
                    loaded = True
                except (EOFError, RuntimeError, OSError) as e:
                    log.debug("worker %d weight reload failed: %s", worker_id, e)
                    time.sleep(poll_s)
                    continue
            if not loaded:
                time.sleep(poll_s)
                continue

            claim = _claim_job(jobs_dir, worker_id)
            if claim is None:
                time.sleep(0.2)   # queue empty — the client mints more on its poll tick
                continue
            seq, claimed_path, job = claim
            jtype = job.get("type", "train")
            try:
                if jtype == "match":
                    if match_runner is None:
                        from chessckers_engine.fleet_match import MatchRunner
                        match_runner = MatchRunner(gate_dir, device=payload["device"])
                    outcome = match_runner.play(job, Path(job["cand_path"]), Path(job["opp_path"]))
                    match_out.mkdir(parents=True, exist_ok=True)
                    _tmp = match_out / f"{seq}.json.tmp"
                    _tmp.write_text(json.dumps({
                        "match_id": job["match_id"], "seed": job["seed"],
                        "opp": job["opponent"], "cand_white": job["cand_white"],
                        "outcome": outcome,
                    }))
                    _tmp.replace(match_out / f"{seq}.json")  # atomic: the client never reads a half-file
                    log.info("worker %d gate game: match %s opp=%s cand_white=%s -> %s",
                             worker_id, job["match_id"], job["opponent"], job["cand_white"], outcome)
                else:  # train (default)
                    ov = job.get("params") or {}
                    games_played += 1
                    game = play_az_game(
                        evaluator, client,
                        n_sims=int(ov.get("sims", payload["n_sims"])),
                        c_puct=float(ov.get("c_puct", payload["c_puct"])),
                        temperature=float(ov.get("temperature", payload["temperature"])),
                        max_plies=int(ov.get("max_plies", max_plies_default)),
                        rng=rng,
                        dirichlet_alpha=ov.get("dirichlet_alpha", payload.get("dirichlet_alpha")),
                        dirichlet_eps=float(ov.get("dirichlet_eps", payload.get("dirichlet_eps", 0.25))),
                        vloss_batch=int(payload.get("vloss_batch", 1)),
                        search_fn=native_search,
                        resign_threshold=float(payload.get("resign_threshold", 0.0)),
                        resign_no_resign_frac=float(payload.get("resign_no_resign_frac", 0.1)),
                        resign_consecutive=int(payload.get("resign_consecutive", 2)),
                        resign_min_ply=int(payload.get("resign_min_ply", 8)),
                    )
                    examples = az_game_to_examples(game)
                    _game_path = buffer.append_game(
                        worker_id=worker_id, game_id=games_played, examples=examples)
                    try:
                        Path(str(_game_path) + ".meta").write_text(json.dumps({
                            "worker_id": worker_id, "machine": machine,
                            "outcome": game.outcome, "plies": len(game.records),
                            "seed_fen": game.records[0].fen if game.records else None,
                        }))
                    except (OSError, AttributeError, TypeError):
                        pass
            except Exception as e:  # noqa: BLE001 — one bad job must never kill the worker
                log.warning("worker %d job %s (%s) failed: %r", worker_id, seq, jtype, e)
            finally:
                try:
                    claimed_path.unlink()  # release the slot so the client mints the next job
                except OSError:
                    pass
            try:
                _hb.write(run_dir, machine=machine, worker_id=worker_id, role="worker",
                          games_played=games_played, incarnation_id=incarnation_id)
            except OSError as e:
                log.debug("worker %d heartbeat write failed: %s", worker_id, e)
    finally:
        client.close()
        if server is not None:
            server.shutdown()
    return games_played


def play_jobs_forever_subprocess(payload: dict) -> None:
    """Subprocess entry point for the job-driven executor — same os._exit hard-exit rationale as
    play_forever_subprocess (skip interpreter shutdown so a lingering thread can't wedge join)."""
    import os as _os
    import sys as _sys
    rc = 0
    try:
        play_jobs_forever(payload)
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
