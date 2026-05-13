"""Async AlphaZero coordinator: workers + trainer + eval, all concurrent.

Architecture:
  - N self-play workers (subprocesses, spawn context). Each owns a model
    copy on `device`, plays games forever, hot-reloads weights from
    `weights.pt` between games, appends each finished game to the
    file-backed ReplayBuffer.
  - 1 trainer (thread in the coordinator process). Samples uniformly
    from the buffer, runs dual-loss SGD, atomically rewrites
    `weights.pt` every `weight_save_every` steps, dumps a durable
    checkpoint every `checkpoint_every` steps.
  - Coordinator main thread: spawns/supervises everything, runs
    periodic eval-vs-random, watches the wall-clock deadline, traps
    SIGINT/SIGTERM by tripping a sentinel file that all components
    poll between safe points.

No iter boundaries, no gating — training runs continuously; the latest
weights are always live to workers.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
# torch.multiprocessing is a drop-in replacement for multiprocessing that adds
# zero-copy shared-memory pickling for tensors. mp.Queue with torch tensors
# would otherwise serialize the bytes through a Unix socket per request.
import torch.multiprocessing as mp

from chessckers_engine.checkpoints import DEFAULT_WEIGHTS_DIR
from chessckers_engine.cross_inference import CrossInferenceServer
from chessckers_engine.device import pick_device
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.replay_buffer import ReplayBuffer
from chessckers_engine.selfplay_az_loop import _run_eval_parallel
from chessckers_engine.selfplay_worker_async import play_forever, play_forever_subprocess
from chessckers_engine.trainer_loop import TrainerLoop

log = logging.getLogger("chessckers_engine.selfplay_az_async")


def _log_eval_to_wandb(wandb_run, opponent_label: str,
                       vs_w: dict, vs_b: dict, trainer_step: int) -> None:
    """Log per-opponent/side eval rates to W&B as scalar series.

    Each opponent gets four series per side: win/loss/draw rate + total games.
    These plot directly in the W&B run dashboard as eval/<opponent>/as_white/...
    keyed by trainer_step on the x-axis."""
    if wandb_run is None:
        return
    metrics: dict[str, float] = {}
    for side_label, result in (("as_white", vs_w), ("as_black", vs_b)):
        games = max(int(result.get("games", 0)), 1)
        # 'white'/'black' counts the *color* that won; our snapshot played the
        # color matching side_label, so snapshot_wins is that color's count.
        snapshot_color = "white" if side_label == "as_white" else "black"
        opponent_color = "black" if side_label == "as_white" else "white"
        prefix = f"eval/{opponent_label}/{side_label}"
        metrics[f"{prefix}/win_rate"] = result.get(snapshot_color, 0) / games
        metrics[f"{prefix}/loss_rate"] = result.get(opponent_color, 0) / games
        metrics[f"{prefix}/draw_rate"] = result.get("draw", 0) / games
        metrics[f"{prefix}/games"] = games
    wandb_run.log(metrics, step=trainer_step)


def _count_games_since(buffer_root: Path, since_mtime: float) -> int:
    """Count *.pkl files in buffer/ with mtime >= since_mtime.

    One .pkl == one finished game. The mtime gate excludes games rsync'd in
    from remote workers' historical buffers (cloud_sync_sidecar uses
    `rsync -a` which preserves source mtime, so stale leftovers come in
    with old timestamps and are skipped). Pruning could in principle drop
    a counted file before the next poll — but the trainer's mtime-based
    eviction only triggers at buffer overflow (>buffer_max_games=4000),
    which is far above any --run-games target we set on a single run."""
    if not buffer_root.exists():
        return 0
    n = 0
    for p in buffer_root.glob("*.pkl"):
        try:
            if p.stat().st_mtime >= since_mtime:
                n += 1
        except OSError:
            continue
    return n


def _build_worker_payload(worker_id: int, *, buffer_root: Path,
                          stop_path: Path, n_sims: int, c_puct: float,
                          temperature: float, dirichlet_alpha: float,
                          dirichlet_eps: float, vloss_batch: int,
                          max_plies: int, seed: int,
                          # Per-worker mode args (omit/None when shared):
                          weights_path: Optional[Path] = None,
                          model_arch: Optional[dict] = None,
                          device: Optional[str] = None,
                          mcts_batch_size: int = 1,
                          # Shared-inference mode args (set both or neither):
                          request_q=None, response_q=None,
                          # Linux-only CPU pin:
                          pin_cpu: Optional[int] = None) -> dict:
    payload: dict = {
        "worker_id": worker_id,
        "buffer_root": str(buffer_root),
        "n_sims": n_sims,
        "c_puct": c_puct,
        "temperature": temperature,
        "dirichlet_alpha": dirichlet_alpha if dirichlet_alpha > 0 else None,
        "dirichlet_eps": dirichlet_eps,
        "vloss_batch": vloss_batch,
        "max_plies": max_plies,
        "seed": seed + worker_id,
        "stop_path": str(stop_path),
        "max_games": None,
        "pin_cpu": pin_cpu,
    }
    if request_q is not None and response_q is not None:
        payload["request_q"] = request_q
        payload["response_q"] = response_q
        # In selfplay_az_async we use worker_id == local index w (no
        # --worker-id-base offset), so q_index defaults to worker_id and
        # this is harmless. Set it explicitly for documentation parity
        # with selfplay_workers_only.py.
        payload["q_index"] = worker_id
    else:
        # Per-worker mode requires its own GPU model + weights mtime poll.
        assert weights_path is not None and model_arch is not None and device is not None
        payload["weights_path"] = str(weights_path)
        payload["model_arch"] = model_arch
        payload["device"] = device
        payload["mcts_batch_size"] = mcts_batch_size
        payload["weights_poll_seconds"] = 5.0
    return payload


def _eval_snapshot_and_log(*, weights_path: Path, snapshot_path: Path,
                           model_arch: dict, device: str, eval_games: int,
                           eval_sims: int, eval_workers: int, eval_log_path: Path,
                           trainer_step: int,
                           eval_opponents: Optional[list[tuple[str, Path]]] = None,
                           wandb_run=None) -> dict:
    """Snapshot current weights and play `eval_games` vs random + each opponent, both sides.
    Appends one JSON line to `eval_log_path` and returns the summary.

    `eval_opponents` is a list of (label, checkpoint_path) pairs. For each, two
    additional series run: snapshot-as-white-vs-opp-as-black and vice versa.
    Results land under keys `as_white_vs_<label>` / `as_black_vs_<label>`."""
    if not weights_path.exists():
        log.info("eval skipped — weights not yet written")
        return {}
    # Snapshot so eval results aren't biased by mid-eval trainer updates.
    tmp = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    tmp.write_bytes(weights_path.read_bytes())
    os.replace(tmp, snapshot_path)
    as_white = _run_eval_parallel(
        str(snapshot_path), None, eval_games, eval_sims, model_arch, device, eval_workers,
    )
    as_black = _run_eval_parallel(
        None, str(snapshot_path), eval_games, eval_sims, model_arch, device, eval_workers,
    )
    summary: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trainer_step": trainer_step,
        "eval_games_per_side": eval_games,
        "eval_sims": eval_sims,
        "as_white_vs_random": as_white,
        "as_black_vs_random": as_black,
    }
    log.info("eval @ step %d vs random | W: %d/%d/%d | B: %d/%d/%d",
             trainer_step,
             as_white["white"], as_white["black"], as_white["draw"],
             as_black["black"], as_black["white"], as_black["draw"])
    _log_eval_to_wandb(wandb_run, "random", as_white, as_black, trainer_step)
    for label, opp_path in (eval_opponents or []):
        opp_str = str(opp_path)
        # snapshot plays white; opponent plays black
        vs_w = _run_eval_parallel(
            str(snapshot_path), opp_str, eval_games, eval_sims,
            model_arch, device, eval_workers,
        )
        # snapshot plays black; opponent plays white
        vs_b = _run_eval_parallel(
            opp_str, str(snapshot_path), eval_games, eval_sims,
            model_arch, device, eval_workers,
        )
        summary[f"as_white_vs_{label}"] = vs_w
        summary[f"as_black_vs_{label}"] = vs_b
        log.info("eval @ step %d vs %s | W: %d/%d/%d | B: %d/%d/%d",
                 trainer_step, label,
                 vs_w["white"], vs_w["black"], vs_w["draw"],
                 vs_b["black"], vs_b["white"], vs_b["draw"])
        _log_eval_to_wandb(wandb_run, label, vs_w, vs_b, trainer_step)
    with eval_log_path.open("a") as f:
        f.write(json.dumps(summary))
        f.write("\n")
    return summary


def run_async_training(
    run_dir: Path,
    *,
    model_arch: dict,
    device: str = "auto",
    n_workers: int = 4,
    n_sims: int = 800,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    dirichlet_alpha: float = 0.5,
    dirichlet_eps: float = 0.40,
    mcts_batch_size: int = 1,
    vloss_batch: int = 1,
    max_plies: int = 400,
    trainer_batch_size: int = 128,
    trainer_lr: float = 1e-3,
    weight_save_every: int = 200,
    checkpoint_every: int = 2000,
    min_buffer_games: int = 20,
    buffer_max_games: int = 4000,
    grad_clip: float = 1.0,
    value_loss_weight: float = 1.0,
    eval_every_seconds: float = 1800.0,
    eval_games: int = 20,
    eval_sims: int = 200,
    eval_workers: int = 4,
    run_seconds: float = 24 * 3600,
    run_games: Optional[int] = None,
    seed: int = 0,
    base_weights: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    eval_opponents: Optional[list[tuple[str, Path]]] = None,
    wandb_project: Optional[str] = "chessckers",
    wandb_run_id: Optional[str] = None,
    wandb_mode: str = "online",
    main_loop_poll_seconds: float = 5.0,
    shared_inference: bool = False,
    shared_max_batch_size: int = 64,
    shared_timeout_ms: float = 5.0,
    pin_cpus: bool = False,
) -> dict:
    """Run async training for `run_seconds`, then shut down cleanly.

    Returns a summary dict with final stats. All artifacts live under `run_dir`.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"
    eval_snapshot_path = run_dir / "weights_eval_snapshot.pt"
    buffer_root = run_dir / "buffer"
    checkpoint_dir = run_dir / "checkpoints"
    eval_log_path = run_dir / "eval.jsonl"
    stop_path = run_dir / "STOP"
    if stop_path.exists():
        stop_path.unlink()

    if device == "auto":
        device = str(pick_device())

    # Initialise wandb run if not disabled. The run id defaults to the
    # run_dir basename (e.g. "local-006") so repeated invocations against
    # the same dir resume the same W&B run — wandb owns the canonical
    # "where did I leave off" answer instead of fragile filesystem scans.
    wandb_run = None
    if wandb_mode not in ("disabled", "off", "none"):
        try:
            import wandb as _wandb
            wandb_run = _wandb.init(
                project=wandb_project or "chessckers",
                id=wandb_run_id or run_dir.name,
                resume="allow",
                mode=wandb_mode,
                dir=str(run_dir),
                config={
                    "n_workers": n_workers, "n_sims": n_sims, "c_puct": c_puct,
                    "temperature": temperature,
                    "dirichlet_alpha": dirichlet_alpha, "dirichlet_eps": dirichlet_eps,
                    "mcts_batch_size": mcts_batch_size, "vloss_batch": vloss_batch,
                    "max_plies": max_plies, "trainer_batch_size": trainer_batch_size,
                    "trainer_lr": trainer_lr, "weight_save_every": weight_save_every,
                    "checkpoint_every": checkpoint_every,
                    "buffer_max_games": buffer_max_games, "grad_clip": grad_clip,
                    "value_loss_weight": value_loss_weight,
                    "eval_every_seconds": eval_every_seconds,
                    "eval_games": eval_games, "eval_sims": eval_sims,
                    "eval_workers": eval_workers, "run_seconds": run_seconds,
                    "run_games": run_games, "seed": seed,
                    "base_weights": str(base_weights) if base_weights else None,
                    "resume_from": str(resume_from) if resume_from else None,
                    "device": device, "model_arch": model_arch,
                    "eval_opponents": [
                        {"label": lbl, "path": str(p)}
                        for lbl, p in (eval_opponents or [])
                    ],
                },
            )
            log.info("wandb run: %s (id=%s, mode=%s)",
                     wandb_run.url if wandb_run else "n/a",
                     wandb_run.id if wandb_run else "n/a", wandb_mode)
        except Exception as e:
            log.warning("wandb init failed (%s); continuing without it", e)
            wandb_run = None

    # Initialize model on `device` and broadcast initial weights so workers
    # don't sit idle waiting for the trainer's first save (which only fires
    # after min_buffer_games).
    torch.manual_seed(seed)
    model = ChesskersScorer(**model_arch).to(device)
    if base_weights is not None:
        from chessckers_engine.checkpoints import load_checkpoint
        load_checkpoint(model, base_weights)
    elif resume_from is not None and Path(resume_from).exists():
        # Resume implies workers should also play with the resumed model, not
        # random weights. Otherwise self-play games for the first ~weight_save_every
        # steps are random — and any cloud sync sidecar will rsync those random
        # weights up to remote workers immediately.
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd)
        log.info("seeded worker weights from resume checkpoint %s", resume_from)
    torch.save(model.state_dict(), weights_path)
    log.info("seeded weights at %s on device=%s", weights_path, device)

    # Stop file as the universal shutdown signal.
    def _on_signal(signum, _frame):
        log.warning("signal %d received — tripping stop file", signum)
        try:
            stop_path.touch()
        except OSError:
            pass

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    ctx = mp.get_context("spawn")

    # Optional shared-inference path: one server thread on the trainer's GPU,
    # batching leaves from all workers. Workers become CPU-only (no model copy).
    cross_server: Optional[CrossInferenceServer] = None
    request_q = None
    response_qs: list = []
    if shared_inference:
        request_q = ctx.Queue()
        response_qs = [ctx.Queue() for _ in range(n_workers)]
        cross_server = CrossInferenceServer(
            model=model, request_q=request_q, response_qs=response_qs,
            max_batch_size=shared_max_batch_size, timeout_ms=shared_timeout_ms,
            log_every=50,
        )
        cross_server.start()
        log.info("shared inference server started (max_batch=%d, timeout=%.1fms)",
                 shared_max_batch_size, shared_timeout_ms)

    # Pin workers to specific cores when requested. We reserve core 0 for
    # the coordinator (trainer thread + inference server thread + main loop)
    # and walk workers across cores 1..N. On macOS (no sched_setaffinity)
    # this is a silent no-op inside the worker.
    n_cpus = (os.cpu_count() or 0) if hasattr(os, "sched_setaffinity") else 0
    cpu_assignments: list[Optional[int]]
    if pin_cpus and n_cpus >= 2:
        cpu_assignments = [1 + (w % (n_cpus - 1)) for w in range(n_workers)]
        log.info("pinning %d workers across cores 1..%d", n_workers, n_cpus - 1)
    else:
        cpu_assignments = [None] * n_workers

    # Spawn workers (subprocesses).
    workers: list[mp.Process] = []
    for w in range(n_workers):
        payload = _build_worker_payload(
            w, buffer_root=buffer_root, stop_path=stop_path,
            n_sims=n_sims, c_puct=c_puct, temperature=temperature,
            dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
            vloss_batch=vloss_batch, max_plies=max_plies, seed=seed,
            weights_path=(None if shared_inference else weights_path),
            model_arch=(None if shared_inference else model_arch),
            device=(None if shared_inference else device),
            mcts_batch_size=mcts_batch_size,
            request_q=(request_q if shared_inference else None),
            response_q=(response_qs[w] if shared_inference else None),
            pin_cpu=cpu_assignments[w],
        )
        p = ctx.Process(target=play_forever_subprocess, args=(payload,), name=f"worker-{w}")
        p.start()
        workers.append(p)
    mode = "shared-inference" if shared_inference else "per-worker"
    log.info("spawned %d self-play workers (%s mode)", n_workers, mode)

    # Trainer in a thread (shares the model object; eval uses the snapshot).
    buffer = ReplayBuffer(buffer_root, max_games=buffer_max_games)
    trainer_stop = threading.Event()
    trainer = TrainerLoop(
        model=model, buffer=buffer,
        weights_path=weights_path, checkpoint_dir=checkpoint_dir, device=device,
        batch_size=trainer_batch_size, lr=trainer_lr,
        weight_save_every=weight_save_every, checkpoint_every=checkpoint_every,
        min_buffer_games=min_buffer_games, value_loss_weight=value_loss_weight,
        grad_clip=grad_clip, log_every=50, stop_event=trainer_stop,
        resume_from=resume_from, wandb_run=wandb_run,
    )
    trainer_thread = threading.Thread(target=trainer.run, name="trainer")
    trainer_thread.start()

    start = time.perf_counter()
    # Wall-clock timestamp for gating the games counter — any .pkl file with
    # mtime >= run_start_mtime was produced during this run; stale leftovers
    # rsync'd from remote workers' previous-run buffers don't count.
    run_start_mtime = time.time()
    last_eval = 0.0  # first eval fires after eval_every_seconds wall-clock
    last_games_log = 0
    try:
        while not stop_path.exists() and (time.perf_counter() - start) < run_seconds:
            elapsed = time.perf_counter() - start
            if run_games is not None:
                games_done = _count_games_since(buffer_root, run_start_mtime)
                if games_done - last_games_log >= 100:
                    log.info("games progress: %d / %d", games_done, run_games)
                    last_games_log = games_done
                    if wandb_run is not None:
                        wandb_run.log({
                            "games/done": games_done,
                            "games/target": run_games,
                            "games/elapsed_min": elapsed / 60.0,
                            "games/per_min": games_done / max(elapsed / 60.0, 1e-9),
                        }, step=trainer.step)
                if games_done >= run_games:
                    log.info("game target hit: %d >= %d — tripping stop file", games_done, run_games)
                    stop_path.touch()
                    break
            if (elapsed - last_eval) >= eval_every_seconds and trainer.step > 0:
                _eval_snapshot_and_log(
                    weights_path=weights_path, snapshot_path=eval_snapshot_path,
                    model_arch=model_arch, device=device,
                    eval_games=eval_games, eval_sims=eval_sims,
                    eval_workers=eval_workers, eval_log_path=eval_log_path,
                    trainer_step=trainer.step,
                    eval_opponents=eval_opponents,
                    wandb_run=wandb_run,
                )
                last_eval = elapsed
            time.sleep(main_loop_poll_seconds)
    finally:
        # Wind everything down cleanly. Order matters: stop workers first
        # so they finish in-flight games, then stop trainer.
        if not stop_path.exists():
            stop_path.touch()
        log.info("shutting down — waiting for workers to finish their games")
        # Don't use Process.join() — when workers use mp.Queue and exit via
        # os._exit, join() can hang indefinitely on macOS waiting for state
        # that never arrives. Poll is_alive() instead, with deadline +
        # escalation to terminate/kill.
        deadline = time.time() + 300.0
        for p in workers:
            while p.is_alive() and time.time() < deadline:
                time.sleep(0.1)
            if p.is_alive():
                log.warning("worker %s still alive after deadline; terminating", p.name)
                p.terminate()
                grace_until = time.time() + 5.0
                while p.is_alive() and time.time() < grace_until:
                    time.sleep(0.1)
                if p.is_alive():
                    log.warning("worker %s still alive after terminate; killing", p.name)
                    p.kill()
        # Workers are gone — no more requests can land. Safe to stop the
        # shared inference server (otherwise blocked clients would wedge).
        if cross_server is not None:
            stats = cross_server.stats()
            log.info(
                "x-inference summary: batches=%d reqs=%d avg_bs=%.2f max_bs=%d gpu_secs=%.2f",
                stats["n_batches"], stats["n_requests"], stats["avg_batch_size"],
                stats["max_batch_size_seen"], stats["inference_secs"],
            )
            cross_server.shutdown()
        trainer_stop.set()
        trainer_thread.join(timeout=120)

    summary = {
        "run_dir": str(run_dir),
        "trainer_steps": trainer.step,
        "buffer_games": buffer.count_games(),
        "buffer_examples": buffer.count_examples(),
        "wall_seconds": time.perf_counter() - start,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done: %s", summary)
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    return summary


def main() -> int:
    from chessckers_engine.runtime import setup_logging
    setup_logging()
    p = argparse.ArgumentParser(description="Async AlphaZero training (workers + trainer concurrent).")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--dirichlet-eps", type=float, default=0.40)
    p.add_argument("--mcts-batch-size", type=int, default=1)
    p.add_argument("--vloss-batch", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--trainer-batch-size", type=int, default=128)
    p.add_argument("--trainer-lr", type=float, default=1e-3)
    p.add_argument("--weight-save-every", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--min-buffer-games", type=int, default=20)
    p.add_argument("--buffer-max-games", type=int, default=4000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--value-loss-weight", type=float, default=1.0)
    p.add_argument("--eval-every-seconds", type=float, default=1800.0)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-sims", type=int, default=200)
    p.add_argument("--eval-workers", type=int, default=4)
    p.add_argument("--run-seconds", type=float, default=24 * 3600)
    p.add_argument("--run-games", type=int, default=None,
                   help="Stop after this many lifetime games across all workers. "
                        "If set, ends the run when reached. --run-seconds still "
                        "applies as a safety cap; whichever fires first wins.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", type=Path, default=None,
                   help="Optional starting weights to load before training (model only).")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Resume from a durable checkpoint dict (model + optimizer + step). "
                        "Use over --base when continuing a preempted run, so Adam moments "
                        "and step counter survive.")
    p.add_argument("--eval-opponent", action="append", default=[], metavar="LABEL=PATH",
                   help="Extra eval opponent (repeatable). Format 'label=path/to/ckpt.pt' "
                        "adds two series per cycle: as_white_vs_<label> and as_black_vs_<label>. "
                        "Bare paths (no '=') derive the label from the filename stem.")
    p.add_argument("--wandb-project", default="chessckers",
                   help="W&B project name. Set --wandb-mode disabled to skip W&B entirely.")
    p.add_argument("--wandb-run-id", default=None,
                   help="W&B run id (defaults to run_dir basename, e.g. 'local-006'). "
                        "Repeated launches against the same run_dir resume the same run.")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"],
                   help="online: stream to wandb.ai; offline: log to local dir for later sync; "
                        "disabled: no wandb at all.")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--shared-inference", action="store_true",
                   help="Single coordinator-side server batches leaves from ALL workers. "
                        "Workers run pure-CPU (no GPU model). Higher GPU util on shared GPU.")
    p.add_argument("--shared-max-batch-size", type=int, default=64,
                   help="Max batch size for the shared inference server.")
    p.add_argument("--shared-timeout-ms", type=float, default=5.0,
                   help="Max wait (ms) for additional requests to coalesce into a batch.")
    p.add_argument("--pin-cpus", action="store_true",
                   help="Pin each worker to a dedicated CPU core (Linux only). "
                        "Coordinator stays on core 0; workers walk cores 1..N. "
                        "Reduces L1/L2 cache invalidation from kernel scheduler migrations.")
    args = p.parse_args()

    model_arch = {"d_hidden": args.d_hidden, "c_filters": args.c_filters, "n_blocks": args.n_blocks}
    eval_opponents: list[tuple[str, Path]] = []
    for token in args.eval_opponent:
        if "=" in token:
            label, raw_path = token.split("=", 1)
        else:
            raw_path = token
            label = Path(raw_path).stem
        eval_opponents.append((label, Path(raw_path)))
    run_async_training(
        run_dir=args.run_dir, model_arch=model_arch, device=args.device,
        n_workers=args.workers, n_sims=args.sims, c_puct=args.c_puct,
        temperature=args.temperature, dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps, mcts_batch_size=args.mcts_batch_size,
        vloss_batch=args.vloss_batch, max_plies=args.max_plies,
        trainer_batch_size=args.trainer_batch_size, trainer_lr=args.trainer_lr,
        weight_save_every=args.weight_save_every, checkpoint_every=args.checkpoint_every,
        min_buffer_games=args.min_buffer_games, buffer_max_games=args.buffer_max_games,
        grad_clip=args.grad_clip, value_loss_weight=args.value_loss_weight,
        eval_every_seconds=args.eval_every_seconds, eval_games=args.eval_games,
        eval_sims=args.eval_sims, eval_workers=args.eval_workers,
        run_seconds=args.run_seconds, run_games=args.run_games,
        seed=args.seed, base_weights=args.base,
        resume_from=args.resume_from,
        eval_opponents=eval_opponents,
        wandb_project=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
        wandb_mode=args.wandb_mode,
        shared_inference=args.shared_inference,
        shared_max_batch_size=args.shared_max_batch_size,
        shared_timeout_ms=args.shared_timeout_ms,
        pin_cpus=args.pin_cpus,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
