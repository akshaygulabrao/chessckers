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
import multiprocessing as mp
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from chessckers_engine.checkpoints import DEFAULT_WEIGHTS_DIR
from chessckers_engine.device import pick_device
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.replay_buffer import ReplayBuffer
from chessckers_engine.selfplay_az_loop import _run_eval_parallel
from chessckers_engine.selfplay_worker_async import play_forever
from chessckers_engine.trainer_loop import TrainerLoop

log = logging.getLogger("chessckers_engine.selfplay_az_async")


def _build_worker_payload(worker_id: int, *, weights_path: Path, buffer_root: Path,
                          stop_path: Path, model_arch: dict, device: str,
                          n_sims: int, c_puct: float, temperature: float,
                          dirichlet_alpha: float, dirichlet_eps: float,
                          mcts_batch_size: int, vloss_batch: int,
                          max_plies: int, seed: int) -> dict:
    return {
        "worker_id": worker_id,
        "device": device,
        "model_arch": model_arch,
        "weights_path": str(weights_path),
        "buffer_root": str(buffer_root),
        "n_sims": n_sims,
        "c_puct": c_puct,
        "temperature": temperature,
        "dirichlet_alpha": dirichlet_alpha if dirichlet_alpha > 0 else None,
        "dirichlet_eps": dirichlet_eps,
        "mcts_batch_size": mcts_batch_size,
        "vloss_batch": vloss_batch,
        "max_plies": max_plies,
        "seed": seed + worker_id,
        "stop_path": str(stop_path),
        "max_games": None,
        "weights_poll_seconds": 5.0,
    }


def _eval_snapshot_and_log(*, weights_path: Path, snapshot_path: Path,
                           model_arch: dict, device: str, eval_games: int,
                           eval_sims: int, eval_workers: int, eval_log_path: Path,
                           trainer_step: int) -> dict:
    """Snapshot current weights and play `eval_games` vs random, both sides.
    Appends one JSON line to `eval_log_path` and returns the summary."""
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
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trainer_step": trainer_step,
        "eval_games_per_side": eval_games,
        "eval_sims": eval_sims,
        "as_white_vs_random": as_white,
        "as_black_vs_random": as_black,
    }
    with eval_log_path.open("a") as f:
        f.write(json.dumps(summary))
        f.write("\n")
    log.info("eval @ step %d | W: %d/%d/%d | B: %d/%d/%d",
             trainer_step,
             as_white["white"], as_white["black"], as_white["draw"],
             as_black["black"], as_black["white"], as_black["draw"])
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
    seed: int = 0,
    base_weights: Optional[Path] = None,
    main_loop_poll_seconds: float = 5.0,
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

    # Initialize model on `device` and broadcast initial weights so workers
    # don't sit idle waiting for the trainer's first save (which only fires
    # after min_buffer_games).
    torch.manual_seed(seed)
    model = ChesskersScorer(**model_arch).to(device)
    if base_weights is not None:
        from chessckers_engine.checkpoints import load_checkpoint
        load_checkpoint(model, base_weights)
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

    # Spawn workers (subprocesses with own MPS/CUDA context).
    ctx = mp.get_context("spawn")
    workers: list[mp.Process] = []
    for w in range(n_workers):
        payload = _build_worker_payload(
            w, weights_path=weights_path, buffer_root=buffer_root,
            stop_path=stop_path, model_arch=model_arch, device=device,
            n_sims=n_sims, c_puct=c_puct, temperature=temperature,
            dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
            mcts_batch_size=mcts_batch_size, vloss_batch=vloss_batch,
            max_plies=max_plies, seed=seed,
        )
        p = ctx.Process(target=play_forever, args=(payload,), name=f"worker-{w}")
        p.start()
        workers.append(p)
    log.info("spawned %d self-play workers", n_workers)

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
    )
    trainer_thread = threading.Thread(target=trainer.run, name="trainer")
    trainer_thread.start()

    start = time.perf_counter()
    last_eval = 0.0  # first eval fires after eval_every_seconds wall-clock
    try:
        while not stop_path.exists() and (time.perf_counter() - start) < run_seconds:
            elapsed = time.perf_counter() - start
            if (elapsed - last_eval) >= eval_every_seconds and trainer.step > 0:
                _eval_snapshot_and_log(
                    weights_path=weights_path, snapshot_path=eval_snapshot_path,
                    model_arch=model_arch, device=device,
                    eval_games=eval_games, eval_sims=eval_sims,
                    eval_workers=eval_workers, eval_log_path=eval_log_path,
                    trainer_step=trainer.step,
                )
                last_eval = elapsed
            time.sleep(main_loop_poll_seconds)
    finally:
        # Wind everything down cleanly. Order matters: stop workers first
        # so they finish in-flight games, then stop trainer.
        if not stop_path.exists():
            stop_path.touch()
        log.info("shutting down — waiting for workers to finish their games")
        for p in workers:
            p.join(timeout=300)
            if p.is_alive():
                log.warning("worker %s did not exit; terminating", p.name)
                p.terminate()
                p.join(timeout=10)
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", type=Path, default=None,
                   help="Optional starting weights to load before training.")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=6)
    args = p.parse_args()

    model_arch = {"d_hidden": args.d_hidden, "c_filters": args.c_filters, "n_blocks": args.n_blocks}
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
        run_seconds=args.run_seconds, seed=args.seed, base_weights=args.base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
