"""Workers-only self-play launcher: no trainer, no eval, no GPU required.

Designed for cloud CPU boxes that just generate games against a slowly-
updating weights snapshot. Periodic rsync (handled outside this script)
brings new weights from the trainer's host. The trainer ingests the
generated games on the way back.

Each worker uses per-worker inference mode (its own model on its own
device, hot-reloaded from `weights_path` when mtime changes) — so there
is no shared inference server. On a 32-vCPU CPU box with a small model,
parallel CPU forwards are competitive with a single batched MPS forward.

Usage (on the cloud box):

    python -m chessckers_engine.selfplay_workers_only \\
      --run-dir /root/run \\
      --weights /root/run/weights.pt \\
      --workers 32 --device cpu --sims 10 \\
      --d-hidden 128 --c-filters 64 --n-blocks 4

Stop: `touch /root/run/STOP` (or send SIGTERM).
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

from chessckers_engine.runtime import setup_logging
from chessckers_engine.selfplay_worker_async import play_forever_subprocess

log = logging.getLogger("chessckers_engine.selfplay_workers_only")


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Workers-only self-play (no trainer).")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Directory to host buffer/, stop sentinel, etc.")
    p.add_argument("--weights", required=True, type=Path,
                   help="Path to weights.pt — workers hot-reload when mtime changes.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--worker-id-base", type=int, default=0,
                   help="First worker_id (filenames start with this prefix). Set non-zero "
                        "on cloud boxes to avoid collisions when rsynced into a local buffer "
                        "that also has worker_ids 0..N from local self-play.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sims", type=int, default=10)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--dirichlet-eps", type=float, default=0.40)
    p.add_argument("--mcts-batch-size", type=int, default=1)
    p.add_argument("--vloss-batch", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--d-hidden", type=int, default=128)
    p.add_argument("--c-filters", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--weights-poll-seconds", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pin-cpu", action="store_true",
                   help="Pin each worker to a CPU core (Linux only).")
    args = p.parse_args()

    run_dir: Path = args.run_dir.resolve()
    buffer_root = run_dir / "buffer"
    stop_path = run_dir / "STOP"
    run_dir.mkdir(parents=True, exist_ok=True)
    buffer_root.mkdir(parents=True, exist_ok=True)
    if stop_path.exists():
        stop_path.unlink()

    weights_path: Path = args.weights.resolve()
    if not weights_path.exists():
        log.warning("weights file does not exist yet at %s — workers will wait", weights_path)

    model_arch = {
        "d_hidden": args.d_hidden,
        "c_filters": args.c_filters,
        "n_blocks": args.n_blocks,
    }

    # SIGTERM → touch stop file, let workers exit cleanly.
    def _on_sigterm(_signum, _frame):
        log.info("SIGTERM received; signaling stop")
        try:
            stop_path.touch()
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    ctx = mp.get_context("spawn")
    workers = []
    for i in range(args.workers):
        wid = args.worker_id_base + i
        payload = {
            "worker_id": wid,
            "buffer_root": str(buffer_root),
            "n_sims": args.sims,
            "c_puct": args.c_puct,
            "temperature": args.temperature,
            "dirichlet_alpha": args.dirichlet_alpha,
            "dirichlet_eps": args.dirichlet_eps,
            "vloss_batch": args.vloss_batch,
            "max_plies": args.max_plies,
            "seed": args.seed + wid,
            "stop_path": str(stop_path),
            "max_games": None,
            # Per-worker inference mode (no shared queues).
            "device": args.device,
            "model_arch": model_arch,
            "weights_path": str(weights_path),
            "mcts_batch_size": args.mcts_batch_size,
            "weights_poll_seconds": args.weights_poll_seconds,
            "pin_cpu": (wid % os.cpu_count()) if args.pin_cpu else None,
        }
        proc = ctx.Process(target=play_forever_subprocess, args=(payload,),
                           name=f"worker-{wid}")
        proc.start()
        workers.append(proc)
    log.info("spawned %d workers; weights=%s buffer=%s", len(workers), weights_path, buffer_root)

    # Heartbeat loop: report game count every 60s. Exit when stop_path appears
    # or all workers die.
    last_count = 0
    last_log = time.time()
    try:
        while not stop_path.exists():
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                log.info("all workers exited; stopping")
                break
            now = time.time()
            if now - last_log >= 60.0:
                count = sum(1 for _ in buffer_root.glob("*.pkl"))
                rate = (count - last_count) / max(1.0, now - last_log) * 60.0
                log.info("alive=%d games=%d (+%.1f/min)", alive, count, rate)
                last_count = count
                last_log = now
            time.sleep(2.0)
    finally:
        if not stop_path.exists():
            stop_path.touch()
        log.info("waiting for workers (300s deadline)")
        deadline = time.time() + 300.0
        for w in workers:
            while w.is_alive() and time.time() < deadline:
                time.sleep(0.5)
            if w.is_alive():
                log.warning("worker %s slow to exit; terminating", w.name)
                w.terminate()
        # Final exit-code marker the launcher script polls for.
        (run_dir / "exit_code").write_text("0\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
