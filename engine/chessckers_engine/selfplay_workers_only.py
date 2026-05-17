"""Workers-only self-play launcher: no trainer, no eval.

Designed for cloud boxes that just generate games against a slowly-
updating weights snapshot. Periodic rsync (handled outside this script)
brings new weights from the trainer's host. The trainer ingests the
generated games on the way back.

Two inference modes:

  * **Per-worker** (default): each worker holds its own model copy on
    `--device` and mtime-polls `weights_path` for hot-reload. Fine for
    pure-CPU boxes; on GPU it serializes N small per-worker forwards
    behind the kernel-launch tax.

  * **Shared** (`--shared-inference`): one server thread in this
    process owns the model on `--device`, batches leaves from all N
    workers via mp.Queue. Workers become CPU-only (no GPU model). On a
    shared GPU this collapses N small forwards into one batched
    forward, saturating the device. A `WeightsWatcher` thread polls
    `--weights` mtime and reloads the model in place when the sidecar
    rsyncs in a fresh snapshot.

Usage (on the cloud box, GPU shared mode):

    python -m chessckers_engine.selfplay_workers_only \\
      --run-dir /root/run --weights /root/run/weights.pt \\
      --workers 14 --device cuda --sims 100 \\
      --shared-inference --shared-max-batch-size 32 \\
      --d-hidden 256 --c-filters 128 --n-blocks 6

Stop: `touch /root/run/STOP` (or send SIGTERM).
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# torch.multiprocessing is a drop-in replacement for multiprocessing that
# adds zero-copy shared-memory pickling for tensors — required for shared
# inference, since workers ship CPU tensors through the request queue.
import torch
import torch.multiprocessing as mp

from chessckers_engine.runtime import setup_logging
from chessckers_engine.selfplay_worker_async import play_forever_subprocess

log = logging.getLogger("chessckers_engine.selfplay_workers_only")


class WeightsWatcher:
    """Polls `weights_path` mtime; reloads the shared model in place when newer.

    Mirrors the per-worker hot-reload logic that lives inside
    `selfplay_worker_async.play_forever`, but runs once in the coordinator
    process so all workers share the update via the shared model object.
    Concurrent with the inference server's forward — `load_state_dict` is
    in-place, same pattern the trainer/server combo uses in
    `selfplay_az_async`.
    """

    def __init__(self, model: torch.nn.Module, weights_path: Path,
                 poll_seconds: float = 30.0) -> None:
        self._model = model
        self._weights_path = weights_path
        self._poll_s = poll_seconds
        self._last_mtime = -1.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="WeightsWatcher")

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        from chessckers_engine.checkpoints import load_checkpoint
        while not self._stop.is_set():
            try:
                mtime = self._weights_path.stat().st_mtime
            except FileNotFoundError:
                self._stop.wait(self._poll_s)
                continue
            if mtime > self._last_mtime:
                try:
                    load_checkpoint(self._model, self._weights_path)
                    self._model.eval()
                    self._last_mtime = mtime
                    log.info("reloaded weights from %s (mtime=%.0f)",
                             self._weights_path, mtime)
                except (EOFError, RuntimeError, OSError) as e:
                    # Sidecar may be mid-rsync — try again next tick.
                    log.debug("weights reload skipped: %s", e)
            self._stop.wait(self._poll_s)

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _build_per_worker_payload(*, wid: int, buffer_root: Path, stop_path: Path,
                              args: argparse.Namespace, model_arch: dict,
                              weights_path: Path) -> dict:
    return {
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
        "device": args.device,
        "model_arch": model_arch,
        "weights_path": str(weights_path),
        "mcts_batch_size": args.mcts_batch_size,
        "weights_poll_seconds": args.weights_poll_seconds,
        "pin_cpu": (wid % os.cpu_count()) if args.pin_cpu else None,
        "machine": os.environ.get("MACHINE", "unknown"),
    }


def _build_shared_payload(*, wid: int, q_index: int, buffer_root: Path, stop_path: Path,
                          args: argparse.Namespace,
                          request_q, response_q) -> dict:
    return {
        "worker_id": wid,
        "q_index": q_index,
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
        "request_q": request_q,
        "response_q": response_q,
        "pin_cpu": (wid % os.cpu_count()) if args.pin_cpu else None,
        "machine": os.environ.get("MACHINE", "unknown"),
    }


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Workers-only self-play (no trainer).")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Directory to host buffer/, stop sentinel, etc.")
    p.add_argument("--weights", required=True, type=Path,
                   help="Path to weights.pt — workers (per-worker mode) or the "
                        "WeightsWatcher (shared mode) hot-reload when mtime changes.")
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
    p.add_argument("--shared-inference", action="store_true",
                   help="One coordinator-side server batches leaves from ALL workers. "
                        "Workers run pure-CPU; the model lives on --device in this process. "
                        "Required for high GPU utilization with N>1 workers.")
    p.add_argument("--shared-max-batch-size", type=int, default=32,
                   help="Max batch size for the shared inference server.")
    p.add_argument("--shared-timeout-ms", type=float, default=5.0,
                   help="Max wait (ms) for additional requests to coalesce into a batch.")
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

    # ---- Optional shared-inference setup ----
    cross_server = None
    weights_watcher: Optional[WeightsWatcher] = None
    request_q = None
    response_qs: list = []
    if args.shared_inference:
        from chessckers_engine.cross_inference import CrossInferenceServer
        from chessckers_engine.model import ChesskersScorer

        device = torch.device(args.device)
        model = ChesskersScorer(**model_arch).to(device).eval()
        # Seed model from disk if available; otherwise the watcher will pick
        # it up the first time it appears.
        if weights_path.exists():
            from chessckers_engine.checkpoints import load_checkpoint
            load_checkpoint(model, weights_path)
            log.info("seeded shared model from %s on %s", weights_path, device)
        else:
            log.warning("no weights at %s yet; serving random init until sidecar pushes one",
                        weights_path)

        request_q = ctx.Queue()
        response_qs = [ctx.Queue() for _ in range(args.workers)]
        cross_server = CrossInferenceServer(
            model=model, request_q=request_q, response_qs=response_qs,
            max_batch_size=args.shared_max_batch_size,
            timeout_ms=args.shared_timeout_ms,
            log_every=50,
        )
        cross_server.start()
        log.info("shared inference server started on %s (max_batch=%d, timeout=%.1fms)",
                 device, args.shared_max_batch_size, args.shared_timeout_ms)

        weights_watcher = WeightsWatcher(
            model=model, weights_path=weights_path,
            poll_seconds=args.weights_poll_seconds,
        )
        weights_watcher.start()

    # ---- Spawn workers ----
    workers = []
    for i in range(args.workers):
        wid = args.worker_id_base + i
        if args.shared_inference:
            payload = _build_shared_payload(
                wid=wid, q_index=i,
                buffer_root=buffer_root, stop_path=stop_path, args=args,
                request_q=request_q, response_q=response_qs[i],
            )
        else:
            payload = _build_per_worker_payload(
                wid=wid, buffer_root=buffer_root, stop_path=stop_path, args=args,
                model_arch=model_arch, weights_path=weights_path,
            )
        proc = ctx.Process(target=play_forever_subprocess, args=(payload,),
                           name=f"worker-{wid}")
        proc.start()
        workers.append(proc)
    mode = "shared-inference" if args.shared_inference else "per-worker"
    log.info("spawned %d workers (%s mode); weights=%s buffer=%s",
             len(workers), mode, weights_path, buffer_root)

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
        # Workers are gone — no more requests can land. Safe to stop the
        # shared inference server (otherwise blocked clients would wedge).
        if cross_server is not None:
            stats = cross_server.stats()
            log.info("x-inference summary: batches=%d reqs=%d avg_bs=%.2f "
                     "max_bs=%d gpu_secs=%.2f",
                     stats["n_batches"], stats["n_requests"], stats["avg_batch_size"],
                     stats["max_batch_size_seen"], stats["inference_secs"])
            cross_server.shutdown()
        if weights_watcher is not None:
            weights_watcher.shutdown()
        # Final exit-code marker the launcher script polls for.
        (run_dir / "exit_code").write_text("0\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
