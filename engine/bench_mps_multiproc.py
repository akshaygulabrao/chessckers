"""Smoke test: does PyTorch MPS work with multiprocessing?

Spawns N processes, each loads a small ChesskersScorer on MPS, runs forward
passes, and reports throughput. We need >1.5x speedup at 4 workers vs 1 to
declare MPS multiproc viable for the async trainer.

Usage:
    cd /Users/ox/AAworkspace/chessckers/engine
    python3 bench_mps_multiproc.py
"""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Tuple

import torch

from chessckers_engine.model import ChesskersScorer

# 5M-param config (the proposed default for the new async run)
ARCH = {"d_hidden": 256, "c_filters": 128, "n_blocks": 6}
N_FORWARDS = 100
BATCH = 8


def _worker(worker_id: int, device: str, results_q) -> None:
    """One forward-pass loop. Reports its own elapsed time + per-batch latency."""
    torch.manual_seed(worker_id)
    model = ChesskersScorer(**ARCH).to(device).eval()
    # Warm up — first forward on MPS is much slower (kernel compile)
    x = torch.randn(BATCH, 14, 8, 8, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model.position_trunk(x)
    # Timed loop
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_FORWARDS):
            x = torch.randn(BATCH, 14, 8, 8, device=device)
            _ = model.position_trunk(x)
    if device.startswith("mps"):
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    results_q.put((worker_id, elapsed, N_FORWARDS / elapsed))


def run(n_workers: int, device: str) -> Tuple[float, float]:
    """Returns (wall_seconds, total_forwards_per_sec)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(i, device, q)) for i in range(n_workers)]
    start = time.perf_counter()
    for p in procs:
        p.start()
    results = [q.get(timeout=120) for _ in range(n_workers)]
    for p in procs:
        p.join(timeout=10)
    wall = time.perf_counter() - start
    total_fwd = sum(r[2] for r in results) * (N_FORWARDS / max(r[1] for r in results) / N_FORWARDS)
    # Aggregate: total forwards across all workers / wall time
    total_throughput = (n_workers * N_FORWARDS) / wall
    return wall, total_throughput


def main() -> None:
    if not torch.backends.mps.is_available():
        print("MPS not available — exiting")
        return
    device = "mps"
    print(f"Model: {ARCH} → 5M-param ChesskersScorer")
    print(f"Per worker: {N_FORWARDS} forwards × batch={BATCH}")
    print()
    print(f"{'workers':>8}  {'wall_s':>8}  {'fwd/s':>10}  {'speedup':>8}")
    print("-" * 50)
    baseline = None
    for n in (1, 2, 4, 6):
        try:
            wall, fwd_per_s = run(n, device)
            if baseline is None:
                baseline = fwd_per_s
            speedup = fwd_per_s / baseline
            print(f"{n:>8}  {wall:>8.2f}  {fwd_per_s:>10.1f}  {speedup:>7.2f}x")
        except Exception as e:
            print(f"{n:>8}  FAILED: {e}")
            break
    print()
    print("Verdict:")
    print("  >= 1.5x at n=4 → MPS multiproc viable")
    print("  ~ 1.0x at n=4 → MPS serializes, async will be CUDA-only")


if __name__ == "__main__":
    main()
