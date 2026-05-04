"""Benchmark how self-play throughput scales with worker count.

Plays N games at each worker setting, reports wall time and games/sec.
Use to pick optimal --workers before committing to a long training run.

Usage on remote:
    cd /root/chessckers/engine
    python3 bench_workers.py --device cuda --games-per-trial 16 \\
        --workers 4 8 16 24
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az_loop import _play_game_subprocess
from chessckers_engine.train_az import save_checkpoint


def bench(
    n_workers: int,
    n_games: int,
    state_path: str,
    arch: dict,
    device: str,
    n_sims: int,
    mcts_batch_size: int,
    vloss_batch: int,
) -> tuple[float, float]:
    """Returns (wall_seconds, games_per_sec)."""
    payloads = [
        {
            "state_path": state_path,
            "model_arch": arch,
            "device": device,
            "mcts_batch_size": mcts_batch_size,
            "n_sims": n_sims,
            "c_puct": 1.5,
            "temperature": 1.0,
            "seed": 1000 + i,
            "dirichlet_alpha": 0.3,
            "dirichlet_eps": 0.25,
            "vloss_batch": vloss_batch,
        }
        for i in range(n_games)
    ]
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(processes=n_workers) as pool:
        # Drain results to ensure all complete; we don't use the games.
        for _ in pool.imap_unordered(_play_game_subprocess, payloads):
            pass
    dt = time.perf_counter() - t0
    return dt, n_games / dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games-per-trial", type=int, default=16,
                    help="Number of games each worker-count config plays. "
                         "Higher = more accurate, slower.")
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16, 24],
                    help="Worker counts to test, e.g. --workers 4 8 16 24")
    ap.add_argument("--model-blocks", type=int, default=20)
    ap.add_argument("--model-filters", type=int, default=256)
    ap.add_argument("--model-hidden", type=int, default=384)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--mcts-batch-size", type=int, default=64)
    ap.add_argument("--vloss-batch", type=int, default=8)
    args = ap.parse_args()

    arch = {
        "d_hidden": args.model_hidden,
        "c_filters": args.model_filters,
        "n_blocks": args.model_blocks,
    }
    n_params = sum(p.numel() for p in ChesskersScorer(**arch).parameters())
    print(f"model: {args.model_blocks} blocks × {args.model_filters} filters → "
          f"{n_params/1e6:.2f}M params, sims={args.sims}, vloss={args.vloss_batch}")
    print(f"device: {args.device}, mcts_batch_size={args.mcts_batch_size}\n")

    # Save a fresh model state for workers to load.
    state_path = Path("/tmp/bench_state.pt")
    torch.manual_seed(0)
    model = ChesskersScorer(**arch)
    save_checkpoint(model, state_path)

    print(f"{'workers':>8} {'games':>6} {'wall_s':>8} {'games/s':>8} {'sec/game':>9}")
    print("-" * 50)
    results = []
    for n_workers in args.workers:
        try:
            dt, gps = bench(
                n_workers=n_workers,
                n_games=args.games_per_trial,
                state_path=str(state_path),
                arch=arch,
                device=args.device,
                n_sims=args.sims,
                mcts_batch_size=args.mcts_batch_size,
                vloss_batch=args.vloss_batch,
            )
            sec_per_game = dt / args.games_per_trial
            print(f"{n_workers:>8} {args.games_per_trial:>6} {dt:>8.1f} {gps:>8.2f} {sec_per_game:>9.2f}")
            results.append((n_workers, gps))
        except Exception as e:  # noqa: BLE001
            print(f"{n_workers:>8} FAILED: {e}")

    if results:
        best_n, best_gps = max(results, key=lambda r: r[1])
        print(f"\nbest: workers={best_n} → {best_gps:.2f} games/s")
        baseline_gps = next((g for n, g in results if n == 8), None)
        if baseline_gps:
            speedup = best_gps / baseline_gps
            print(f"speedup vs workers=8: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
