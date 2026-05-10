"""Stable benchmark harness for MCTS throughput.

Runs N MCTS calls of S sims each on a fixed set of seed positions and reports
sims/sec averaged across runs, plus per-position wall-clock variance. Use as
the metric to chase when iterating on engine optimizations.

Usage:
    uv run python bench/bench_mcts.py
    uv run python bench/bench_mcts.py --sims 200 --positions 5 --warmup 1 --runs 3
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from chessckers_engine.mcts_puct import run_mcts
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.variant_py import PyVariantClient

# A small set of distinct positions: opening, mid-Black-development,
# mid-tactical, late. Picked so the bench exercises all four Black
# move-gen pipelines (quiet/charge/deploy/capture) under realistic
# stack distributions.
# Real Chessckers FENs derived by walking the engine forward a few plies.
# Stack overlays included so MCTS has actual Black material to expand;
# without them the position is "Black eliminated" and MCTS exits instantly.
POSITIONS = [
    # 0. Starting position.
    None,
    # 1. After 1.e2-e4 ... <black quiet> ... 1.d2-d4 — Black to move, mid-opening.
    "pppppppp/kkkkkkkk/1ppppppp/1p6/3PP3/8/PPP2PPP/RNBQKBNR"
    "[b5:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,a7:k,b7:k,c7:k,d7:k,e7:k,"
    "f7:k,g7:k,h7:k,a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] b KQ - 0 1",
    # 2. Few moves deeper — Black to move with a Stone advanced to a4.
    "pppppppp/kkkkkkkk/1ppppppp/8/p2PP3/5N2/PPP2PPP/RNBQKB1R"
    "[a4:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,a7:k,b7:k,c7:k,d7:k,e7:k,"
    "f7:k,g7:k,h7:k,a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] b KQ - 1 1",
    # 3. Mid-position with two White knights developed; White to move.
    "pppppppp/kkkkkkkk/2pppppp/p7/3PP3/1pN2N2/PPP2PPP/R1BQKB1R"
    "[b3:s,a5:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,a7:k,b7:k,c7:k,d7:k,e7:k,"
    "f7:k,g7:k,h7:k,a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQ - 2 1",
]


def _bench_one(
    model: ChesskersScorer, client: PyVariantClient, fen: str | None, sims: int
) -> float:
    state = client.new_game(fen)
    t0 = time.perf_counter()
    run_mcts(state, client, model, n_sims=sims, c_puct=1.5)
    return time.perf_counter() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--positions", type=int, default=len(POSITIONS))
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--d_hidden", type=int, default=64)
    p.add_argument("--c_filters", type=int, default=32)
    p.add_argument("--n_blocks", type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(0)
    model = ChesskersScorer(
        d_hidden=args.d_hidden, c_filters=args.c_filters, n_blocks=args.n_blocks,
    ).eval()
    client = PyVariantClient()

    fens = POSITIONS[: args.positions]

    # Warmup: prime python-chess tables, JIT, etc.
    for _ in range(args.warmup):
        for fen in fens:
            _bench_one(model, client, fen, max(args.sims // 4, 10))

    per_run_totals: list[float] = []
    per_position: dict[int, list[float]] = {i: [] for i in range(len(fens))}
    for r in range(args.runs):
        run_total = 0.0
        for i, fen in enumerate(fens):
            elapsed = _bench_one(model, client, fen, args.sims)
            per_position[i].append(elapsed)
            run_total += elapsed
        per_run_totals.append(run_total)

    total_sims = args.sims * len(fens) * args.runs
    grand_total = sum(per_run_totals)
    sims_per_sec = total_sims / grand_total

    print(f"=== MCTS bench (sims={args.sims}, positions={len(fens)}, runs={args.runs}) ===")
    print(f"net model: d_hidden={args.d_hidden} c_filters={args.c_filters} n_blocks={args.n_blocks}")
    print(f"total sims: {total_sims}   total wall: {grand_total:.3f}s")
    print(f"throughput: {sims_per_sec:.1f} sims/sec")
    print()
    print("per-position avg ms/sim across runs:")
    for i, samples in per_position.items():
        avg_ms = (statistics.mean(samples) / args.sims) * 1000
        std_ms = (statistics.pstdev(samples) / args.sims) * 1000 if len(samples) > 1 else 0.0
        label = "starting" if fens[i] is None else fens[i][:30] + "..."
        print(f"  {i}: {avg_ms:6.2f} ± {std_ms:4.2f}   {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
