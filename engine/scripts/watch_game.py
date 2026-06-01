#!/usr/bin/env python
"""Watch a self-play game from any Chessckers FEN.

The trained net plays BOTH sides at --sims (default 400) MCTS sims/move. The
MOVE played is always the argmax of the visit counts (the "calculation");
exploration is injected only as root Dirichlet noise (--explore, default
0.30 = 30%), so different runs (or --seed) give varied games while each move
stays the search's best. --explore 0 = pure greedy/deterministic. Each ply
renders the 10x10 board live as MCTS finishes it. Defaults to the latest ckpt.

  cd engine
  .venv/bin/python scripts/watch_game.py "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1"
  # options: --weights X.pt  --sims 400  --max-plies 80  --device cpu|mps  --delay 0.5
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

_ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../engine


def _resolve_weights(arg: str) -> str:
    if arg:
        return arg
    cks = sorted(
        glob.glob(os.path.join(_ENG, "weights/run/iter-az-*.pt")),
        key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)),
        reverse=True,
    )
    for p in cks + [os.path.join(_ENG, "weights/base_live.pt")]:
        if os.path.exists(p):
            return p
    raise SystemExit("no weights found (weights/run/iter-az-*.pt or base_live.pt); pass --weights")


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch a greedy (argmax) self-play game from a FEN.")
    ap.add_argument("fen", help="Chessckers start FEN, e.g. '8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1'")
    ap.add_argument("--weights", default="", help="checkpoint .pt (default: latest weights/run/iter-az-*.pt)")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--explore", type=float, default=0.30,
                    help="root Dirichlet exploration-noise fraction (default 0.30 = 30 pct); the "
                         "played move stays argmax of visits. 0 = pure greedy/deterministic.")
    ap.add_argument("--seed", type=int, default=-1,
                    help="rng seed (default: random each run, so games vary)")
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--device", default="cpu", help="cpu|mps|cuda (default cpu)")
    ap.add_argument("--delay", type=float, default=0.0, help="extra pause between plies, seconds")
    args = ap.parse_args()

    import torch
    from chessckers_engine.checkpoints import load_checkpoint
    from chessckers_engine.model import ChesskersScorer
    from chessckers_engine.render_board import render_board
    from chessckers_engine.selfplay_az import play_az_game
    from chessckers_engine.variant_py import PyVariantClient

    model = ChesskersScorer(d_hidden=256, c_filters=96, n_blocks=4).to(args.device).eval()
    weights = _resolve_weights(args.weights)
    try:
        load_checkpoint(model, weights)
    except Exception:  # newest checkpoint may be mid-write; fall back
        weights = os.path.join(_ENG, "weights/base_live.pt")
        load_checkpoint(model, weights)
    seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), "big")
    print(f"weights: {weights}\nsims: {args.sims} | device: {args.device} | "
          f"explore (root noise): {args.explore:.0%} | move pick: argmax | seed: {seed}\n")

    os.environ["CHESSCKERS_START_FEN"] = args.fen  # play_az_game's new_game() reads this

    class WatchSink:
        def on_move(self, d: dict) -> None:
            uci = d.get("last_uci")
            head = f"ply {d.get('ply', 0)}: {uci}" if uci else "start"
            print(f"\n=== {head} ===")
            print(render_board(d["fen"]))
            if args.delay:
                time.sleep(args.delay)

        def on_game_end(self, d: dict) -> None:
            print(f"\n######## {str(d.get('outcome')).upper()} "
                  f"({d.get('final_status')}) in {len(d.get('history', []))} plies ########")

    play_az_game(
        model, PyVariantClient(),
        n_sims=args.sims, c_puct=1.5,
        temperature=0.0, temp_cutoff_plies=0,                 # argmax MOVE pick ("calculation")
        dirichlet_alpha=(0.3 if args.explore > 0 else None),  # root exploration noise...
        dirichlet_eps=args.explore,                           # ...at --explore fraction (30%)
        max_plies=args.max_plies,
        rng=torch.Generator().manual_seed(seed),
        sink=WatchSink(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
