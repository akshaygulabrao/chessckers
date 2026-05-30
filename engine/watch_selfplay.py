"""Watch one self-play game in the terminal: the net plays BOTH sides via PUCT,
and the 10x10 board is re-rendered after every move (capture chains overlaid as
a numbered path).

Usage:
    python watch_selfplay.py                      # fresh-init (random) net — works now
    python watch_selfplay.py --weights weights/iter-az-001.pt
    python watch_selfplay.py --sims 30 --delay 0.6 --max-plies 120

Runs on CPU by default so it doesn't contend with a training run using MPS.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def _move_path(move: dict) -> list[str] | None:
    """For a capture/chain move, the ordered grid cells to overlay (from → … → to)."""
    wps = move.get("waypoints") or []
    if not wps and move.get("capture") is None:
        return None
    return [move["from"], *wps, move["to"]]


def _describe(move: dict) -> str:
    tags = []
    if move.get("capture") is not None:
        tags.append("capture")
    if move.get("waypoints"):
        tags.append(f"chain[{len(move['waypoints'])}]")
    if move.get("deployCount") is not None:
        tags.append(f"deploy{move['deployCount']}")
    if move.get("demotionsRequired") is not None:
        tags.append(f"charge/demote{move['demotionsRequired']}")
    return move.get("uci", "?") + (f"  ({', '.join(tags)})" if tags else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None, help="checkpoint .pt (default: fresh-init random net)")
    ap.add_argument("--sims", type=int, default=30, help="PUCT sims per move (lower = faster to watch)")
    ap.add_argument("--delay", type=float, default=0.6, help="seconds to pause between moves")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-hidden", type=int, default=256)
    ap.add_argument("--model-filters", type=int, default=96)
    ap.add_argument("--model-blocks", type=int, default=4)
    args = ap.parse_args()

    from chessckers_engine.checkpoints import load_checkpoint
    from chessckers_engine.mcts_puct import pick_puct
    from chessckers_engine.model import ChesskersScorer
    from chessckers_engine.render_board import render_board
    from chessckers_engine.variant_py import PyVariantClient

    device = torch.device(args.device)
    model = ChesskersScorer(d_hidden=args.model_hidden, c_filters=args.model_filters,
                            n_blocks=args.model_blocks).to(device).eval()
    if args.weights:
        if not Path(args.weights).exists():
            print(f"weights not found: {args.weights}", file=sys.stderr)
            return 2
        load_checkpoint(model, args.weights)
        tag = args.weights
    else:
        tag = "fresh-init (random) net"

    client = PyVariantClient()
    state = client.new_game()
    print(f"self-play — {tag}, sims={args.sims}\n")

    last_move: dict | None = None
    ply = 0
    while not state.get("status") and ply < args.max_plies:
        legal = state.get("legalMoves") or []
        if not legal:
            break
        print(f"\n=== ply {ply}  —  {state['turn']} to move ===")
        path = _move_path(last_move) if last_move else None
        print(render_board(state["fen"], path=path))
        if last_move is not None:
            print(f"(last move overlaid above: {_describe(last_move)})")

        move = pick_puct(state, client, model, n_sims=args.sims)
        if move is None:
            break
        print(f"→ {state['turn']} plays: {_describe(move)}")
        state = client.make_move(state["fen"], move["uci"])
        last_move = move
        ply += 1
        time.sleep(args.delay)

    print(f"\n=== final position (ply {ply}) ===")
    print(render_board(state["fen"], path=_move_path(last_move) if last_move else None))
    winner = state.get("winner")
    print(f"\nstatus={state.get('status')}  winner={winner}  plies={ply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
