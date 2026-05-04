"""Play one game of best.pt vs random and append to chessground/watch/games.jsonl
so spectate.html can replay it.

Usage:
    uv run python play_vs_random.py --weights weights/local-mps-003/best.pt --side white
    uv run python play_vs_random.py --weights weights/local-mps-003/best.pt --side black
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.mcts_puct import pick_puct
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.random_player import pick_random
from chessckers_engine.variant_py import PyVariantClient


def play_one(model, side: str, n_sims: int, max_plies: int) -> dict:
    client = PyVariantClient()
    state = client.new_game()
    history: list[dict] = []
    plies = 0
    while not state.get("status") and plies < max_plies:
        legal = state.get("legalMoves") or []
        if not legal:
            break
        if state["turn"] == side:
            move = pick_puct(state, client, model, n_sims=n_sims)
        else:
            move = pick_random(legal)
        if move is None:
            break
        history.append({"fen": state["fen"], "uci": move["uci"]})
        state = client.make_move(state["fen"], move["uci"])
        plies += 1
    outcome = state.get("winner") or (
        "white" if state.get("status") == "variantEnd" else
        "black" if state.get("status") == "mate" else
        "draw"
    )
    return {
        "history": history,
        "final_fen": state["fen"],
        "final_status": state.get("status"),
        "outcome": outcome,
        "controllers": {
            "white": "nn" if side == "white" else "random",
            "black": "nn" if side == "black" else "random",
        },
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "n_sims": n_sims,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--side", choices=["white", "black"], default="white",
                    help="which side the NN plays")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--model-blocks", type=int, default=8)
    ap.add_argument("--model-filters", type=int, default=128)
    ap.add_argument("--model-hidden", type=int, default=256)
    ap.add_argument("--out", default="../chessground/watch/games.jsonl",
                    help="path to games.jsonl (relative to engine/)")
    ap.add_argument("--reset", action="store_true",
                    help="truncate games.jsonl before writing (don't append)")
    ap.add_argument("--max-plies", type=int, default=400)
    args = ap.parse_args()

    model = ChesskersScorer(
        d_hidden=args.model_hidden,
        c_filters=args.model_filters,
        n_blocks=args.model_blocks,
    ).eval()
    load_checkpoint(model, args.weights)

    print(f"playing {args.side}=NN vs {('black' if args.side=='white' else 'white')}=random "
          f"with {args.sims} sims...")
    game = play_one(model, args.side, args.sims, args.max_plies)
    print(f"  → outcome={game['outcome']} status={game['final_status']} "
          f"plies={len(game['history'])}")
    nn_won = (game["outcome"] == args.side)
    print(f"  NN {'WON' if nn_won else 'did NOT win'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and out.exists():
        out.unlink()
    with out.open("a") as f:
        f.write(json.dumps(game))
        f.write("\n")
    print(f"wrote game to {out.resolve()}")
    print(f"open chessground/spectate.html in your browser to view")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
