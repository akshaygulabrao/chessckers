"""Play and emit one or more NN-vs-random games.

Two output modes:
  --to chessground/watch/games.jsonl   (append games for the spectate viewer)
  --to -                                (print plain text to stdout: move-by-move)

Each game record matches what `chessground/spectate.html` expects:
    {"history":[{"fen":"...","uci":"..."}, ...], "final_fen":"...",
     "outcome":"...", "iter":0, "game_idx":N}

Default: play 1 game with NN as white, write to stdout in text form.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.mcts_puct import run_mcts
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.random_player import pick_random
from chessckers_engine.variant_py import PyVariantClient


def _nn_picker(weights: Path, arch: dict, device: str, sims: int, temp: float, seed: int):
    model = ChesskersScorer(**arch).to(device)
    load_checkpoint(model, weights)
    model.eval()
    client = PyVariantClient()
    rng = random.Random(seed)

    def picker(state):
        result = run_mcts(state, client, model, n_sims=sims)
        if not result.visit_distribution or result.chosen is None:
            return result.chosen
        if temp <= 0:
            return result.chosen
        ucis = list(result.visit_distribution.keys())
        visits = [result.visit_distribution[u] for u in ucis]
        invT = 1.0 / temp
        weights_ = [v ** invT for v in visits]
        s = sum(weights_)
        if s <= 0:
            return result.chosen
        probs = [w / s for w in weights_]
        chosen_uci = rng.choices(ucis, weights=probs, k=1)[0]
        for uci, child in result.root.children.items():
            if uci == chosen_uci:
                return child.move_to_here
        return result.chosen
    return picker, client


def _play_one(white_picker, black_picker, client, max_plies: int = 400):
    state = client.new_game()
    history = []
    ply = 0
    while not state.get("status") and ply < max_plies:
        cur_fen = state["fen"]
        picker = white_picker if state["turn"] == "white" else black_picker
        move = picker(state)
        if move is None:
            break
        state = client.make_move(cur_fen, move["uci"])
        history.append({"fen": cur_fen, "uci": move["uci"]})
        ply += 1
    final_fen = state["fen"]
    if state.get("status"):
        outcome = state.get("winner") or state["status"]
    elif ply >= max_plies:
        outcome = "draw-max-plies"
    else:
        outcome = "incomplete"
    return {
        "history": history,
        "final_fen": final_fen,
        "outcome": outcome,
    }


def _print_text(game: dict, label: str) -> None:
    print(f"=== {label} — outcome: {game['outcome']} ({len(game['history'])} plies) ===")
    for i, hop in enumerate(game["history"], 1):
        side = "W" if i % 2 == 1 else "B"
        print(f"  ply {i:3d} {side}: {hop['uci']}")
    print(f"  final fen: {game['final_fen']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path,
                   default=Path("runs/local-001/weights.pt"))
    p.add_argument("--games", type=int, default=1, help="how many games to play")
    p.add_argument("--nn-color", choices=["white", "black", "alternate"],
                   default="alternate")
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="0 = argmax (deterministic), 1.0 = proportional")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu",
                   help="cpu/mps/cuda — keep cpu so MPS training isn't disturbed")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--to", default="-",
                   help="path to append game JSONL (use '-' for text-stdout)")
    args = p.parse_args()

    arch = dict(d_hidden=args.d_hidden, c_filters=args.c_filters, n_blocks=args.n_blocks)
    nn_picker, _ = _nn_picker(
        args.weights, arch, args.device, args.sims, args.temperature, args.seed,
    )

    def random_picker(state):
        return pick_random(state.get("legalMoves") or [])

    play_client = PyVariantClient()
    out_path: Path | None = None if args.to == "-" else Path(args.to)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(args.games):
        if args.nn_color == "white" or (args.nn_color == "alternate" and i % 2 == 0):
            white, black, label = nn_picker, random_picker, f"game {i+1}: NN(white) vs random(black)"
        else:
            white, black, label = random_picker, nn_picker, f"game {i+1}: random(white) vs NN(black)"
        game = _play_one(white, black, play_client)
        game["iter"] = 0
        game["game_idx"] = i + 1
        game["total_games"] = args.games
        if out_path is None:
            _print_text(game, label)
        else:
            with out_path.open("a") as f:
                f.write(json.dumps(game) + "\n")
            print(f"appended game {i+1}/{args.games} ({game['outcome']}, {len(game['history'])} plies) → {out_path}",
                  file=sys.stderr)
    play_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
