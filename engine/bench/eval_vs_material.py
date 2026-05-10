"""On-demand stronger eval: play the current weights against the material
1-ply player. Doesn't disrupt the running training — snapshots `weights.pt`
and runs in a separate process, all-Python (PyVariantClient, no Scala server).

Usage:
    uv run python bench/eval_vs_material.py \
        --weights runs/local-001/weights.pt --games 8 --sims 50

Output: one line per side ("as white", "as black") with W/D/L vs material,
plus an Elo-style win-rate so progress against a fixed baseline is easy to
read. (Random opponent runs to ~95% win rate quickly; material opponent is
the next rung up — actually plays opening moves with intent.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import random as _random

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.evaluate import play_game
from chessckers_engine.material_player import pick_material
from chessckers_engine.mcts_puct import run_mcts
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.random_player import pick_random
from chessckers_engine.variant_py import PyVariantClient


def _build_nn_picker(weights: Path, arch: dict, device: str, sims: int,
                     temperature: float, seed: int):
    """NN picker that samples from MCTS visit distribution with `temperature`.

    Pure argmax (temperature → 0) is deterministic, so eval results don't move
    until the model crosses a hard outcome threshold — bad for tracking
    progress. A small temperature (~0.3) injects enough variance for the
    metric to drift smoothly with weight changes."""
    model = ChesskersScorer(**arch).to(device)
    load_checkpoint(model, weights)
    model.eval()
    client = PyVariantClient()
    rng = _random.Random(seed)

    def picker(state):
        result = run_mcts(state, client, model, n_sims=sims)
        if not result.visit_distribution or result.chosen is None:
            return result.chosen
        if temperature <= 0:
            return result.chosen
        ucis = list(result.visit_distribution.keys())
        visits = [result.visit_distribution[u] for u in ucis]
        # Temperature-scaled sampling: probs ∝ visits ** (1/T).
        invT = 1.0 / temperature
        weights_ = [v ** invT for v in visits]
        s = sum(weights_)
        if s <= 0:
            return result.chosen
        probs = [w / s for w in weights_]
        chosen_uci = rng.choices(ucis, weights=probs, k=1)[0]
        # Look up the move dict by uci among the children.
        for uci, child in result.root.children.items():
            if uci == chosen_uci:
                return child.move_to_here
        return result.chosen
    return picker, client


def _build_material_picker():
    client = PyVariantClient()
    def picker(state):
        return pick_material(state, client)
    return picker, client


def _build_random_picker():
    def picker(state):
        return pick_random(state.get("legalMoves") or [])
    return picker, None


def _run(opponent_name: str, weights: Path, arch: dict, device: str,
         sims: int, games: int, temperature: float, seed: int) -> dict:
    """Play `games` as white and `games` as black vs `opponent_name`."""
    nn_picker, nn_client = _build_nn_picker(weights, arch, device, sims, temperature, seed)
    if opponent_name == "material":
        opp_picker, _ = _build_material_picker()
    elif opponent_name == "random":
        opp_picker, _ = _build_random_picker()
    else:
        raise ValueError(f"unknown opponent: {opponent_name}")

    play_client = PyVariantClient()
    summary = {"white": {"w": 0, "b": 0, "d": 0}, "black": {"w": 0, "b": 0, "d": 0}}
    for i in range(games):
        outcome = play_game(nn_picker, opp_picker, play_client)
        if outcome == "white":
            summary["white"]["w"] += 1
        elif outcome == "black":
            summary["white"]["b"] += 1
        else:
            summary["white"]["d"] += 1
    for i in range(games):
        outcome = play_game(opp_picker, nn_picker, play_client)
        if outcome == "white":
            summary["black"]["w"] += 1
        elif outcome == "black":
            summary["black"]["b"] += 1
        else:
            summary["black"]["d"] += 1
    play_client.close()
    nn_client.close()
    return summary


def _winrate(side_summary: dict, nn_color: str) -> float:
    """`nn_color` ∈ {white, black}: returns NN's win fraction (draws count 0.5)."""
    w, b, d = side_summary["w"], side_summary["b"], side_summary["d"]
    n = w + b + d
    if n == 0:
        return 0.0
    nn_wins = w if nn_color == "white" else b
    return (nn_wins + 0.5 * d) / n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--opponent", choices=["material", "random"], default="material")
    p.add_argument("--games", type=int, default=8,
                   help="games per side (so total = 2*games)")
    p.add_argument("--sims", type=int, default=50, help="MCTS sims per nn move")
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.3,
                   help="sampling temperature on MCTS visits (0 = argmax, deterministic)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arch = dict(d_hidden=args.d_hidden, c_filters=args.c_filters, n_blocks=args.n_blocks)
    print(f"=== nn @ {args.weights} (sims={args.sims}, temp={args.temperature}) vs {args.opponent} ===")

    summary = _run(args.opponent, args.weights, arch, args.device, args.sims,
                   args.games, args.temperature, args.seed)

    w_wr = _winrate(summary["white"], "white")
    b_wr = _winrate(summary["black"], "black")
    n_per_side = sum(summary["white"].values())

    print(f"as white ({n_per_side} games): "
          f"W={summary['white']['w']} B={summary['white']['b']} D={summary['white']['d']}  "
          f"win-rate={w_wr:.0%}")
    print(f"as black ({n_per_side} games): "
          f"W={summary['black']['w']} B={summary['black']['b']} D={summary['black']['d']}  "
          f"win-rate={b_wr:.0%}")
    overall = (w_wr + b_wr) / 2
    print(f"overall win-rate: {overall:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
