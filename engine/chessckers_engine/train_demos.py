"""Demonstration-bootstrapped training.

Loads human-played games from a JSONL file, extracts the (fen, move, target)
triples for the specified color's moves, and runs supervised training (using
the existing `train.train`) to imitate them. The trained checkpoint is
intended to be used as `--base` for the trainer (`train_continuous` / `train_az`),
giving AZ self-play a non-random starting policy that already encodes whatever
strategic patterns the human demonstrated.

CLI:
    uv run python -m chessckers_engine.train_demos \
        --games engine/games/games.jsonl \
        --color black \
        [--base path/to/checkpoint.pt] \
        [--out path/to/output.pt] \
        --epochs 20

Note: this trains the policy head only (against move-played targets) via
the M4-phase-1-style train(); the value head stays at random init. AZ
self-play will train the value head from outcomes once it kicks in.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from chessckers_engine.checkpoints import default_checkpoint_path, load_checkpoint
from chessckers_engine.demos import extract_examples, filter_games_for_color, load_games
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.train import save_checkpoint, train
from chessckers_engine.variant_py import PyVariantClient

log = logging.getLogger("chessckers_engine.train_demos")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True, help="path to games.jsonl")
    p.add_argument("--color", default="black", choices=["white", "black"])
    p.add_argument("--out", default=None,
                   help="output .pt path (default: weights/demo-bootstrap-<timestamp>.pt)")
    p.add_argument("--base", default=None,
                   help="warm-start from this .pt (default: random init)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    games = load_games(args.games)
    log.info("loaded %d games from %s", len(games), args.games)
    games = filter_games_for_color(games, args.color)
    log.info("kept %d games where %s was a player and the game finished", len(games), args.color)
    if not games:
        log.error("no usable games; nothing to train on")
        return 1

    # Game-outcome breakdown for the chosen color.
    outcomes = {"win": 0, "loss": 0, "draw": 0}
    for g in games:
        if g["outcome"] == args.color:
            outcomes["win"] += 1
        elif g["outcome"] == "draw":
            outcomes["draw"] += 1
        else:
            outcomes["loss"] += 1
    log.info("%s outcomes: %d wins, %d losses, %d draws", args.color, outcomes["win"], outcomes["loss"], outcomes["draw"])

    client = PyVariantClient()

    examples = extract_examples(games, args.color, client)
    log.info("extracted %d (fen, move, target) examples for %s", len(examples), args.color)
    if not examples:
        log.error("no examples extracted; nothing to train on")
        client.close()
        return 1

    torch.manual_seed(args.seed)
    model = ChesskersScorer()
    if args.base:
        log.info("warm-starting from %s", args.base)
        load_checkpoint(model, args.base)

    result = train(model, examples, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    log.info("training complete; final loss=%.4f", result.final_loss)

    out_path = Path(args.out) if args.out else (
        default_checkpoint_path().with_name(f"demo-bootstrap-{args.color}.pt")
    )
    save_checkpoint(model, out_path)
    log.info("saved demo-bootstrap checkpoint to %s", out_path)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
