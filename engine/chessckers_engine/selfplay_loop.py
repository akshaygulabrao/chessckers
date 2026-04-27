"""Iterated self-play training loop.

Each iteration:
  1. Play `games_per_iter` self-play games with the current model on both sides
     (stochastic policy via softmax(logits/τ)).
  2. Convert decisions → outcome-targeted training examples.
  3. Train the current model on those examples (MSE, reusing train.train()).
  4. Save the updated weights as `weights/iter-{N:03d}.pt`.
  5. Run a small eval vs random to produce a per-iteration win-rate datapoint.

CLI:
    uv run python -m chessckers_engine.selfplay_loop \
        --iterations 20 --games-per-iter 50 --epochs 3 \
        [--base path/to/start.pt]   # warm start; if omitted, random init
        [--temperature 1.0] [--temperature-final 0.2]  # anneal across iterations
        [--eval-games 10]            # games per iteration vs random for the table

Prints a table after each iteration:
    iter | examples | train_loss | nn vs random (W/D/L)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from chessckers_engine.checkpoints import DEFAULT_WEIGHTS_DIR
from chessckers_engine.evaluate import evaluate as run_eval
from chessckers_engine.evaluate import format_results
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.random_player import pick_random
from chessckers_engine.selfplay import decisions_to_examples, play_self_game, sample_move
from chessckers_engine.server_client import ServerClient
from chessckers_engine.train import save_checkpoint, train

log = logging.getLogger("chessckers_engine.selfplay_loop")


def _temperature_for(iter_idx: int, n_iters: int, t0: float, t_final: float) -> float:
    """Linear anneal from t0 (iter 0) to t_final (iter n_iters-1)."""
    if n_iters <= 1:
        return t0
    frac = iter_idx / (n_iters - 1)
    return t0 + (t_final - t0) * frac


def _make_nn_picker(model: ChesskersScorer):
    def picker(state):
        # τ=0 here: deterministic argmax for evaluation, regardless of training τ.
        return sample_move(model, state, rng=None, temperature=0.0)

    return picker


def _make_random_picker():
    def picker(state):
        return pick_random(state.get("legalMoves") or [])

    return picker


def run_iterations(
    model: ChesskersScorer,
    client: ServerClient,
    n_iters: int,
    games_per_iter: int,
    epochs: int,
    batch_size: int,
    lr: float,
    t_initial: float,
    t_final: float,
    eval_games: int,
    weights_dir: Path,
    seed: int = 0,
) -> list[dict]:
    """Run the iteration loop. Returns a list of per-iteration summary dicts."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    g = torch.Generator().manual_seed(seed)

    for it in range(n_iters):
        temp = _temperature_for(it, n_iters, t_initial, t_final)
        log.info("iter %d/%d: playing %d games at temperature=%.3f",
                 it + 1, n_iters, games_per_iter, temp)
        examples: list[dict] = []
        outcomes = {"white": 0, "black": 0, "draw": 0}
        for _ in range(games_per_iter):
            game = play_self_game(model, client, temperature=temp, rng=g)
            outcomes[game.outcome] += 1
            examples.extend(decisions_to_examples(game))

        log.info("iter %d/%d: collected %d examples; outcomes %s",
                 it + 1, n_iters, len(examples), outcomes)

        result = train(model, examples, epochs=epochs, batch_size=batch_size, lr=lr,
                       seed=seed + it, log_every=0)
        log.info("iter %d/%d: trained; final_loss=%.4f", it + 1, n_iters, result.final_loss)

        ckpt = weights_dir / f"iter-{it + 1:03d}.pt"
        save_checkpoint(model, ckpt)

        # Quick eval vs random both ways
        nn = _make_nn_picker(model)
        rnd = _make_random_picker()
        as_white = run_eval(nn, rnd, client, eval_games)
        as_black = run_eval(rnd, nn, client, eval_games)
        # Black wins (score from black's perspective) when 'black' field of black-as-second-arg counts
        # In as_black, NN is black; black wins → "black" key
        nn_white_score = (as_white["white"] + 0.5 * as_white["draw"]) / max(eval_games, 1)
        nn_black_score = (as_black["black"] + 0.5 * as_black["draw"]) / max(eval_games, 1)

        summary = {
            "iter": it + 1,
            "temperature": round(temp, 3),
            "examples": len(examples),
            "train_loss": round(result.final_loss, 4),
            "self_play": outcomes,
            "nn_as_white_vs_random": as_white,
            "nn_as_black_vs_random": as_black,
            "nn_white_score": round(nn_white_score, 3),
            "nn_black_score": round(nn_black_score, 3),
            "checkpoint": str(ckpt),
        }
        summaries.append(summary)

        log.info(
            "iter %d/%d done | self-play W/B/D=%d/%d/%d | nn(W)vs.random %d/%d/%d | nn(B)vs.random %d/%d/%d | loss=%.4f",
            it + 1, n_iters,
            outcomes["white"], outcomes["black"], outcomes["draw"],
            as_white["white"], as_white["black"], as_white["draw"],
            as_black["black"], as_black["white"], as_black["draw"],
            result.final_loss,
        )

    return summaries


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--games-per-iter", type=int, default=50)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0, help="initial sampling temperature")
    p.add_argument("--temperature-final", type=float, default=0.3, help="final sampling temperature")
    p.add_argument("--eval-games", type=int, default=10, help="games vs random per iteration")
    p.add_argument("--base", default=None, help="warm-start from this .pt file (default: random init)")
    p.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    client = ServerClient()
    try:
        client.new_game()
    except Exception as e:  # noqa: BLE001
        log.error("cannot reach API: %s", e)
        return 1

    torch.manual_seed(args.seed)
    model = ChesskersScorer()
    if args.base:
        log.info("warm-starting from %s", args.base)
        from chessckers_engine.checkpoints import load_checkpoint
        load_checkpoint(model, args.base)

    summaries = run_iterations(
        model=model,
        client=client,
        n_iters=args.iterations,
        games_per_iter=args.games_per_iter,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        t_initial=args.temperature,
        t_final=args.temperature_final,
        eval_games=args.eval_games,
        weights_dir=Path(args.weights_dir),
        seed=args.seed,
    )

    print("\n  iter | τ     | examples | train_loss | self-play W/B/D | nn(W)vs.rand | nn(B)vs.rand")
    print("  -----|-------|----------|-----------|-----------------|--------------|--------------")
    for s in summaries:
        sp = s["self_play"]
        w = s["nn_as_white_vs_random"]
        b = s["nn_as_black_vs_random"]
        print(
            f"  {s['iter']:4d} | {s['temperature']:.3f} | {s['examples']:8d} | "
            f"{s['train_loss']:9.4f} | {sp['white']:3d}/{sp['black']:3d}/{sp['draw']:3d}        | "
            f"{w['white']:2d}/{w['black']:2d}/{w['draw']:2d}        | "
            f"{b['black']:2d}/{b['white']:2d}/{b['draw']:2d}"
        )
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
