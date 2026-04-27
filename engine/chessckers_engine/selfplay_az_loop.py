"""AlphaZero-style iteration loop for Chessckers.

Each iteration:
  1. Play `games_per_iter` self-play games using PUCT MCTS at every move.
  2. Convert each game's records into AZExamples (policy + value targets).
  3. Train the model on the combined examples (dual loss).
  4. Save a numbered checkpoint to `weights/iter-az-{N:03d}.pt`.
  5. Quick eval vs random both ways for a per-iteration win-rate datapoint.

CLI:
    uv run python -m chessckers_engine.selfplay_az_loop \
        --iterations 5 --games-per-iter 8 --sims 25 --epochs 3 \
        [--base path/to/start.pt]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from chessckers_engine.checkpoints import DEFAULT_WEIGHTS_DIR, load_checkpoint
from chessckers_engine.evaluate import evaluate as run_eval
from chessckers_engine.mcts_puct import pick_puct
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.random_player import pick_random
from chessckers_engine.selfplay_az import az_game_to_examples, play_az_game
from chessckers_engine.server_client import ServerClient
from chessckers_engine.train_az import save_checkpoint, train_az

log = logging.getLogger("chessckers_engine.selfplay_az_loop")


def _temperature_for(iter_idx: int, n_iters: int, t0: float, t_final: float) -> float:
    if n_iters <= 1:
        return t0
    return t0 + (t_final - t0) * (iter_idx / (n_iters - 1))


def _make_puct_picker(model: ChesskersScorer, client: ServerClient, n_sims: int):
    def picker(state):
        return pick_puct(state, client, model, n_sims=n_sims)

    return picker


def _make_random_picker():
    def picker(state):
        return pick_random(state.get("legalMoves") or [])

    return picker


def run_az_iterations(
    model: ChesskersScorer,
    client: ServerClient,
    n_iters: int,
    games_per_iter: int,
    n_sims: int,
    c_puct: float,
    epochs: int,
    lr: float,
    t_initial: float,
    t_final: float,
    eval_games: int,
    weights_dir: Path,
    seed: int = 0,
) -> list[dict]:
    weights_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    g = torch.Generator().manual_seed(seed)

    for it in range(n_iters):
        temp = _temperature_for(it, n_iters, t_initial, t_final)
        log.info("iter %d/%d: playing %d games (sims=%d, τ=%.3f)",
                 it + 1, n_iters, games_per_iter, n_sims, temp)
        outcomes = {"white": 0, "black": 0, "draw": 0}
        examples = []
        for gi in range(games_per_iter):
            game = play_az_game(model, client, n_sims=n_sims, c_puct=c_puct, temperature=temp, rng=g)
            outcomes[game.outcome] += 1
            examples.extend(az_game_to_examples(game))
            log.info("  game %d/%d done: %s, %d records", gi + 1, games_per_iter, game.outcome, len(game.records))

        log.info("iter %d/%d: %d examples; outcomes %s",
                 it + 1, n_iters, len(examples), outcomes)
        result = train_az(model, examples, epochs=epochs, lr=lr, seed=seed + it, log_every=0)
        ckpt = weights_dir / f"iter-az-{it + 1:03d}.pt"
        save_checkpoint(model, ckpt)

        # eval vs random both ways
        puct_pick = _make_puct_picker(model, client, n_sims)
        rnd = _make_random_picker()
        as_white = run_eval(puct_pick, rnd, client, eval_games)
        as_black = run_eval(rnd, puct_pick, client, eval_games)

        summary = {
            "iter": it + 1,
            "temperature": round(temp, 3),
            "examples": len(examples),
            "policy_loss": round(result.epoch_losses[-1]["policy"], 4),
            "value_loss": round(result.epoch_losses[-1]["value"], 4),
            "self_play": outcomes,
            "puct_as_white_vs_random": as_white,
            "puct_as_black_vs_random": as_black,
            "checkpoint": str(ckpt),
        }
        summaries.append(summary)
        log.info(
            "iter %d/%d done | self-play W/B/D=%d/%d/%d | puct(W)vs.rand %d/%d/%d | puct(B)vs.rand %d/%d/%d | policy=%.4f value=%.4f",
            it + 1, n_iters,
            outcomes["white"], outcomes["black"], outcomes["draw"],
            as_white["white"], as_white["black"], as_white["draw"],
            as_black["black"], as_black["white"], as_black["draw"],
            result.epoch_losses[-1]["policy"], result.epoch_losses[-1]["value"],
        )

    return summaries


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--games-per-iter", type=int, default=8)
    p.add_argument("--sims", type=int, default=25, help="PUCT simulations per move")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-final", type=float, default=0.3)
    p.add_argument("--eval-games", type=int, default=5)
    p.add_argument("--base", default=None)
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
        load_checkpoint(model, args.base)

    summaries = run_az_iterations(
        model=model,
        client=client,
        n_iters=args.iterations,
        games_per_iter=args.games_per_iter,
        n_sims=args.sims,
        c_puct=args.c_puct,
        epochs=args.epochs,
        lr=args.lr,
        t_initial=args.temperature,
        t_final=args.temperature_final,
        eval_games=args.eval_games,
        weights_dir=Path(args.weights_dir),
        seed=args.seed,
    )

    print("\n  iter | τ     | examples | policy | value  | self-play W/B/D | puct(W)vs.rand | puct(B)vs.rand")
    print("  -----|-------|----------|--------|--------|-----------------|----------------|----------------")
    for s in summaries:
        sp = s["self_play"]
        w = s["puct_as_white_vs_random"]
        b = s["puct_as_black_vs_random"]
        print(
            f"  {s['iter']:4d} | {s['temperature']:.3f} | {s['examples']:8d} | "
            f"{s['policy_loss']:6.3f} | {s['value_loss']:6.3f} | "
            f"{sp['white']:3d}/{sp['black']:3d}/{sp['draw']:3d}        | "
            f"{w['white']:2d}/{w['black']:2d}/{w['draw']:2d}          | "
            f"{b['black']:2d}/{b['white']:2d}/{b['draw']:2d}"
        )
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
