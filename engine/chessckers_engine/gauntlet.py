"""Iter-vs-iter gauntlet: pit two PUCT-NN agents against each other over N
games (half each color) and report W/D/L from the challenger's perspective.

Two modes:
  - Pairwise: --challenger PATH --champion PATH
  - Ladder:   --ladder-dir DIR     (matches iter-az-N vs iter-az-{N-1}
                                    for every consecutive pair found)

The point: training metrics (loss curves, vs-random win-rate) can lie, but
"is the new model stronger than the previous one?" is the gold-standard
question for AlphaZero-style training. If iter-N consistently beats
iter-{N-1} > 50%, the loop is doing its job.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.evaluate import play_game
from chessckers_engine.mcts_puct import pick_puct
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.server_client import ServerClient
from chessckers_engine.variant_py import PyVariantClient

log = logging.getLogger("chessckers_engine.gauntlet")

Picker = Callable[[dict], dict | None]


def _load_model(path: Path) -> ChesskersScorer:
    model = ChesskersScorer()
    load_checkpoint(model, path)
    model.eval()
    return model


def _make_puct_picker(model: ChesskersScorer, client: ServerClient,
                      n_sims: int, c_puct: float) -> Picker:
    def picker(state: dict) -> dict | None:
        return pick_puct(state, client, model, n_sims=n_sims, c_puct=c_puct)
    return picker


def _empty_record() -> dict[str, int]:
    return {"win": 0, "loss": 0, "draw": 0}


def gauntlet_match(
    challenger: ChesskersScorer,
    champion: ChesskersScorer,
    client: ServerClient,
    n_games: int,
    n_sims: int,
    c_puct: float,
    max_plies: int = 400,
) -> dict:
    """Play `n_games` between challenger and champion. Half games the
    challenger plays White, half plays Black. Returns counts from the
    challenger's perspective with per-color breakdown."""
    challenger_pick = _make_puct_picker(challenger, client, n_sims, c_puct)
    champion_pick = _make_puct_picker(champion, client, n_sims, c_puct)

    overall = _empty_record()
    by_color = {"as_white": _empty_record(), "as_black": _empty_record()}

    for i in range(n_games):
        challenger_is_white = (i % 2 == 0)
        if challenger_is_white:
            outcome = play_game(challenger_pick, champion_pick, client, max_plies=max_plies)
            res = "win" if outcome == "white" else "loss" if outcome == "black" else "draw"
            by_color["as_white"][res] += 1
        else:
            outcome = play_game(champion_pick, challenger_pick, client, max_plies=max_plies)
            res = "win" if outcome == "black" else "loss" if outcome == "white" else "draw"
            by_color["as_black"][res] += 1
        overall[res] += 1
        log.info(
            "  game %d/%d (challenger=%s) -> challenger %s  (running: W=%d L=%d D=%d)",
            i + 1, n_games, "white" if challenger_is_white else "black",
            res, overall["win"], overall["loss"], overall["draw"],
        )

    score = (overall["win"] + 0.5 * overall["draw"]) / max(n_games, 1)
    return {"overall": overall, "by_color": by_color, "score": score, "games": n_games}


def format_match(challenger_name: str, champion_name: str, result: dict) -> str:
    o = result["overall"]
    bw, bb = result["by_color"]["as_white"], result["by_color"]["as_black"]
    n = result["games"]
    return (
        f"\n  Challenger: {challenger_name}\n"
        f"  Champion:   {champion_name}\n"
        f"  Games: {n}   Score (challenger's): {result['score']:.3f}\n"
        f"  Overall  W/L/D = {o['win']}/{o['loss']}/{o['draw']}\n"
        f"  As White W/L/D = {bw['win']}/{bw['loss']}/{bw['draw']}\n"
        f"  As Black W/L/D = {bb['win']}/{bb['loss']}/{bb['draw']}\n"
    )


def run_ladder(
    ladder_dir: Path,
    client: ServerClient,
    games_per_match: int,
    n_sims: int,
    c_puct: float,
    max_plies: int = 400,
) -> list[dict]:
    """For each consecutive iter-az-N vs iter-az-{N-1} pair in `ladder_dir`,
    run a gauntlet_match and collect per-pair scores. Returns a list of dicts
    suitable for tabulation."""
    ckpts = sorted(ladder_dir.glob("iter-az-*.pt"))
    if len(ckpts) < 2:
        log.error("need ≥2 iter-az-*.pt in %s; found %d", ladder_dir, len(ckpts))
        return []

    rows: list[dict] = []
    for i in range(1, len(ckpts)):
        champ_path, chal_path = ckpts[i - 1], ckpts[i]
        log.info("ladder rung %d/%d: %s vs %s",
                 i, len(ckpts) - 1, chal_path.name, champ_path.name)
        champ = _load_model(champ_path)
        chal = _load_model(chal_path)
        result = gauntlet_match(chal, champ, client,
                                n_games=games_per_match, n_sims=n_sims,
                                c_puct=c_puct, max_plies=max_plies)
        rows.append({
            "challenger": chal_path.name,
            "champion": champ_path.name,
            **result["overall"],
            "as_white_w": result["by_color"]["as_white"]["win"],
            "as_white_l": result["by_color"]["as_white"]["loss"],
            "as_white_d": result["by_color"]["as_white"]["draw"],
            "as_black_w": result["by_color"]["as_black"]["win"],
            "as_black_l": result["by_color"]["as_black"]["loss"],
            "as_black_d": result["by_color"]["as_black"]["draw"],
            "score": result["score"],
            "games": result["games"],
        })
    return rows


def format_ladder(rows: list[dict]) -> str:
    if not rows:
        return "  (no rungs)\n"
    lines = ["", "  rung                                 W   L   D   score   verdict"]
    lines.append("  " + "-" * 70)
    for r in rows:
        # short label e.g. "iter-az-002 vs 001"
        chal = r["challenger"].replace("iter-az-", "").replace(".pt", "")
        champ = r["champion"].replace("iter-az-", "").replace(".pt", "")
        label = f"iter-{chal} vs iter-{champ}"
        verdict = ("improved" if r["score"] > 0.55
                   else "regressed" if r["score"] < 0.45
                   else "noisy / no signal")
        lines.append(
            f"  {label:36s}  {r['win']:3d} {r['loss']:3d} {r['draw']:3d}   "
            f"{r['score']:.3f}   {verdict}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    from chessckers_engine.runtime import setup_logging
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--challenger", help="Path to .pt for challenger (newer model)")
    p.add_argument("--champion", help="Path to .pt for champion (older model)")
    p.add_argument("--ladder-dir", help="Run ladder over consecutive iter-az-*.pt in this dir")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--sims", type=int, default=25)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--api-url", default="http://localhost:8080")
    p.add_argument("--use-server", action="store_true",
                   help="use the scalachess HTTP server instead of the in-process Python variant")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not (args.ladder_dir or (args.challenger and args.champion)):
        p.error("must provide either --ladder-dir or both --challenger and --champion")

    torch.manual_seed(args.seed)
    if args.use_server:
        client = ServerClient(base_url=args.api_url)
        try:
            client.new_game()
        except Exception as e:  # noqa: BLE001
            log.error("cannot reach API at %s: %s", args.api_url, e)
            return 1
    else:
        client = PyVariantClient()

    if args.ladder_dir:
        rows = run_ladder(Path(args.ladder_dir), client,
                          games_per_match=args.games, n_sims=args.sims,
                          c_puct=args.c_puct, max_plies=args.max_plies)
        print(format_ladder(rows))
    else:
        chal = _load_model(Path(args.challenger))
        champ = _load_model(Path(args.champion))
        result = gauntlet_match(chal, champ, client,
                                n_games=args.games, n_sims=args.sims,
                                c_puct=args.c_puct, max_plies=args.max_plies)
        print(format_match(args.challenger, args.champion, result))

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
