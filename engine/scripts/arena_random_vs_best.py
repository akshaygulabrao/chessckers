"""One-off sanity arena: random mover vs the best net, 5 games each color.

Concrete bug-floor check — if the trained net can't crush a random mover,
something is badly broken (move-gen, encoding, sign of value, etc.).

Records each game's UCI move line (replayable via scripts/watch_game.py
--moves "...") and prints the final board so you can SEE the games.
"""
from __future__ import annotations

import argparse
import logging
import random

import torch

from chessckers_engine.checkpoints import load_scorer
from chessckers_engine.mcts_puct import pick_puct
from chessckers_engine.render_board import render_board
from chessckers_engine.selfplay_az import _outcome_from_state
from chessckers_engine.variant_py import PyVariantClient


def play_recorded(white_pick, black_pick, client, max_plies, render=False, delay=0.0,
                  start_fen=None):
    """Like evaluate.play_game but returns (outcome, uci_moves, final_fen).

    render=True prints the board after every ply (watch the game live)."""
    import time

    state = client.new_game(fen=start_fen)
    moves: list[str] = []
    ply = 0
    if render:
        print(render_board(state["fen"]), flush=True)
    while not state.get("status") and ply < max_plies:
        mover = state["turn"]
        picker = white_pick if mover == "white" else black_pick
        chosen = picker(state)
        if chosen is None:
            break
        moves.append(chosen["uci"])
        state = client.make_move(state["fen"], chosen["uci"])
        ply += 1
        if render:
            print(f"\n  ply {ply}: {mover} plays {chosen['uci']}", flush=True)
            print(render_board(state["fen"]), flush=True)
            if delay:
                time.sleep(delay)
    return _outcome_from_state(state), moves, state["fen"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    # The simplified training start: White's 8 pawns + king vs three 2-King towers.
    DEFAULT_START_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] w - - 0 1"
    p.add_argument("--net", default="weights/run/best.pt")
    p.add_argument("--start-fen", default=DEFAULT_START_FEN,
                   help="start position (default: the simplified training start)")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show-board", action="store_true",
                   help="render the final board of each game")
    p.add_argument("--render", action="store_true",
                   help="render the board after EVERY ply (watch the game live)")
    p.add_argument("--delay", type=float, default=0.0,
                   help="seconds to pause between plies when --render")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    client = PyVariantClient()
    model = load_scorer(args.net)
    model.eval()

    def net_pick(state):
        return pick_puct(state, client, model, n_sims=args.sims, c_puct=args.c_puct)

    def random_pick(state):
        legal = state.get("legalMoves") or []
        return rng.choice(legal) if legal else None

    rec = {"win": 0, "loss": 0, "draw": 0}
    by = {"as_white": dict(rec), "as_black": dict(rec)}
    half = args.games // 2
    for i in range(args.games):
        net_white = i < half
        print(f"\n=== game {i+1}/{args.games}  net={'White' if net_white else 'Black'} ===", flush=True)
        if net_white:
            outcome, moves, fen = play_recorded(net_pick, random_pick, client, args.max_plies,
                                                render=args.render, delay=args.delay,
                                                start_fen=args.start_fen)
            res = "win" if outcome == "white" else "loss" if outcome == "black" else "draw"
            by["as_white"][res] += 1
        else:
            outcome, moves, fen = play_recorded(random_pick, net_pick, client, args.max_plies,
                                                render=args.render, delay=args.delay,
                                                start_fen=args.start_fen)
            res = "win" if outcome == "black" else "loss" if outcome == "white" else "draw"
            by["as_black"][res] += 1
        rec[res] += 1
        print(f"\n=== game {i+1}/{args.games}  net={'White' if net_white else 'Black'}  "
              f"-> net {res.upper()} ({outcome}, {len(moves)} plies) ===", flush=True)
        print(f"  moves: {' '.join(moves)}")
        if args.show_board:
            print(render_board(fen))

    score = (rec["win"] + 0.5 * rec["draw"]) / max(args.games, 1)
    bw, bb = by["as_white"], by["as_black"]
    print(f"\n  Net:   {args.net}  (sims={args.sims})")
    print(f"  Games: {args.games}   Net score: {score:.3f}")
    print(f"  Overall  W/L/D = {rec['win']}/{rec['loss']}/{rec['draw']}")
    print(f"  As White W/L/D = {bw['win']}/{bw['loss']}/{bw['draw']}")
    print(f"  As Black W/L/D = {bb['win']}/{bb['loss']}/{bb['draw']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
