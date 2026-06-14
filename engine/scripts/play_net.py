#!/usr/bin/env python
"""Play a human-vs-net game of Chessckers from any FEN.

YOU pick your moves from a numbered legal-move menu (so you never hand-type the
cadence/deploy UCI); the trained net replies via PUCT MCTS. Each ply renders the
10x10 board + the net's raw WDL eval, and on the net's turn its top lines. Loads
ANY checkpoint via its .arch.json sidecar (V1/V2/V4), like watch_game.py.

  cd engine
  .venv/bin/python scripts/play_net.py --color black --sims 200 --device mps
  .venv/bin/python scripts/play_net.py "<FEN>" --weights X.pt --color white
  # or play the LIVE fleet champion in one command:  cc play --color black

At your turn: type a move's number, or its raw UCI, 'u' to undo your last move,
'q' to quit. Options: --color white|black (the side YOU play) --sims 200
--explore 0 (net root noise; 0 = strongest) --device cpu|mps --weights X.pt
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # so `import watch_game` resolves regardless of cwd

# Reuse watch_game's net-loading + eval/analysis renderers (single source of truth;
# its heavy imports are lazy under main(), so importing it here is cheap).
from watch_game import (  # noqa: E402
    DEFAULT_START_FEN,
    _print_analysis,
    _print_net_eval,
    _resolve_weights,
)


def _ask_human(legal: list[dict]) -> str | None:
    """Numbered legal-move menu. Returns a chosen UCI, None to quit, or 'UNDO'."""
    print(f"  your {len(legal)} legal moves:")
    for i, m in enumerate(legal):
        print(f"    [{i:2}] {m['uci']}")
    ucis = {m["uci"] for m in legal}
    while True:
        sel = input("  pick # (or uci, u=undo, q=quit): ").strip()
        if sel in ("q", "Q", ""):
            return None
        if sel in ("u", "U"):
            return "UNDO"
        if sel.isdigit():
            i = int(sel)
            if 0 <= i < len(legal):
                return legal[i]["uci"]
            print("  out of range.")
            continue
        if sel in ucis:  # raw UCI
            return sel
        print(f"  not a legal move/selection: {sel!r}")


def _undo(history: list[dict], you: str) -> tuple[dict, int]:
    """Roll back to YOUR previous turn (drops the net's reply + your last move).
    Uses the FEN turn field as truth, so it's correct under White's double-move too."""
    if len(history) <= 1:
        print("  (nothing to undo)")
        return history[-1], len(history) - 1
    history.pop()  # drop current (your) state
    while len(history) > 1 and history[-1]["turn"] != you:
        history.pop()  # drop back through the net's move(s)
    print("  (undone)")
    return history[-1], len(history) - 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Play a human-vs-net Chessckers game.")
    ap.add_argument("fen", nargs="?", default=DEFAULT_START_FEN,
                    help=f"start FEN (default: the training start {DEFAULT_START_FEN!r})")
    ap.add_argument("--color", choices=["white", "black"], default="black",
                    help="the side YOU play (default black = the towers). The net plays the other.")
    ap.add_argument("--weights", default="",
                    help="checkpoint .pt (default: latest local weights/run/*; --latest for the fleet net)")
    ap.add_argument("--latest", action="store_true",
                    help="use the live fleet net (lczero-server/trainer/run1/weights.pt) if present locally")
    ap.add_argument("--sims", type=int, default=200, help="net MCTS sims/move (>=50 recommended)")
    ap.add_argument("--explore", type=float, default=0.0,
                    help="net root Dirichlet noise fraction (0 = strongest/greedy; >0 varies its play)")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--device", default="cpu", help="cpu|mps|cuda (default cpu)")
    args = ap.parse_args()

    from chessckers_engine.checkpoints import load_scorer
    from chessckers_engine.mcts_puct import run_mcts
    from chessckers_engine.render_board import render_board
    from chessckers_engine.selfplay_az import _outcome_from_state
    from chessckers_engine.variant_py import PyVariantClient

    model = weights = None
    last_err: Exception | None = None
    for cand in _resolve_weights(args.weights, args.latest):  # freshest first
        try:
            model = load_scorer(cand).to(args.device).eval()
            weights = cand
            break
        except Exception as e:  # noqa: BLE001 — try the next durable candidate
            last_err = e
    if model is None:
        raise SystemExit(f"could not load any checkpoint; last error: {last_err}")

    you = args.color
    net_color = "white" if you == "black" else "black"
    alpha = 0.3 if args.explore > 0 else None  # AlphaZero-chess concentration
    os.environ["CHESSCKERS_START_FEN"] = args.fen  # new_game() reads this
    client = PyVariantClient()
    state = client.new_game()
    print(f"weights: {weights}\nYOU play: {you} | net: {net_color} | sims: {args.sims} | "
          f"device: {args.device} | explore: {args.explore:.0%}\n")

    history = [state]  # FEN-state stack for undo
    ply = 0
    while not state.get("status") and ply < args.max_plies:
        legal = state.get("legalMoves") or []
        if not legal:
            break
        mover = state["turn"]
        print(f"\n=== ply {ply + 1} — {mover} to move"
              + (" (YOU)" if mover == you else " (net)") + " ===")
        print(render_board(state["fen"]))
        if state.get("check"):
            print("  ** CHECK **")
        _print_net_eval(model, state["fen"], mover, args.device)

        if mover == you:
            mv = _ask_human(legal)
            if mv is None:
                print("bye.")
                return 0
            if mv == "UNDO":
                state, ply = _undo(history, you)
                continue
            state = client.make_move(state["fen"], mv)
        else:
            result = run_mcts(state, client, model, n_sims=args.sims, c_puct=1.5,
                              dirichlet_alpha=alpha, dirichlet_eps=args.explore)
            _print_analysis(mover, result, args.sims)
            chosen = result.chosen
            if chosen is None:
                break
            print(f"  >> net plays: {chosen['uci']}")
            state = client.make_move(state["fen"], chosen["uci"])
        history.append(state)
        ply += 1

    print("\n" + render_board(state["fen"]))
    status = state.get("status")
    if status:
        outcome = _outcome_from_state(state)
        if outcome == "draw":
            print(f"\n######## DRAW ({status}) in {ply} plies ########")
        else:
            who = "YOU WIN" if outcome == you else "NET WINS"
            print(f"\n######## {who} — {outcome.upper()} ({status}) in {ply} plies ########")
    else:
        print(f"\n######## stopped at {ply} plies (max-plies={args.max_plies}) ########")
    return 0


if __name__ == "__main__":
    sys.exit(main())
