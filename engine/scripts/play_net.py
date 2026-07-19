#!/usr/bin/env python
"""Play a human-vs-net game of Chessckers from any FEN.

YOU pick your moves from a numbered legal-move menu (so you never hand-type the
cadence/deploy UCI). The net replies via the REAL akshay-chessckers-0 lc0 fork
(the default whenever a fork binary is found; .pt nets auto-export to .bin),
driven with FULL GAME HISTORY (`position fen <start> moves ...`) so it keeps its
search tree between its moves and sees the true game ply — the production
operating point (run22.md 07-17; the stateless per-FEN driving used before is a
different, White-collapsing one). `--mcts` forces the old in-repo Python PUCT
opponent, which also renders the net's WDL eval + top lines each ply.

  cd engine
  .venv/bin/python scripts/play_net.py --color black                # vs the fork @128v
  .venv/bin/python scripts/play_net.py "<FEN>" --weights X.pt --color white
  .venv/bin/python scripts/play_net.py --mcts --sims 200 --device mps
  # or play the LIVE fleet net in one command:  cc play --color black

At your turn: type a move's number, or its raw UCI, 'u' to undo your last move,
'q' to quit. Options: --color white|black (the side YOU play) --engine PATH
--visits 128 (do NOT use 800: the fork's UCI mode hard-crashes there — run22.md)
--mcts  --sims 200  --explore 0 (python-mode noise; 0 = strongest)
--device cpu|mps  --weights X.pt|X.bin
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
from engine_uci import DEFAULT_BINARY, UciEngine  # noqa: E402  (fork opponent)
from ladder import _ensure_bin  # noqa: E402  (.pt -> fork-loadable .bin on demand)


def _find_fork_binary() -> str:
    """First runnable fork build: box layout (fork is a sibling of engine), Mac
    layout (fork is a sibling of the chessckers repo), then the box default."""
    eng = os.path.dirname(_HERE)
    rel = ("akshay-chessckers-0", "build", "release", "akshay-chessckers-0")
    for p in (os.path.join(eng, "..", *rel),
              os.path.join(eng, "..", "..", *rel),
              DEFAULT_BINARY):
        p = os.path.abspath(p)
        if os.access(p, os.X_OK):
            return p
    return ""


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
    ap.add_argument("--engine", default="",
                    help="fork binary for the net's moves (default: auto-detect the "
                         "sibling build, then the box path). Only used without --mcts.")
    ap.add_argument("--mcts", action="store_true",
                    help="use the in-repo Python PUCT opponent instead of the fork "
                         "(renders WDL eval + top-line panels; --sims/--explore/--device)")
    ap.add_argument("--visits", type=int, default=128,
                    help="fork nodes/move (default 128 = the gate operating point; do "
                         "NOT use 800 — the fork's UCI mode hard-crashes, run22.md)")
    ap.add_argument("--sims", type=int, default=200, help="python-mode MCTS sims/move")
    ap.add_argument("--explore", type=float, default=0.0,
                    help="python-mode root Dirichlet noise (0 = strongest/greedy)")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--device", default="cpu", help="python-mode device: cpu|mps|cuda")
    args = ap.parse_args()

    from chessckers_engine.render_board import render_board
    from chessckers_engine.selfplay_az import _outcome_from_state
    from chessckers_engine.variant_py import PyVariantClient

    engine_bin = "" if args.mcts else (args.engine or _find_fork_binary())
    if not args.mcts and not engine_bin:
        print("(no fork binary found — falling back to the Python MCTS opponent)")

    model = weights = None
    last_err: Exception | None = None
    if engine_bin:
        for cand in _resolve_weights(args.weights, args.latest):  # freshest first
            try:
                weights = cand if cand.endswith(".bin") else _ensure_bin(cand)
                break
            except Exception as e:  # noqa: BLE001 — try the next durable candidate
                last_err = e
        if weights is None:
            raise SystemExit(f"could not resolve a net .bin; last error: {last_err}")
    else:
        from chessckers_engine.checkpoints import load_scorer
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
    eng = None
    if engine_bin:
        eng = UciEngine(weights, binary=engine_bin, visits=args.visits)
        eng.new_game()
        opp = f"lc0 fork @{args.visits}v, history-driven ({engine_bin})"
    else:
        opp = f"python MCTS {args.sims} sims on {args.device} | explore {args.explore:.0%}"
    print(f"weights: {weights}\nYOU play: {you} | net: {net_color} | opponent: {opp}\n")

    history = [state]  # FEN-state stack for undo
    moves: list[str] = []  # UCI history from args.fen (fork driving + undo, both modes)
    ply = 0
    try:
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
            if model is not None:
                _print_net_eval(model, state["fen"], mover, args.device)

            if mover == you:
                mv = _ask_human(legal)
                if mv is None:
                    print("bye.")
                    return 0
                if mv == "UNDO":
                    state, ply = _undo(history, you)
                    del moves[len(history) - 1:]  # keep the UCI history aligned
                    continue
                state = client.make_move(state["fen"], mv)
                moves.append(mv)
            elif eng is not None:
                try:
                    mv = eng.bestmove(args.fen, moves=moves)
                except RuntimeError as e:
                    # Intermittent fork crash: full-history driving makes this
                    # lossless — respawn and replay the same game.
                    print(f"  (engine died — restarting: {str(e).splitlines()[0]})")
                    eng.restart()
                    eng.new_game()
                    mv = eng.bestmove(args.fen, moves=moves)
                if mv is None:
                    break
                if mv not in {m["uci"] for m in legal}:
                    raise SystemExit(
                        f"fork played {mv!r} at ply {ply + 1}, which PyVariant says is "
                        "illegal — rules-parity bug; capture this FEN + history and "
                        "run cc verify-chunks")
                print(f"  >> net plays: {mv}")
                state = client.make_move(state["fen"], mv)
                moves.append(mv)
            else:
                from chessckers_engine.mcts_puct import run_mcts
                result = run_mcts(state, client, model, n_sims=args.sims, c_puct=1.5,
                                  dirichlet_alpha=alpha, dirichlet_eps=args.explore)
                _print_analysis(mover, result, args.sims)
                chosen = result.chosen
                if chosen is None:
                    break
                print(f"  >> net plays: {chosen['uci']}")
                state = client.make_move(state["fen"], chosen["uci"])
                moves.append(chosen["uci"])
            history.append(state)
            ply += 1
    finally:
        if eng is not None:
            eng.close()

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
