#!/usr/bin/env python
"""Watch a self-play game from any Chessckers FEN.

The trained net plays BOTH sides at --sims (default 400) MCTS sims/move. The
MOVE played is always the argmax of the visit counts (the "calculation");
exploration is injected only as root Dirichlet noise (--explore, default
0.30 = 30%), so different runs (or --seed) give varied games while each move
stays the search's best. --explore 0 = pure greedy/deterministic. Each ply
renders the 10x10 board live, then the net's raw WDL eval + moves-left estimate
(both from the mover's POV), then the engine's top 3 lines for that position:
MCTS policy probability (visits), value (mover's POV), and the principal
variation. Defaults to the latest ckpt.

  cd engine
  .venv/bin/python scripts/watch_game.py "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1"
  # options: --weights X.pt  --sims 400  --max-plies 200  --device cpu|mps  --delay 0.5
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

_ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../engine

TOP_N = 3       # candidate moves ("lines") to show per position
PV_MAX = 8      # plies of principal-variation continuation to show per line


def _resolve_weights(arg: str) -> list[str]:
    """Ordered checkpoint candidates, freshest first. The async/continuous
    trainer (train_continuous.py) writes weights/run/{weights.pt, iter-async-
    NNNN.pt}; weights.pt is the live rolling publish (newest but may be mid-
    write), so it leads and the numbered durable snapshots follow as fallbacks.
    base_wdl_v* are the WDL-arch bases. (The pre-WDL base_live.pt / iter-az-*.pt
    are deliberately excluded — old scalar value head, won't load here.)"""
    if arg:
        return [arg]
    run = os.path.join(_ENG, "weights/run")
    numbered = glob.glob(os.path.join(run, "iter-async-[0-9]*.pt"))
    iters = sorted(
        numbered,
        key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)),
        reverse=True,
    )
    cands = [
        os.path.join(run, "weights.pt"),
        *iters,
        os.path.join(_ENG, "weights/base_wdl_v2.pt"),
        os.path.join(_ENG, "weights/base_wdl_v1.pt"),
    ]
    found = [p for p in cands if os.path.exists(p)]
    if not found:
        raise SystemExit("no WDL weights found (weights/run/weights.pt, iter-async-*.pt, "
                         "or weights/base_wdl_v*.pt); pass --weights")
    return found


def _pv_ucis(child, max_len: int = PV_MAX) -> list[str]:
    """Principal variation starting at a root child: descend by most-visited
    child each ply. Returns [child's move, best reply, ...]. Stops at a
    terminal/unexpanded node or an unexplored (0-visit) child."""
    ucis: list[str] = []
    node = child
    for _ in range(max_len):
        if node is None or node.move_to_here is None:
            break
        ucis.append(node.move_to_here["uci"])
        if node.is_terminal or not node.children:
            break
        nxt = max(node.children.values(), key=lambda c: c.visits)
        if nxt.visits == 0:
            break
        node = nxt
    return ucis


def _print_net_eval(model, fen: str, turn: str, device: str) -> None:
    """Show the value head's raw WDL distribution and the moves-left head's
    plies-to-end estimate for this position (both from the mover's POV) — the
    network's un-searched read, distinct from the MCTS-backed `ev` per move."""
    import torch

    from chessckers_engine.encoding import encode_position

    pos = encode_position(fen).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model._position_embedding(pos)
        wdl = torch.softmax(model.value_head(emb), dim=-1).reshape(-1)
        mlh = float(model.moves_left_head(emb).reshape(()).item())
    w, d, lose = (100.0 * wdl[0].item(), 100.0 * wdl[1].item(), 100.0 * wdl[2].item())
    print(f"  {turn} net eval: W {w:4.1f}% / D {d:4.1f}% / L {lose:4.1f}%  moves-left ~{mlh:.0f}")


def _print_analysis(turn: str, result, n_sims: int, top_n: int = TOP_N) -> None:
    """Show the top-N moves the search considered at this position: MCTS policy
    probability (proportional to visits), value from the mover's perspective
    (-childQ), visit count, and the principal-variation continuation."""
    children = sorted(result.root.children.values(), key=lambda c: c.visits, reverse=True)
    if not children:
        return
    total = sum(c.visits for c in children) or 1
    print(f"  {turn} to move — top {min(top_n, len(children))} of {len(children)} ({n_sims} sims):")
    for i, c in enumerate(children[:top_n], 1):
        uci = c.move_to_here["uci"] if c.move_to_here else "?"
        pct = 100.0 * c.visits / total
        ev = -c.q or 0.0  # child.q is from the child's POV; negate -> mover's POV (avoid -0.00)
        pv = _pv_ucis(c)
        cont = " ".join(pv[1:]) if len(pv) > 1 else ("# mate" if c.is_terminal else "")
        print(f"    {i}. {uci:<14} {pct:5.1f}%  ev {ev:+.2f}  n={c.visits:<4} {cont}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch a greedy (argmax) self-play game from a FEN.")
    ap.add_argument("fen", help="Chessckers start FEN, e.g. '8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1'")
    ap.add_argument("--weights", default="", help="checkpoint .pt (default: latest weights/run/iter-az-*.pt)")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--explore", type=float, default=0.30,
                    help="root Dirichlet exploration-noise fraction (default 0.30 = 30 pct); the "
                         "played move stays argmax of visits. 0 = pure greedy/deterministic.")
    ap.add_argument("--seed", type=int, default=-1,
                    help="rng seed (default: random each run, so games vary)")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--device", default="cpu", help="cpu|mps|cuda (default cpu)")
    ap.add_argument("--delay", type=float, default=0.0, help="extra pause between plies, seconds")
    args = ap.parse_args()

    import torch
    from chessckers_engine.checkpoints import load_checkpoint
    from chessckers_engine.mcts_puct import run_mcts
    from chessckers_engine.model import ChesskersScorer
    from chessckers_engine.render_board import render_board
    from chessckers_engine.selfplay_az import _outcome_from_state
    from chessckers_engine.variant_py import PyVariantClient

    model = ChesskersScorer(d_hidden=256, c_filters=96, n_blocks=4).to(args.device).eval()
    weights = None
    last_err: Exception | None = None
    for cand in _resolve_weights(args.weights):  # freshest first; weights.pt may be mid-write
        try:
            load_checkpoint(model, cand)
            weights = cand
            break
        except Exception as e:  # noqa: BLE001 — try the next durable candidate
            last_err = e
    if weights is None:
        raise SystemExit(f"could not load any candidate checkpoint; last error: {last_err}")

    seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), "big")
    # run_mcts draws root Dirichlet noise from the GLOBAL torch RNG, so seed it
    # here: this makes --seed actually reproduce a game (and --seed -1 vary it).
    torch.manual_seed(seed)
    print(f"weights: {weights}\nsims: {args.sims} | device: {args.device} | "
          f"explore (root noise): {args.explore:.0%} | move pick: argmax | seed: {seed}\n")

    os.environ["CHESSCKERS_START_FEN"] = args.fen  # new_game() reads this
    client = PyVariantClient()
    state = client.new_game()
    alpha = 0.3 if args.explore > 0 else None  # AlphaZero-chess default concentration

    def show_board(ply: int, uci: str | None, fen: str) -> None:
        head = f"ply {ply}: {uci}" if uci else "start"
        print(f"\n=== {head} ===")
        print(render_board(fen))

    show_board(0, None, state["fen"])
    ply = 0
    while not state.get("status") and ply < args.max_plies:
        if not (state.get("legalMoves") or []):
            break
        result = run_mcts(
            state, client, model,
            n_sims=args.sims, c_puct=1.5,
            dirichlet_alpha=alpha,      # root exploration noise...
            dirichlet_eps=args.explore,  # ...at --explore fraction (30%)
        )
        _print_net_eval(model, state["fen"], state["turn"], args.device)  # raw net WDL + moves-left
        _print_analysis(state["turn"], result, args.sims)  # top-3 lines for this position
        chosen = result.chosen
        if chosen is None:
            break
        state = client.make_move(state["fen"], chosen["uci"])
        ply += 1
        show_board(ply, chosen["uci"], state["fen"])
        if args.delay:
            time.sleep(args.delay)

    status = state.get("status")
    if status:
        outcome = _outcome_from_state(state)
        print(f"\n######## {outcome.upper()} WINS ({status}) in {ply} plies ########"
              if outcome != "draw" else f"\n######## DRAW ({status}) in {ply} plies ########")
    else:
        print(f"\n######## UNFINISHED — stopped at {ply} plies (max-plies={args.max_plies}) ########")
    return 0


if __name__ == "__main__":
    sys.exit(main())
