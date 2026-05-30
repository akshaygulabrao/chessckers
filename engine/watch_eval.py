"""Play ONE eval game from the endgame start position and, at every ply, show:
  - the 10x10 board (last move overlaid as a numbered path),
  - the network's value head output for the side to move,
  - the network's policy prior for EVERY legal move (and the MCTS visit % if
    --sims > 0), sorted, with the move actually played marked '*'.

Defaults: the net plays Black (the e6 tower), White (the lone king) plays
random — this is the `vs.rand B` detector matchup. Use --white net for a
net-vs-net (self-play) eval game.

Usage:
    python watch_eval.py --weights weights/endgame/iter-az-005.pt
    python watch_eval.py                      # fresh-init (random) net
    python watch_eval.py --white net --sims 50
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ENDGAME_FEN = "8/8/4p3/8/8/8/8/4K3[e6:ssss] b - - 0 1"


def _fmt_move(m: dict) -> str:
    tags = []
    if m.get("capture") is not None:
        tags.append("x")
    if m.get("waypoints"):
        tags.append(f"chain[{len(m['waypoints'])}]")
    if m.get("deployCount") is not None:
        tags.append(f"deploy{m['deployCount']}")
    if m.get("demotionsRequired") is not None:
        tags.append(f"charge{m['demotionsRequired']}")
    return m.get("uci", "?") + (f" {','.join(tags)}" if tags else "")


def _move_path(m: dict | None) -> list[str] | None:
    if m is None:
        return None
    wps = m.get("waypoints") or []
    if not wps and m.get("capture") is None:
        return None
    return [m["from"], *wps, m["to"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None, help="checkpoint (default: fresh-init random net)")
    ap.add_argument("--fen", default=os.environ.get("CHESSCKERS_START_FEN", ENDGAME_FEN))
    ap.add_argument("--white", choices=["random", "net"], default="random",
                    help="who plays White, the lone king (default: random)")
    ap.add_argument("--sims", type=int, default=50,
                    help="MCTS sims for the net's move + visit display (0 = raw policy argmax)")
    ap.add_argument("--max-plies", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0, help="seed for random White")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-hidden", type=int, default=256)
    ap.add_argument("--model-filters", type=int, default=96)
    ap.add_argument("--model-blocks", type=int, default=4)
    args = ap.parse_args()

    import random

    from chessckers_engine.checkpoints import load_checkpoint
    from chessckers_engine.encoding import encode_move, encode_position
    from chessckers_engine.mcts_puct import run_mcts
    from chessckers_engine.model import ChesskersScorer
    from chessckers_engine.random_player import pick_random
    from chessckers_engine.render_board import render_board
    from chessckers_engine.variant_py import PyVariantClient

    device = torch.device(args.device)
    model = ChesskersScorer(d_hidden=args.model_hidden, c_filters=args.model_filters,
                            n_blocks=args.model_blocks).to(device).eval()
    tag = "fresh-init (random) net"
    if args.weights:
        load_checkpoint(model, args.weights)
        tag = args.weights

    client = PyVariantClient()
    rng = random.Random(args.seed)

    state = client.new_game(args.fen)
    print(f"eval game — net={tag} | White={args.white} | sims={args.sims}")
    print(f"start: {args.fen}\n")

    last: dict | None = None
    ply = 0
    while not state.get("status") and ply < args.max_plies:
        legal = list(state.get("legalMoves") or [])
        if not legal:
            break
        stm = state["turn"]
        net_controls = (stm == "black") or (stm == "white" and args.white == "net")

        print(f"\n========== ply {ply} — {stm} to move ==========")
        print(render_board(state["fen"], path=_move_path(last)))
        if last is not None:
            print(f"(last: {_fmt_move(last)})")

        # Network's view of this position: value + policy prior for every move.
        pos_t = encode_position(state["fen"]).unsqueeze(0).to(device)
        move_t = torch.stack([encode_move(m) for m in legal]).to(device)
        with torch.no_grad():
            logits, value = model.policy_and_value(pos_t, move_t)
            priors = torch.softmax(logits, dim=0).cpu().tolist()
        print(f"net value (from {stm} POV): {float(value):+.3f}")

        # MCTS visit distribution (also picks the net's move when net controls).
        visits = None
        chosen = None
        if args.sims > 0:
            res = run_mcts(state, client, model, n_sims=args.sims, c_puct=1.5)
            visits = res.visit_distribution
            total = sum(visits.values()) or 1
            if net_controls:
                chosen = res.chosen
        if net_controls and chosen is None:  # sims==0 -> raw-policy argmax
            chosen = max(zip(legal, priors), key=lambda r: r[1])[0]
        if not net_controls:
            chosen = pick_random(legal, rng)

        rows = sorted(zip(legal, priors), key=lambda r: r[1], reverse=True)
        print(f"all {len(rows)} legal moves (policy prior" + (", mcts visit%" if visits else "") + "):")
        for m, p in rows:
            mark = "*" if chosen is not None and m["uci"] == chosen["uci"] else " "
            line = f"  {mark} {p * 100:5.1f}%  {_fmt_move(m)}"
            if visits is not None:
                vc = visits.get(m["uci"], 0)
                line += f"   visits {vc:>3} ({vc / total * 100:4.1f}%)"
            print(line)
        print(f"-> {stm} plays: {_fmt_move(chosen)}"
              + ("" if net_controls else "  [random]"))

        state = client.make_move(state["fen"], chosen["uci"])
        last = chosen
        ply += 1

    print(f"\n========== final (ply {ply}) ==========")
    print(render_board(state["fen"], path=_move_path(last)))
    winner = state.get("winner")
    print(f"\nstatus={state.get('status')}  winner={winner}  plies={ply}")
    if winner == "black":
        print("=> BLACK (the tower) won — captured the king.")
    elif winner == "white":
        print("=> WHITE won — the king captured the tower (Black blundered).")
    else:
        print("=> draw (ply cap) — Black failed to convert.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
