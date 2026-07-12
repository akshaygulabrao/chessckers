#!/usr/bin/env python3
"""Gauntlet: play the CURRENT net against all previous snapshots, oldest→newest.

Answers "is the live net actually stronger than the nets that came before it?" —
the oracle-free strength/regression check the arena gate can't give you (the gate
only compares each new net to its immediate predecessor, and on this fleet the
lenient calcElo>-20 gate rubber-stamps ~everything). Plays the current net
(`weights.pt`) vs each sampled checkpoint from the run's `iter-async-*.pt` lineage,
both colors, via PUCT MCTS on the box, and prints the current net's score + Elo
lead vs each opponent, a strength curve, and a REGRESSION flag if any older net
turns out stronger than current.

  cc gauntlet                          # current vs ~6 sampled snapshots (~10-20 min)
  cc gauntlet --n 16 --games 6 --sims 200
  cc gauntlet --all                    # vs EVERY snapshot (hours)
  cc gauntlet a.pt b.pt                 # current vs explicit nets
options: --run-dir DIR  --current PATH  --n N  --all  --games G  --sims S
         --c-puct 1.5  --max-plies 160  --start-fen FEN  --device auto|cuda|mps|cpu  --seed 0
         --temperature 1.0  --temp-plies 20

Games are diversified by sampling moves from the visit distribution at
--temperature for the first --temp-plies plies (then argmax) — without it, every
"game" between the same two nets from the fixed start is the SAME deterministic
game, and the verdict is vacuous (the run-14 postmortem's gauntlet bug).
--temperature 0 restores the old deterministic behavior.

SLOW: pure-Python PyVariant MCTS is CPU-bound (~tens of evals/s) and shares the box
with the live fleet, so even modest runs take many minutes — background it, or cut
--sims/--games/--n. (Same per-game cost as `cc ladder`.)

Note: uses the trainer's `iter-async-*.pt` checkpoint lineage as the stand-in for
"previous nets" — the published .bin champions aren't loadable in the Python MCTS
path. `weights.pt` is the live EMA net = the latest published best.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

# Default to the live fleet run dir. lczero-server is a SIBLING of engine on the
# box (/workspace/chessckers/{engine,lczero-server}) but two levels up on the Mac
# (engine nested in chessckers/, lczero-server its sibling). Pick whichever exists.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENG = os.path.dirname(_HERE)
_SERVER_DIR = next(
    (p for p in (os.path.join(_ENG, "..", "lczero-server"),
                 os.path.join(_ENG, "..", "..", "lczero-server"))
     if os.path.isdir(p)),
    os.path.join(_ENG, "..", "lczero-server"),
)
_DEFAULT_RUN_DIR = os.path.join(_SERVER_DIR, "trainer", "run1")
sys.path.insert(0, _HERE)  # so `import watch_game` resolves regardless of cwd
import _run_ident  # noqa: E402  (RUN_NAME for the header)
from watch_game import DEFAULT_START_FEN  # noqa: E402  (the training start FEN, read from the fork)


def _label(path: str) -> str:
    """Short label: 'i<iter>' for a snapshot, 'best' for weights.pt, else basename."""
    b = os.path.basename(path)
    m = re.search(r"iter-async-0*(\d+)\.pt$", b)
    if m:
        return f"i{m.group(1)}"
    if b == "weights.pt":
        return "best"
    return b.replace(".pt", "")[:8]


def _iter_index(path: str) -> int | None:
    m = re.search(r"iter-async-0*(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def pick_opponents(run_dir, n, use_all, explicit, current_path):
    """Opponent net paths: explicit if given, else the iter-async lineage (every one
    with --all, else N sampled evenly oldest→newest), minus the current net itself."""
    if explicit:
        return explicit
    paths = [p for p in glob.glob(os.path.join(run_dir, "iter-async-*.pt"))
             if _iter_index(p) is not None]
    paths.sort(key=_iter_index)
    cur = os.path.abspath(current_path)
    paths = [p for p in paths if os.path.abspath(p) != cur]
    if not paths:
        raise SystemExit(f"gauntlet: no iter-async-*.pt under {run_dir} (pass explicit nets or --run-dir)")
    if not use_all and len(paths) > n:
        idx = sorted({round(k * (len(paths) - 1) / (n - 1)) for k in range(n)})
        paths = [paths[i] for i in idx]
    return paths


def play_game(white_model, black_model, client, pick, sims, cpuct, max_plies, start_fen,
              temperature=0.0, temp_plies=0) -> tuple[str, bool]:
    """One game from start_fen; returns ('white'|'black'|'draw', truncated), where
    truncated=True means it hit the ply cap with no win condition (so the 'draw' is
    a timeout, not a real draw). For the first temp_plies plies moves are SAMPLED
    from visits at `temperature` (game diversity); after that, argmax."""
    from chessckers_engine.selfplay_az import _outcome_from_state
    state = client.new_game(fen=start_fen)
    ply = 0
    while not state.get("status") and ply < max_plies:
        model = white_model if state["turn"] == "white" else black_model
        temp = temperature if ply < temp_plies else 0.0
        chosen = pick(state, client, model, n_sims=sims, c_puct=cpuct, temperature=temp)
        if chosen is None:
            break
        state = client.make_move(state["fen"], chosen["uci"])
        ply += 1
    truncated = not state.get("status")   # exited on the ply cap with no terminal result
    return _outcome_from_state(state), truncated


def _elo(score: float) -> float:
    """Elo lead implied by a score fraction in [0,1] (capped at ±800)."""
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return max(-800.0, min(800.0, -400.0 * math.log10(1.0 / score - 1.0)))


def render(rows, agg_pts, agg_g, curve=True):
    """Per-opponent table (+ optional strength sparkline) + regression verdict."""
    w = max(6, max(len(r[0]) for r in rows))
    print(f"\n  {'opponent':>{w}}   W-D-L    cur%   Elo±")
    print("  " + "─" * (w + 23))
    for lbl, ww, dd, ll, sc in rows:
        print(f"  {lbl:>{w}}  {ww:>2}-{dd}-{ll:<2}  {100 * sc:>4.0f}%  {_elo(sc):>+5.0f}")
    if curve:
        blocks = "▁▂▃▄▅▆▇█"
        spark = "".join(blocks[min(7, int(sc * 7.999))] for *_, sc in rows)
        print(f"\n  strength curve (current's score vs opponent, oldest→newest):\n    {spark}")
        print("    (full bar = current crushes that old net; mid bar ≈ 50% = indistinguishable)")
    agg = 100 * agg_pts / agg_g if agg_g else 0
    print(f"  aggregate: current scored {agg:.0f}% over {int(agg_g)} games vs the field")
    regress = [(lbl, sc) for lbl, _, _, _, sc in rows if sc < 0.5]
    if regress:
        worst = min(regress, key=lambda x: x[1])
        print(f"  \033[31m⚠ REGRESSION\033[0m: current is weaker than {len(regress)} snapshot(s) — "
              f"worst vs {worst[0]} ({100 * worst[1]:.0f}%, {_elo(worst[1]):+.0f} Elo). "
              f"A later net went backwards and the lenient gate promoted it.")
    else:
        print("  \033[32m✓ no regression\033[0m: current scored ≥50% vs every snapshot tested.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Current-net-vs-all-previous gauntlet (strength + regression).")
    ap.add_argument("nets", nargs="*", help="explicit opponent .pt paths (else sample the run dir)")
    ap.add_argument("--run-dir", default=_DEFAULT_RUN_DIR)
    ap.add_argument("--current", default="", help="current net path (default: <run-dir>/weights.pt)")
    ap.add_argument("--n", type=int, default=6, help="snapshots to sample as opponents (ignored with --all/explicit)")
    ap.add_argument("--all", action="store_true", help="play EVERY iter-async snapshot, not a sample")
    ap.add_argument("--games", type=int, default=2, help="games per opponent (colors split)")
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--max-plies", type=int, default=160, help="ply cap; games past it score as draws (weak old nets rarely finish)")
    ap.add_argument("--start-fen", default=DEFAULT_START_FEN, help="start FEN (default: the training start)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="visit-sampling temperature for the opening plies (0 = deterministic argmax, the old vacuous behavior)")
    ap.add_argument("--temp-plies", type=int, default=20,
                    help="plies of temperature before argmax kicks in (fleet matches use tempdecay 10 moves = 20 plies)")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-curve", action="store_true", help="omit the strength sparkline; show only the table")
    ap.add_argument("--out", default="", help="append one JSON history row (current vs opponents) to this file")
    args = ap.parse_args()

    import torch
    from chessckers_engine.checkpoints import load_scorer
    from chessckers_engine.mcts_puct import pick_puct
    from chessckers_engine.variant_py import PyVariantClient

    dev = args.device
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    current = args.current or os.path.join(args.run_dir, "weights.pt")
    if not os.path.exists(current):
        raise SystemExit(f"gauntlet: current net not found: {current} (pass --current)")
    opps = pick_opponents(args.run_dir, args.n, args.all, args.nets, current)

    cur_label = _label(current)
    rn = _run_ident.run_name()
    print(f"gauntlet{f' [{rn}]' if rn else ''}: current '{cur_label}' vs {len(opps)} snapshots on {dev} | "
          f"{args.games} games/opp | {args.sims} sims | temp {args.temperature} for {args.temp_plies} plies"
          f"\n  current: {current}", flush=True)

    cur_model = load_scorer(current).to(dev).eval()
    client = PyVariantClient()

    rows = []
    agg_pts = agg_g = 0.0
    n_trunc = 0
    for opp in opps:
        lbl = _label(opp)
        opp_model = load_scorer(opp).to(dev).eval()
        w = d = l = 0
        for gi in range(args.games):
            cur_white = gi % 2 == 0
            wm, bm = (cur_model, opp_model) if cur_white else (opp_model, cur_model)
            out, trunc = play_game(wm, bm, client, pick_puct, args.sims, args.c_puct,
                                   args.max_plies, args.start_fen,
                                   temperature=args.temperature, temp_plies=args.temp_plies)
            n_trunc += trunc
            if out == "draw":
                d += 1
            elif (out == "white") == cur_white:
                w += 1
            else:
                l += 1
        del opp_model
        if dev == "cuda":
            torch.cuda.empty_cache()
        ng = w + d + l
        sc = (w + 0.5 * d) / ng if ng else 0.0
        rows.append((lbl, w, d, l, sc))
        agg_pts += w + 0.5 * d
        agg_g += ng
        print(f"  vs {lbl:>6}: {w}-{d}-{l}  ({100 * sc:.0f}%)", flush=True)

    render(rows, agg_pts, agg_g, curve=not args.no_curve)
    if n_trunc:
        frac = 100 * n_trunc / agg_g if agg_g else 0
        print(f"  \033[33m⚠ {n_trunc}/{int(agg_g)} games ({frac:.0f}%) hit the {args.max_plies}-ply cap "
              f"→ scored DRAW (no win condition reached).\033[0m")
        if frac >= 50:
            print("    Result is TRUNCATION-DOMINATED, not real draws — raise --sims "
                  "(self-play uses 800) and/or --max-plies for decisive games.")
    if args.out:
        import json, time
        agg_score = agg_pts / agg_g if agg_g else 0.0
        row = {
            "ts": int(time.time()),
            "current": cur_label,
            "agg_score": round(agg_score, 4),
            "agg_elo": round(_elo(agg_score), 1),
            "regression": any(sc < 0.5 for *_, sc in rows),
            "opponents": [
                {"label": lbl, "w": ww, "d": dd, "l": ll,
                 "score": round(sc, 4), "elo": round(_elo(sc), 1)}
                for lbl, ww, dd, ll, sc in rows
            ],
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  appended history row → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
