#!/usr/bin/env python3
"""Round-robin ladder: play champion nets head-to-head, print an Elo + score matrix.

Plays each PAIR of nets `--games` games (colors split evenly) from the start FEN
via PUCT MCTS, then prints a pairwise score matrix, a Bradley-Terry Elo ranking,
and a chronological Elo curve — all in the terminal. Runs on the box (nets + GPU
are there): `cc ladder`.

  cc ladder                              # ~6 snapshots sampled from the run dir, round-robin
  cc ladder --n 8 --games 6 --sims 200
  cc ladder --vs-best                    # everyone vs the NEWEST only (quick anchor ladder)
  cc ladder a.pt b.pt c.pt               # explicit nets
options: --run-dir DIR  --n N  --games G  --sims S  --c-puct 1.5  --max-plies 400
         --start-fen FEN  --device auto|cuda|mps|cpu  --seed 0  --vs-best
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
from watch_game import DEFAULT_START_FEN  # noqa: E402  (the training start; reads the fork's board.cc)


def _label(path: str) -> str:
    """Short column label for a net: the iter number, 'best' for weights.pt."""
    b = os.path.basename(path)
    m = re.search(r"iter-async-0*(\d+)\.pt$", b)
    if m:
        return m.group(1)
    if b == "weights.pt":
        return "best"
    return b.replace(".pt", "")[:6]


def discover_nets(run_dir: str, n: int, explicit: list[str]) -> list[tuple[str, str]]:
    """Return [(label, path)] either from explicit paths or by sampling N snapshots
    evenly across the run dir's iter-async-*.pt lineage (+ the newest weights.pt)."""
    if explicit:
        return [(_label(p), p) for p in explicit]
    paths = [p for p in glob.glob(os.path.join(run_dir, "iter-async-*.pt"))
             if re.search(r"(\d+)\.pt$", os.path.basename(p))]  # skip any non-numeric snapshot
    paths.sort(key=lambda p: int(re.search(r"(\d+)\.pt$", os.path.basename(p)).group(1)))
    if not paths:
        raise SystemExit(f"ladder: no iter-async-*.pt under {run_dir} (pass explicit nets or --run-dir)")
    if len(paths) > n:
        idx = sorted({round(k * (len(paths) - 1) / (n - 1)) for k in range(n)})
        paths = [paths[i] for i in idx]
    best = os.path.join(run_dir, "weights.pt")
    if os.path.exists(best) and best not in paths:
        paths.append(best)
    return [(_label(p), p) for p in paths]


def play_game(white_model, black_model, client, pick, sims, cpuct, max_plies, start_fen) -> str:
    """One game; returns 'white' | 'black' | 'draw' (via the canonical outcome helper)."""
    from chessckers_engine.selfplay_az import _outcome_from_state
    state = client.new_game(fen=start_fen)
    ply = 0
    while not state.get("status") and ply < max_plies:
        model = white_model if state["turn"] == "white" else black_model
        chosen = pick(state, client, model, n_sims=sims, c_puct=cpuct)
        if chosen is None:
            break
        state = client.make_move(state["fen"], chosen["uci"])
        ply += 1
    return _outcome_from_state(state)


def bradley_terry_elo(score: list[list[float]], n_games: list[list[int]]) -> list[float]:
    """Elo from a results matrix via Bradley-Terry MM (draws = half a win each).
    score[i][j] = i's score vs j; n_games[i][j] = games i vs j (symmetric). Anchored
    to mean 0 (the absolute level is arbitrary — this is a RELATIVE ladder)."""
    n = len(score)
    wins = [sum(score[i]) for i in range(n)]            # total score of each net
    p = [1.0] * n
    for _ in range(500):
        new = []
        for i in range(n):
            den = sum(n_games[i][j] / (p[i] + p[j]) for j in range(n) if j != i and n_games[i][j])
            new.append(wins[i] / den if den > 0 else p[i])
        new = [max(x, 1e-12) for x in new]
        gm = math.exp(sum(math.log(x) for x in new) / n)  # normalize geometric mean -> 1
        p = [x / gm for x in new]
    elo = [400.0 * math.log10(pi) for pi in p]
    mean = sum(elo) / n
    return [e - mean for e in elo]


def render(labels, score, n_games, elo):
    """Print the pairwise score matrix (row vs col, %), the Elo ranking, and a
    chronological Elo curve."""
    n = len(labels)
    order = sorted(range(n), key=lambda i: -elo[i])     # best first
    w = max(4, max(len(l) for l in labels))
    cell = lambda v: f"{v:>{w}}"

    print("\nScore matrix — row's score % vs column (rows/cols sorted by Elo, best first):")
    print(" " * (w + 2) + " ".join(cell(labels[j]) for j in order))
    for i in order:
        cells = []
        for j in order:
            if i == j:
                cells.append(cell("·"))
            elif n_games[i][j]:
                cells.append(cell(f"{100 * score[i][j] / n_games[i][j]:.0f}%"))
            else:
                cells.append(cell("-"))
        print(f"{labels[i]:>{w}}  " + " ".join(cells))

    print("\nElo ranking (Bradley-Terry, mean=0; relative only):")
    print(f"  {'#':>2}  {'net':>{w}}  {'elo':>6}  {'pts':>9}  {'score':>6}")
    for rank, i in enumerate(order, 1):
        g = sum(n_games[i])
        pts = sum(score[i])
        print(f"  {rank:>2}  {labels[i]:>{w}}  {elo[i]:>+6.0f}  {pts:>5.1f}/{g:<3}  "
              f"{100 * pts / g if g else 0:>5.0f}%")

    # chronological curve: nets in training order (numeric label asc; 'best' last)
    chrono = sorted(range(n), key=lambda i: (labels[i] == "best", _num(labels[i])))
    lo, hi = min(elo), max(elo)
    blocks = "▁▂▃▄▅▆▇█"
    spark = "".join(blocks[min(7, int((elo[i] - lo) / (hi - lo + 1e-9) * 7))] for i in chrono)
    print("\nElo over training order (oldest→newest):")
    print(f"  {spark}   [{', '.join(labels[i] for i in chrono)}]   range {hi - lo:.0f} Elo")


def _num(lbl: str) -> int:
    return int(lbl) if lbl.isdigit() else 1 << 30


def _record(score, n_games, i) -> str:
    """W-D-L over all of i's games (draws inferred from half-points)."""
    n = len(score)
    w = d = l = 0
    for j in range(n):
        g = n_games[i][j]
        if not g:
            continue
        s = score[i][j]
        di = round((s - math.floor(s)) * 2)  # 0 or 1 fractional point per draw-pair is lossy; recompute below
    # recompute exactly from stored draws is cleaner; fall back to aggregate
    return f"{_agg(score, n_games, i)}"


def _agg(score, n_games, i) -> str:
    n = len(score)
    g = sum(n_games[i])
    s = sum(score[i])
    # draws aren't stored separately here; show score/games as W(.5D) summary
    return f"{s:.1f}/{g}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Round-robin net ladder with an Elo/score matrix.")
    ap.add_argument("nets", nargs="*", help="explicit net .pt paths (else sample the run dir)")
    ap.add_argument("--run-dir", default=_DEFAULT_RUN_DIR)
    ap.add_argument("--n", type=int, default=6, help="how many snapshots to sample when none given")
    ap.add_argument("--games", type=int, default=4, help="games per pairing (colors split)")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--start-fen", default=DEFAULT_START_FEN, help="start FEN (default: the training start)")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vs-best", action="store_true",
                    help="play everyone vs the NEWEST net only (anchor ladder), not full round-robin")
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

    nets = discover_nets(args.run_dir, args.n, args.nets)
    labels = [lbl for lbl, _ in nets]
    print(f"ladder: {len(nets)} nets on {dev} | {args.games} games/pair | {args.sims} sims | "
          f"{'vs-best' if args.vs_best else 'round-robin'}\n  nets: {', '.join(labels)}")
    models = [load_scorer(p).to(dev).eval() for _, p in nets]
    client = PyVariantClient()

    n = len(nets)
    score = [[0.0] * n for _ in range(n)]
    n_games = [[0] * n for _ in range(n)]

    pairs = [(i, n - 1) for i in range(n - 1)] if args.vs_best else \
            [(i, j) for i in range(n) for j in range(i + 1, n)]
    for (i, j) in pairs:
        for g in range(args.games):
            i_white = g % 2 == 0
            wi, bi = (i, j) if i_white else (j, i)
            out = play_game(models[wi], models[bi], client, pick_puct,
                            args.sims, args.c_puct, args.max_plies, args.start_fen)
            si = 1.0 if (out == "white") == i_white else 0.0 if out != "draw" else 0.5
            score[i][j] += si
            score[j][i] += 1.0 - si
            n_games[i][j] += 1
            n_games[j][i] += 1
        print(f"  {labels[i]} vs {labels[j]}: {score[i][j]:.1f}-{score[j][i]:.1f}", flush=True)

    elo = bradley_terry_elo(score, n_games)
    render(labels, score, n_games, elo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
