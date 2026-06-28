#!/usr/bin/env python3
"""cc strength — net-vs-past-selves strength from the in-fleet GATE (no games played).

The promotion gate (lczero-server) already plays every new net vs the current best
with the FAST C++ engine and records W-L-D + the promote/reject verdict in the DB
`matches` table (a 40-game match takes ~1 min). This just READS those rows and prints
a table — per gate match: candidate net#, opponent (best) net#, W-L-D, Elo (the gate's
calcElo), PROMOTED/rejected — plus the running cumulative Elo over promotions.

Instant (one read-only DB query), and it's the engine's real strength signal — unlike
the slow Python `cc gauntlet`, which replays games through PyVariant MCTS and starves
on a box that's busy self-playing. Use `cc gauntlet` only for occasional deep offline
checks (fleet paused).
"""
import argparse
import math
import os
import sqlite3
import sys

# lczero-server is a SIBLING of engine on the box (/workspace/chessckers/{engine,
# lczero-server}) but two levels up on the Mac. Pick whichever exists.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENG = os.path.dirname(_HERE)
_SERVER_DIR = next(
    (p for p in (os.path.join(_ENG, "..", "lczero-server"),
                 os.path.join(_ENG, "..", "..", "lczero-server"))
     if os.path.isdir(p)),
    os.path.join(_ENG, "..", "lczero-server"),
)


def _elo(score: float) -> float:
    """The gate's calcElo: -400*log10(1/score - 1), capped ±800 (mirrors main.go)."""
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return max(-800.0, min(800.0, -400.0 * math.log10(1.0 / score - 1.0)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Strength table from the in-fleet gate matches (no games played).")
    ap.add_argument("--db", default=os.path.join(_SERVER_DIR, "chessckers.db"),
                    help="server SQLite db (default: <server>/chessckers.db)")
    ap.add_argument("--last", type=int, default=25,
                    help="show the last N gate matches (0 = all)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"cc strength: db not found: {args.db} (pass --db)")

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    num = {r["id"]: r["network_number"]
           for r in con.execute("SELECT id, network_number FROM networks")}
    rows = con.execute(
        "SELECT candidate_id, current_best_id, wins, losses, draws, passed "
        "FROM matches WHERE done = 1 AND test_only = 0 AND deleted_at IS NULL "
        "ORDER BY id").fetchall()
    con.close()

    if not rows:
        print("cc strength: no completed gate matches yet "
              "(run just started, or auto-promote / bootstrap only).")
        return 0

    # cumulative Elo over PROMOTED matches = running strength of the best net above
    # net 1 (each promoted calcElo is the new best's edge over the prior best).
    cum, cum_at = 0.0, []
    for r in rows:
        n = r["wins"] + r["losses"] + r["draws"]
        e = _elo((r["wins"] + 0.5 * r["draws"]) / n) if n else 0.0
        if r["passed"]:
            cum += e
        cum_at.append(cum)

    start = 0 if args.last <= 0 else max(0, len(rows) - args.last)
    shown = rows[start:]
    promoted = sum(1 for r in rows if r["passed"])

    print(f"\n  in-fleet gate: {len(rows)} matches, {promoted} promoted "
          f"(showing last {len(shown)}) — each net vs the then-current best")
    print(f"  {'cand':>5} {'vs best':>8}  {'W-L-D':>9}  {'Elo':>5}   verdict  {'cumElo':>7}")
    print("  " + "─" * 50)
    for i, r in enumerate(shown):
        n = r["wins"] + r["losses"] + r["draws"]
        e = _elo((r["wins"] + 0.5 * r["draws"]) / n) if n else 0.0
        cand = num.get(r["candidate_id"], f"id{r['candidate_id']}")
        best = num.get(r["current_best_id"], f"id{r['current_best_id']}")
        wld = f"{r['wins']}-{r['losses']}-{r['draws']}"
        verdict = ("\033[32mPROMOTE\033[0m" if r["passed"]
                   else "\033[31mreject \033[0m")
        print(f"  {cand:>5} {best:>8}  {wld:>9}  {e:>+5.0f}   {verdict}  "
              f"{cum_at[start + i]:>+7.0f}")
    print(f"\n  cumulative Elo over promotions: {cum_at[-1]:+.0f}  "
          f"(approx; sums each promoted match's calcElo)")
    print("  Elo<0 promoted = the lenient gate (calcElo>-20) let a slightly-worse net "
          "through;\n  sustained drift is what the regression-ladder follow-on would catch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
