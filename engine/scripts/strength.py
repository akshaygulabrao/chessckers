#!/usr/bin/env python3
"""cc strength — net-vs-past-selves strength from the in-fleet GATE (no games played).

The promotion gate (lczero-server) already plays every new net vs the current best
with the FAST C++ engine and records W-L-D + the promote/reject verdict in the DB
`matches` table (a 40-game match takes ~1 min). This just READS those rows and prints
a table — per gate match: candidate net#, opponent (best) net#, W-L-D, Elo (the gate's
calcElo), PROMOTED/rejected — plus the running cumulative Elo over promotions.
Regression-panel legs (run 20+: candidate vs past champions, pass = calcElo above
`matches.panel.threshold`) print dim + indented under their gate match, so a reject
whose main Elo passed −20 is self-explanatory (a panel leg regressed).

Instant (one read-only DB query), and it's the engine's real strength signal — unlike
the slow Python `cc gauntlet`, which replays games through PyVariant MCTS and starves
on a box that's busy self-playing. Use `cc gauntlet` only for occasional deep offline
checks (fleet paused).
"""
import argparse
import datetime
import json
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


def _humanize_elapsed(seconds: float) -> str:
    """Format elapsed seconds as '3d 4h' / '5h 12m' / '48m'."""
    s = int(seconds)
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    minutes = s // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Strength table from the in-fleet gate matches (no games played).")
    ap.add_argument("--db", default=os.path.join(_SERVER_DIR, "chessckers.db"),
                    help="server SQLite db (default: <server>/chessckers.db)")
    ap.add_argument("--last", type=int, default=25,
                    help="show the last N gate matches (0 = all)")
    ap.add_argument("--since-net", type=int, default=None,
                    help="only matches whose candidate network_number >= this (default: "
                         "auto-detect the current run's start = the most recent bootstrap-promoted net)")
    ap.add_argument("--all", action="store_true",
                    help="show matches across ALL runs (disable current-run scoping)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"cc strength: db not found: {args.db} (pass --db)")

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    num = {r["id"]: r["network_number"]
           for r in con.execute("SELECT id, network_number FROM networks")}
    rows = con.execute(
        "SELECT id, candidate_id, current_best_id, wins, losses, draws, passed "
        "FROM matches WHERE done = 1 AND test_only = 0 AND deleted_at IS NULL "
        "ORDER BY id").fetchall()
    # Regression-panel legs (run 20+): test_only rows tied to their gate match via
    # panel_parent_id. A leg's own `passed` flag is never set by the server — its
    # verdict is calcElo vs the panel threshold. Older DBs lack the column entirely.
    legs = {}
    try:
        for leg in con.execute(
                "SELECT panel_parent_id, current_best_id, wins, losses, draws, done "
                "FROM matches WHERE test_only = 1 AND panel_parent_id != 0 "
                "AND deleted_at IS NULL ORDER BY id"):
            legs.setdefault(leg["panel_parent_id"], []).append(leg)
    except sqlite3.OperationalError:
        pass
    con.close()

    panel_thr = -50.0
    try:
        with open(os.path.join(_SERVER_DIR, "serverconfig.json")) as f:
            panel_thr = float(json.load(f)["matches"]["panel"]["threshold"])
    except Exception:  # noqa: BLE001
        pass

    if not rows:
        print("cc strength: no completed gate matches yet "
              "(run just started, or auto-promote / bootstrap only).")
        return 0

    # Scope to the CURRENT run. A buffer-preserving scale-up (e.g. run 11: c48/b5 -> c64/b6)
    # reuses training_run 1 and only resets best_network_id, so the DB still holds the PRIOR
    # run's nets + matches. The current run starts at the most recent BOOTSTRAP-promoted net —
    # one that was "best" but never a match candidate (best=0 -> set directly, no gate match).
    # Auto-detect that boundary so strength doesn't conflate the old c48 run with the new c64 one.
    run_start = args.since_net
    if not args.all:
        if run_start is None:
            cand_ids = {r["candidate_id"] for r in rows}
            boots = [i for i in ({r["current_best_id"] for r in rows} - cand_ids) if i in num]
            if boots:
                run_start = num[max(boots, key=lambda i: num[i])]
        if run_start is not None:
            rows = [r for r in rows if num.get(r["candidate_id"], -1) >= run_start]
        if not rows:
            print(f"cc strength: no completed gate matches yet for the current run "
                  f"(since net #{run_start}) — the new net's first gate match may still be "
                  f"running. Use --all (or --since-net N) to see prior runs.")
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

    # Fetch run identity and start time for display in the header
    _run_label = ""
    _clock_line = ""
    try:
        _rcon = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        _run_row = _rcon.execute(
            "SELECT id, description, datetime(created_at), datetime('now') "
            "FROM training_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        _rcon.close()
        if _run_row:
            _run_label = f" — {_run_row[1]} (db run {_run_row[0]})"
            if _run_row[2]:
                _start = datetime.datetime.fromisoformat(_run_row[2]).replace(
                    tzinfo=datetime.timezone.utc)
                _now = datetime.datetime.fromisoformat(_run_row[3]).replace(
                    tzinfo=datetime.timezone.utc)
                _elapsed = _humanize_elapsed((_now - _start).total_seconds())
                _clock_line = (f"  clock:      {_elapsed}"
                               f"  (run started {_run_row[2][:16]} UTC)")
    except Exception:  # noqa: BLE001
        pass

    scope = "ALL runs" if args.all else (f"current run (since net #{run_start})"
                                         if run_start is not None else "all matches")
    print(f"\n  in-fleet gate{_run_label} [{scope}]: {len(rows)} matches, {promoted} promoted "
          f"(showing last {len(shown)}) — each net vs the then-current best")
    if _clock_line:
        print(_clock_line)
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
        for leg in legs.get(r["id"], ()):
            opp = num.get(leg["current_best_id"], f"id{leg['current_best_id']}")
            label = f"└ vs {opp}"
            ln = leg["wins"] + leg["losses"] + leg["draws"]
            if ln == 0:
                note = ("skipped (verdict already sealed)" if leg["done"]
                        else "queued")
                print(f"  {'':>5} \033[2m{label:>8}  {'—':>9}  {'':>5}   "
                      f"panel {note}\033[0m")
                continue
            le = _elo((leg["wins"] + 0.5 * leg["draws"]) / ln)
            lwld = f"{leg['wins']}-{leg['losses']}-{leg['draws']}"
            tag = (f"\033[31mpanel FAIL (≤{panel_thr:g})\033[0m" if le <= panel_thr
                   else "\033[2mpanel ok\033[0m")
            running = "" if leg["done"] else " \033[33m(running)\033[0m"
            print(f"  {'':>5} \033[2m{label:>8}  {lwld:>9}  {le:>+5.0f}\033[0m   "
                  f"{tag}{running}")
    print(f"\n  cumulative Elo over promotions: {cum_at[-1]:+.0f}  "
          f"(approx; sums each promoted match's calcElo)")
    print("  Elo<0 promoted = the lenient gate (calcElo>-20) let a slightly-worse net "
          "through;\n  sustained drift is what the regression-ladder follow-on would catch.")
    if any(legs.get(r["id"]) for r in shown):
        print(f"  reject with a passing main Elo = a panel leg (vs a past champion) came "
              f"in ≤{panel_thr:g};\n  a 'skipped' leg never played — an earlier leg had "
              f"already sealed the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
