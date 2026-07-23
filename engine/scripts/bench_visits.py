#!/usr/bin/env python3
"""bench_visits — ops-noise-immune A/B metric for mate_bench trials:
SEARCH VISITS TO THRESHOLD-CROSSING, computed from archived trial DBs +
per-trial chunk tars (mate_bench save_trial_db writes both).

Why: wall-clock on a shared cloud box is dominated by tenant contention, gate
pauses, and OOM recoveries (run 25: throughput swung 777-3,378 games/h);
games-to-crossing is unfair to PCR by construction (its games are deliberately
cheap). Visits = plies x visits/move is the currency PCR actually trades in:
~NN forward passes ~ GPU-seconds, immune to everything above, hardware-portable.

Per-game accounting (exact, from the ccz1 chunks):
  - records          = len(examples) = moves that ran FULL search (dense runs:
                       every ply; PCR runs: only the pcr-full-prob moves)
  - total plies      = fen_ply(first record) + moves_left_target(first record)
                       (moves_left is plies-to-end, so ANY one record pins the
                       game length even in PCR-sparse chunks; first/last records
                       must agree or the game is flagged estimated)
  - visits(game)     = records*FULL_V + (plies - records)*FAST_V
FULL_V/FAST_V are read from the run's train_params (--visits / --pcr-fast-visits;
no PCR flag => all moves full). League games are accounted identically from
their own chunks. Gate-match games are EXCLUDED (gating overhead, not learning).

Crossings are recomputed per threshold (default 0.5, 0.75, 0.9) over the
trailing --window self-play games — multi-threshold robustness check on top of
mate_bench's single 0.9 stopping rule.

Usage (box or anywhere the bench_trials tree is copied):
  bench_visits.py [--trials-dir /workspace/chessckers/bench_trials]
                  [--window 1000] [--thresholds 0.5,0.75,0.9]
Prints per-trial + per-arm tables and writes bench_visits.json into
--trials-dir. Stdlib-only on purpose (archive-portable, like mate_bench).
"""
import argparse
import glob
import gzip
import json
import os
import re
import sqlite3
import statistics
import sys
import tarfile
from collections import deque

RES_WHITE, RES_BLACK, RES_DRAW = 1, 2, 3


def fen_ply(fen: str) -> int:
    """0-based ply index from the FEN's side-to-move + fullmove counter."""
    t = fen.split()
    return 2 * (int(t[5]) - 1) + (0 if t[1] == "w" else 1)


def chunk_stats(raw: bytes):
    """(records, total_plies|None) for one ccz1 chunk. total_plies is None when
    the first/last records disagree on game length (>2 plies) — flagged, and the
    caller falls back to a records-based estimate."""
    payload = json.loads(gzip.decompress(raw))
    ex = payload.get("examples", [])
    if not ex:
        return 0, None
    ests = [fen_ply(x["fen"]) + x["moves_left_target"] for x in (ex[0], ex[-1])]
    total = int(round(ests[0])) if abs(ests[0] - ests[1]) <= 2 else None
    return len(ex), total


def visit_params(train_params: str):
    """(full_visits, fast_visits, pcr_full_prob) from the run's trainParams."""
    def flag(name, default):
        m = re.search(rf"--{name}=([0-9.]+)", train_params or "")
        return float(m.group(1)) if m else default
    return (int(flag("visits", 800)), int(flag("pcr-fast-visits", 100)),
            flag("pcr-full-prob", 1.0))


def crossings(con, window: int, thresholds):
    """{thr: {game_id, games}} — earliest full-window Black-share >= thr over
    self-play games (opponent_network_id=0), single streaming pass."""
    remaining = sorted(thresholds)
    out = {}
    dq, b_in, n_seen = deque(), 0, 0
    cur = con.execute(
        "SELECT id, result FROM training_games WHERE result > 0 "
        "AND opponent_network_id = 0 ORDER BY id")
    for gid, res in cur:
        n_seen += 1
        dq.append(res)
        if res == RES_BLACK:
            b_in += 1
        if len(dq) > window:
            if dq.popleft() == RES_BLACK:
                b_in -= 1
        if len(dq) == window:
            while remaining and b_in / window >= remaining[0]:
                out[remaining.pop(0)] = {"game_id": gid, "games": n_seen}
        if not remaining:
            break
    return out


def analyze_trial(db_path: str, tar_path: str, window: int, thresholds):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    run_name, params = con.execute(
        "SELECT description, train_parameters FROM training_runs "
        "ORDER BY id DESC LIMIT 1").fetchone()
    full_v, fast_v, pcr = visit_params(params)
    cross = crossings(con, window, thresholds)
    league_ids = {r[0] for r in con.execute(
        "SELECT id FROM training_games WHERE opponent_network_id != 0")}
    max_gid = max((c["game_id"] for c in cross.values()), default=None)

    trial = {"db": db_path, "run_name": run_name,
             "full_visits": full_v, "fast_visits": fast_v, "pcr_full_prob": pcr,
             "thresholds": {str(t): dict(c) for t, c in cross.items()}}
    if not os.path.exists(tar_path):
        trial["warning"] = "no chunk tar — visits not computed"
        con.close()
        return trial

    # cumulative per-game visit cost, ordered by game id, up to the highest
    # crossing (or the whole trial if it never crossed — censored total)
    per_game = []  # (gid, visits, plies, records, estimated, is_league)
    n_bad = 0
    with tarfile.open(tar_path) as tf:
        members = {}
        for m in tf.getmembers():
            mm = re.search(r"training\.(\d+)\.gz$", m.name)
            if mm:
                members[int(mm.group(1))] = m
        for gid in sorted(members):
            if max_gid is not None and gid > max_gid:
                break
            try:
                recs, plies = chunk_stats(tf.extractfile(members[gid]).read())
            except Exception:  # noqa: BLE001 — torn/foreign chunk
                n_bad += 1
                continue
            est = plies is None
            if est:  # fall back: expected plies from the PCR record share
                plies = int(round(recs / max(pcr, 1e-9)))
            fast = max(0, plies - recs)
            per_game.append((gid, recs * full_v + fast * fast_v, plies, recs,
                             est, gid in league_ids))
    con.close()

    cum_v = cum_p = cum_r = 0
    it = iter(sorted(cross.items()))
    nxt = next(it, None)
    for gid, v, p, r, _e, _lg in per_game:
        cum_v += v
        cum_p += p
        cum_r += r
        while nxt and gid >= nxt[1]["game_id"]:
            trial["thresholds"][str(nxt[0])].update(
                visits=cum_v, plies=cum_p, records=cum_r)
            nxt = next(it, None)
    trial.update(
        games_archived=len(per_game), bad_chunks=n_bad,
        estimated_games=sum(1 for g in per_game if g[4]),
        league_games=sum(1 for g in per_game if g[5]),
        total_visits=cum_v, total_plies=cum_p, total_records=cum_r)
    return trial


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials-dir", default="/workspace/chessckers/bench_trials")
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--thresholds", default="0.5,0.75,0.9")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]

    trials = []
    for db in sorted(glob.glob(os.path.join(args.trials_dir, "*", "trial*.db"))):
        tar = db.replace(".db", "_games.tar.gz")
        t = analyze_trial(db, tar, args.window, thresholds)
        t["trial_file"] = os.path.relpath(db, args.trials_dir)
        trials.append(t)
    if not trials:
        sys.exit(f"bench_visits: no trial DBs under {args.trials_dir}")

    key = str(max(thresholds))
    print(f"== bench_visits — visits-to-crossing (window {args.window}) ==")
    hdr = (f"  {'trial':<42} {'games':>7} {'plies':>9} "
           + "".join(f"{'V@' + str(int(t * 100)) + '%':>10}" for t in sorted(thresholds)))
    print(hdr)
    by_arm = {}
    for t in trials:
        cells = ""
        for thr in sorted(thresholds):
            c = t["thresholds"].get(str(thr))
            v = c.get("visits") if c else None
            cells += f"{v / 1e6:>9.1f}M" if v else f"{'-':>10}"
        g = t["thresholds"].get(key, {}).get("games", "-")
        print(f"  {t['trial_file']:<42} {g:>7} {t.get('total_plies', 0):>9,} {cells}"
              + ("  [EST]" if t.get("estimated_games") else "")
              + ("  [" + t["warning"] + "]" if t.get("warning") else ""))
        v90 = t["thresholds"].get(key, {}).get("visits")
        if v90:
            by_arm.setdefault(t["run_name"], []).append(v90)
    print()
    for arm, vs in sorted(by_arm.items()):
        med = statistics.median(vs)
        print(f"  {arm}: median V@{key} = {med / 1e6:.1f}M visits "
              f"({len(vs)} crossed trials: "
              + ", ".join(f"{v / 1e6:.1f}M" for v in sorted(vs)) + ")")

    out = os.path.join(args.trials_dir, "bench_visits.json")
    with open(out, "w") as f:
        json.dump({"window": args.window, "thresholds": thresholds,
                   "trials": trials}, f, indent=1)
    print(f"\n  full detail → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
