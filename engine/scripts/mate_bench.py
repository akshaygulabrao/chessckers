#!/usr/bin/env python3
"""mate_bench — wall-clock benchmark: how fast does a run find the Black mate?

Measures time-to-convergence of the current training run: the first moment the
trailing --window games' Black-win share reaches --threshold (default ≥90% of
ALL window games — a draw means the mate was NOT found; the decisive-only share
is also reported). The crossing is recomputed retroactively from
training_games.created_at, so the stamped number is exact regardless of when the
watcher started (or whether it restarted).

Modes:
  --report (default)  one-shot: current share, exact crossing if already crossed,
                      and the cross-run comparison table from BENCH_RESULTS.jsonl.
                      Works on archived DBs too: --db <path> [--results <path>].
  --watch             poll every --interval s; on crossing (or at --max-hours),
                      stamp the result and AUTO-END the run: stop the self-play
                      client + trainer (clean STOP-file flush), leave the server
                      up for status/strength/archive. `cc restart` resumes.
  --no-stop           with --watch: stamp but leave the fleet running.
  --stamp             with --report: append the crossing to the results ledger
                      (retro-stamping an archived run into the comparison table).

Stamps append one JSON line per run to /workspace/chessckers/BENCH_RESULTS.jsonl
(outside lczero-server, so it survives reset_fleet — like ALERTS.log).

Run via the command center:  cc bench | cc bench --watch | cc bench --stop
Stdlib-only on purpose: runs against archive DBs with any python3.
"""
import argparse
import datetime
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import deque

DB = "/workspace/chessckers/lczero-server/chessckers.db"
RESULTS = "/workspace/chessckers/BENCH_RESULTS.jsonl"
TRAINER_RUN_DIR = "/workspace/chessckers/lczero-server/trainer/run1"

# training_games.result codes (lczero-server main.go gameready handler)
RES_WHITE, RES_BLACK, RES_DRAW = 1, 2, 3


def _humanize(seconds: float) -> str:
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


def _utc(dt_str: str) -> datetime.datetime:
    """Parse sqlite datetime() output (UTC) into an aware datetime."""
    return datetime.datetime.fromisoformat(dt_str).replace(
        tzinfo=datetime.timezone.utc)


def _connect(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        sys.exit(f"mate_bench: no DB at {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)


def run_info(con) -> dict:
    row = con.execute(
        "SELECT id, description, datetime(created_at), datetime('now') "
        "FROM training_runs ORDER BY id DESC LIMIT 1").fetchone()
    if not row or not row[2]:
        sys.exit("mate_bench: no training_runs row (empty/foreign DB?)")
    return {"id": row[0], "name": row[1] or f"db-run-{row[0]}",
            "created_at": row[2], "now": row[3]}


def window_counts(con, window: int) -> dict:
    """Counts over the trailing `window` games with a recorded result."""
    rows = con.execute(
        "SELECT result FROM training_games WHERE result > 0 "
        "ORDER BY id DESC LIMIT ?", (window,)).fetchall()
    b = sum(1 for (r,) in rows if r == RES_BLACK)
    w = sum(1 for (r,) in rows if r == RES_WHITE)
    d = sum(1 for (r,) in rows if r == RES_DRAW)
    total = con.execute(
        "SELECT count(*) FROM training_games WHERE result > 0").fetchone()[0]
    return {"n": len(rows), "b": b, "w": w, "d": d, "total": total}


def _share_str(b: int, w: int, d: int, n: int) -> str:
    dec = b + w
    return (f"B {100 * b / max(1, n):.1f}% (dec {100 * b / max(1, dec):.1f}%)"
            f"  W {100 * w / max(1, n):.1f}%  D {100 * d / max(1, n):.1f}%")


def retro_crossing(con, window: int, threshold: float) -> dict | None:
    """Earliest game at which the FULL trailing window's blackwon/window ≥
    threshold. Streams id-ordered rows with O(window) memory; returns None if
    the run never crossed."""
    dq: deque[int] = deque()
    b_in = 0
    n_seen = 0
    cur = con.execute(
        "SELECT id, result, datetime(created_at) FROM training_games "
        "WHERE result > 0 ORDER BY id")
    for gid, res, created in cur:
        n_seen += 1
        dq.append(res)
        if res == RES_BLACK:
            b_in += 1
        if len(dq) > window:
            if dq.popleft() == RES_BLACK:
                b_in -= 1
        if len(dq) == window and b_in / window >= threshold:
            w_in = sum(1 for r in dq if r == RES_WHITE)
            return {"game_id": gid, "games": n_seen, "created_at": created,
                    "b": b_in, "w": w_in, "d": window - b_in - w_in}
    return None


def _pgrep(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern],
                          capture_output=True).returncode == 0


def stop_fleet(log) -> None:
    """END the run: stop the self-play client, then the trainer via a clean
    STOP-file flush. The server (and its DB/networks) stays up so `cc status`
    / `cc strength` / archiving keep working; `cc restart` warm-resumes.
    pkill patterns are bracket-escaped so nothing can self-match."""
    def run(cmd):
        log(f"  $ {' '.join(cmd)}")
        subprocess.run(cmd, check=False, capture_output=True)

    log("ending run: stopping self-play client ...")
    run(["tmux", "kill-session", "-t", "cc-client"])
    run(["pkill", "-f", "lc0-clien[t]"])
    run(["pkill", "-f", "akshay-chessckers-[0] selfplay"])

    log("stopping trainer (STOP file → graceful flush) ...")
    stop_path = os.path.join(TRAINER_RUN_DIR, "STOP")
    try:
        open(stop_path, "w").close()
    except OSError as e:
        log(f"  (STOP touch failed: {e})")
    deadline = time.time() + 90
    while time.time() < deadline and _pgrep("train_continuou[s]"):
        time.sleep(5)
    if _pgrep("train_continuou[s]"):
        log("  trainer still alive after 90s — hard pkill")
        run(["pkill", "-f", "train_continuou[s]"])
    deadline = time.time() + 30
    while time.time() < deadline and _pgrep("trainer_bridg[e]"):
        time.sleep(3)
    if _pgrep("trainer_bridg[e]"):
        run(["pkill", "-TERM", "-f", "trainer_bridg[e]"])
    # Clear STOP once everything is down: restart_fleet.sh does NOT remove it,
    # so leaving it behind would make a later `cc restart` trainer exit on boot.
    try:
        os.remove(stop_path)
    except OSError:
        pass
    log("fleet ended (server left running; `cc restart` resumes the run)")


def build_stamp(con, args, converged: bool, note: str = "") -> dict:
    ri = run_info(con)
    wc = window_counts(con, args.window)
    now = _utc(ri["now"])
    started = _utc(ri["created_at"])
    stamp = {
        "ts": now.strftime("%Y-%m-%d %H:%M"),
        "run_id": ri["id"], "run_name": ri["name"],
        "run_started_utc": ri["created_at"][:16],
        "threshold": args.threshold, "window": args.window,
        "converged": converged, "note": note,
        "games_total": wc["total"],
        "window_now": {"b": wc["b"], "w": wc["w"], "d": wc["d"], "n": wc["n"]},
    }
    cross = retro_crossing(con, args.window, args.threshold) if converged else None
    if cross:
        elapsed = (_utc(cross["created_at"]) - started).total_seconds()
        stamp.update({
            "elapsed_s": int(elapsed), "elapsed": _humanize(elapsed),
            "games": cross["games"], "game_id": cross["game_id"],
            "crossed_utc": cross["created_at"][:16],
            "games_per_h": round(cross["games"] / max(1e-9, elapsed / 3600)),
            "black_all": round(cross["b"] / args.window, 4),
            "black_decisive": round(cross["b"] / max(1, cross["b"] + cross["w"]), 4),
            "window_cross": {"b": cross["b"], "w": cross["w"], "d": cross["d"]},
        })
    else:
        elapsed = (now - started).total_seconds()
        stamp.update({"elapsed_s": int(elapsed), "elapsed": _humanize(elapsed)})
    return stamp


def append_stamp(stamp: dict, path: str, log) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(stamp) + "\n")
    log(f"stamped → {path}")


def print_stamp(stamp: dict) -> None:
    if stamp["converged"]:
        print("\n== MATE FOUND — benchmark complete ==")
        print(f"  run:      {stamp['run_name']} (db run {stamp['run_id']})")
        print(f"  crossed:  {stamp['crossed_utc']} UTC — elapsed {stamp['elapsed']} "
              f"from run start ({stamp['run_started_utc']} UTC)")
        wcx = stamp["window_cross"]
        print(f"  games:    {stamp['games']:,} at crossing  (~{stamp['games_per_h']:,} games/h)")
        print(f"  window@cross: {_share_str(wcx['b'], wcx['w'], wcx['d'], stamp['window'])}"
              f"   [thr {stamp['threshold']:.0%} of all, window {stamp['window']}]")
    else:
        print(f"\n== NOT CONVERGED — {stamp['note'] or 'benchmark incomplete'} ==")
        print(f"  run:      {stamp['run_name']} (db run {stamp['run_id']})")
        print(f"  elapsed:  {stamp['elapsed']}   games {stamp['games_total']:,}")
        wn = stamp["window_now"]
        print(f"  window now: {_share_str(wn['b'], wn['w'], wn['d'], max(1, wn['n']))}")


def print_table(path: str) -> None:
    if not os.path.exists(path):
        return
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return
    print(f"\n  -- benchmark ledger ({path}) --")
    print(f"  {'run':<38} {'result':<9} {'elapsed':>8} {'games':>7} {'g/h':>6}  B%all(dec)")
    for r in rows:
        name = (r.get("run_name") or "?")[:38]
        if r.get("converged"):
            res, games = "MATE", f"{r.get('games', 0):,}"
            gph = f"{r.get('games_per_h', 0):,}"
            share = f"{100 * r.get('black_all', 0):.0f}% ({100 * r.get('black_decisive', 0):.0f}%)"
        else:
            res, gph = "DNF", "-"
            games = f"{r.get('games_total', 0):,}"
            share = "-"
        print(f"  {name:<38} {res:<9} {r.get('elapsed', '?'):>8} {games:>7} {gph:>6}  {share}")


def report(args) -> int:
    con = _connect(args.db)
    ri = run_info(con)
    wc = window_counts(con, args.window)
    started = _utc(ri["created_at"])
    # Anchor the clock on the NEWEST GAME, not datetime('now'): on a live DB they
    # agree to within seconds, and on an archived DB "now" is meaningless.
    newest = con.execute(
        "SELECT datetime(created_at) FROM training_games WHERE result > 0 "
        "ORDER BY id DESC LIMIT 1").fetchone()
    asof = newest[0] if newest and newest[0] else ri["now"]
    elapsed = (_utc(asof) - started).total_seconds()
    print(f"== mate_bench — {ri['name']} (db run {ri['id']}) ==")
    print(f"  clock:   {_humanize(elapsed)}  (run start {ri['created_at'][:16]} → "
          f"newest game {asof[:16]} UTC)")
    gph = wc["total"] / max(1e-9, elapsed / 3600)
    print(f"  games:   {wc['total']:,}  (~{gph:,.0f} games/h)")
    print(f"  window {wc['n']}: {_share_str(wc['b'], wc['w'], wc['d'], max(1, wc['n']))}"
          f"   [thr {args.threshold:.0%} of all]")
    cross = retro_crossing(con, args.window, args.threshold)
    if cross:
        c_elapsed = (_utc(cross["created_at"]) - started).total_seconds()
        print(f"  CROSSED: {_humanize(c_elapsed)} / {cross['games']:,} games "
              f"({cross['created_at'][:16]} UTC, game #{cross['game_id']}) — "
              f"{_share_str(cross['b'], cross['w'], cross['d'], args.window)}")
        if args.stamp:
            stamp = build_stamp(con, args, converged=True, note="retro-stamp")
            append_stamp(stamp, args.results, lambda m: print(f"  {m}"))
    else:
        print(f"  not crossed yet at thr {args.threshold:.0%} (window {args.window})")
        if args.stamp:
            print("  (--stamp ignored: no crossing)")
    print_table(args.results)
    return 0


def watch(args) -> int:
    def log(msg):
        print(f"[mate_bench {datetime.datetime.now(datetime.timezone.utc):%H:%M}] {msg}",
              flush=True)

    log(f"watching {args.db}: thr {args.threshold:.0%} of ALL over window {args.window}, "
        f"poll {args.interval}s, max {args.max_hours or '∞'}h"
        + (" (--no-stop: will NOT end the fleet)" if args.no_stop else
           " — will AUTO-END the run on crossing"))
    prev_total = None
    while True:
        try:
            con = _connect(args.db)
            ri = run_info(con)
            wc = window_counts(con, args.window)
            elapsed = (_utc(ri["now"]) - _utc(ri["created_at"])).total_seconds()
            delta = f" (+{wc['total'] - prev_total})" if prev_total is not None else ""
            prev_total = wc["total"]
            log(f"games={wc['total']:,}{delta}  window {wc['n']}: "
                f"{_share_str(wc['b'], wc['w'], wc['d'], max(1, wc['n']))}  "
                f"elapsed {_humanize(elapsed)}")
            # Authoritative check is the retro scan (cheap: one indexed pass), not
            # the live window: a run that crossed BEFORE the watcher was armed (or
            # crossed and then regressed) has already produced its benchmark number
            # — keeping the fleet burning past it adds nothing to this metric.
            converged = retro_crossing(con, args.window, args.threshold) is not None
            timed_out = bool(args.max_hours) and elapsed >= args.max_hours * 3600
            if converged or timed_out:
                note = "" if converged else f"max-hours {args.max_hours}h reached"
                stamp = build_stamp(con, args, converged=converged, note=note)
                con.close()
                append_stamp(stamp, args.results, log)
                print_stamp(stamp)
                if args.no_stop:
                    log("--no-stop: leaving the fleet running")
                else:
                    stop_fleet(log)
                return 0
            con.close()
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 — transient DB lock/reset mid-poll
            log(f"poll failed ({e}) — retrying")
        time.sleep(args.interval)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Time-to-mate benchmark: report or auto-ending watcher.")
    ap.add_argument("--db", default=DB)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="Black share of ALL window games to trigger (default 0.90)")
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--interval", type=int, default=60, help="watch poll seconds")
    ap.add_argument("--max-hours", type=float, default=24,
                    help="auto-end DNF after this many hours from RUN START (0 = never)")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--report", action="store_true", help="(default mode)")
    ap.add_argument("--no-stop", action="store_true",
                    help="with --watch: stamp but don't end the fleet")
    ap.add_argument("--stamp", action="store_true",
                    help="with --report: retro-stamp the crossing into --results")
    args = ap.parse_args()
    return watch(args) if args.watch else report(args)


if __name__ == "__main__":
    sys.exit(main())
