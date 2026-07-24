#!/usr/bin/env python3
"""mate_bench — wall-clock benchmark: how fast does a run find the Black mate?

Measures time-to-convergence of the current training run: the first moment the
trailing --window SELF-PLAY games' Black-win share reaches --threshold (default
≥90% of ALL window games — a draw means the mate was NOT found; the decisive-only
share is also reported). The crossing is recomputed retroactively from
training_games.created_at, so the stamped number is exact regardless of when the
watcher started (or whether it restarted).

League games (opponent_network_id != 0) are EXCLUDED from the metric: they mix in
learner-vs-old-champ results, so their Black share tracks the pool composition,
not position mastery (run 24: the fully-converged frozen net read 84% with league
games included — fossil pool nets farmed as White — vs 98% pure self-play).
League share is still printed as context. Pre-league archive DBs (no
opponent_network_id column) fall back to all games.

Modes:
  --report (default)  one-shot: current share, exact crossing if already crossed,
                      and the cross-run comparison table from BENCH_RESULTS.jsonl.
                      Works on archived DBs too: --db <path> [--results <path>].
  --watch             poll every --interval s; on crossing (or at --max-hours),
                      stamp the result and AUTO-END the run: stop the self-play
                      client + trainer (clean STOP-file flush), leave the server
                      up for status/strength/archive. `cc restart` resumes.
  --trials N          run the FULL experiment N times (default 5) to average out
                      run-to-run randomness: watch the current run to its crossing
                      (trial 1), then for each further trial reset_fleet (WIPES
                      fleet state — archive first!) and relaunch with the config
                      from the @reboot cron line, watch, stamp. Each trial's DB is
                      saved to /workspace/chessckers/bench_trials/<label>/ before
                      the wipe, and a per-trial trainParams parity guard aborts if
                      a relaunch drifts from trial 1's config. Ends with a summary
                      stamp (median/mean±sd) in BENCH_RESULTS.jsonl. Each trial
                      gets a DISTINCT trainer seed (cron SEED + trial - 1, via
                      restart_fleet SEED → launch_trainer → bridge →
                      train_continuous --seed): independent net inits + replay
                      sampling on top of self-play stochasticity.
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
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import deque

SERVER_DIR = "/workspace/chessckers/lczero-server"
DB = f"{SERVER_DIR}/chessckers.db"
RESULTS = "/workspace/chessckers/BENCH_RESULTS.jsonl"
TRAINER_RUN_DIR = f"{SERVER_DIR}/trainer/run1"
TRIALS_DIR = "/workspace/chessckers/bench_trials"

# Env keys that define a run's config in the @reboot cron line (fresh-run installs
# it; restart_fleet.sh consumes it) — the persisted source of truth trials relaunch
# from. PCR_* reach bootstrap only on the post-reset empty DB.
CRON_KEYS = ("RUN_NAME", "ARCH_VERSION", "C_FILTERS", "N_BLOCKS", "SE_RATIO",
             "POLICY_TARGET", "VALUE_Q_RATIO", "EMA_DECAY", "PUBLISH_GAMES",
             "PARALLELISM", "PCR_FULL_PROB", "PCR_FAST_VISITS",
             "GUMBEL_SH", "GUMBEL_M", "VISITS", "SEED")

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
        "SELECT id, description, datetime(created_at), datetime('now'), "
        "train_parameters FROM training_runs ORDER BY id DESC LIMIT 1").fetchone()
    if not row or not row[2]:
        sys.exit("mate_bench: no training_runs row (empty/foreign DB?)")
    return {"id": row[0], "name": row[1] or f"db-run-{row[0]}",
            "created_at": row[2], "now": row[3], "train_params": row[4] or ""}


def self_filter(con) -> str:
    """SQL fragment excluding league games; empty on pre-league schemas."""
    cols = [r[1] for r in con.execute("PRAGMA table_info(training_games)")]
    return " AND opponent_network_id = 0" if "opponent_network_id" in cols else ""


def window_counts(con, window: int) -> dict:
    """Counts over the trailing `window` self-play games with a recorded result."""
    flt = self_filter(con)
    rows = con.execute(
        f"SELECT result FROM training_games WHERE result > 0{flt} "
        "ORDER BY id DESC LIMIT ?", (window,)).fetchall()
    b = sum(1 for (r,) in rows if r == RES_BLACK)
    w = sum(1 for (r,) in rows if r == RES_WHITE)
    d = sum(1 for (r,) in rows if r == RES_DRAW)
    total = con.execute(
        "SELECT count(*) FROM training_games WHERE result > 0").fetchone()[0]
    league = con.execute(
        "SELECT count(*) FROM training_games WHERE result > 0 "
        "AND opponent_network_id != 0").fetchone()[0] if flt else 0
    return {"n": len(rows), "b": b, "w": w, "d": d,
            "total": total, "league": league}


def _share_str(b: int, w: int, d: int, n: int) -> str:
    dec = b + w
    return (f"B {100 * b / max(1, n):.1f}% (dec {100 * b / max(1, dec):.1f}%)"
            f"  W {100 * w / max(1, n):.1f}%  D {100 * d / max(1, n):.1f}%")


def retro_crossing(con, window: int, threshold: float) -> dict | None:
    """Earliest self-play game at which the FULL trailing window's
    blackwon/window ≥ threshold. Streams id-ordered rows with O(window) memory;
    returns None if the run never crossed."""
    dq: deque[int] = deque()
    b_in = 0
    n_seen = 0
    cur = con.execute(
        "SELECT id, result, datetime(created_at) FROM training_games "
        f"WHERE result > 0{self_filter(con)} ORDER BY id")
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
        "basis": "self-play-only",
        "converged": converged, "note": note,
        "games_total": wc["total"], "league_games": wc["league"],
        "train_params": ri["train_params"],
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
    trial = (f" (trial {stamp['trial']}/{stamp['trials']})"
             if stamp.get("trial") else "")
    if stamp["converged"]:
        print(f"\n== MATE FOUND{trial} ==")
        print(f"  run:      {stamp['run_name']} (db run {stamp['run_id']})")
        print(f"  crossed:  {stamp['crossed_utc']} UTC — elapsed {stamp['elapsed']} "
              f"from run start ({stamp['run_started_utc']} UTC)")
        wcx = stamp["window_cross"]
        print(f"  games:    {stamp['games']:,} self-play games at crossing  "
              f"(~{stamp['games_per_h']:,}/h)")
        print(f"  window@cross: {_share_str(wcx['b'], wcx['w'], wcx['d'], stamp['window'])}"
              f"   [thr {stamp['threshold']:.0%} of all, window {stamp['window']}]")
    else:
        print(f"\n== NOT CONVERGED{trial} — {stamp['note'] or 'benchmark incomplete'} ==")
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
        name = r.get("run_name") or "?"
        if r.get("type") == "summary":
            med = _humanize(r["median_s"]) if "median_s" in r else "-"
            sd = f" ±{_humanize(r['sd_s'])}" if r.get("sd_s") else ""
            mg = f"{r['median_games']:,}" if "median_games" in r else "-"
            label = f"{name} [{r.get('trials', '?')} trials]"[:38]
            score = f"{r.get('converged', '?')}/{r.get('trials', '?')}"
            print(f"  {label:<38} {score:<9} {med + sd:>8} {mg:>7} {'':>6}  (median)")
            continue
        if r.get("trial"):
            name = f"{name[:30]} [t{r['trial']}/{r.get('trials', '?')}]"
        name = name[:38]
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
    lg = (f", {100 * wc['league'] / max(1, wc['total']):.0f}% league — excluded "
          f"from the metric" if wc["league"] else "")
    print(f"  games:   {wc['total']:,}  (~{gph:,.0f} games/h{lg})")
    print(f"  self-play window {wc['n']}: "
          f"{_share_str(wc['b'], wc['w'], wc['d'], max(1, wc['n']))}"
          f"   [thr {args.threshold:.0%} of all]")
    cross = retro_crossing(con, args.window, args.threshold)
    if cross:
        c_elapsed = (_utc(cross["created_at"]) - started).total_seconds()
        print(f"  CROSSED: {_humanize(c_elapsed)} / {cross['games']:,} self-play games "
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


def _log(msg: str) -> None:
    print(f"[mate_bench {datetime.datetime.now(datetime.timezone.utc):%H:%M}] {msg}",
          flush=True)


def _watch_until_crossed(args, trial: int = 0, trials: int = 0,
                         seed: int | None = None) -> dict:
    """Poll until crossing (or max-hours DNF), stamp, stop client+trainer
    (unless --no-stop), and return the stamp. The trials loop calls this once
    per trial (passing that trial's trainer seed for the stamp); single --watch
    mode calls it once."""
    tag = f"[trial {trial}/{trials}] " if trial else ""
    _log(f"{tag}watching {args.db}: thr {args.threshold:.0%} of ALL over window "
         f"{args.window}, poll {args.interval}s, max {args.max_hours or '∞'}h"
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
            _log(f"{tag}games={wc['total']:,}{delta}  self window {wc['n']}: "
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
                if trial:
                    stamp["trial"], stamp["trials"] = trial, trials
                if seed is not None:
                    stamp["seed"] = seed
                append_stamp(stamp, args.results, _log)
                print_stamp(stamp)
                if args.no_stop:
                    _log("--no-stop: leaving the fleet running")
                else:
                    stop_fleet(_log)
                return stamp
            con.close()
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 — transient DB lock/reset mid-poll
            _log(f"{tag}poll failed ({e}) — retrying")
        time.sleep(args.interval)


def watch(args) -> int:
    _watch_until_crossed(args)
    return 0


def cron_env() -> dict:
    """The run's config env from the @reboot cron line (installed by cc fresh-run,
    consumed by restart_fleet.sh) — the persisted source of truth trials relaunch
    from. Hard-fails if absent: silently relaunching with restart_fleet.sh
    DEFAULTS would benchmark the wrong config."""
    out = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    line = next((ln for ln in out.splitlines() if "restart_fleet.sh" in ln), "")
    env = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k in CRON_KEYS:
                env[k] = v
    if "RUN_NAME" not in env:
        sys.exit("mate_bench: no restart_fleet.sh cron line with RUN_NAME — trials "
                 "can't replicate the run config (launch the run via cc fresh-run)")
    return env


def relaunch_trial(env: dict) -> None:
    """DESTRUCTIVE: wipe fleet state (reset_fleet.sh) and relaunch server +
    trainer + client with the cron-line config. tmux sessions are killed first so
    restart_fleet.sh's already-running check can't no-op on leftover shells."""
    for sess in ("cc", "cc-client"):
        subprocess.run(["tmux", "kill-session", "-t", sess],
                       check=False, capture_output=True)
    _log("reset_fleet: WIPING db/networks/games/pgns/trainer ...")
    r = subprocess.run(["bash", f"{SERVER_DIR}/scripts/reset_fleet.sh"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"mate_bench: reset_fleet.sh failed:\n{r.stdout}\n{r.stderr}")
    _log("relaunching fleet: " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))
    r = subprocess.run(["bash", f"{SERVER_DIR}/scripts/restart_fleet.sh"],
                       env={**os.environ, **env}, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"mate_bench: restart_fleet.sh failed:\n{r.stdout}\n{r.stderr}")


def wait_train_params(args, timeout: int = 300) -> str:
    """Wait for the fresh run's training_runs row after a relaunch; return its
    train_parameters (for the per-trial config parity guard)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(args.db):
            try:
                con = _connect(args.db)
                row = con.execute(
                    "SELECT train_parameters FROM training_runs "
                    "ORDER BY id DESC LIMIT 1").fetchone()
                con.close()
                if row and row[0]:
                    return row[0]
            except Exception:  # noqa: BLE001 — bootstrap mid-write
                pass
        time.sleep(5)
    sys.exit(f"mate_bench: no training_runs row {timeout}s after relaunch — "
             "fleet didn't come up (check tmux cc:server)")


def save_trial_db(dest_dir: str, trial: int, args) -> None:
    """Preserve the trial's DB before the next reset wipes it (WAL-checkpointed —
    the run-22 archive lost a day of rows to a bare-.db copy)."""
    os.makedirs(dest_dir, exist_ok=True)
    try:
        con = sqlite3.connect(args.db, timeout=10)
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.close()
    except Exception as e:  # noqa: BLE001 — busy checkpoint is non-fatal
        _log(f"(wal checkpoint failed: {e} — copying -wal alongside)")
    dst = os.path.join(dest_dir, f"trial{trial}.db")
    shutil.copyfile(args.db, dst)
    if os.path.exists(args.db + "-wal") and os.path.getsize(args.db + "-wal"):
        shutil.copyfile(args.db + "-wal", dst + "-wal")
    _log(f"trial DB saved → {dst}")
    # Archive the trial's ccz1 chunks (server games dir) alongside the DB: they
    # carry per-game record counts + moves_left (→ exact plies + full/fast move
    # split), which bench_visits.py turns into the ops-noise-immune metric
    # (search visits to crossing). reset_fleet wipes the dir — this is the only
    # moment the data exists.
    games_dir = os.path.join(SERVER_DIR, "games")
    if os.path.isdir(games_dir):
        tar = os.path.join(dest_dir, f"trial{trial}_games.tar.gz")
        r = subprocess.run(["tar", "czf", tar, "-C", games_dir, "."],
                           capture_output=True, text=True)
        if r.returncode == 0:
            _log(f"trial chunks archived → {tar}")
        else:
            _log(f"(chunk archive FAILED rc={r.returncode}: {r.stderr.strip()})")


def run_trials(args) -> int:
    if args.no_stop:
        sys.exit("mate_bench: --no-stop is incompatible with --trials "
                 "(each trial must stop + reset the fleet)")
    env = cron_env()
    label = env["RUN_NAME"]
    stamp_dir = os.path.join(
        TRIALS_DIR,
        f"{label}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%d-%H%M}")
    _log(f"=== {args.trials} TRIALS of [{label}] — between-trial resets WIPE fleet "
         f"state; per-trial DBs → {stamp_dir}/ ===")
    stamps: list[dict] = []
    base_params = ""
    # Distinct trainer seed per trial => independent net inits + replay sampling.
    # Trial 1 keeps the seed the run was LAUNCHED with (cron SEED, default 0);
    # later trials increment from it.
    base_seed = int(env.get("SEED", "0") or 0)
    for t in range(1, args.trials + 1):
        trial_seed = base_seed + t - 1
        if t > 1:
            relaunch_trial({**env, "SEED": str(trial_seed)})
            params = wait_train_params(args)
            if base_params and params != base_params:
                sys.exit(f"mate_bench: trial {t} trainParams DIVERGED from trial 1 "
                         f"— aborting the experiment.\n  trial 1: {base_params}\n"
                         f"  trial {t}: {params}")
            _log(f"[trial {t}/{args.trials}] fleet up (seed {trial_seed}), "
                 "trainParams verified — watching")
        stamp = _watch_until_crossed(args, trial=t, trials=args.trials,
                                     seed=trial_seed)
        if not base_params:
            base_params = stamp.get("train_params", "")
        stamps.append(stamp)
        save_trial_db(stamp_dir, t, args)
    conv = [s for s in stamps if s["converged"]]
    elap = [s["elapsed_s"] for s in conv]
    summary = {
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "type": "summary", "run_name": label,
        "trials": args.trials, "converged": len(conv),
        "threshold": args.threshold, "window": args.window,
        "basis": "self-play-only",
        "elapsed_s": [s["elapsed_s"] for s in stamps],
        "games": [s.get("games") for s in stamps],
        "seeds": [s.get("seed") for s in stamps],
        "trial_dbs": stamp_dir,
    }
    if elap:
        summary["median_s"] = int(statistics.median(elap))
        summary["mean_s"] = int(statistics.mean(elap))
        summary["sd_s"] = int(statistics.stdev(elap)) if len(elap) > 1 else 0
        summary["median_games"] = int(statistics.median(
            s["games"] for s in conv))
    append_stamp(summary, args.results, _log)
    print("\n== TRIALS COMPLETE ==")
    print(f"  {label}: {len(conv)}/{args.trials} converged")
    for s in stamps:
        res = _humanize(s["elapsed_s"]) if s["converged"] else f"DNF ({s['note']})"
        games = f"  {s['games']:,} games" if s["converged"] else ""
        print(f"    trial {s.get('trial', '?')} (seed {s.get('seed', '?')}): {res}{games}")
    if elap:
        print(f"  median {_humanize(summary['median_s'])}   "
              f"mean {_humanize(summary['mean_s'])} ± {_humanize(summary['sd_s'])}   "
              f"median games {summary['median_games']:,}")
    print(f"  per-trial DBs: {stamp_dir}/")
    return 0


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
    ap.add_argument("--trials", type=int, default=None,
                    help="run N full trials (reset_fleet + relaunch between each; "
                         "N=1 runs a single stamped trial — used by resume drivers)")
    ap.add_argument("--no-stop", action="store_true",
                    help="with --watch: stamp but don't end the fleet")
    ap.add_argument("--stamp", action="store_true",
                    help="with --report: retro-stamp the crossing into --results")
    args = ap.parse_args()
    if args.trials:
        return run_trials(args)
    return watch(args) if args.watch else report(args)


if __name__ == "__main__":
    sys.exit(main())
