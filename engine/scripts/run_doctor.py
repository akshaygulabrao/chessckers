#!/usr/bin/env python3
"""run_doctor — one-shot health + convergence report for a Chessckers run.

Runs ON the box (via `cc doctor`). Consolidates everything I used to poke for by
hand: process liveness, trainer step/rate/lr + agreement signals, game count +
TRUE newest-game age (mtime, not the lexicographic-buggy field), the W/B/draw
trend over the whole run, and the anchor-gauntlet strength trend (last Elo ± CI,
Elo/24h slope, saturation/PLATEAU flags, ALERTS.log tail, gate stall screen).

  cc doctor                         # full report
  cc doctor --csv run_metrics.csv   # also append one metrics row (use in a loop = sampler)
  cc doctor --block 2000            # trend bucket size
  cc doctor --rate-window 60        # minutes to average steps/s + games/s over
"""
import argparse, datetime, glob, json, os, re, subprocess, sys, time

SERVER = "/workspace/chessckers/lczero-server"
LEDGER = "/workspace/chessckers/engine/docs/runs"  # synced repo ledger — human run numbers
PROCS = {"cc-server": "server", "trainer_bridge": "bridge", "train_continuous": "trainer",
         "selfplay": "selfplay", "lc0-client": "client"}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def latest_run(root):
    runs = sorted(glob.glob(os.path.join(root, "trainer", "run*")),
                  key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    return os.path.basename(runs[-1]) if runs else "run1"


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


def run_clock(root) -> str | None:
    """Return a clock line for the current training run, or None on failure."""
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{root}/chessckers.db?mode=ro", uri=True, timeout=2)
        row = con.execute(
            "SELECT datetime(created_at), datetime('now') FROM training_runs "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        con.close()
        if not row or not row[0]:
            return None
        start = datetime.datetime.fromisoformat(row[0]).replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.fromisoformat(row[1]).replace(tzinfo=datetime.timezone.utc)
        elapsed = _humanize_elapsed((now - start).total_seconds())
        return f"clock:   {elapsed}  (run started {row[0][:16]} UTC)"
    except Exception:
        return None


def run_label(root):
    """Human run identity. The on-disk dir is always run1 (lc0 TrainingRun id, resets
    with the DB every fresh run) — the real identity is the ledger number (newest
    docs/runs/runN.md) + the RUN_NAME the server was bootstrapped with (DB description)."""
    name = None
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{root}/chessckers.db?mode=ro", uri=True, timeout=2)
        row = con.execute("select description from training_runs order by id desc limit 1").fetchone()
        con.close()
        name = row[0] if row else None
    except Exception:
        pass
    nums = [int(m.group(1)) for p in glob.glob(f"{LEDGER}/run[0-9]*.md")
            if (m := re.match(r"run(\d+)\.md$", os.path.basename(p)))]
    return " — ".join(x for x in [f"run{max(nums)}" if nums else None, name] if x)


def window_rates(metrics_f, minutes):
    """steps/s + games/s averaged over the last `minutes` of train_metrics.jsonl.
    The train_stats.json heartbeat is a 60s point-sample: games arrive in chunk
    bursts minutes apart and the replay throttle stalls steps between them, so
    the instantaneous rates read 0.0 nearly always. Returns (sps, gps, span_min)."""
    try:
        with open(metrics_f) as f:
            lines = f.readlines()[-720:]
    except OSError:
        return None
    cut = time.time() - minutes * 60
    rows = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("updated", 0) >= cut:
            rows.append(r)
    if len(rows) < 2 or rows[-1]["updated"] <= rows[0]["updated"]:
        return None
    dt = rows[-1]["updated"] - rows[0]["updated"]
    return (max(0, rows[-1]["steps"] - rows[0]["steps"]) / dt,
            max(0, rows[-1]["games_seen"] - rows[0]["games_seen"]) / dt,
            dt / 60)


def proc_state():
    out = sh("ps -eo args | grep -iE '%s' | grep -v grep" % "|".join(PROCS))
    return {label: any(key in l for l in out.splitlines()) for key, label in PROCS.items()}


def newest_age_min(pgn_dir):
    now = time.time()
    t = sh(f"find {pgn_dir} -name '*.pgn' -printf '%T@\\n' 2>/dev/null | sort -n | tail -1").strip()
    return max(0.0, now - float(t)) / 60 if t else None


def trend(pgn_dir, block):
    cmd = (f"cd {pgn_dir} && find . -name '*.pgn' | xargs grep -HoE '1/2-1/2|1-0|0-1' 2>/dev/null "
           f"| sed 's#^\\./##; s#\\.pgn:# #' "
           f"| awk '{{n=$1;r=$2;b=int(n/{block}); t[b]++; "
           f"if(r==\"1-0\")w[b]++; else if(r==\"0-1\")k[b]++; else d[b]++}} "
           f"END{{for(b in t) printf \"%d %d %.1f %.1f %.1f\\n\", b*{block},t[b],"
           f"100*w[b]/t[b],100*k[b]/t[b],100*d[b]/t[b]}}'")
    rows = [tuple(float(x) for x in ln.split()) for ln in sh(cmd).splitlines() if len(ln.split()) == 5]
    return sorted(rows, key=lambda r: r[0])


def strength_trend(root, run):
    """Anchor-gauntlet strength trend: per anchor the last Elo ± CI, the Elo/24h
    slope over its last 3 rows, saturation + PLATEAU flags (rules imported from
    anchor_gauntlet.py so this report can never drift from the alarm), then the
    ALERTS.log tail and the gate stall-floor screen. Read-only, fully non-fatal."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import anchor_gauntlet as ag
        rows = ag.load_history(f"{root}/trainer/{run}/anchor_gauntlet.jsonl")
        if not rows:
            print("anchors: no anchor data")
        else:
            newest = rows[-1]
            names = [a.get("anchor") for a in newest.get("anchors", []) if a.get("anchor")]
            sat = ag.saturated_anchors(rows, names)
            bn = newest.get("best_net")
            print(f"anchors: {len(rows)} rows, newest {(time.time() - newest.get('ts', 0))/3600:.1f}h ago"
                  f"  current={newest.get('current')}"
                  + (f"  best_net=#{bn}" if bn is not None else ""))
            for name in names:
                series = ag.anchor_series(rows, name)
                last = series[-1]
                ci = (last.get("elo_hi", 0) - last.get("elo_lo", 0)) / 2
                s3 = series[-3:]
                slope = ((s3[-1].get("elo", 0) - s3[0].get("elo", 0))
                         / (s3[-1]["ts"] - s3[0]["ts"]) * 86400
                         if len(s3) >= 2 and s3[-1]["ts"] > s3[0]["ts"] else None)
                pc = ag.plateau_check(rows, name)
                # a saturated anchor is flat because it's CAPPED, not stalled — no PLATEAU
                flags = ("  saturated" if name in sat
                         else "  \033[33mPLATEAU\033[0m" if pc and pc[0] else "")
                print(f"   {name:>10}  elo {last.get('elo', 0):>+7.1f} ±{ci:<4.0f} "
                      f"slope {f'{slope:+.0f}/24h' if slope is not None else 'n/a':>9}{flags}")
        alerts = os.path.abspath(os.path.join(root, "..", "ALERTS.log"))
        if os.path.exists(alerts):
            lines = open(alerts).readlines()
            # ALERTS.log deliberately outlives fleet resets, so after an archive it
            # opens with the PREVIOUS run's alarms — show only this run's (+untagged).
            import _run_ident
            rn = _run_ident.run_name()
            n_other = 0
            if rn:
                mine = [ln for ln in lines if f"run={rn} " in ln or "run=" not in ln]
                n_other = len(lines) - len(mine)
                lines = mine
            tail = lines[-5:]
            print(f"alerts ({alerts}, last {len(tail)}"
                  + (f"; {n_other} from other runs suppressed" if n_other else "") + "):")
            for ln in tail:
                print(f"   {ln.rstrip()}")
        screen = ag.gate_stall_screen(f"{root}/chessckers.db")
        if screen is None:
            print("gate screen: n/a (<10 done matches or no db)")
        else:
            print(f"gate screen: {'STALL-FLOOR' if screen[0] else 'ok'} ({screen[1]})")
    except Exception as e:
        print(f"anchors: no anchor data ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=SERVER)
    ap.add_argument("--run", default=None)
    ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--csv", default=None, help="append one metrics row here (sampler mode)")
    ap.add_argument("--rate-window", type=int, default=30,
                    help="minutes to average steps/s + games/s over (train_metrics.jsonl)")
    a = ap.parse_args()
    run = a.run or latest_run(a.root)
    pgn_dir = f"{a.root}/pgns/{run}"
    stats_f = f"{a.root}/trainer/{run}/train_stats.json"
    label = run_label(a.root)

    up = proc_state()
    st = json.load(open(stats_f)) if os.path.exists(stats_f) else {}
    rates = window_rates(f"{a.root}/trainer/{run}/train_metrics.jsonl", a.rate_window)
    age = newest_age_min(pgn_dir)
    rows = trend(pgn_dir, a.block)

    print(f"== {label}  (dir {run}) ==" if label else f"== run {run} ==")
    clk = run_clock(a.root)
    if clk:
        print(clk)
    print("procs:   " + "  ".join(f"{lbl}{'✓' if up[lbl] else '✗'}" for lbl in PROCS.values()))
    if st:
        stale = time.time() - st.get("updated", 0)
        if rates:
            rate_s = f"{rates[0]:.2f} steps/s  {rates[1]*3600:.1f} games/h [{rates[2]:.0f}m avg]"
        else:
            rate_s = f"{st.get('steps_per_s')}/s  {st.get('games_per_s')} games/s"
        print(f"trainer: step {st.get('steps')}  {rate_s}  lr={st.get('lr')}  "
              f"vsign={st.get('value_sign_agree')} ptop1={st.get('policy_top1_agree')}  "
              f"(stats {stale/60:.0f}m old)")
    print(f"games:   {int(sum(r[1] for r in rows))} scored  |  newest "
          + (f"{age:.0f}m ago" if age is not None else "n/a")
          + ("  \033[33m[generation STOPPED]\033[0m" if age and age > 10 else ""))
    if rows:
        print("trend (block start | n | W% B% draw%):")
        for r in rows[::max(1, len(rows) // 14)] + ([rows[-1]] if len(rows) > 1 else []):
            print(f"   #{int(r[0]):>7}  n={int(r[1]):<5} W{r[2]:5.1f}  B{r[3]:5.1f}  d{r[4]:5.1f}")
    strength_trend(a.root, run)

    if a.csv:
        new = not os.path.exists(a.csv)
        last = rows[-1] if rows else (0, 0, 0, 0, 0)
        with open(a.csv, "a") as f:
            if new:
                f.write("ts,step,games_seen,games_per_s,lr,vsign,ptop1,W,B,draw\n")
            gps = round(rates[1], 4) if rates else st.get('games_per_s', '')
            f.write(f"{int(time.time())},{st.get('steps','')},{st.get('games_seen','')},"
                    f"{gps},{st.get('lr','')},{st.get('value_sign_agree','')},"
                    f"{st.get('policy_top1_agree','')},{last[2]},{last[3]},{last[4]}\n")
        print(f"(appended metrics row to {a.csv})")


if __name__ == "__main__":
    main()
