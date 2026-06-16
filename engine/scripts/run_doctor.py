#!/usr/bin/env python3
"""run_doctor — one-shot health + convergence report for a Chessckers run.

Runs ON the box (via `cc doctor`). Consolidates everything I used to poke for by
hand: process liveness, trainer step/rate/lr + agreement signals, game count +
TRUE newest-game age (mtime, not the lexicographic-buggy field), and the W/B/draw
trend over the whole run.

  cc doctor                         # full report
  cc doctor --csv run_metrics.csv   # also append one metrics row (use in a loop = sampler)
  cc doctor --block 2000            # trend bucket size
"""
import argparse, glob, json, os, re, subprocess, time

SERVER = "/workspace/chessckers/lczero-server"
PROCS = {"cc-server": "server", "trainer_bridge": "bridge", "train_continuous": "trainer",
         "selfplay": "selfplay", "lc0-client": "client"}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def latest_run(root):
    runs = sorted(glob.glob(os.path.join(root, "trainer", "run*")),
                  key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    return os.path.basename(runs[-1]) if runs else "run1"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=SERVER)
    ap.add_argument("--run", default=None)
    ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--csv", default=None, help="append one metrics row here (sampler mode)")
    a = ap.parse_args()
    run = a.run or latest_run(a.root)
    pgn_dir = f"{a.root}/pgns/{run}"
    stats_f = f"{a.root}/trainer/{run}/train_stats.json"

    up = proc_state()
    st = json.load(open(stats_f)) if os.path.exists(stats_f) else {}
    age = newest_age_min(pgn_dir)
    rows = trend(pgn_dir, a.block)

    print(f"== run {run} ==")
    print("procs:   " + "  ".join(f"{lbl}{'✓' if up[lbl] else '✗'}" for lbl in PROCS.values()))
    if st:
        stale = time.time() - st.get("updated", 0)
        print(f"trainer: step {st.get('steps')}  {st.get('steps_per_s')}/s  "
              f"{st.get('games_per_s')} games/s  lr={st.get('lr')}  "
              f"vsign={st.get('value_sign_agree')} ptop1={st.get('policy_top1_agree')}  "
              f"(stats {stale/60:.0f}m old)")
    print(f"games:   {int(sum(r[1] for r in rows))} scored  |  newest "
          + (f"{age:.0f}m ago" if age is not None else "n/a")
          + ("  \033[33m[generation STOPPED]\033[0m" if age and age > 10 else ""))
    if rows:
        print("trend (block start | n | W% B% draw%):")
        for r in rows[::max(1, len(rows) // 14)] + ([rows[-1]] if len(rows) > 1 else []):
            print(f"   #{int(r[0]):>7}  n={int(r[1]):<5} W{r[2]:5.1f}  B{r[3]:5.1f}  d{r[4]:5.1f}")

    if a.csv:
        new = not os.path.exists(a.csv)
        last = rows[-1] if rows else (0, 0, 0, 0, 0)
        with open(a.csv, "a") as f:
            if new:
                f.write("ts,step,games_seen,games_per_s,lr,vsign,ptop1,W,B,draw\n")
            f.write(f"{int(time.time())},{st.get('steps','')},{st.get('games_seen','')},"
                    f"{st.get('games_per_s','')},{st.get('lr','')},{st.get('value_sign_agree','')},"
                    f"{st.get('policy_top1_agree','')},{last[2]},{last[3]},{last[4]}\n")
        print(f"(appended metrics row to {a.csv})")


if __name__ == "__main__":
    main()
