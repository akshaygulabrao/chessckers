#!/usr/bin/env python3
"""memguard — keep the container's cgroup RAM usage far from memory.max.

The box's OOM killer (cgroup memory.max ~197GiB, page-cache-accounted) has a
kill history (oom_kill 117 as of 2026-07-20, incl. the run-23/24 trainer
SIGKILLs and the run-25 tmux massacre). Our fleet's anon memory is a few GB;
what reaches the ceiling is file cache from the write-once-read-once chunk
pipeline plus dirty-page bursts at bulk deletes. Every minute this:

  1. sync                       — dirty pages stay small => always reclaimable;
  2. samples cgroup memory + PSI + oom_kill into memguard.jsonl (time series
     for postmortems — dmesg is blocked in the container);
  3. fadvise(DONTNEED)-sweeps the fleet's IO dirs when file cache > CACHE_HIGH;
  4. re-asserts OOM victim steering: heavyweights (trainer/client/engine) very
     killable, plumbing (tmux/cron/drivers) less so (lowering may EPERM in an
     unprivileged container — best-effort);
  5. appends to ALERTS.log on new oom_kills, >ALERT_FRAC of limit, or PSI spike.

Cron (flock so a slow sweep can't stack):
  * * * * * flock -n /workspace/chessckers/memguard.lock python3 /workspace/chessckers/memguard.py >> /workspace/chessckers/memguard.err 2>&1

Stdlib-only; safe to run any time (read-only apart from cache drops + adj).
"""
import json
import os
import re
import subprocess
import time

WS = "/workspace/chessckers"
LOG = f"{WS}/memguard.jsonl"
ALERTS = f"{WS}/ALERTS.log"
CG = "/sys/fs/cgroup"
SWEEP_DIRS = [f"{WS}/lczero-server/games", f"{WS}/lczero-server/pgns",
              f"{WS}/lczero-server/networks", f"{WS}/lczero-server/trainer"]
CACHE_HIGH = 4 << 30          # sweep when cgroup file cache exceeds 4GiB
ALERT_FRAC = 0.5              # alert when memory.current > 50% of memory.max
PSI_ALERT = 10.0              # memory.pressure some avg10 (%)


def rd(path: str) -> str:
    try:
        return open(path).read()
    except OSError:
        return ""


def stat_map(txt: str) -> dict:
    d = {}
    for ln in txt.splitlines():
        k, _, v = ln.partition(" ")
        if v.strip().isdigit():
            d[k] = int(v)
    return d


def adj(pattern: str, val: int) -> None:
    """Set oom_score_adj on processes matching pattern (bracketed patterns so
    pgrep can't self-match). Lowering below 0 may EPERM — ignore."""
    pids = subprocess.run(["pgrep", "-f", pattern],
                          capture_output=True, text=True).stdout.split()
    for pid in pids:
        try:
            open(f"/proc/{pid}/oom_score_adj", "w").write(str(val))
        except OSError:
            pass


def main() -> None:
    os.sync()
    cur = int(rd(f"{CG}/memory.current") or 0)
    mx_raw = rd(f"{CG}/memory.max").strip()
    mx = int(mx_raw) if mx_raw.isdigit() else 0
    st = stat_map(rd(f"{CG}/memory.stat"))
    ev = stat_map(rd(f"{CG}/memory.events"))
    m = re.search(r"some avg10=([\d.]+)", rd(f"{CG}/memory.pressure"))
    psi = float(m.group(1)) if m else 0.0

    adj(r"train_continuou[s]", 900)
    adj(r"lc0-clien[t]", 900)
    adj(r"akshay-chessckers-[0]", 900)
    adj(r"cc-serve[r]", 300)
    for pat in (r"tmux", r"cron", r"mate_benc[h].py", r"bench_resum[e].sh"):
        adj(pat, -400)

    swept = 0
    if st.get("file", 0) > CACHE_HIGH:
        for root in SWEEP_DIRS:
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    try:
                        fd = os.open(os.path.join(dp, fn), os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                            swept += 1
                        finally:
                            os.close(fd)
                    except OSError:
                        pass

    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           "current": cur, "anon": st.get("anon", 0), "file": st.get("file", 0),
           "dirty": st.get("file_dirty", 0), "psi10": psi,
           "oom_kill": ev.get("oom_kill", 0), "swept": swept}
    prev = {}
    try:
        with open(LOG) as f:
            last = ""
            for last in f:
                pass
        if last.strip():
            prev = json.loads(last)
    except (OSError, json.JSONDecodeError):
        pass
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

    alerts = []
    if prev and rec["oom_kill"] > prev.get("oom_kill", rec["oom_kill"]):
        alerts.append(f"OOM KILL: +{rec['oom_kill'] - prev['oom_kill']} "
                      f"(lifetime {rec['oom_kill']})")
    if mx and cur > ALERT_FRAC * mx:
        alerts.append(f"memory.current {cur >> 30}GiB > "
                      f"{int(ALERT_FRAC * 100)}% of limit {mx >> 30}GiB")
    if psi > PSI_ALERT:
        alerts.append(f"memory pressure some avg10={psi}")
    if alerts:
        with open(ALERTS, "a") as f:
            for a in alerts:
                f.write(f"{rec['ts']} [memguard] {a}\n")


if __name__ == "__main__":
    main()
