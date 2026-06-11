"""Live training monitor — progress, ETA, and a liveness heartbeat for a
training run (e.g. train_continuous), by tailing its log file (no changes to the trainer).

Answers the three things a raw log doesn't: how far along, when it will
finish, and whether it's still actually advancing (or hung/crashed).

Usage:
    python watch_training.py /tmp/curr_disc2.log     # live, refresh 2s
    python watch_training.py                         # newest /tmp/*.log
    python watch_training.py LOG --once              # one snapshot, no loop
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
ITER_START = re.compile(r"iter (\d+)/(\d+): playing (\d+) games \(sims=(\d+)")
SP_GAME = re.compile(r"  game (\d+)/(\d+): (\w+) in (\d+) plies")
EPOCH = re.compile(r"epoch (\d+) done")
EVAL_GAME = re.compile(r"evaluate: game (\d+)/(\d+) ->")
ITER_DONE = re.compile(
    r"iter (\d+)/(\d+) done \| self-play (\d+)W/(\d+)B/(\d+)D.*?"
    r"vs\.rand W:(\d+)/(\d+)/(\d+) B:(\d+)/(\d+)/(\d+)"
)
MODEL = re.compile(r"model: .*?(\d+) params")
START_FEN = re.compile(r"start FEN\s*:\s*(.+)")


def _ts(line: str) -> datetime | None:
    m = TS.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


def _bar(frac: float, width: int = 22) -> str:
    frac = max(0.0, min(1.0, frac))
    f = int(round(frac * width))
    return "█" * f + "░" * (width - f)


def _hms(secs: float) -> str:
    secs = int(max(0, secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def parse(path: str) -> dict:
    st: dict = {
        "total": None, "gpi": None, "sims": None, "params": None, "fen": None,
        "iter": 0, "phase": "starting", "sub": (0, 0), "epoch": 0,
        "done": 0, "done_times": [], "trend": [], "start_time": None,
        "last_time": None,
    }
    for raw in open(path, encoding="utf-8", errors="replace"):
        t = _ts(raw)
        if t:
            st["last_time"] = t
            if st["start_time"] is None:
                st["start_time"] = t
        if m := START_FEN.search(raw):
            st["fen"] = m.group(1).strip()
        if m := MODEL.search(raw):
            st["params"] = int(m.group(1))
        if m := ITER_START.search(raw):
            st["iter"], st["total"] = int(m.group(1)), int(m.group(2))
            st["gpi"], st["sims"] = int(m.group(3)), int(m.group(4))
            st["phase"], st["sub"], st["epoch"] = "self-play", (0, st["gpi"]), 0
        elif m := SP_GAME.search(raw):
            st["phase"], st["sub"] = "self-play", (int(m.group(1)), int(m.group(2)))
        elif m := EPOCH.search(raw):
            st["phase"], st["epoch"] = "train", int(m.group(1))
        elif m := EVAL_GAME.search(raw):
            st["phase"], st["sub"] = "eval", (int(m.group(1)), int(m.group(2)))
        elif m := ITER_DONE.search(raw):
            it, tot = int(m.group(1)), int(m.group(2))
            st["total"] = tot
            st["done"] = it
            st["phase"] = "between iters"
            if t:
                st["done_times"].append(t)
            st["trend"].append((it, m.group(3), m.group(4), m.group(5),
                                 m.group(8), m.group(9), m.group(10)))
    return st


def render(path: str, st: dict, running: bool) -> str:
    now = datetime.now()
    total = st["total"] or 0
    done = st["done"]
    L = []
    finished = total and done >= total
    status = "✅ DONE" if finished else ("🟢 RUNNING" if running else "🔴 EXITED (not running)")
    L.append(f"═══ training monitor · {os.path.basename(path)} · {status} · {now:%H:%M:%S} ═══")
    cfg = []
    if st["total"]: cfg.append(f"{st['total']} iters")
    if st["gpi"]: cfg.append(f"{st['gpi']} games/iter")
    if st["sims"]: cfg.append(f"sims={st['sims']}")
    if st["params"]: cfg.append(f"model {st['params']/1e6:.2f}M")
    if cfg: L.append("  " + " · ".join(cfg))
    if st["fen"]: L.append(f"  start: {st['fen']}")
    L.append("")

    # Iteration progress
    if total:
        L.append(f"iters    [{_bar(done/total)}] {done}/{total}  ({done/total*100:.0f}%)")
    # Current phase + sub-progress
    if not finished:
        g, G = st["sub"]
        if st["phase"] == "self-play" and G:
            L.append(f"iter {st['iter']}/{total} · SELF-PLAY [{_bar(g/G,10)}] game {g}/{G}")
        elif st["phase"] == "train":
            L.append(f"iter {st['iter']}/{total} · TRAIN · epoch {st['epoch']}")
        elif st["phase"] == "eval" and G:
            L.append(f"iter {st['iter']}/{total} · EVAL [{_bar(g/G,10)}] game {g}/{G}")
        else:
            L.append(f"iter {st['iter']}/{total} · {st['phase']}")
    L.append("")

    # Timing + ETA
    dts = st["done_times"]
    elapsed = (st["last_time"] - st["start_time"]).total_seconds() if st["start_time"] and st["last_time"] else 0
    durs = []
    if st["start_time"] and dts:
        prev = st["start_time"]
        for d in dts:
            durs.append((d - prev).total_seconds()); prev = d
    mean = sum(durs) / len(durs) if durs else None
    line = f"timing: elapsed {_hms(elapsed)}"
    if mean:
        line += f" · mean {_hms(mean)}/iter"
        if not finished and total:
            in_iter = (now - dts[-1]).total_seconds() if dts else elapsed
            eta = max(0, mean * (total - done) - in_iter)
            fin = now.timestamp() + eta
            line += f" · ETA ~{_hms(eta)} · est finish {datetime.fromtimestamp(fin):%H:%M}"
    L.append(line)

    # Heartbeat / liveness
    if st["last_time"]:
        quiet = (now - st["last_time"]).total_seconds()
        if finished:
            hb = "run complete."
        elif not running:
            hb = f"⚠ process not running and only {done}/{total} iters done — exited early; check log tail."
        elif quiet > 600:
            hb = f"⚠ no log line for {_hms(quiet)} — possibly stalled (or a slow high-sim eval game)."
        else:
            hb = f"last log {_hms(quiet)} ago · ✓ advancing"
        L.append(f"heartbeat: {hb}")
    L.append("")

    # Win-rate trend (vs.rand B = net-as-Black, the detector)
    if st["trend"]:
        L.append("trend (vs.rand B = net plays Black):")
        L.append("  iter | self-play W/B/D | vs.rand B W/L/D")
        for row in st["trend"][-12:]:
            it, w, b, d, bw, bl, bd = row
            L.append(f"  {it:>4} | {w:>2}/{b:>2}/{d:>2}        |  {bw}/{bl}/{bd}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default=None, help="log file (default: newest /tmp/*.log)")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--proc", default="selfplay_az_loop", help="process name for the liveness check")
    args = ap.parse_args()

    path = args.log
    if path is None:
        logs = sorted(glob.glob("/tmp/*.log"), key=os.path.getmtime, reverse=True)
        if not logs:
            print("no log given and no /tmp/*.log found", file=sys.stderr); return 2
        path = logs[0]
    if not os.path.exists(path):
        print(f"log not found: {path}", file=sys.stderr); return 2

    def running() -> bool:
        return subprocess.run(["pgrep", "-f", args.proc], capture_output=True).returncode == 0

    if args.once:
        print(render(path, parse(path), running()))
        return 0
    try:
        while True:
            st = parse(path)
            alive = running()
            sys.stdout.write("\033[2J\033[H" + render(path, st, alive) + "\n")
            sys.stdout.flush()
            if st["total"] and st["done"] >= st["total"]:
                print("\n(run complete)"); break
            if not alive and st["done"] > 0 and st["done"] < (st["total"] or 1):
                print("\n(process is no longer running)"); break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
