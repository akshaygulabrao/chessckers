#!/usr/bin/env python3
"""Per-source game throughput from the local replay buffer.

Counts files in <run-dir>/buffer/*.pkl, groups by worker_id prefix
(0-99 local, 200-299 vast, 300+ leena), and reports:
  - cumulative games in the buffer
  - games written in the last --window minutes (default 10)
  - per-minute rate, total and per-worker
  - how stale the freshest file from each source is

Worker counts are inferred from the actual unique worker_id prefixes
seen (rather than reading env files) so the per-worker rate stays
honest when N_WORKERS=auto on vast or when workers crash.

Usage:
  scripts/buffer_rate.py
  scripts/buffer_rate.py --window 5
  scripts/buffer_rate.py --run-dir engine/runs/local-006
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path


def source_for_wid(wid: int) -> str:
    if wid < 100:
        return "local"
    if wid < 200:
        # 100-199 was the old local WID_BASE before the bundled-mode
        # rewrite; keep a label for back-compat buffers.
        return "local-old"
    if wid < 300:
        return "vast"
    return "leena"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir", type=Path,
        default=Path("engine/runs/local-005"),
        help="run dir containing buffer/ (default: engine/runs/local-005)",
    )
    p.add_argument(
        "--window", type=int, default=10,
        help="window (minutes) for rate computation (default 10)",
    )
    args = p.parse_args()

    buf = args.run_dir / "buffer"
    if not buf.is_dir():
        print(f"no buffer dir: {buf}", file=sys.stderr)
        return 2

    now = time.time()
    cutoff = now - args.window * 60

    workers_seen: dict[str, set[int]] = defaultdict(set)
    total: dict[str, int] = defaultdict(int)
    recent: dict[str, int] = defaultdict(int)
    latest_mt: dict[str, float] = defaultdict(float)

    for f in buf.glob("*.pkl"):
        try:
            wid = int(f.name.split("_", 1)[0])
        except ValueError:
            continue
        src = source_for_wid(wid)
        workers_seen[src].add(wid)
        mt = f.stat().st_mtime
        total[src] += 1
        if mt > cutoff:
            recent[src] += 1
        if mt > latest_mt[src]:
            latest_mt[src] = mt

    print(f"buffer:  {buf}")
    print(f"window:  last {args.window} min  (cutoff {time.strftime('%H:%M:%S', time.localtime(cutoff))})")
    print(f"now:     {time.strftime('%H:%M:%S', time.localtime(now))}")
    print()
    cols = ("source", "workers", "total", "in-window", "games/min", "per-worker", "latest")
    fmt = "{:<10} {:>8} {:>7} {:>10} {:>11} {:>11} {:>10}"
    print(fmt.format(*cols))
    print(fmt.format(*("-" * len(c) for c in cols)))

    order = ("local", "leena", "vast", "local-old")
    grand_recent = 0
    grand_workers = 0
    for s in order:
        if s not in total:
            continue
        n_workers = len(workers_seen[s])
        rate = recent[s] / args.window
        per_w = (rate / n_workers) if n_workers else 0.0
        age = (now - latest_mt[s]) / 60 if latest_mt[s] else float("inf")
        latest_str = f"{age:>5.1f}m ago" if age != float("inf") else "never"
        print(fmt.format(s, n_workers, total[s], recent[s], f"{rate:.2f}", f"{per_w:.3f}", latest_str))
        grand_recent += recent[s]
        grand_workers += n_workers

    print(fmt.format(*("-" * len(c) for c in cols)))
    print(fmt.format(
        "TOTAL", grand_workers, sum(total.values()), grand_recent,
        f"{grand_recent/args.window:.2f}",
        f"{grand_recent/args.window/grand_workers:.3f}" if grand_workers else "—",
        "",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
