"""File-backed heartbeat protocol for the distributed self-play stack.

Each worker writes a tiny JSON file every game (and at startup) summarizing
its liveness + cumulative games. The coordinator reads the directory to
derive both:

  - **liveness**: which workers have heartbeated in the last N seconds.
  - **authoritative game counter**: sum of `games_played` across workers
    *for this coordinator's incarnation*, robust against rsync-of-stale-
    files and against buffer pruning (which broke our previous mtime-
    based counter).

Layout::

    <run_dir>/heartbeats/<machine>_<worker_id>.json     # one file per worker

JSON shape::

    {
      "wall_ts":        1747469000.0,    # time.time() at write
      "machine":        "vast",          # tag (local / leena / vast / ...)
      "worker_id":      201,             # numeric id, unique within machine
      "role":           "worker",        # worker | trainer | sidecar | coord
      "games_played":   412,             # monotonic, this incarnation
      "incarnation_id": 1747468500.0,    # time.time() at worker startup
    }

The coordinator captures its own start wall-clock and uses it to decide
whether each heartbeat is "from this run" (`incarnation_id >= coord_start`)
or "stale leftover from a previous run" (`incarnation_id < coord_start`).
Stale files contribute 0 to the game counter — this is what fixes the
local-006 day-one bug where rsync'd Leena files made the coordinator
think 17k games had been played in 68s.

Atomic writes via `.tmp + os.replace` so a torn write is never visible
to readers.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional


def write(
    run_dir: str | os.PathLike,
    *,
    machine: str,
    worker_id: int,
    role: str,
    games_played: int,
    incarnation_id: float,
) -> Path:
    """Atomically write a heartbeat file. Returns the final path."""
    run_dir = Path(run_dir)
    hb_dir = run_dir / "heartbeats"
    hb_dir.mkdir(parents=True, exist_ok=True)
    target = hb_dir / f"{machine}_{worker_id}.json"
    tmp = target.with_suffix(".json.tmp")
    payload = {
        "wall_ts": time.time(),
        "machine": machine,
        "worker_id": worker_id,
        "role": role,
        "games_played": games_played,
        "incarnation_id": incarnation_id,
    }
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, target)
    return target


def read_all(run_dir: str | os.PathLike) -> list[dict]:
    """Return a list of all heartbeat dicts. Silently skips unreadable or
    half-written files (next read will catch the completed version)."""
    hb_dir = Path(run_dir) / "heartbeats"
    if not hb_dir.exists():
        return []
    out: list[dict] = []
    for p in hb_dir.glob("*.json"):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def count_games_for_run(run_dir: str | os.PathLike, coord_start_ts: float) -> int:
    """Authoritative games-played counter for the current coord run.

    Sums `games_played` across all heartbeats whose `incarnation_id` is
    `>= coord_start_ts` — i.e. workers that started during or after this
    coordinator booted. Stale heartbeats from prior runs are excluded.

    This replaces the old mtime-based buffer-file-counting heuristic,
    which broke when:
      a) rsync preserved old mtimes from a prior run's buffer files;
      b) buffer pruning evicted counted files once max_games was hit.
    """
    return sum(
        int(hb.get("games_played", 0))
        for hb in read_all(run_dir)
        if float(hb.get("incarnation_id", 0)) >= coord_start_ts
    )


def liveness(run_dir: str | os.PathLike, fresh_window_s: float = 90.0) -> list[dict]:
    """Annotate every heartbeat with `alive: bool` for status displays.

    `alive` = wall_ts of the heartbeat is within `fresh_window_s` of now.
    A heartbeat that hasn't been re-written in the window is considered
    dead — either the worker crashed, the sync sidecar stalled, or the
    box is offline."""
    now = time.time()
    out = []
    for hb in read_all(run_dir):
        hb["age_s"] = now - float(hb.get("wall_ts", 0))
        hb["alive"] = hb["age_s"] <= fresh_window_s
        out.append(hb)
    return out
