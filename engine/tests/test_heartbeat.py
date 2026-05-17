"""Tests for the heartbeat protocol — write/read round-trip + the
authoritative game counter's filtering of stale incarnations."""
from __future__ import annotations

import time
from pathlib import Path

from chessckers_engine import heartbeat as hb


def test_write_read_roundtrip(tmp_path: Path) -> None:
    hb.write(tmp_path, machine="local", worker_id=0, role="worker",
             games_played=5, incarnation_id=1000.0)
    rows = hb.read_all(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["machine"] == "local"
    assert r["worker_id"] == 0
    assert r["games_played"] == 5
    assert r["incarnation_id"] == 1000.0
    assert r["role"] == "worker"
    assert "wall_ts" in r


def test_overwrite_same_worker(tmp_path: Path) -> None:
    hb.write(tmp_path, machine="leena", worker_id=300, role="worker",
             games_played=1, incarnation_id=2000.0)
    hb.write(tmp_path, machine="leena", worker_id=300, role="worker",
             games_played=42, incarnation_id=2000.0)
    rows = hb.read_all(tmp_path)
    assert len(rows) == 1
    assert rows[0]["games_played"] == 42


def test_count_filters_stale_incarnations(tmp_path: Path) -> None:
    """A heartbeat with incarnation_id < coord_start_ts is a leftover from
    a prior run — its games must not contribute to the current run's count."""
    # Pretend the coord started at wall_ts=5000.
    coord_start = 5000.0

    # Stale heartbeat — incarnation from before coord boot. Don't count.
    hb.write(tmp_path, machine="leena", worker_id=300, role="worker",
             games_played=4500, incarnation_id=3000.0)
    # Fresh heartbeat — incarnation after coord boot. Count it.
    hb.write(tmp_path, machine="local", worker_id=0, role="worker",
             games_played=120, incarnation_id=6000.0)
    # Boundary case — equal counts as fresh.
    hb.write(tmp_path, machine="vast", worker_id=201, role="worker",
             games_played=42, incarnation_id=5000.0)

    total = hb.count_games_for_run(tmp_path, coord_start)
    assert total == 120 + 42, f"expected fresh sum, got {total}"


def test_liveness_marks_old_heartbeats_stale(tmp_path: Path) -> None:
    """A heartbeat whose wall_ts is older than fresh_window_s is dead."""
    # Write a heartbeat, then manually backdate the file's wall_ts by patching
    # the JSON content. (Writing real-past timestamps is the simplest way.)
    import json
    p = tmp_path / "heartbeats"
    p.mkdir()
    old_payload = {
        "wall_ts": time.time() - 600,  # 10 min ago
        "machine": "leena", "worker_id": 300, "role": "worker",
        "games_played": 50, "incarnation_id": 1.0,
    }
    new_payload = {**old_payload, "wall_ts": time.time(), "worker_id": 301}
    (p / "leena_300.json").write_text(json.dumps(old_payload))
    (p / "leena_301.json").write_text(json.dumps(new_payload))

    rows = hb.liveness(tmp_path, fresh_window_s=90.0)
    by_wid = {r["worker_id"]: r for r in rows}
    assert by_wid[300]["alive"] is False
    assert by_wid[301]["alive"] is True


def test_no_heartbeats_dir_returns_zero(tmp_path: Path) -> None:
    """Counter must work cleanly on a freshly created run dir."""
    assert hb.read_all(tmp_path) == []
    assert hb.count_games_for_run(tmp_path, 0.0) == 0
    assert hb.liveness(tmp_path) == []
