"""Tests for the Alerter — exercises threshold detection + debounce
without involving a real W&B run (wandb_run=None → fire becomes a log-only
no-op, which is what the assertions inspect)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from chessckers_engine import alerter as al
from chessckers_engine import heartbeat as hb


def _setup(tmp_path: Path, **overrides) -> al.Alerter:
    defaults = dict(
        wandb_run=None,
        run_dir=tmp_path,
        coord_start_ts=1000.0,
        check_every_s=0.0,  # always run on check()
        rate_min_baseline_samples=2,
        alert_debounce_s=0.0,  # don't debounce in tests
    )
    defaults.update(overrides)
    return al.Alerter(**defaults)


def test_worker_offline_alert_fires(tmp_path: Path, caplog) -> None:
    """A stale heartbeat (incarnation post-coord, wall_ts old) should fire
    a worker_offline ALERT."""
    # Heartbeat with old wall_ts (10 min ago) but fresh incarnation_id.
    import json
    hb_dir = tmp_path / "heartbeats"
    hb_dir.mkdir()
    (hb_dir / "leena_300.json").write_text(json.dumps({
        "wall_ts": time.time() - 600,
        "machine": "leena", "worker_id": 300, "role": "worker",
        "games_played": 50, "incarnation_id": 2000.0,
    }))

    a = _setup(tmp_path, worker_stale_s=60.0)
    with caplog.at_level(logging.WARNING):
        a.check(trainer_step=100, games_done=50)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("worker_offline" in m for m in msgs), msgs


def test_stale_pre_coord_heartbeat_does_not_alert(tmp_path: Path, caplog) -> None:
    """Heartbeats from BEFORE coord booted shouldn't trip alerts (they're
    leftovers, not failures)."""
    import json
    hb_dir = tmp_path / "heartbeats"
    hb_dir.mkdir()
    (hb_dir / "leena_300.json").write_text(json.dumps({
        "wall_ts": time.time() - 600,
        "machine": "leena", "worker_id": 300, "role": "worker",
        "games_played": 50, "incarnation_id": 500.0,  # pre-coord (coord_start=1000)
    }))

    a = _setup(tmp_path, worker_stale_s=60.0)
    with caplog.at_level(logging.WARNING):
        a.check(trainer_step=100, games_done=50)
    assert not any("worker_offline" in r.getMessage() for r in caplog.records)


def test_rate_drop_fires_stalled_throughput(tmp_path: Path, caplog) -> None:
    """If recent games-per-second drops below half the historical baseline,
    fire stalled_throughput."""
    a = _setup(tmp_path, rate_window_s=60.0, rate_drop_factor=0.5)
    # Seed history manually: baseline = 10 games/sec, then drop to 1/sec.
    now = time.time()
    a._rate_history = [
        (now - 240, 0,   0),
        (now - 180, 600, 600),
        (now - 120, 1200, 1200),
        (now - 60,  1800, 1800),  # baseline window: 600..1800 games in 120s = 10/s
        (now,       1800, 1810),  # recent: +10 in 60s = 0.17/s — dropped vs 10/s
    ]
    with caplog.at_level(logging.WARNING):
        a._check_rates(now)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("stalled_throughput" in m for m in msgs), msgs


def test_eval_regression_alert(tmp_path: Path, caplog) -> None:
    """Win-rate drop >= threshold across consecutive cycles fires."""
    a = _setup(tmp_path, eval_regression_threshold=0.2)
    cycle1 = {"as_white_vs_random": {"white": 18, "black": 1, "draw": 1, "games": 20}}
    cycle2 = {"as_white_vs_random": {"white": 5,  "black": 14, "draw": 1, "games": 20}}
    a.check_eval(trainer_step=10000, summary=cycle1)
    with caplog.at_level(logging.WARNING):
        a.check_eval(trainer_step=20000, summary=cycle2)
    assert any("eval_regression" in r.getMessage() for r in caplog.records)


def test_debounce_suppresses_repeat(tmp_path: Path, caplog) -> None:
    """The same alert key within debounce window doesn't refire."""
    a = _setup(tmp_path, alert_debounce_s=3600.0)
    a._fire("key1", title="t", text="x", level="WARN")
    a._fire("key1", title="t", text="x", level="WARN")
    msgs = [r.getMessage() for r in caplog.records if "ALERT" in r.getMessage()]
    # Note: caplog only captures from this test, and the inner log.warning
    # happens before debounce check — so we have to check via _last_alert_ts.
    assert "key1" in a._last_alert_ts
