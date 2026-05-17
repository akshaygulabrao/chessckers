"""W&B-backed alerter for the chessckers training stack.

Periodically checked from the coordinator's main loop. Triggers
`wandb.alert(...)` on threshold violations:

  - **stalled_trainer**: trainer step rate over last N seconds drops below
    half the historical baseline. Catches deadlocked trainer threads.
  - **stalled_throughput**: games/min over last N seconds drops below half
    the historical baseline. Catches sidecar deaths, worker crashes,
    remote-box outages.
  - **worker_offline**: any heartbeat older than `worker_stale_s` seconds.
    Catches per-worker failures the throughput aggregate masks.
  - **eval_regression**: current eval cycle's win rate vs any opponent
    drops by `eval_regression_threshold` (0.2 default) from the previous
    cycle. Catches training divergence.

Each alert is debounced — same alert won't refire within
`alert_debounce_s` seconds.

W&B routes alerts to email + Slack per the user's account settings; no
extra infra needed beyond a working wandb run.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from chessckers_engine import heartbeat as _hb

log = logging.getLogger("chessckers_engine.alerter")


@dataclass
class Alerter:
    """Stateful threshold watcher. Pass a wandb_run to enable; otherwise
    becomes a no-op so callers don't need to branch."""

    wandb_run: object = None
    run_dir: Path = Path(".")
    coord_start_ts: float = 0.0

    # Sampling: how often check() should do work (caller can call more often).
    check_every_s: float = 60.0

    # Liveness threshold.
    worker_stale_s: float = 300.0

    # Throughput / step-rate baseline window (rolling).
    rate_window_s: float = 300.0
    rate_min_baseline_samples: int = 3
    # Alert if current rate < baseline * rate_drop_factor.
    rate_drop_factor: float = 0.5

    # Eval-regression threshold (absolute win-rate drop).
    eval_regression_threshold: float = 0.2

    # Debounce: don't refire the same alert within this window.
    alert_debounce_s: float = 1800.0

    # Internal state.
    _last_check_ts: float = 0.0
    _rate_history: list[tuple[float, int, int]] = field(default_factory=list)
    # (wall_ts, trainer_step, games_done)
    _last_alert_ts: dict[str, float] = field(default_factory=dict)
    _last_eval_winrates: dict[str, float] = field(default_factory=dict)

    def _fire(self, key: str, title: str, text: str, level: str = "WARN") -> None:
        """Send an alert, respecting per-key debounce. No-op without wandb."""
        now = time.time()
        if now - self._last_alert_ts.get(key, 0) < self.alert_debounce_s:
            return
        self._last_alert_ts[key] = now
        log.warning("ALERT [%s] %s — %s", key, title, text)
        if self.wandb_run is None:
            return
        try:
            # wandb.alert is on the top-level module, not the run; map levels.
            import wandb as _wandb
            wandb_level = {"INFO": _wandb.AlertLevel.INFO,
                           "WARN": _wandb.AlertLevel.WARN,
                           "ERROR": _wandb.AlertLevel.ERROR}.get(level, _wandb.AlertLevel.WARN)
            _wandb.alert(title=title, text=text, level=wandb_level,
                         wait_duration=int(self.alert_debounce_s))
        except Exception as e:
            log.debug("wandb.alert failed: %s", e)

    def check(self, trainer_step: int, games_done: int) -> None:
        """Call from the main loop. Cheap if invoked more often than
        check_every_s (returns early)."""
        now = time.time()
        if now - self._last_check_ts < self.check_every_s:
            return
        self._last_check_ts = now

        self._rate_history.append((now, trainer_step, games_done))
        # Keep only the last `rate_window_s` of samples plus a small head room
        # so baseline math works near startup.
        cutoff = now - max(self.rate_window_s * 3, 600.0)
        self._rate_history = [t for t in self._rate_history if t[0] >= cutoff]

        self._check_liveness(now)
        self._check_rates(now)

    def _check_liveness(self, now: float) -> None:
        rows = _hb.liveness(self.run_dir, fresh_window_s=self.worker_stale_s)
        # Only flag workers from THIS run (incarnation post-coord-start) —
        # otherwise we'd alert on every stale heartbeat from prior runs.
        dead = [r for r in rows
                if float(r.get("incarnation_id", 0)) >= self.coord_start_ts
                and not r.get("alive", False)]
        for r in dead:
            key = f"worker_offline/{r['machine']}/{r['worker_id']}"
            self._fire(
                key,
                title=f"Worker offline: {r['machine']}/{r['worker_id']}",
                text=(f"No heartbeat in {r['age_s']:.0f}s "
                      f"(threshold {self.worker_stale_s:.0f}s). "
                      f"Last games_played={r.get('games_played')}."),
                level="WARN",
            )

    def _check_rates(self, now: float) -> None:
        # Find the oldest sample within the window.
        window_start = now - self.rate_window_s
        window = [t for t in self._rate_history if t[0] >= window_start]
        if len(window) < 2:
            return
        # Recent rate: diff between window endpoints.
        t0, s0, g0 = window[0]
        t1, s1, g1 = window[-1]
        dt = max(t1 - t0, 1e-6)
        recent_step_rate = (s1 - s0) / dt
        recent_game_rate = (g1 - g0) / dt

        # Baseline: rate over the older half of history (if we have enough).
        if len(self._rate_history) >= self.rate_min_baseline_samples * 2:
            half = len(self._rate_history) // 2
            older = self._rate_history[:half]
            ot0, os0, og0 = older[0]
            ot1, os1, og1 = older[-1]
            base_step_rate = (os1 - os0) / max(ot1 - ot0, 1e-6)
            base_game_rate = (og1 - og0) / max(ot1 - ot0, 1e-6)
            if base_step_rate > 0 and recent_step_rate < base_step_rate * self.rate_drop_factor:
                self._fire(
                    "stalled_trainer",
                    title="Trainer step rate dropped",
                    text=(f"recent={recent_step_rate*60:.1f} steps/min vs "
                          f"baseline={base_step_rate*60:.1f}. "
                          f"Trainer may be stalled."),
                    level="WARN",
                )
            if base_game_rate > 0 and recent_game_rate < base_game_rate * self.rate_drop_factor:
                self._fire(
                    "stalled_throughput",
                    title="Game throughput dropped",
                    text=(f"recent={recent_game_rate*60:.1f} games/min vs "
                          f"baseline={base_game_rate*60:.1f}. "
                          f"Sidecar dead? Remote workers offline?"),
                    level="WARN",
                )

    def check_eval(self, trainer_step: int, summary: dict) -> None:
        """Call after each eval cycle. Compares per-(opponent, side) win
        rates to the previous cycle and alerts on a meaningful regression."""
        for key, result in summary.items():
            if not (isinstance(result, dict) and key.startswith("as_")):
                continue
            games = max(int(result.get("games", 0)), 1)
            side_color = "white" if "as_white" in key else "black"
            wr = result.get(side_color, 0) / games
            prev = self._last_eval_winrates.get(key)
            self._last_eval_winrates[key] = wr
            if prev is None:
                continue
            if prev - wr >= self.eval_regression_threshold:
                self._fire(
                    f"eval_regression/{key}",
                    title=f"Eval regression on {key}",
                    text=(f"win_rate dropped {prev:.2f} -> {wr:.2f} at step "
                          f"{trainer_step} (>= {self.eval_regression_threshold} drop)."),
                    level="WARN",
                )
