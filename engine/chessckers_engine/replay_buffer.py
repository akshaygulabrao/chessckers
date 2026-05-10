"""File-backed replay buffer for async AlphaZero training.

Each finished game is pickled to its own file under `root/`. Workers append;
the trainer samples positions uniformly across all games in the buffer.

Cross-process coordination is filesystem-only — no locks, no shared memory.
Atomic appends use the `.tmp` + rename pattern so a partially-written game is
never visible to readers. The trainer caches loaded games in memory and
refreshes when the directory mtime changes.

Layout:
    root/
        000_0000000001.pkl    # worker_id=0, game_id=1
        000_0000000002.pkl
        001_0000000001.pkl    # worker_id=1, game_id=1
        ...

Each .pkl is a `list[AZExample]` (one game's positions).
"""
from __future__ import annotations

import os
import pickle
import random
from pathlib import Path

from chessckers_engine.selfplay_az import AZExample


class ReplayBuffer:
    """File-backed sliding-window buffer of AZExamples, keyed by game.

    `max_games` bounds disk + memory usage; oldest files (by mtime) are
    pruned when the buffer overflows.
    """

    def __init__(self, root: str | os.PathLike, max_games: int = 4000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_games = max_games
        self._cache: dict[str, list[AZExample]] = {}
        self._flat: list[AZExample] = []
        self._last_mtime: float = -1.0

    def append_game(self, worker_id: int, game_id: int, examples: list[AZExample]) -> Path:
        """Atomically write one game's examples. Returns the final path."""
        name = f"{worker_id:03d}_{game_id:010d}.pkl"
        target = self.root / name
        tmp = target.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)  # atomic on POSIX
        return target

    def _maybe_refresh(self) -> None:
        """Re-scan the root dir if its mtime changed; prune to `max_games`."""
        try:
            cur_mtime = self.root.stat().st_mtime
        except FileNotFoundError:
            return
        if cur_mtime == self._last_mtime:
            return
        # Files can vanish between glob() and stat() under concurrent worker
        # writes / sidecar churn — treat a missing file as oldest so it falls
        # off in the prune step (or is silently skipped by the load loop).
        def _safe_mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except FileNotFoundError:
                return 0.0
        files = sorted(self.root.glob("*.pkl"), key=_safe_mtime)
        # Prune oldest if over cap.
        if len(files) > self.max_games:
            for old in files[: len(files) - self.max_games]:
                try:
                    old.unlink()
                except FileNotFoundError:
                    pass
            files = files[-self.max_games :]
        present = {p.name for p in files}
        for name in list(self._cache.keys()):
            if name not in present:
                del self._cache[name]
        for p in files:
            if p.name in self._cache:
                continue
            try:
                with open(p, "rb") as f:
                    self._cache[p.name] = pickle.load(f)
            except (EOFError, pickle.UnpicklingError, FileNotFoundError,
                    ValueError, MemoryError, OSError):
                # Partial write or vanished file — skip; we'll retry next refresh.
                continue
        self._flat = [ex for examples in self._cache.values() for ex in examples]
        self._last_mtime = cur_mtime

    def count_games(self) -> int:
        self._maybe_refresh()
        return len(self._cache)

    def count_examples(self) -> int:
        self._maybe_refresh()
        return len(self._flat)

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[AZExample]:
        """Uniform sample over positions across all cached games (with replacement)."""
        self._maybe_refresh()
        if not self._flat:
            return []
        r = rng or random
        return r.choices(self._flat, k=batch_size)
