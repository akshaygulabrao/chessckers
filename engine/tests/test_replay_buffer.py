"""Tests for the file-backed ReplayBuffer used by async self-play training."""
from __future__ import annotations

import os
import pickle
import random
from pathlib import Path

import pytest

from chessckers_engine.replay_buffer import ReplayBuffer
from chessckers_engine.selfplay_az import AZExample


def _ex(value: float = 0.0) -> AZExample:
    """Tiny AZExample for tests; identifying value lets us trace through sampling."""
    return AZExample(
        fen=f"fen-{value}",
        legal_moves=[{"uci": "a1a2"}],
        visit_distribution=[1.0],
        wdl_target=[0.0, 1.0, 0.0],
        moves_left_target=value,  # identifier for tracing through sampling
    )


def test_append_and_count(tmp_path: Path):
    buf = ReplayBuffer(tmp_path)
    buf.append_game(worker_id=0, game_id=1, examples=[_ex(0.0), _ex(1.0)])
    buf.append_game(worker_id=0, game_id=2, examples=[_ex(-1.0)])
    assert buf.count_games() == 2
    assert buf.count_examples() == 3


def test_sample_returns_correct_count(tmp_path: Path):
    buf = ReplayBuffer(tmp_path)
    buf.append_game(0, 1, [_ex(0.0), _ex(1.0), _ex(-1.0)])
    sample = buf.sample(batch_size=10, rng=random.Random(0))
    assert len(sample) == 10
    # All sampled examples come from the appended set.
    values = {ex.moves_left_target for ex in sample}
    assert values <= {0.0, 1.0, -1.0}


def test_sample_empty_buffer_returns_empty(tmp_path: Path):
    buf = ReplayBuffer(tmp_path)
    assert buf.sample(batch_size=4) == []
    assert buf.count_games() == 0
    assert buf.count_examples() == 0


def test_atomic_write_no_partial_file_visible(tmp_path: Path):
    """After append_game, no .tmp file should be visible (rename happened)."""
    buf = ReplayBuffer(tmp_path)
    buf.append_game(0, 1, [_ex()])
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_partial_tmp_file_ignored(tmp_path: Path):
    """A leftover .tmp file (from a crashed write) must not break the reader."""
    # Simulate a partial write: a .tmp file with garbage bytes.
    (tmp_path / "999_0000000099.pkl.tmp").write_bytes(b"\x80\x04garbage")
    # And one valid game.
    buf = ReplayBuffer(tmp_path)
    buf.append_game(0, 1, [_ex(0.5)])
    assert buf.count_games() == 1
    assert buf.count_examples() == 1


def test_corrupted_pkl_skipped_not_raised(tmp_path: Path):
    """If a .pkl is mid-write or truncated, we skip it rather than crash."""
    (tmp_path / "000_0000000007.pkl").write_bytes(b"\x80\x04\x95not-a-real-pickle")
    buf = ReplayBuffer(tmp_path)
    # Doesn't raise; corrupted file just contributes nothing.
    assert buf.count_examples() == 0


def test_prune_drops_oldest_when_over_max(tmp_path: Path):
    buf = ReplayBuffer(tmp_path, max_games=3)
    # Append 5 games. We need distinct mtimes so prune can order them.
    for gid in range(1, 6):
        path = buf.append_game(0, gid, [_ex(float(gid))])
        os.utime(path, (gid * 100.0, gid * 100.0))
    # Force re-scan.
    buf._last_mtime = -1.0
    buf._maybe_refresh()
    assert buf.count_games() == 3
    # Oldest (game_id 1, 2) should be gone; 3,4,5 remain.
    remaining = sorted(p.name for p in tmp_path.glob("*.pkl"))
    assert remaining == [
        "000_0000000003.pkl",
        "000_0000000004.pkl",
        "000_0000000005.pkl",
    ]


def test_multiple_workers_distinct_files(tmp_path: Path):
    buf = ReplayBuffer(tmp_path)
    buf.append_game(0, 1, [_ex(0.0)])
    buf.append_game(1, 1, [_ex(1.0)])  # same game_id, different worker
    buf.append_game(2, 1, [_ex(-1.0)])
    assert buf.count_games() == 3
    files = sorted(p.name for p in tmp_path.glob("*.pkl"))
    assert files == [
        "000_0000000001.pkl",
        "001_0000000001.pkl",
        "002_0000000001.pkl",
    ]


def test_refresh_picks_up_new_files(tmp_path: Path):
    """Buffer A writes; buffer B (separate instance, same dir) sees the writes."""
    writer = ReplayBuffer(tmp_path)
    reader = ReplayBuffer(tmp_path)
    assert reader.count_games() == 0
    writer.append_game(0, 1, [_ex(0.5), _ex(-0.5)])
    # Force a refresh on reader (bypass mtime caching, since same-second writes
    # may not bump dir mtime on all filesystems).
    reader._last_mtime = -1.0
    assert reader.count_examples() == 2


def test_pickle_roundtrip_preserves_az_example(tmp_path: Path):
    """AZExample should pickle/unpickle losslessly through the buffer."""
    original = AZExample(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        legal_moves=[{"uci": "e2e4", "from": "e2", "to": "e4"}],
        visit_distribution=[1.0],
        wdl_target=[0.2, 0.3, 0.5],
        moves_left_target=0.7,
    )
    buf = ReplayBuffer(tmp_path)
    buf.append_game(0, 1, [original])
    sampled = buf.sample(batch_size=1, rng=random.Random(0))[0]
    assert sampled.fen == original.fen
    assert sampled.legal_moves == original.legal_moves
    assert sampled.visit_distribution == original.visit_distribution
    assert sampled.wdl_target == original.wdl_target
    assert sampled.moves_left_target == pytest.approx(original.moves_left_target)
