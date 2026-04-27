import os
from pathlib import Path

from chessckers_engine.checkpoints import default_checkpoint_path, latest_checkpoint


def test_latest_returns_none_for_missing_directory(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert latest_checkpoint(missing) is None


def test_latest_returns_none_for_empty_directory(tmp_path):
    assert latest_checkpoint(tmp_path) is None


def test_latest_returns_only_pt_files(tmp_path):
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / "data.jsonl").write_text("ignored")
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"\x00")
    assert latest_checkpoint(tmp_path) == pt


def test_latest_picks_most_recently_modified(tmp_path):
    older = tmp_path / "older.pt"
    newer = tmp_path / "newer.pt"
    older.write_bytes(b"\x00")
    newer.write_bytes(b"\x00")
    # Force older to actually be older
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    assert latest_checkpoint(tmp_path) == newer


def test_default_path_is_under_given_dir_and_creates_dir(tmp_path):
    target = tmp_path / "fresh_dir"
    p = default_checkpoint_path(target)
    assert p.parent == target
    assert target.exists()
    assert p.suffix == ".pt"
    assert p.name.startswith("model-")


def test_default_path_each_call_returns_a_pt_under_the_dir(tmp_path):
    p1 = default_checkpoint_path(tmp_path)
    p2 = default_checkpoint_path(tmp_path)
    # Same dir
    assert p1.parent == tmp_path
    assert p2.parent == tmp_path
    # Both look like model-<...>.pt
    assert p1.name.startswith("model-") and p1.suffix == ".pt"
    assert p2.name.startswith("model-") and p2.suffix == ".pt"
