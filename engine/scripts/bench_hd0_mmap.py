"""Benchmark numpy.memmap throughput on a target drive.

The endgame tablebase stores one byte per position in numpy.memmap files
under a root dir (default the exFAT external drive at /Volumes/hd0). Generation
does large sequential writes; probing does scattered random reads. This script
measures all three access patterns so we can plan how high we can generate.

Usage:
    python3 scripts/bench_hd0_mmap.py --size-mb 512
    python3 scripts/bench_hd0_mmap.py --root /tmp/cc_bench --size-mb 128

Exits non-zero with a clear message if --root is missing or not writable
(so it degrades gracefully when the drive isn't mounted).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

MB = 1024 * 1024


def _check_writable(root: Path) -> str | None:
    """Returns an error message if root can't be used, else None."""
    if not root.exists():
        return f"root does not exist: {root} (is the drive mounted?)"
    if not root.is_dir():
        return f"root is not a directory: {root}"
    probe = root / ".bench_writable_probe"
    try:
        probe.write_bytes(b"\x00")
        probe.unlink()
    except OSError as e:
        return f"root is not writable: {root} ({e})"
    return None


def bench_sequential_write(path: Path, n_bytes: int) -> float:
    """Create + fill a uint8 memmap, flush + fsync. Returns MB/s."""
    t0 = time.perf_counter()
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(n_bytes,))
    mm[:] = 0xFF
    mm.flush()
    with open(path, "rb") as f:
        os.fsync(f.fileno())
    dt = time.perf_counter() - t0
    del mm
    return (n_bytes / MB) / dt


def bench_sequential_read(path: Path, n_bytes: int) -> float:
    """Reopen read-only and scan the whole file. Returns MB/s."""
    t0 = time.perf_counter()
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n_bytes,))
    total = int(mm.sum(dtype=np.uint64))
    dt = time.perf_counter() - t0
    del mm
    assert total == n_bytes * 0xFF  # guard against a no-op read
    return (n_bytes / MB) / dt


def bench_random_read(path: Path, n_bytes: int, n_reads: int) -> tuple[float, float]:
    """Read n_reads random single bytes. Returns (reads/sec, effective MB/s)."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_bytes, size=n_reads, dtype=np.int64)
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n_bytes,))
    t0 = time.perf_counter()
    acc = 0
    for i in idx:
        acc += int(mm[i])
    dt = time.perf_counter() - t0
    del mm
    assert acc == n_reads * 0xFF  # every byte is 0xFF; guard against no-op
    return n_reads / dt, (n_reads / MB) / dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/Volumes/hd0/chessckers_tb",
                    help="Tablebase root dir; bench file goes under <root>/_bench/")
    ap.add_argument("--size-mb", type=int, default=512,
                    help="Size of the memmap to write/read, in MB")
    ap.add_argument("--random-reads", type=int, default=1_000_000,
                    help="Number of random single-byte reads to time")
    args = ap.parse_args()

    root = Path(args.root)
    err = _check_writable(root)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    bench_dir = root / "_bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / f"bench_{os.getpid()}.bin"
    n_bytes = args.size_mb * MB

    print(f"root: {root}")
    print(f"size: {args.size_mb} MB, random reads: {args.random_reads:,}\n")
    try:
        w = bench_sequential_write(path, n_bytes)
        r = bench_sequential_read(path, n_bytes)
        rps, rmb = bench_random_read(path, n_bytes, args.random_reads)

        print(f"{'pattern':<20} {'MB/s':>10} {'extra':>16}")
        print("-" * 48)
        print(f"{'sequential write':<20} {w:>10.1f}")
        print(f"{'sequential read':<20} {r:>10.1f}")
        print(f"{'random read':<20} {rmb:>10.3f} {rps:>12,.0f} r/s")
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
