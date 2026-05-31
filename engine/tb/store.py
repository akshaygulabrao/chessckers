"""On-disk storage for solved tablebase classes (Phase 1).

One file per material class under `<root>/phase1/`, holding exactly one byte
per index slot (see `tb.index.class_size`). The byte packs the position value:

    0x00          -> DRAW
    0xFF          -> VOID (no real position: mirror twin / illegal / overlap)
    0x40 | dtm    -> WIN  in `dtm` plies (dtm in 0..63)
    0x80 | dtm    -> LOSS in `dtm` plies (dtm in 0..63)

A fixed 64-byte header makes each file self-describing; the data region (memmap
target) begins at `HEADER_SIZE`. A top-level `manifest.json` (written
tmp-then-rename for atomicity) indexes every class and its solve state.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

from tb.index import class_size
from tb.model import Value

HEADER_SIZE = 64
_MAGIC = b"CCKTB1\0\0"  # 8 bytes
_VERSION = 1
# struct: magic(8s) version(I) total(I) class_size(Q)  -> 24 bytes, rest reserved
_HEADER_STRUCT = struct.Struct("<8sIIQ")

VOID = 0xFF
DRAW = 0x00
_WIN = 0x40
_LOSS = 0x80
MAX_DTM = 63


# --------------------------------------------------------------------------- #
# Byte codec
# --------------------------------------------------------------------------- #

def byte_encode(value: Value | None) -> int:
    """Pack a `(wdl, dtm)` value (or None for VOID) into one byte."""
    if value is None:
        return VOID
    wdl, dtm = value
    if wdl == 0:
        return DRAW
    d = dtm or 0
    if not (0 <= d <= MAX_DTM):
        raise ValueError(f"dtm {d} out of 6-bit range (max {MAX_DTM})")
    return (_WIN if wdl > 0 else _LOSS) | d


def byte_decode(b: int) -> Value | None:
    """Inverse of `byte_encode`. Returns None for VOID. Raises on a byte that
    `byte_encode` never produces (corruption guard)."""
    if b == VOID:
        return None
    if b == DRAW:
        return (0, None)
    flag = b & 0xC0  # exactly one of WIN/LOSS must be set
    dtm = b & MAX_DTM
    if flag == _WIN:
        return (1, dtm)
    if flag == _LOSS:
        return (-1, dtm)
    raise ValueError(f"invalid tablebase byte {b:#04x}")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def phase1_dir(root: str | os.PathLike) -> Path:
    return Path(root) / "phase1"


def class_path(root: str | os.PathLike, total: int) -> Path:
    return phase1_dir(root) / f"N{total}.tb"


# --------------------------------------------------------------------------- #
# Class files
# --------------------------------------------------------------------------- #

def _write_header(path: Path, total: int, n: int) -> None:
    hdr = _HEADER_STRUCT.pack(_MAGIC, _VERSION, total, n)
    hdr = hdr + b"\0" * (HEADER_SIZE - len(hdr))
    with open(path, "r+b") as f:
        f.write(hdr)


def _read_header(path: Path) -> tuple[int, int]:
    """Return (total, class_size) from a class file header; validate magic."""
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    magic, version, total, n = _HEADER_STRUCT.unpack(raw[: _HEADER_STRUCT.size])
    if magic != _MAGIC:
        raise ValueError(f"bad magic in {path}: {magic!r}")
    if version != _VERSION:
        raise ValueError(f"unsupported version {version} in {path}")
    return total, n


def create_class(root: str | os.PathLike, total: int) -> Path:
    """Create (or truncate) the class file for `total`, header + all-VOID data.
    Returns the path. The data region is initialized to 0xFF."""
    n = class_size(total)
    path = class_path(root, total)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Allocate header + data, then fill data with VOID.
    with open(path, "wb") as f:
        f.truncate(HEADER_SIZE + n)
    _write_header(path, total, n)
    data = np.memmap(path, dtype=np.uint8, mode="r+", offset=HEADER_SIZE, shape=(n,))
    data[:] = VOID
    data.flush()
    del data
    return path


def open_class(
    root: str | os.PathLike, total: int, mode: str = "r"
) -> np.memmap:
    """Memmap the data region of a class file (`mode` 'r' or 'r+'). Validates
    the header and that the file size matches the expected class size."""
    path = class_path(root, total)
    hdr_total, n = _read_header(path)
    if hdr_total != total:
        raise ValueError(f"{path} header total {hdr_total} != requested {total}")
    expected = class_size(total)
    if n != expected:
        raise ValueError(f"{path} header size {n} != computed {expected}")
    return np.memmap(path, dtype=np.uint8, mode=mode, offset=HEADER_SIZE, shape=(n,))


# --------------------------------------------------------------------------- #
# Manifest (atomic)
# --------------------------------------------------------------------------- #

def manifest_path(root: str | os.PathLike) -> Path:
    return Path(root) / "manifest.json"


def read_manifest(root: str | os.PathLike) -> dict:
    p = manifest_path(root)
    if not p.exists():
        return {"phase": 1, "classes": {}}
    with open(p) as f:
        return json.load(f)


def write_manifest(root: str | os.PathLike, manifest: dict) -> None:
    """Atomically replace the manifest (write tmp in the same dir, then rename;
    fsync to survive a crash)."""
    p = manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def mark_class(
    root: str | os.PathLike, total: int, status: str, **extra
) -> None:
    """Record a class's solve state in the manifest (e.g. status='solved')."""
    m = read_manifest(root)
    m.setdefault("classes", {})[f"N{total}"] = {
        "total": total,
        "status": status,
        "file": class_path(root, total).name,
        "class_size": class_size(total),
        **extra,
    }
    write_manifest(root, m)
