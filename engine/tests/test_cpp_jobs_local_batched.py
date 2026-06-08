"""Phase 6f (lc0-split migration): the GPU-batched local-job ENGINE. `run_jobs_local`
with batch_size>1 claims that many train jobs and plays them concurrently through one
shared batched trunk (use_gpu => Metal). Because each game is still seeded
base_seed+(train index), the buffer chunks a batched engine writes are BYTE-IDENTICAL to
the serial (batch_size=1) engine under a byte-identical trunk forward (CPU is). This is
the same parity gate the rest of the C++ port rides on — batching the fleet engine does
NOT change the training data, just how fast it's produced.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from chessckers_engine.model import ChesskersScorerV2
from chessckers_engine.native_net import export_state_dict

cpp = pytest.importorskip("chessckers_cpp")

SEED_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"
PARAMS = dict(sims=24, c_puct=1.5, temperature=1.0, temp_cutoff_plies=8, max_plies=40,
              dirichlet_alpha=0.3, dirichlet_eps=0.25)


def _weights_bin(tmp_path):
    torch.manual_seed(0)
    m = ChesskersScorerV2(n_blocks=2, n_tf_blocks=1, n_heads=4, tf_ff_mult=2)
    m.eval()
    p = tmp_path / "weights.bin"
    export_state_dict(m.state_dict(), str(p))
    return p


def _make_run(tmp_path, weights, n_jobs):
    run = tmp_path
    (run / "jobs").mkdir(parents=True)
    shutil.copy(weights, run / "weights.bin")
    for i in range(n_jobs):
        job = {"type": "train", "bin_sha": "x", "params": PARAMS}
        (run / "jobs" / f"{i:010d}.json").write_text(json.dumps(job))
    return run


def _buffer_chunks(run):
    """{filename: bytes} for the .pkl training chunks the engine wrote."""
    return {p.name: p.read_bytes() for p in sorted((run / "buffer").glob("*.pkl"))}


def test_batched_engine_chunks_match_serial(tmp_path):
    N = 8
    weights = _weights_bin(tmp_path)

    serial = _make_run(tmp_path / "serial", weights, N)
    cpp.run_jobs_local(str(serial), SEED_FEN, worker_id=300, base_seed=7,
                       max_jobs=N, batch_size=1, use_gpu=False)
    serial_chunks = _buffer_chunks(serial)

    batched = _make_run(tmp_path / "batched", weights, N)
    cpp.run_jobs_local(str(batched), SEED_FEN, worker_id=300, base_seed=7,
                       max_jobs=N, batch_size=4, use_gpu=False)
    batched_chunks = _buffer_chunks(batched)

    assert len(serial_chunks) == N, f"serial wrote {len(serial_chunks)} chunks, want {N}"
    assert batched_chunks.keys() == serial_chunks.keys(), "filenames diverged"
    for name in serial_chunks:
        assert batched_chunks[name] == serial_chunks[name], f"chunk {name} bytes diverged"


def test_oversubscribed_engine_chunks_match_serial(tmp_path):
    """concurrency > batch_size (GPU pipelining): the engine claims up to `concurrency`
    train jobs and plays them as that many concurrent games over a width-`batch_size`
    batch. Game seeds are by train index (not thread), so the chunks stay BYTE-IDENTICAL
    to the serial engine — the +24% throughput is pure inference transport."""
    N = 12
    weights = _weights_bin(tmp_path)

    serial = _make_run(tmp_path / "serial", weights, N)
    cpp.run_jobs_local(str(serial), SEED_FEN, worker_id=300, base_seed=7,
                       max_jobs=N, batch_size=1, use_gpu=False)
    serial_chunks = _buffer_chunks(serial)

    over = _make_run(tmp_path / "over", weights, N)
    cpp.run_jobs_local(str(over), SEED_FEN, worker_id=300, base_seed=7,
                       max_jobs=N, batch_size=4, use_gpu=False, concurrency=N)  # 3x oversub
    over_chunks = _buffer_chunks(over)

    assert len(serial_chunks) == N, f"serial wrote {len(serial_chunks)} chunks, want {N}"
    assert over_chunks.keys() == serial_chunks.keys(), "filenames diverged"
    for name in serial_chunks:
        assert over_chunks[name] == serial_chunks[name], f"chunk {name} bytes diverged"


def test_batched_engine_runs_on_gpu_if_present(tmp_path):
    """use_gpu=True (Metal): the engine still produces N valid chunks (float-close, not
    byte-identical to serial). Falls back to CPU when no Metal device, so this just
    asserts the batched GPU path completes and writes well-formed chunks."""
    N = 6
    weights = _weights_bin(tmp_path)
    run = _make_run(tmp_path / "gpu", weights, N)
    handled = cpp.run_jobs_local(str(run), SEED_FEN, worker_id=300, base_seed=7,
                                 max_jobs=N, batch_size=N, use_gpu=True)
    assert handled == N
    chunks = _buffer_chunks(run)
    assert len(chunks) == N
    for b in chunks.values():
        assert b[:2] == b"\x1f\x8b"  # gzip magic — a well-formed ccz chunk
