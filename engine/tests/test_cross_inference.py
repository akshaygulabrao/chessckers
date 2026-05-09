"""Tests for the cross-process inference server + client.

The server runs in-thread in the test process; clients run either
in-thread or in spawned subprocesses to exercise the IPC path."""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest
import torch

from chessckers_engine.cross_inference import (
    CrossInferenceClient,
    CrossInferenceServer,
)
from chessckers_engine.encoding import encode_move, encode_position
from chessckers_engine.model import ChesskersScorer


TINY_ARCH = {"d_hidden": 32, "c_filters": 8, "n_blocks": 1}
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
START_MOVES = [
    {"uci": "e2e4", "from": "e2", "to": "e4", "piece": "P"},
    {"uci": "d2d4", "from": "d2", "to": "d4", "piece": "P"},
    {"uci": "g1f3", "from": "g1", "to": "f3", "piece": "N"},
]


def _make_qs(ctx, n_workers: int):
    return ctx.Queue(), [ctx.Queue() for _ in range(n_workers)]


def test_single_request_in_thread(tmp_path: Path):
    """One in-thread client + server, simplest end-to-end roundtrip."""
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH).eval()
    ctx = mp.get_context("spawn")
    request_q, response_qs = _make_qs(ctx, n_workers=1)
    server = CrossInferenceServer(model, request_q, response_qs, max_batch_size=4)
    server.start()
    try:
        client = CrossInferenceClient(0, request_q, response_qs[0])
        future = client.submit(START_FEN, START_MOVES)
        value, priors = future.result(timeout=10)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0
        assert len(priors) == len(START_MOVES)
        assert all(isinstance(p, float) for p in priors)
        # Priors are softmax-normalized → sum ≈ 1.
        assert abs(sum(priors) - 1.0) < 1e-4
    finally:
        server.shutdown()


def test_empty_legal_moves_returns_empty_priors(tmp_path: Path):
    """A request with no legal moves should still return value, with priors=[]."""
    model = ChesskersScorer(**TINY_ARCH).eval()
    ctx = mp.get_context("spawn")
    request_q, response_qs = _make_qs(ctx, n_workers=1)
    server = CrossInferenceServer(model, request_q, response_qs, max_batch_size=4)
    server.start()
    try:
        client = CrossInferenceClient(0, request_q, response_qs[0])
        value, priors = client.submit(START_FEN, []).result(timeout=10)
        assert isinstance(value, float)
        assert priors == []
    finally:
        server.shutdown()


def test_server_batches_concurrent_requests(tmp_path: Path):
    """Multiple in-thread clients submit roughly simultaneously; server should
    coalesce them into batches > 1."""
    model = ChesskersScorer(**TINY_ARCH).eval()
    ctx = mp.get_context("spawn")
    n = 4
    request_q, response_qs = _make_qs(ctx, n_workers=n)
    server = CrossInferenceServer(
        model, request_q, response_qs, max_batch_size=8, timeout_ms=20.0,
    )
    server.start()
    try:
        clients = [CrossInferenceClient(i, request_q, response_qs[i]) for i in range(n)]
        # Fire all submits, then collect — submits are blocking, so do them
        # from threads to overlap.
        import threading
        results: dict[int, tuple[float, list[float]]] = {}

        def worker(i: int):
            v, p = clients[i].submit(START_FEN, START_MOVES).result(timeout=10)
            results[i] = (v, p)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert len(results) == n
        # At least one batch should have been > 1 (i.e. some concurrency seen).
        stats = server.stats()
        assert stats["max_batch_size_seen"] >= 2, stats
    finally:
        server.shutdown()


def test_request_id_ordering_per_worker(tmp_path: Path):
    """A single client submitting multiple times in sequence should get back
    responses in submit-order (per-worker FIFO is the contract)."""
    model = ChesskersScorer(**TINY_ARCH).eval()
    ctx = mp.get_context("spawn")
    request_q, response_qs = _make_qs(ctx, n_workers=1)
    server = CrossInferenceServer(model, request_q, response_qs, max_batch_size=4)
    server.start()
    try:
        client = CrossInferenceClient(0, request_q, response_qs[0])
        # Submit 5 sequentially; each .result() check validates the rid match.
        for _ in range(5):
            v, p = client.submit(START_FEN, START_MOVES).result(timeout=10)
            assert -1.0 <= v <= 1.0
            assert len(p) == len(START_MOVES)
    finally:
        server.shutdown()


# ---- Subprocess integration test ----


def _subprocess_client_run(worker_id: int, request_q, response_q,
                            n_requests: int, results_q):
    """Run as a spawned subprocess. Submits N requests and returns the results."""
    from chessckers_engine.cross_inference import CrossInferenceClient

    client = CrossInferenceClient(worker_id, request_q, response_q)
    out = []
    for _ in range(n_requests):
        v, p = client.submit(START_FEN, START_MOVES).result(timeout=30)
        out.append((v, len(p)))
    results_q.put((worker_id, out))


@pytest.mark.slow
def test_subprocess_workers_with_shared_server(tmp_path: Path):
    """End-to-end: 3 worker subprocesses + 1 in-thread server. Verify all
    workers get correct-shape responses and the server's stats show batching."""
    torch.manual_seed(0)
    model = ChesskersScorer(**TINY_ARCH).eval()
    ctx = mp.get_context("spawn")
    n_workers = 3
    request_q, response_qs = _make_qs(ctx, n_workers=n_workers)
    server = CrossInferenceServer(
        model, request_q, response_qs, max_batch_size=8, timeout_ms=30.0,
    )
    server.start()
    results_q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_subprocess_client_run,
            args=(i, request_q, response_qs[i], 5, results_q),
            name=f"client-{i}",
        )
        for i in range(n_workers)
    ]
    try:
        for p in procs:
            p.start()
        # Collect results.
        collected = {}
        for _ in range(n_workers):
            wid, out = results_q.get(timeout=60)
            collected[wid] = out
        for p in procs:
            p.join(timeout=10)
        assert len(collected) == n_workers
        for wid, out in collected.items():
            assert len(out) == 5
            for v, n_priors in out:
                assert -1.0 <= v <= 1.0
                assert n_priors == len(START_MOVES)
        # Should have seen some batching across the 3 subprocess workers.
        stats = server.stats()
        # Total requests = 3 workers × 5 reqs = 15.
        assert stats["n_requests"] == n_workers * 5
        assert stats["max_batch_size_seen"] >= 2, stats
    finally:
        server.shutdown()
        for p in procs:
            if p.is_alive():
                p.terminate()
