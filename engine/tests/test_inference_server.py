"""Correctness tests for InferenceServer: batched results must match the
direct (per-position) model calls within floating-point tolerance, and the
server must handle empty-moves and concurrent-thread submission."""
from __future__ import annotations

import threading
import time

import torch

from chessckers_engine.encoding import encode_move, encode_position, encoders_for
from chessckers_engine.inference_server import InferenceServer
from chessckers_engine.model import ChesskersScorer, build_model

FEN_W = "8/8/8/8/8/8/8/4K3 w - - 0 1"
FEN_START = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s] w KQkq - 0 1"
)


def _move(uci: str, fr: str, to: str) -> dict:
    return {
        "uci": uci, "from": fr, "to": to,
        "capture": None, "waypoints": None, "deployCount": None,
        "demotionsRequired": None, "promotion": None,
    }


def _direct_eval(model, fen, legal_moves):
    """Reference: same code path as `_eval_and_priors` for the model branch."""
    pos = encode_position(fen).unsqueeze(0)
    if not legal_moves:
        with torch.no_grad():
            v = model.value(pos)
        return float(v.item()), []
    moves = torch.stack([encode_move(m) for m in legal_moves])
    with torch.no_grad():
        logits, v = model.policy_and_value(pos, moves)
        priors = torch.softmax(logits, dim=0)
    return float(v.item()), priors.tolist()


def test_server_matches_direct_single_request():
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    moves = [_move("e1e2", "e1", "e2")]
    expected_v, expected_priors = _direct_eval(model, FEN_W, moves)
    with InferenceServer(model, max_batch_size=4) as srv:
        v, priors = srv.submit(FEN_W, moves).result(timeout=5)
    assert abs(v - expected_v) < 1e-5
    assert len(priors) == len(expected_priors)
    for p, ep in zip(priors, expected_priors):
        assert abs(p - ep) < 1e-5


def _direct_batch_eval(model, version, fen, legal_moves):
    """Reference using the SAME method (batch_eval) the server uses, at batch 1.
    Version-aware: encodes with the model's own encoders."""
    enc_pos, _, enc_move = encoders_for(version)
    pos = enc_pos(fen).unsqueeze(0)
    mv = torch.stack([enc_move(m) for m in legal_moves]) if legal_moves else None
    with torch.no_grad():
        values, priors_list = model.batch_eval(pos, [mv])
    pr = priors_list[0].tolist() if priors_list[0].numel() > 0 else []
    return float(values[0].item()), pr


def test_server_matches_direct_v2_net():
    """The InferenceServer must encode for the MODEL's arch VERSION. A V2/V3
    (gather-head) net needs the 16ch/10x10 position + 114-dim move encoding; the
    pre-fix server hardcoded V1's 15ch/8x8 + 240-dim and shape-mismatched a
    transformer net. Server output must match the direct V2 batch_eval."""
    torch.manual_seed(0)
    model = build_model(
        version="v2", d_hidden=32, c_filters=16, n_blocks=1,
        n_tf_blocks=1, n_heads=4, tf_ff_mult=2,
    ).eval()
    assert getattr(model, "VERSION", None) == "v2"
    moves = [_move("e1e2", "e1", "e2"), _move("e1f2", "e1", "f2")]
    exp_v, exp_p = _direct_batch_eval(model, "v2", FEN_W, moves)
    with InferenceServer(model, max_batch_size=4) as srv:
        v, priors = srv.submit(FEN_W, moves).result(timeout=10)
    assert abs(v - exp_v) < 1e-5
    assert len(priors) == len(exp_p)
    for p, ep in zip(priors, exp_p):
        assert abs(p - ep) < 1e-5


def test_server_matches_direct_empty_moves():
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    expected_v, _ = _direct_eval(model, FEN_W, [])
    with InferenceServer(model, max_batch_size=4) as srv:
        v, priors = srv.submit(FEN_W, []).result(timeout=5)
    assert abs(v - expected_v) < 1e-5
    assert priors == []


def test_server_batches_concurrent_submissions():
    """8 threads submit simultaneously; all results must match the direct
    (single-position) reference. Verifies batching doesn't corrupt outputs."""
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    moves_a = [_move("e1e2", "e1", "e2")]
    moves_b = [_move("e2e4", "e2", "e4"), _move("d2d4", "d2", "d4")]
    exp_va, exp_pa = _direct_eval(model, FEN_W, moves_a)
    exp_vb, exp_pb = _direct_eval(model, FEN_START, moves_b)

    results: dict[int, tuple[float, list[float]]] = {}

    def submit(srv, idx, fen, moves):
        results[idx] = srv.submit(fen, moves).result(timeout=5)

    with InferenceServer(model, max_batch_size=8, timeout_ms=20) as srv:
        threads = []
        for i in range(8):
            fen = FEN_W if i % 2 == 0 else FEN_START
            mv = moves_a if i % 2 == 0 else moves_b
            threads.append(threading.Thread(target=submit, args=(srv, i, fen, mv)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    for i in range(8):
        v, p = results[i]
        if i % 2 == 0:
            assert abs(v - exp_va) < 1e-5
            assert all(abs(a - b) < 1e-5 for a, b in zip(p, exp_pa))
        else:
            assert abs(v - exp_vb) < 1e-5
            assert all(abs(a - b) < 1e-5 for a, b in zip(p, exp_pb))


def test_server_stats_track_batch_sizes():
    """Submit several requests with timing that forces a mix of singleton
    and grouped batches; verify n_batches/n_requests/histogram make sense."""
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    moves = [_move("e1e2", "e1", "e2")]

    # Tiny timeout: a quiet 50ms gap between submissions guarantees each lands
    # in its own batch (the inference thread won't wait for stragglers).
    with InferenceServer(model, max_batch_size=8, timeout_ms=1.0) as srv:
        for _ in range(3):
            srv.submit(FEN_W, moves).result(timeout=5)
            time.sleep(0.05)
        s_singleton = srv.stats()
        assert s_singleton["n_batches"] == 3
        assert s_singleton["n_requests"] == 3
        assert s_singleton["avg_batch_size"] == 1.0
        # All in the singleton bucket [0].
        assert s_singleton["hist_1_2_4_8_16_32+"][0] == 3

    # Concurrent submissions should produce at least one batch with size > 1.
    with InferenceServer(model, max_batch_size=16, timeout_ms=20) as srv:
        threads = [
            threading.Thread(target=lambda: srv.submit(FEN_W, moves).result(timeout=5))
            for _ in range(8)
        ]
        for t in threads: t.start()
        for t in threads: t.join(timeout=10)
        s_concurrent = srv.stats()
    assert s_concurrent["n_requests"] == 8
    assert s_concurrent["max_batch_size_seen"] >= 2, (
        f"expected at least one merged batch, got stats={s_concurrent}"
    )


def test_server_propagates_exceptions_to_futures():
    """If batch_eval raises, every future in that batch must be resolved with
    the exception (rather than hanging forever)."""
    torch.manual_seed(0)
    model = ChesskersScorer().eval()

    # Force a failure by sabotaging batch_eval.
    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated batch failure")
    model.batch_eval = boom

    with InferenceServer(model, max_batch_size=2) as srv:
        f1 = srv.submit(FEN_W, [_move("e1e2", "e1", "e2")])
        f2 = srv.submit(FEN_W, [_move("e1e2", "e1", "e2")])
        try:
            f1.result(timeout=5)
            assert False, "expected exception"
        except RuntimeError as e:
            assert "simulated batch failure" in str(e)
        try:
            f2.result(timeout=5)
            assert False, "expected exception"
        except RuntimeError as e:
            assert "simulated batch failure" in str(e)
