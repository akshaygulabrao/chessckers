"""Parity gate for cpp.ChesskersNet.eval_batch (batched-trunk leaf inference).

eval_batch(K positions) must be BYTE-IDENTICAL to K serial eval() calls — it only
routes the conv stack through a batched GEMM (conv3x3_batch); every other op is the
same per-board code. The batching is a CPU-BLAS efficiency play (see the
project-batched-selfplay-mps-finding memory): measured ~1.1x on the real V2 net, so it
is NOT wired into search (which would need parity-breaking virtual-loss collection) — but
the primitive is kept + parity-locked for a future GPU backend (Phase 6).
"""
from __future__ import annotations

import random

import torch

import chessckers_cpp as cpp
from chessckers_engine.model import build_model
from chessckers_engine.native_net import export_state_dict
from chessckers_engine.variant_py import PyVariantClient


def _real_leaves(n=16):
    """n real (position, legal-move-features) leaves from a random walk off the start."""
    cl = PyVariantClient()
    st = cl.new_game()
    random.seed(1)
    positions, moves_per = [], []
    for _ in range(60):
        leg = st["legalMoves"]
        if not leg:
            break
        positions.append(cpp.encode_position_v2(cpp.parse_fen(st["fen"])))
        moves_per.append([cpp.encode_move_v2(mv) for mv in leg])
        st = cl.make_move(st["fen"], random.choice(leg)["uci"])
        if st.get("winner"):
            break
    return positions[:n], moves_per[:n]


def _v2_net(tmp_path):
    arch = dict(version="v2", d_hidden=256, c_filters=96, n_blocks=9,
                n_tf_blocks=7, n_heads=4, tf_ff_mult=4)
    torch.manual_seed(0)
    m = build_model(**arch)
    m.eval()
    binp = tmp_path / "w.bin"
    export_state_dict(m.state_dict(), binp)
    return cpp.ChesskersNet(str(binp))


def test_eval_batch_byte_identical_to_serial(tmp_path):
    net = _v2_net(tmp_path)
    positions, moves_per = _real_leaves()
    assert len(positions) >= 4 and any(len(m) > 1 for m in moves_per)

    serial = [net.eval(p, mv) for p, mv in zip(positions, moves_per)]
    batched = net.eval_batch(positions, moves_per)

    assert len(batched) == len(serial)
    for (sv, sp), (bv, bp) in zip(serial, batched):
        assert bv == sv  # exact: same ops, batched GEMM only
        assert list(bp) == list(sp)
