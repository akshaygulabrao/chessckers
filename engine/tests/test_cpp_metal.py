"""Phase 6a: Metal/MPSGraph backend toolchain spike.

Confirms the Apple-only MPSGraph backend links, runs on the GPU, and agrees with the CPU
BLAS forward (the parity oracle). Skips on non-Apple / headless boxes (no Metal device, or
the module built without CC_HAVE_METAL).
"""
from __future__ import annotations

import os
import random
import sys
import tempfile

import pytest

import chessckers_cpp as cpp


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal backend is Apple-only")
@pytest.mark.parametrize("shape", [(64, 96, 256), (800, 96, 288), (1600, 864, 96)])
def test_metal_matmul_matches_cpu(shape):
    m, k, n = shape
    diff = cpp.metal_matmul_selftest(m, k, n)
    if diff == -2.0:
        pytest.skip("built without CC_HAVE_METAL")
    if diff == -1.0:
        pytest.skip("no Metal GPU on this box")
    assert diff < 1e-3, f"MPSGraph matmul diverged from CPU: max|diff|={diff}"


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal backend is Apple-only")
@pytest.mark.parametrize("n_tf", [0, 7])  # 0 = conv path only; 7 = full transformer trunk
def test_metal_trunk_v2_matches_cpu(n_tf):
    """Phase 6b: the MPSGraph V2 spatial trunk matches the torch reference (the CPU oracle)."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from chessckers_engine.model import build_model
    from chessckers_engine.native_net import export_state_dict
    from chessckers_engine.variant_py import PyVariantClient

    arch = dict(version="v2", d_hidden=256, c_filters=96, n_blocks=9,
                n_tf_blocks=n_tf, n_heads=4, tf_ff_mult=4)
    torch.manual_seed(0)
    m = build_model(**arch)
    m.eval()
    binp = os.path.join(tempfile.mkdtemp(), "w.bin")
    export_state_dict(m.state_dict(), binp)
    net = cpp.ChesskersNet(binp)

    cl = PyVariantClient()
    st = cl.new_game()
    random.seed(1)
    P = []
    for _ in range(60):
        leg = st["legalMoves"]
        if not leg:
            break
        P.append(cpp.encode_position_v2(cpp.parse_fen(st["fen"])))
        st = cl.make_move(st["fen"], random.choice(leg)["uci"])
        if st.get("winner"):
            break
    P = P[:16]

    ok, feats = cpp.metal_trunk_v2(net, P)
    if not ok:
        pytest.skip("no Metal GPU / unsupported trunk")
    feats = np.array(feats)
    with torch.no_grad():
        pos = torch.stack([torch.tensor(p).reshape(16, 10, 10) for p in P])
        cpu = m.position_trunk(pos).reshape(len(P), -1).numpy()
    diff = float(np.abs(feats - cpu).max())
    assert diff < 1e-3, f"Metal trunk (n_tf={n_tf}) diverged from CPU: max|diff|={diff}"
