"""Slice 6 oracle test: the native C++ NN forward (cpp.ChesskersNet) must match
PyTorch's ChesskersScorer within a tight tolerance, given the same encoded
inputs.

Weights are exported via native_net.export_state_dict and loaded in C++. For
each position we feed the SAME encoded planes + move features to both, and
compare the WDL->Q value and the softmax policy priors. (The encoders are ported
to C++ in a follow-up; here we isolate the forward by encoding in Python.)
"""
from __future__ import annotations

import pytest
import torch

from chessckers_engine.encoding import (
    encode_move,
    encode_move_v2,
    encode_position,
    encode_position_v2,
)
from chessckers_engine.model import ChesskersScorer, ChesskersScorerV2
from chessckers_engine.native_net import export_state_dict
from chessckers_engine.variant_py.client import PyVariantClient

cpp = pytest.importorskip("chessckers_cpp")

SEEDS = [
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] w - - 0 1",
]

ATOL = 2e-4
# V2 adds 7 transformer blocks (attention softmax + GELU) on top of the residual trunk,
# but the double-accumulated norms/softmax keep it within ~1e-6 of PyTorch — as tight as V1.
ATOL_V2 = 2e-4


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    torch.manual_seed(0)
    m = ChesskersScorer()
    m.eval()
    wpath = str(tmp_path_factory.mktemp("net") / "net.bin")
    export_state_dict(m.state_dict(), wpath)
    return m, cpp.ChesskersNet(wpath)


@pytest.mark.parametrize("seed", SEEDS)
def test_native_forward_matches_pytorch(net, seed: str):
    model, cnet = net
    client = PyVariantClient()
    legal = client.new_game(seed)["legalMoves"]
    assert legal, seed

    pos = encode_position(seed)               # (14, 8, 8)
    move_feats = [encode_move(mv) for mv in legal]  # each (240,)

    v_cpp, priors_cpp = cnet.eval(
        pos.flatten().tolist(), [mf.tolist() for mf in move_feats]
    )

    with torch.no_grad():
        logits, value = model.policy_and_value(pos.unsqueeze(0), torch.stack(move_feats))
        probs = torch.softmax(logits, dim=0).tolist()

    assert abs(v_cpp - float(value.item())) < ATOL, (
        f"value: cpp={v_cpp} torch={float(value.item())}"
    )
    max_dp = max(abs(a - b) for a, b in zip(priors_cpp, probs))
    assert max_dp < ATOL, f"max prior diff {max_dp} (seed={seed})"


def test_value_only_eval(net):
    """A position with no moves yields the value with empty priors."""
    _, cnet = net
    pos = encode_position(SEEDS[0])
    v, priors = cnet.eval(pos.flatten().tolist(), [])
    assert priors == []
    assert -1.0 <= v <= 1.0


# --- V2 (ChesskersScorerV2: transformer trunk + gather head) native parity ---


@pytest.fixture(scope="module")
def net_v2(tmp_path_factory):
    torch.manual_seed(0)
    # The 2.52M A/B config: 9 residual + 7 transformer blocks, 4 heads.
    m = ChesskersScorerV2(n_blocks=9, n_tf_blocks=7, n_heads=4, tf_ff_mult=4)
    m.eval()
    wpath = str(tmp_path_factory.mktemp("net_v2") / "net.bin")
    export_state_dict(m.state_dict(), wpath)
    return m, cpp.ChesskersNet(wpath)


@pytest.mark.parametrize("seed", SEEDS)
def test_native_v2_forward_matches_pytorch(net_v2, seed: str):
    model, cnet = net_v2
    client = PyVariantClient()
    legal = client.new_game(seed)["legalMoves"]
    assert legal, seed

    pos = encode_position_v2(seed)                     # (16, 10, 10)
    move_feats = [encode_move_v2(mv) for mv in legal]  # each (114,)

    v_cpp, priors_cpp = cnet.eval(
        pos.flatten().tolist(), [mf.tolist() for mf in move_feats]
    )

    with torch.no_grad():
        logits, value = model.policy_and_value(pos.unsqueeze(0), torch.stack(move_feats))
        probs = torch.softmax(logits, dim=0).tolist()

    dv = abs(v_cpp - float(value.item()))
    max_dp = max(abs(a - b) for a, b in zip(priors_cpp, probs))
    print(f"\n[v2 {seed[:24]}…] value Δ={dv:.2e}  max prior Δ={max_dp:.2e}  (N={len(legal)})")
    assert dv < ATOL_V2, f"value: cpp={v_cpp} torch={float(value.item())} (Δ={dv:.2e})"
    assert max_dp < ATOL_V2, f"max prior diff {max_dp:.2e} (seed={seed})"


def test_native_v2_value_only(net_v2):
    _, cnet = net_v2
    pos = encode_position_v2(SEEDS[0])
    v, priors = cnet.eval(pos.flatten().tolist(), [])
    assert priors == []
    assert -1.0 <= v <= 1.0


@pytest.mark.parametrize("seed", SEEDS)
def test_native_v2_encoders_match_python(seed: str):
    """C++ V2 encoders must be BIT-exact to the Python ones (no float math beyond
    the same f64-divide-then-narrow), for both position planes and every move."""
    client = PyVariantClient()
    legal = client.new_game(seed)["legalMoves"]
    assert legal, seed

    pos_py = encode_position_v2(seed).flatten().tolist()
    pos_cpp = cpp.encode_position_v2(cpp.parse_fen(seed))
    assert len(pos_cpp) == len(pos_py) == 16 * 100
    assert pos_cpp == pos_py, f"position planes mismatch (seed={seed})"

    for mv in legal:
        m_py = encode_move_v2(mv).tolist()
        m_cpp = cpp.encode_move_v2(mv)
        assert m_cpp == m_py, f"move {mv['uci']} mismatch (seed={seed})"
