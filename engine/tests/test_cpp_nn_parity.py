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

from chessckers_engine.encoding import encode_move, encode_position
from chessckers_engine.model import ChesskersScorer
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
