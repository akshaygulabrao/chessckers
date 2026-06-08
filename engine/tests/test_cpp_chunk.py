"""Phase 3A (lc0-split migration): C++-side training-chunk encoding.

`cpp.play_game_chunk(seed=S)` plays one native game and encodes it to a gzipped
"ccz1" chunk entirely in C++. The gate is a tensor-identical round-trip through
the Python decoder:

    decode_chunk(cpp.play_game_chunk(seed=S))
        ==  az_game_to_examples(<AZGame from cpp.play_game_native(seed=S)>)

i.e. the C++ encoder reproduces, field-for-field, the AZExamples the Python
self-play path would have produced for the same game — so a native self-play
client can encode+upload chunks the Python trainer decodes byte-for-byte into
the same position/move/target tensors. Field equality (fen + full legal-move
dicts + visit_distribution + wdl_target + moves_left_target) is strictly stronger
than tensor equality. play_game_chunk uses play_game_pure(seed) which Phase 2
proved byte-identical to play_game_native(seed), so the two games are the same.
"""
from __future__ import annotations

import pytest
import torch

from chessckers_engine.model import ChesskersScorer, ChesskersScorerV2
from chessckers_engine.native_net import export_state_dict
from chessckers_engine.selfplay_az import AZGame, AZRecord, az_game_to_examples
from chessckers_engine.training_chunk import SCHEMA, decode_chunk

cpp = pytest.importorskip("chessckers_cpp")

SEED_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"


def _bin(m, tmp_path_factory, name):
    m.eval()
    wpath = str(tmp_path_factory.mktemp(name) / "net.bin")
    export_state_dict(m.state_dict(), wpath)
    return cpp.ChesskersNet(wpath)


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    torch.manual_seed(0)
    return _bin(ChesskersScorer(), tmp_path_factory, "net")


@pytest.fixture(scope="module")
def net_v2(tmp_path_factory):
    torch.manual_seed(0)
    return _bin(ChesskersScorerV2(n_blocks=2, n_tf_blocks=1, n_heads=4, tf_ff_mult=2),
                tmp_path_factory, "net_v2")


def _params(**over):
    p = dict(n_sims=24, c_puct=1.5, temperature=1.0, temp_cutoff_plies=8, max_plies=40,
             dirichlet_alpha=0.3, dirichlet_eps=0.25)
    p.update(over)
    return p


def _oracle_examples(net, seed, p):
    """Examples the Python path produces for the same native game."""
    records_raw, outcome, final_status = cpp.play_game_native(
        cpp.parse_fen(SEED_FEN), net, seed=seed, **p)
    records = [AZRecord(fen=fen, legal_moves=list(legal), visit_counts=list(vc), side_to_move=side)
               for (fen, legal, vc, side) in records_raw]
    return az_game_to_examples(AZGame(records=records, final_status=final_status, outcome=outcome))


@pytest.mark.parametrize("seed", [0, 100, 7])
def test_chunk_roundtrip_matches_oracle_v1(net, seed):
    p = _params()
    chunk = cpp.play_game_chunk(cpp.parse_fen(SEED_FEN), net, seed=seed, **p)
    assert isinstance(chunk, bytes) and len(chunk) > 0
    got = decode_chunk(chunk)
    assert got == _oracle_examples(net, seed, p)


@pytest.mark.parametrize("seed", [0, 100, 7])
def test_chunk_roundtrip_matches_oracle_v2(net_v2, seed):
    p = _params()
    chunk = cpp.play_game_chunk(cpp.parse_fen(SEED_FEN), net_v2, seed=seed, **p)
    got = decode_chunk(chunk)
    assert got == _oracle_examples(net_v2, seed, p)


def test_chunk_is_data_only_ccz1(net_v2):
    """The chunk is gzip (magic 1f 8b) and decodes to a ccz1 schema — never pickle."""
    import gzip
    import json

    chunk = cpp.play_game_chunk(cpp.parse_fen(SEED_FEN), net_v2, seed=3, **_params())
    assert chunk[:2] == b"\x1f\x8b"  # gzip magic; a pickle would not start here
    payload = json.loads(gzip.decompress(chunk))
    assert payload["schema"] == SCHEMA
    assert isinstance(payload["examples"], list) and payload["examples"]
    ex0 = payload["examples"][0]
    assert set(ex0) == {"fen", "legal_moves", "visit_distribution", "wdl_target",
                        "moves_left_target"}
    # visit distribution normalized; targets are JSON floats (not ints)
    assert abs(sum(ex0["visit_distribution"]) - 1.0) < 1e-9
    assert all(isinstance(x, float) for x in ex0["wdl_target"])
    assert isinstance(ex0["moves_left_target"], float)
