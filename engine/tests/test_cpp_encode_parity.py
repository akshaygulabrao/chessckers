"""Slice 6c oracle test: the native C++ encoders match the Rust-backed Python
encoders byte-for-byte.

  * cpp.encode_position(Board)  vs  encoding.encode_position(fen)  — 14*8*8 planes
  * cpp.encode_move(move_dict)  vs  encoding.encode_move(move_dict) — 240-dim

Over self-play rollout positions (both colors) so every move type is exercised:
quiets, deploys, charges (with demotions / overshoot waypoints), capture chains
(multi-hop waypoints), White captures / promotions / castling.
"""
from __future__ import annotations

import random

import chess
import pytest

from chessckers_engine import encoding
from chessckers_engine.variant_py.client import PyVariantClient
from chessckers_engine.variant_py.state import STARTING_FEN, parse_fen

cpp = pytest.importorskip("chessckers_cpp")

SEEDS = [
    STARTING_FEN,
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
    "k7/8/8/8/8/8/8/R3K2R[a8:kk,h8:kk] w KQ - 0 1",
]


def _collect(n_games=12, max_plies=45, seed=606):
    client = PyVariantClient()
    rng = random.Random(seed)
    out = []
    for g in range(n_games):
        st = client.new_game(SEEDS[g % len(SEEDS)])
        for _ in range(max_plies):
            out.append((st["fen"], st.get("legalMoves") or []))
            legal = st.get("legalMoves") or []
            if st.get("status") or not legal:
                break
            st = client.make_move(st["fen"], rng.choice(legal)["uci"])
    return out


CORPUS = _collect()


def test_encode_position_parity():
    for fen, _ in CORPUS:
        cpp_pos = cpp.encode_position(cpp.parse_fen(fen))
        py_pos = encoding.encode_position(fen).flatten().tolist()
        assert cpp_pos == py_pos, f"position planes differ at {fen}"


def test_encode_move_parity():
    n = 0
    for fen, legal in CORPUS:
        for mv in legal:
            cpp_m = cpp.encode_move(mv)
            py_m = encoding.encode_move(mv).tolist()
            assert cpp_m == py_m, f"move features differ: {mv['uci']} @ {fen}"
            n += 1
    assert n > 500, f"too few moves encoded ({n})"
