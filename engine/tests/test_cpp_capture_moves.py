"""Slice 2a oracle test: C++ Black diagonal capture moves (chains + first-hop
rams) vs the pure-Python reference (moves_black) — SET-equality: the canonical
move-set must match.

Positions exercise multi-hop chains, branching, first-hop rams, off-grid
overshoot in a chain, and a king-capturing chain (the white-king short-circuit).
"""
from __future__ import annotations

import chess
import pytest

from chessckers_engine.variant_py import moves_black as mb
from chessckers_engine.variant_py.state import parse_fen

cpp = pytest.importorskip("chessckers_cpp")

CORPUS = [
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",      # dense, short chains
    "k7/1N6/2N5/3N4/4N3/5N2/8/4K3[a8:kkkkk] b - - 0 1",                  # long single chain
    "8/8/2k1k3/1P1P1P2/8/8/8/4K3[c6:kkk,e6:kkk] b - - 0 1",             # branching towers
    "8/8/8/8/2k5/1P6/K7/8[c4:kkkk] b - - 0 1",                          # chain captures White king
    "7Q/6k1/8/8/8/8/8/4K3[g7:kkk] b - - 0 1",                          # overshoot in a chain
    "8/8/3kkk2/8/8/8/1PPPPPP1/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",        # seed-mix shape
]


def _canon(m: dict) -> tuple:
    wp = m.get("waypoints")
    return (
        m["uci"],
        m["from"],
        m["to"],
        m["piece"],
        m.get("color"),
        m.get("capture"),
        tuple(wp) if wp is not None else None,
        tuple(m.get("chainHops") or []),
        tuple(m.get("_chain_all_captures") or []),
        m.get("cadence"),
        bool(m.get("_is_suicide")),
        bool(m.get("_chain_promotes")),
    )


def _bb(state):
    occ = int(state.board.occupied)
    occw = int(state.board.occupied_co[chess.WHITE])
    wk = state.board.king(chess.WHITE)
    king_sq = -1 if wk is None else int(wk)
    stacks = {int(s): p for s, p in state.stacks.items()}
    return occ, occw, king_sq, stacks


def _py_moves(state):
    return mb.black_diagonal_capture_moves(state)


@pytest.mark.parametrize("fen", CORPUS)
def test_capture_moves_setdiff_vs_python(fen: str):
    state = parse_fen(fen)
    occ, occw, king_sq, stacks = _bb(state)
    cpp_moves = cpp.black_diagonal_capture_moves(occ, occw, king_sq, stacks)
    py_set = {_canon(m) for m in _py_moves(state)}
    cpp_set = {_canon(m) for m in cpp_moves}
    assert cpp_set == py_set, (
        f"\n only in cpp: {sorted(cpp_set - py_set)}\n only in py:  {sorted(py_set - cpp_set)}"
    )


def test_corpus_actually_exercises_captures():
    # Guard against an all-empty bug (which would make every set-diff trivially
    # pass). The first five positions are built to yield real captures/chains.
    for fen in CORPUS[:5]:
        state = parse_fen(fen)
        occ, occw, king_sq, stacks = _bb(state)
        assert cpp.black_diagonal_capture_moves(occ, occw, king_sq, stacks), fen
