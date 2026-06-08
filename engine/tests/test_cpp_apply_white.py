"""Slice 5a oracle test: C++ White apply-move vs PyVariant, plus a C++-driven
full-game replay of BOTH colors' apply.

White apply ports python-chess board.push for the search-relevant fields
(piece move + promotion, capture incl. en passant, castling king+rook, castling
-rights clean/clear, ep target, turn). Verified by:
  * per-position: apply every legal White move in C++ and PyVariant, compare the
    child Board's fields + stacks;
  * C++-driven replay: advance state ENTIRELY with C++ apply (both colors) over
    full games, comparing to PyVariant's State at every ply. State objects are
    advanced directly (no FEN round-trip), so raw ep/castling are compared — the
    serialization canonicalization that would muddy a FEN diff is sidestepped.
"""
from __future__ import annotations

import random

import chess
import pytest

from chessckers_engine.variant_py import moves_black as mb
from chessckers_engine.variant_py import moves_white as mw
from chessckers_engine.variant_py.client import PyVariantClient
from chessckers_engine.variant_py.state import STARTING_FEN, parse_fen

cpp = pytest.importorskip("chessckers_cpp")

SEEDS = [
    STARTING_FEN,
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] w - - 0 1",
    "k7/8/8/8/8/8/8/R3K2R[a8:kk,h8:kk] w KQ - 0 1",   # castling-rich
    "7p/8/8/8/8/8/8/4K3[h8:ssss] w - - 0 1",
]


def _py_fields(b: chess.Board) -> tuple:
    return (
        int(b.occupied), int(b.occupied_co[chess.WHITE]), int(b.occupied_co[chess.BLACK]),
        int(b.pawns), int(b.knights), int(b.bishops), int(b.rooks), int(b.queens), int(b.kings),
        int(b.castling_rights), (-1 if b.ep_square is None else int(b.ep_square)),
        b.turn == chess.WHITE,
    )


def _cpp_fields(board) -> tuple:
    return (
        board.occupied, board.occupied_white, board.occupied_black,
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        board.castling_rights, board.ep_square, board.turn_white,
    )


def _collect(n_games=12, max_plies=50, seed=909, only_white=False):
    client = PyVariantClient()
    rng = random.Random(seed)
    fens = []
    for g in range(n_games):
        st = client.new_game(SEEDS[g % len(SEEDS)])
        for _ in range(max_plies):
            fen = st["fen"]
            if not only_white or parse_fen(fen).board.turn == chess.WHITE:
                fens.append(fen)
            legal = st.get("legalMoves") or []
            if st.get("status") or not legal:
                break
            st = client.make_move(fen, rng.choice(legal)["uci"])
    return fens


def test_apply_white_matches_pyvariant():
    fens = _collect(only_white=True)
    n = 0
    for fen in fens:
        for mv in mw.white_legal_moves(parse_fen(fen)):
            cb = cpp.apply_white_move(cpp.parse_fen(fen), mv)
            ps = mw.apply_white_move(parse_fen(fen), mv["uci"])
            assert _cpp_fields(cb) == _py_fields(ps.board), f"fen={fen} uci={mv['uci']}"
            assert dict(cb.stacks) == dict(ps.stacks), f"stacks fen={fen} uci={mv['uci']}"
            n += 1
    assert n > 500, f"too few White moves applied ({n})"


def _legal_dicts(state):
    if state.board.turn == chess.WHITE:
        return mw.white_legal_moves(state)
    return mb._all_black_legal(state)


def test_cpp_driven_replay_matches_pyvariant():
    rng = random.Random(20260603)
    total = 0
    for g in range(24):
        seed = SEEDS[g % len(SEEDS)]
        py_state = parse_fen(seed)
        cb = cpp.parse_fen(seed)
        for ply in range(220):
            assert _cpp_fields(cb) == _py_fields(py_state.board), f"g{g} ply{ply}"
            assert dict(cb.stacks) == {int(s): p for s, p in py_state.stacks.items()}, \
                f"stacks g{g} ply{ply}"
            total += 1
            status, _ = cpp.detect_status(cb)
            legal = _legal_dicts(py_state)
            if status or not legal:
                break
            mv = rng.choice(legal)
            if py_state.board.turn == chess.WHITE:
                cb = cpp.apply_white_move(cb, mv)
                py_state = mw.apply_white_move(py_state, mv["uci"])
            else:
                cb = cpp.apply_black_move(cb, mv)
                py_state = mb.apply_black_move_known(py_state, mv)
    assert total > 1000, f"replay too short ({total})"
