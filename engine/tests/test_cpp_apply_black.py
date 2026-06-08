"""Slice 5a oracle test: C++ Black apply-move + status detection vs PyVariant.

Black apply: for every legal Black move from many positions, apply in C++ and in
PyVariant (apply_black_move_known) and compare the resulting child Board's
move-gen-relevant fields (occupancies, per-piece bitboards, castling, ep, turn)
and the stacks overlay. (Clocks and FEN serialization canonicalization are
irrelevant to search and not compared.)

Status: cpp.detect_status vs the engine's (status, winner) over rollout +
hand-crafted terminal positions.
"""
from __future__ import annotations

import random

import chess
import pytest

from chessckers_engine.variant_py import moves_black as mb
from chessckers_engine.variant_py.client import PyVariantClient
from chessckers_engine.variant_py.state import STARTING_FEN, parse_fen

cpp = pytest.importorskip("chessckers_cpp")

SEEDS = [
    "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/3kkk2/2PPPP2/2PPPP2/8/8/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
    STARTING_FEN,
]


def _collect(n_games=12, max_plies=60, seed=4242, only_black=False):
    client = PyVariantClient()
    rng = random.Random(seed)
    fens = []
    for g in range(n_games):
        st = client.new_game(SEEDS[g % len(SEEDS)])
        for _ in range(max_plies):
            fen = st["fen"]
            if not only_black or parse_fen(fen).board.turn == chess.BLACK:
                fens.append(fen)
            legal = st.get("legalMoves") or []
            if st.get("status") or not legal:
                break
            st = client.make_move(fen, rng.choice(legal)["uci"])
    return fens


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


def test_apply_black_matches_pyvariant():
    fens = _collect(only_black=True)
    n = 0
    for fen in fens:
        state = parse_fen(fen)
        for mv in mb._all_black_legal(state):
            cb = cpp.apply_black_move(cpp.parse_fen(fen), mv)
            ps = mb.apply_black_move_known(parse_fen(fen), mv)
            assert _cpp_fields(cb) == _py_fields(ps.board), f"fen={fen} mv={mv['uci']}"
            assert dict(cb.stacks) == dict(ps.stacks), f"stacks fen={fen} mv={mv['uci']}"
            n += 1
    assert n > 500, f"too few moves applied ({n})"


TERMINAL = [
    ("8/8/8/8/8/8/8/4K3 w - - 0 1", ("variantEnd", "white")),       # Black eliminated
    ("7p/8/8/8/8/8/8/8[h8:s] b - - 0 1", ("variantEnd", "black")),  # no White king
]


def test_detect_status_matches_pyvariant_over_rollout():
    client = PyVariantClient()
    for fen in _collect():
        st = client.new_game(fen)
        assert cpp.detect_status(cpp.parse_fen(fen)) == (st["status"], st["winner"]), fen


@pytest.mark.parametrize("fen,expected", TERMINAL)
def test_detect_status_terminal_cheap_checks(fen: str, expected: tuple):
    assert cpp.detect_status(cpp.parse_fen(fen)) == expected
