"""Slice 0 oracle test: the C++ `chessckers_cpp` FEN round-trip must match the
PyVariant reference byte-for-byte over a golden corpus.

Oracle: chessckers_engine.variant_py.state.parse_fen / serialize_fen (PyVariant).
C++:    chessckers_cpp.parse_fen / serialize_fen.
Assert: cpp.serialize_fen(cpp.parse_fen(fen)) == py.serialize_fen(py.parse_fen(fen)),
and that STARTING_FEN round-trips verbatim.

Skipped when the C++ extension isn't built (cpp/build.sh). The corpus is the
canonical-FEN set the roadmap specifies (all ep='-'); en-passant "legal"
canonicalization is a later, move-gen-dependent slice.
"""
from __future__ import annotations

import pytest

from chessckers_engine.variant_py.state import STARTING_FEN, parse_fen, serialize_fen

cpp = pytest.importorskip("chessckers_cpp")


# Golden corpus: starting position, curriculum seeds, tall/all-king/moved-stone/
# stone-over-king overlays (both turns), an empty-overlay edge case, and a
# castling-bearing position to exercise the castling-field override.
CORPUS = [
    STARTING_FEN,
    "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
    "8/8/3kkk2/8/8/8/3PPP2/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/3kkk2/8/8/8/1PPPPPP1/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
    "8/8/8/8/8/3kk3/8/4K3[d3:kk,e3:kk] b - - 0 1",
    "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1",
    "8/8/8/8/8/2k2k2/8/4K3[c3:kk,f3:kk] b - - 0 1",
    "8/8/pppppppp/8/8/8/8/4K3"
    "[a6:s,b6:S,c6:Sk,d6:sk,e6:Sksk,f6:kkk,g6:sk,h6:sS] w - - 0 1",
    "8/8/pppppppp/8/8/8/8/4K3"
    "[a6:s,b6:S,c6:Sk,d6:sk,e6:Sksk,f6:kkk,g6:sk,h6:sS] b - - 0 1",
    "8/8/8/8/8/3kkk2/8/4K3[d3:kkkk,e3:k,f3:kk] w - - 0 1",
    "8/8/8/8/8/8/8/4K3[] w - - 0 1",  # empty overlay -> serialized without brackets
    "rnbqkbnr/8/8/8/8/8/PPPPPPPP/RNBQKBNR[a8:k,h8:k] w KQ - 5 12",
]


@pytest.mark.parametrize("fen", CORPUS)
def test_cpp_fen_roundtrip_matches_pyvariant(fen: str):
    expected = serialize_fen(parse_fen(fen))
    got = cpp.serialize_fen(cpp.parse_fen(fen))
    assert got == expected, f"\n cpp: {got!r}\n  py: {expected!r}"


def test_starting_fen_castling_normalizes_like_pyvariant():
    # STARTING_FEN is NOT a serialize/parse fixed point: python-chess resolves
    # "KQkq" position-awarely and, because Black has no rooks on a8/h8, the black
    # queenside right resolves to nothing while kingside falls back to file-H ->
    # the castling field becomes "KQk". The C++ parser must reproduce that quirk.
    py_out = serialize_fen(parse_fen(STARTING_FEN))
    cpp_out = cpp.serialize_fen(cpp.parse_fen(STARTING_FEN))
    assert cpp_out == py_out
    assert " w KQk - 0 1" in py_out


def test_board_bb_decomposition_matches_python_chess():
    """The C++ Board's bb fields must agree with python-chess for White and with
    the Black-pawn/Black-king encoding for Black tops — the contract later slices
    (encoders, move-gen) build on."""
    import chess

    b = cpp.parse_fen(STARTING_FEN)
    ref = parse_fen(STARTING_FEN).board
    assert b.occupied == int(ref.occupied)
    assert b.occupied_white == int(ref.occupied_co[chess.WHITE])
    assert b.occupied_black == int(ref.occupied_co[chess.BLACK])
    assert b.pawns == int(ref.pawns)
    assert b.kings == int(ref.kings)
    assert b.castling_rights == int(ref.castling_rights)
    assert b.turn_white is True
    assert dict(b.stacks) == {sq: pc for sq, pc in parse_fen(STARTING_FEN).stacks.items()}
