"""Tests for the Chessckers FEN parser/serializer (variant_py.state)."""
from __future__ import annotations

import chess
import pytest

from chessckers_engine.variant_py.state import parse_fen, serialize_fen


# Initial position FEN. NOTE: scalachess preserves `KQkq` in the castling
# field at game start as a literal, even though Black has no chess king to
# castle. After any move, scalachess strips `kq` (we see this in saved
# games: `w KQkq` → `b KQ` after move 1). python-chess always strips `kq`
# when no Black king is on e8. The engine's hot path doesn't see initial
# positions during play, only post-move positions, where both engines agree
# on canonical form. Test uses the canonical post-first-move form.
INITIAL_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQ - 0 1"
)

# Mid-game position from a real saved game (after ~20 plies of self-play).
MID_FEN = (
    "1ppp3p/k1kk2kk/pp1ppkkp/8/8/PPPPPPPP/R7/1NBQKBNR"
    "[a6:s,b6:s,d6:skS,e6:skS,f6:sk,g6:sSk,h6:s,a7:k,c7:k,d7:k,"
    "b8:s,c8:s,d8:s,h8:s] b K - 0 1"
)

# King-capture endgame.
KING_CAPTURE_FEN = "8/8/8/1p6/2K5/8/8/8[b5:s] b - - 0 1"

# Position with no overlay (all-empty board with one White king).
NO_OVERLAY_FEN = "8/8/8/8/8/8/8/4K3 w - - 0 1"


@pytest.mark.parametrize("fen", [INITIAL_FEN, MID_FEN, KING_CAPTURE_FEN, NO_OVERLAY_FEN])
def test_fen_roundtrip(fen):
    """Parsing then serializing must yield byte-identical canonical form."""
    assert serialize_fen(parse_fen(fen)) == fen


def test_initial_position_has_24_stacks():
    state = parse_fen(INITIAL_FEN)
    assert len(state.stacks) == 24  # 8 squares × 3 ranks (6, 7, 8)


def test_initial_position_pieces_by_rank():
    state = parse_fen(INITIAL_FEN)
    # Rank 6 and 8: Stones (unmoved)
    for f in range(8):
        assert state.stacks[chess.square(f, 5)] == "s"  # rank 6
        assert state.stacks[chess.square(f, 7)] == "s"  # rank 8
    # Rank 7: Kings
    for f in range(8):
        assert state.stacks[chess.square(f, 6)] == "k"


def test_overlay_with_multipiece_stack():
    """Stacks like `d6:skS` should preserve the full bottom-to-top string."""
    state = parse_fen(MID_FEN)
    assert state.stacks[chess.D6] == "skS"
    assert state.stacks[chess.E6] == "skS"
    assert state.stacks[chess.G6] == "sSk"  # Stone-bottom, moved-Stone, King-top


def test_no_overlay_yields_empty_stacks():
    state = parse_fen(NO_OVERLAY_FEN)
    assert state.stacks == {}


def test_turn_castling_etc_preserved():
    state = parse_fen(MID_FEN)
    assert state.board.turn == chess.BLACK
    assert state.board.has_kingside_castling_rights(chess.WHITE)
    assert not state.board.has_queenside_castling_rights(chess.WHITE)


def test_invalid_fen_raises():
    with pytest.raises(ValueError):
        parse_fen("totally-not-a-fen")
    with pytest.raises(ValueError):
        parse_fen("8/8/8/8/8/8/8/4K3[zz9:s] w - - 0 1")  # bad square in overlay


def test_state_copy_is_independent():
    state = parse_fen(INITIAL_FEN)
    copy = state.copy()
    copy.stacks[chess.A6] = "S"  # mutate the copy
    assert state.stacks[chess.A6] == "s"  # original untouched
