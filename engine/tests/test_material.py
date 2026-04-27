from chessckers_engine.material import material, material_for_side_to_move

INITIAL_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)

# White starting: 8P + 2N + 2B + 2R + Q + K = 8 + 6 + 6 + 10 + 9 + 1000 = 1039
# Black starting: 16 Stones (ranks 6 & 8) + 8 Kings (rank 7) = 16 + 16 = 32
WHITE_START_TOTAL = 1039
BLACK_START_TOTAL = 32


def test_starting_position_balance_is_white_total_minus_black_total():
    assert material(INITIAL_FEN) == WHITE_START_TOTAL - BLACK_START_TOTAL


def test_white_to_move_perspective_matches_raw_material():
    assert material_for_side_to_move(INITIAL_FEN) == material(INITIAL_FEN)


def test_black_to_move_perspective_flips_sign():
    fen = INITIAL_FEN.replace(" w ", " b ")
    assert material_for_side_to_move(fen) == -material(INITIAL_FEN)


def test_empty_board_has_zero_material():
    assert material("8/8/8/8/8/8/8/8 w - - 0 1") == 0


def test_no_overlay_means_no_black_material():
    fen = "RNBQKBNR/PPPPPPPP/8/8/8/8/8/8 w - - 0 1"
    assert material(fen) == WHITE_START_TOTAL


def test_unmoved_and_moved_stones_have_equal_value():
    a = "8/8/8/8/8/8/8/8[d4:s] w - - 0 1"
    b = "8/8/8/8/8/8/8/8[d4:S] w - - 0 1"
    assert material(a) == material(b) == -1


def test_single_king_tower_counts_two():
    assert material("8/8/8/8/8/8/8/8[d4:k] w - - 0 1") == -2


def test_tower_kk_counts_four_for_black():
    assert material("8/8/8/8/8/8/8/8[d4:kk] w - - 0 1") == -4


def test_tower_sk_counts_one_stone_plus_two_king():
    # bottom-to-top: Stone, King → 1 + 2 = 3 worth of Black material
    assert material("8/8/8/8/8/8/8/8[d4:sk] w - - 0 1") == -3


def test_white_pawn_capture_increases_material_by_one():
    """Removing one Black Stone (from any square) increases material by 1."""
    base = material(INITIAL_FEN)
    fen_minus_a6 = INITIAL_FEN.replace("a6:s,", "")
    assert material(fen_minus_a6) == base + 1


def test_white_pawn_capture_of_king_increases_material_by_two():
    base = material(INITIAL_FEN)
    fen_minus_a7_king = INITIAL_FEN.replace("a7:k,", "")
    assert material(fen_minus_a7_king) == base + 2


def test_invalid_fen_raises():
    import pytest

    with pytest.raises(ValueError):
        material("not a fen")


def test_king_value_override_zeros_out_king_for_training_targets():
    # King is the only White piece; default value -> 1000, override -> 0
    fen = "8/8/8/8/8/8/8/4K3 w - - 0 1"
    assert material(fen) == 1000
    assert material(fen, king_value=0) == 0


def test_king_value_override_propagates_through_side_to_move_helper():
    fen = "8/8/8/8/8/8/8/4K3 b - - 0 1"  # black to move; sign flips
    assert material_for_side_to_move(fen) == -1000
    assert material_for_side_to_move(fen, king_value=0) == 0
