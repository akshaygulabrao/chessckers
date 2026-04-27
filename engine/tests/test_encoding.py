import torch

from chessckers_engine.encoding import (
    CH_KING_COUNT,
    CH_KING_TOP,
    CH_SECOND_IS_KING,
    CH_SIDE_TO_MOVE,
    CH_STONE_COUNT,
    CH_STONE_TOP,
    CH_TOP_IS_UNMOVED_STONE,
    CH_TOWER_HEIGHT,
    CH_W_KNIGHT,
    CH_W_PAWN,
    MOVE_D,
    MV_CAPTURE,
    MV_CHAIN,
    MV_CHAIN_LEN,
    MV_DEMOTIONS_REQ,
    MV_DEPLOY,
    MV_DEPLOY_COUNT,
    MV_ORTHO,
    MV_PROMO_BASE,
    POS_C,
    encode_move,
    encode_position,
    square_index,
)

INITIAL_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)


def test_position_tensor_has_expected_shape():
    t = encode_position(INITIAL_FEN)
    assert t.shape == (POS_C, 8, 8)
    assert t.dtype == torch.float32


def test_starting_fen_white_pawns_hot_on_rank_2():
    t = encode_position(INITIAL_FEN)
    pawn_plane = t[CH_W_PAWN]
    assert pawn_plane.sum().item() == 8.0
    # rank 2 is y=1; all 8 files hot
    assert pawn_plane[1].tolist() == [1.0] * 8
    # every other rank is empty
    for y in [0, 2, 3, 4, 5, 6, 7]:
        assert pawn_plane[y].sum().item() == 0.0


def test_starting_fen_white_knights_hot_at_b1_and_g1():
    t = encode_position(INITIAL_FEN)
    knight_plane = t[CH_W_KNIGHT]
    assert knight_plane.sum().item() == 2.0
    assert knight_plane[0, 1].item() == 1.0  # b1
    assert knight_plane[0, 6].item() == 1.0  # g1


def test_starting_fen_stone_top_marker_hot_on_ranks_6_and_8():
    t = encode_position(INITIAL_FEN)
    stone_plane = t[CH_STONE_TOP]
    # 8 stones on rank 6 (y=5) + 8 on rank 8 (y=7) = 16 hot
    assert stone_plane.sum().item() == 16.0
    assert stone_plane[5].tolist() == [1.0] * 8
    assert stone_plane[7].tolist() == [1.0] * 8
    # rank 7 is kings, not stones
    assert stone_plane[6].sum().item() == 0.0


def test_starting_fen_king_top_marker_hot_on_rank_7():
    t = encode_position(INITIAL_FEN)
    king_plane = t[CH_KING_TOP]
    assert king_plane.sum().item() == 8.0
    assert king_plane[6].tolist() == [1.0] * 8


def test_starting_fen_tower_height_is_one_over_24_on_all_24_tower_squares():
    t = encode_position(INITIAL_FEN)
    height_plane = t[CH_TOWER_HEIGHT]
    expected = 1.0 / 24.0
    nonzero = (height_plane > 0).sum().item()
    assert nonzero == 24
    # Every nonzero entry equals the singleton-tower height
    for y in [5, 6, 7]:
        for x in range(8):
            assert abs(height_plane[y, x].item() - expected) < 1e-6


def test_starting_fen_top_is_unmoved_stone_set_for_initial_stones_only():
    t = encode_position(INITIAL_FEN)
    plane = t[CH_TOP_IS_UNMOVED_STONE]
    # 16 unmoved stones (ranks 6 and 8), 0 elsewhere
    assert plane.sum().item() == 16.0
    assert plane[5].tolist() == [1.0] * 8
    assert plane[7].tolist() == [1.0] * 8
    assert plane[6].sum().item() == 0.0  # rank 7 is kings


def test_starting_fen_stone_count_and_king_count_normalize_by_24():
    t = encode_position(INITIAL_FEN)
    # All 24 starting towers have height 1: stones on 6/8, kings on 7
    one_over_24 = 1.0 / 24.0
    assert abs(t[CH_STONE_COUNT, 5, 0].item() - one_over_24) < 1e-6  # a6 stone
    assert abs(t[CH_KING_COUNT, 6, 0].item() - one_over_24) < 1e-6  # a7 king
    assert t[CH_STONE_COUNT, 6, 0].item() == 0.0  # a7 has no stones
    assert t[CH_KING_COUNT, 5, 0].item() == 0.0  # a6 has no kings


def test_starting_fen_second_is_king_is_zero_everywhere_for_singleton_towers():
    t = encode_position(INITIAL_FEN)
    assert t[CH_SECOND_IS_KING].sum().item() == 0.0


def test_side_to_move_plane_zero_when_white_to_move():
    t = encode_position(INITIAL_FEN)
    assert t[CH_SIDE_TO_MOVE].sum().item() == 0.0


def test_side_to_move_plane_all_ones_when_black_to_move():
    fen = INITIAL_FEN.replace(" w ", " b ")
    t = encode_position(fen)
    assert t[CH_SIDE_TO_MOVE].sum().item() == 64.0


def test_overlay_with_taller_tower_encodes_height_and_second_is_king():
    # Single tower at e7: bottom King, then Stone(moved), then Stone(unmoved) on top
    # pieces string is bottom-to-top: "kSs" → height 3, top=s, second=S
    fen = "8/8/8/8/8/8/8/8[e7:kSs] w - - 0 1"
    t = encode_position(fen)
    x, y = 4, 6  # e7
    assert abs(t[CH_TOWER_HEIGHT, y, x].item() - 3 / 24) < 1e-6
    assert abs(t[CH_STONE_COUNT, y, x].item() - 2 / 24) < 1e-6
    assert abs(t[CH_KING_COUNT, y, x].item() - 1 / 24) < 1e-6
    assert t[CH_TOP_IS_UNMOVED_STONE, y, x].item() == 1.0
    # Second-from-top is S (moved stone), not k → plane stays 0
    assert t[CH_SECOND_IS_KING, y, x].item() == 0.0


def test_overlay_with_king_under_top_sets_second_is_king():
    # Tower kS: King at bottom, moved Stone on top → second-from-top is k
    fen = "8/8/8/8/8/8/8/8[d4:kS] w - - 0 1"
    t = encode_position(fen)
    x, y = 3, 3  # d4
    assert t[CH_SECOND_IS_KING, y, x].item() == 1.0
    # Top is S, not s, so unmoved-stone marker stays 0
    assert t[CH_TOP_IS_UNMOVED_STONE, y, x].item() == 0.0


# ---- Move encoding ----


def _bare_move(**overrides):
    base = {"from": "e2", "to": "e4", "uci": "e2e4"}
    base.update(overrides)
    return base


def test_move_tensor_has_expected_shape_and_dtype():
    v = encode_move(_bare_move())
    assert v.shape == (MOVE_D,)
    assert v.dtype == torch.float32


def test_move_e2e4_has_correct_from_and_to_one_hots():
    v = encode_move(_bare_move())
    e2 = square_index("e2")
    e4 = square_index("e4")
    assert v[e2].item() == 1.0
    assert v[64 + e4].item() == 1.0
    # Exactly two one-hot bits set in the from/to region
    assert v[:128].sum().item() == 2.0


def test_move_quiet_move_has_no_flags_set():
    v = encode_move(_bare_move())
    for bit in [MV_CAPTURE, MV_CHAIN, MV_DEPLOY, MV_ORTHO]:
        assert v[bit].item() == 0.0
    # Promotion 'none' bit (index 0 of the 5) is hot
    assert v[MV_PROMO_BASE].item() == 1.0


def test_move_capture_sets_capture_flag():
    v = encode_move(_bare_move(capture="e4"))
    assert v[MV_CAPTURE].item() == 1.0


def test_move_chain_sets_chain_flag_and_length():
    v = encode_move(_bare_move(waypoints=["g5", "h4"], capture="g5"))
    assert v[MV_CHAIN].item() == 1.0
    assert abs(v[MV_CHAIN_LEN].item() - 2 / 8) < 1e-6


def test_move_deploy_sets_deploy_flag_and_count():
    v = encode_move(_bare_move(deployCount=3))
    assert v[MV_DEPLOY].item() == 1.0
    assert abs(v[MV_DEPLOY_COUNT].item() - 3 / 24) < 1e-6


def test_move_ortho_sets_ortho_flag_and_demotions():
    v = encode_move(_bare_move(demotionsRequired=2))
    assert v[MV_ORTHO].item() == 1.0
    assert abs(v[MV_DEMOTIONS_REQ].item() - 2 / 8) < 1e-6


def test_move_promotion_to_queen_hot_at_q_slot():
    v = encode_move(_bare_move(promotion="q"))
    # 'none' = base, q = base+1
    assert v[MV_PROMO_BASE].item() == 0.0
    assert v[MV_PROMO_BASE + 1].item() == 1.0


def test_move_promotion_to_knight_hot_at_n_slot():
    v = encode_move(_bare_move(promotion="n"))
    assert v[MV_PROMO_BASE + 4].item() == 1.0
