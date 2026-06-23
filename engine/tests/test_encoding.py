import torch

from chessckers_engine.encoding import (
    CH_DEPTH_BASE,
    CH_KING_TOP,
    CH_SIDE_TO_MOVE,
    CH_STONE_TOP,
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
    MV_WAYPOINT_BASE,
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


def test_starting_fen_tower_stack_depth_is_one():
    """All 24 starting towers have height 1 → only channel CH_DEPTH_BASE+0 is non-zero."""
    t = encode_position(INITIAL_FEN)
    depth0 = t[CH_DEPTH_BASE]  # ch 8 = piece at stack[0] (bottom = top for height-1)
    nonzero = (depth0 > 0).sum().item()
    assert nonzero == 24
    # Ranks 6 (stones) and 8 (stones) have value ~0.33; rank 7 (kings) has value 1.0
    for x in range(8):
        assert abs(depth0[5, x].item() - 1.0/3.0) < 1e-6  # a6..h6 = stone 's'
        assert abs(depth0[6, x].item() - 1.0) < 1e-6      # a7..h7 = king 'k'
        assert abs(depth0[7, x].item() - 1.0/3.0) < 1e-6  # a8..h8 = stone 's'
    # Deeper channels are all zero
    for d in range(1, 5):
        assert t[CH_DEPTH_BASE + d].sum().item() == 0.0


def test_per_depth_encodes_tower_order():
    """Tower 'kSs' at e7: bottom=k (1.0), middle=S (0.67), top=s (0.33)."""
    fen = "8/8/8/8/8/8/8/8[e7:kSs] w - - 0 1"
    t = encode_position(fen)
    x, y = 4, 6  # e7
    assert abs(t[CH_DEPTH_BASE + 0, y, x].item() - 1.0) < 1e-6      # bottom = k
    assert abs(t[CH_DEPTH_BASE + 1, y, x].item() - 2.0/3.0) < 1e-6   # middle = S
    assert abs(t[CH_DEPTH_BASE + 2, y, x].item() - 1.0/3.0) < 1e-6   # top = s
    assert t[CH_DEPTH_BASE + 3, y, x].item() == 0.0  # beyond height
    assert t[CH_DEPTH_BASE + 4, y, x].item() == 0.0


def test_per_depth_king_under_top():
    """Tower 'kS': bottom=k (1.0), top=S (0.67)."""
    fen = "8/8/8/8/8/8/8/8[d4:kS] w - - 0 1"
    t = encode_position(fen)
    x, y = 3, 3  # d4
    assert abs(t[CH_DEPTH_BASE + 0, y, x].item() - 1.0) < 1e-6       # bottom = k
    assert abs(t[CH_DEPTH_BASE + 1, y, x].item() - 2.0/3.0) < 1e-6    # top = S
    assert t[CH_DEPTH_BASE + 2, y, x].item() == 0.0


def test_side_to_move_plane_zero_when_white_to_move():
    t = encode_position(INITIAL_FEN)
    assert t[CH_SIDE_TO_MOVE].sum().item() == 0.0


def test_side_to_move_plane_all_ones_when_black_to_move():
    fen = INITIAL_FEN.replace(" w ", " b ")
    t = encode_position(fen)
    assert t[CH_SIDE_TO_MOVE].sum().item() == 64.0


def test_overlay_with_taller_tower_encodes_per_depth():
    """Tower 'kSs' at e7: bottom=k, middle=S, top=s, the rest zero."""
    fen = "8/8/8/8/8/8/8/8[e7:kSs] w - - 0 1"
    t = encode_position(fen)
    x, y = 4, 6  # e7
    assert abs(t[CH_DEPTH_BASE + 0, y, x].item() - 1.0) < 1e-6
    assert abs(t[CH_DEPTH_BASE + 1, y, x].item() - 2.0/3.0) < 1e-6
    assert abs(t[CH_DEPTH_BASE + 2, y, x].item() - 1.0/3.0) < 1e-6
    assert t[CH_DEPTH_BASE + 3, y, x].item() == 0.0
    assert t[CH_DEPTH_BASE + 4, y, x].item() == 0.0


def test_overlay_with_king_under_top_sets_second_depth():
    """Tower 'kS': bottom=k, top=S."""
    fen = "8/8/8/8/8/8/8/8[d4:kS] w - - 0 1"
    t = encode_position(fen)
    x, y = 3, 3  # d4
    assert abs(t[CH_DEPTH_BASE + 0, y, x].item() - 1.0) < 1e-6
    assert abs(t[CH_DEPTH_BASE + 1, y, x].item() - 2.0/3.0) < 1e-6
    assert t[CH_DEPTH_BASE + 2, y, x].item() == 0.0


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


def test_chain_waypoint_mask_distinguishes_paths_with_same_endpoints():
    """Two chains a8→a4 that share endpoints but visit different squares
    must encode to different vectors. Without the waypoint mask the policy
    head can't tell them apart and trains on contradictory targets."""
    m1 = {"from": "a8", "to": "a4", "uci": "a8a4", "waypoints": ["b7", "c6", "b5"]}
    m2 = {"from": "a8", "to": "a4", "uci": "a8a4", "waypoints": ["b7", "a6", "b5"]}
    via_diag = encode_move(m1)
    via_other = encode_move(m2)
    diff = (via_diag != via_other).nonzero().flatten().tolist()
    # m1 has c6 set but not a6, m2 has a6 set but not c6 → exactly 2 differing bits.
    assert len(diff) == 2
    assert all(MV_WAYPOINT_BASE <= b < MV_WAYPOINT_BASE + 100 for b in diff)


def test_chain_waypoint_mask_includes_rim_squares():
    """A chain that hops over a rim square (e.g. 'z6') sets a bit in the
    waypoint mask. Rim landings are legal post-no-bounce, and they're
    often the distinguishing feature between two chain paths."""
    m = {"from": "b8", "to": "b6", "uci": "b8b6", "waypoints": ["a7", "z6", "a5"]}
    v = encode_move(m)
    # 'z6' → file10=0, rank10=6 → offset 60 of the 100-bit region.
    assert v[MV_WAYPOINT_BASE + 60].item() == 1.0


def test_move_deploy_sets_deploy_flag_and_count():
    v = encode_move(_bare_move(deployCount=3))
    assert v[MV_DEPLOY].item() == 1.0
    assert abs(v[MV_DEPLOY_COUNT].item() - 3 / 5) < 1e-6


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
