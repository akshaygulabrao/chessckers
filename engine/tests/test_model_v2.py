"""ChesskersScorerV2 (square-grounded gather head) + V2 encoders.

Structural / correctness tests for the research-backed redesign (see the
`project-policy-head-redesign` memory): the 10x10 spatial position encoding,
the gather-indexed move encoding, and that the model's gather is coordinate-
aligned with the position fill — i.e. the cell a move's from_idx gathers is the
same cell the position encoder wrote that piece into.
"""

import torch

from chessckers_engine.encoding import (
    CH_KING_TOP,
    CH_V2_ONBOARD,
    MOVE_D_V2,
    MV2_PATH_BASE,
    MV2_SCALAR_BASE,
    POS_C_V2,
    _sq10,
    encode_move_v2,
    encode_position_state_v2,
    encode_position_v2,
)
from chessckers_engine.model import ChesskersScorerV2
from chessckers_engine.variant_py.client import PyVariantClient

# Black king-top tower on d4; White pawns on c3/e3 → a real diagonal capture
# CHAIN exists (d4 hops e3, overshoots through the rim, settles on e1).
CHAIN_FEN = "8/8/8/8/3k4/2P1P3/8/4K3[d4:k] b - - 0 1"


def _client_moves(fen=None):
    c = PyVariantClient()
    g = c.new_game(fen) if fen else c.new_game()
    return g["fen"], g["legalMoves"]


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
def test_position_v2_shape_and_onboard_mask():
    fen, _ = _client_moves(CHAIN_FEN)
    pos = encode_position_v2(fen)
    assert pos.shape == (POS_C_V2, 10, 10)
    # On-board mask: 1 on the 8x8 interior, 0 everywhere on the rim ring.
    assert pos[CH_V2_ONBOARD, 1:9, 1:9].sum().item() == 64.0
    assert pos[CH_V2_ONBOARD, 0, :].sum().item() == 0.0
    assert pos[CH_V2_ONBOARD, 9, :].sum().item() == 0.0
    assert pos[CH_V2_ONBOARD, :, 0].sum().item() == 0.0
    assert pos[CH_V2_ONBOARD, :, 9].sum().item() == 0.0


def test_position_v2_piece_lands_at_gather_aligned_cell():
    # The d4 king-top tower must sit at the SAME 10x10 flat index that a move
    # from d4 will gather (this alignment is the whole point of V2).
    pos = encode_position_v2(CHAIN_FEN)
    idx = _sq10("d4")  # 44
    r10, f10 = divmod(idx, 10)
    assert pos[CH_KING_TOP, r10, f10].item() == 1.0


def test_position_v2_fen_and_state_agree():
    c = PyVariantClient()
    g = c.new_game(CHAIN_FEN)
    from_fen = encode_position_v2(g["fen"])
    # parse() exposes the State for the state-based encoder.
    state = c.parse(g["fen"])
    from_state = encode_position_state_v2(state)
    assert torch.equal(from_fen, from_state)


def test_move_v2_encoding_of_real_chain():
    _, moves = _client_moves(CHAIN_FEN)
    chain = next(m for m in moves if m.get("waypoints"))
    assert chain["from"] == "d4" and chain["to"] == "e1"
    v = encode_move_v2(chain)
    assert v.shape == (MOVE_D_V2,)
    assert int(v[0].item()) == _sq10("d4")  # from_idx
    assert int(v[1].item()) == _sq10("e1")  # to_idx
    # Path mask marks the INTERMEDIATE waypoints (e3, f2, and the rim square d0)
    # but NOT the endpoints d4/e1 — even though e1 appears in the raw waypoints.
    path = v[MV2_PATH_BASE:MV2_SCALAR_BASE]
    for wp in ("e3", "f2", "d0"):
        assert path[_sq10(wp)].item() == 1.0
    assert path[_sq10("e1")].item() == 0.0  # to excluded
    assert path[_sq10("d4")].item() == 0.0  # from excluded
    # Type scalars: capture + chain flags set, chain_len = 4/8.
    s = MV2_SCALAR_BASE
    assert v[s + 0].item() == 1.0  # is_capture
    assert v[s + 1].item() == 1.0  # is_chain
    assert abs(v[s + 4].item() - 4 / 8.0) < 1e-6  # chain_len


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _model():
    torch.manual_seed(0)
    return ChesskersScorerV2(d_hidden=64, c_filters=32, n_blocks=2).eval()


def test_forward_one_finite_logit_per_move():
    fen, moves = _client_moves(CHAIN_FEN)
    model = _model()
    pos = encode_position_v2(fen).unsqueeze(0)
    mv = torch.stack([encode_move_v2(m) for m in moves])
    with torch.no_grad():
        logits = model(pos, mv)
    assert logits.shape == (len(moves),)
    assert torch.isfinite(logits).all().item()


def test_value_in_range_and_policy_and_value_matches():
    fen, moves = _client_moves(CHAIN_FEN)
    model = _model()
    pos = encode_position_v2(fen).unsqueeze(0)
    mv = torch.stack([encode_move_v2(m) for m in moves])
    with torch.no_grad():
        v = model.value(pos)
        logits = model(pos, mv)
        logits2, v2 = model.policy_and_value(pos, mv)
    assert -1.0 <= v.item() <= 1.0
    assert torch.allclose(v, v2)
    assert torch.allclose(logits, logits2, atol=1e-6)


def test_batch_eval_matches_per_position_forward():
    # Two positions (White-to-move start + the Black chain position); batched
    # priors must equal the per-position softmax(forward), proving the padded
    # per-position gather is contamination-free.
    fen_a, moves_a = _client_moves()             # White to move, 20 moves, no chains
    fen_b, moves_b = _client_moves(CHAIN_FEN)    # Black to move, includes a chain
    model = _model()
    pos_a = encode_position_v2(fen_a).unsqueeze(0)
    pos_b = encode_position_v2(fen_b).unsqueeze(0)
    mv_a = torch.stack([encode_move_v2(m) for m in moves_a])
    mv_b = torch.stack([encode_move_v2(m) for m in moves_b])
    with torch.no_grad():
        ref_a = torch.softmax(model(pos_a, mv_a), dim=0)
        ref_b = torch.softmax(model(pos_b, mv_b), dim=0)
        positions = torch.cat([pos_a, pos_b], dim=0)
        values, priors = model.batch_eval(positions, [mv_a, mv_b])
    assert values.shape == (2,)
    assert torch.allclose(priors[0], ref_a, atol=1e-6)
    assert torch.allclose(priors[1], ref_b, atol=1e-6)
    assert abs(priors[0].sum().item() - 1.0) < 1e-5
    assert abs(priors[1].sum().item() - 1.0) < 1e-5


def test_gather_is_coordinate_aligned_with_spatial_map():
    # The crux: F.reshape(1, C, 100)[:, :, idx] must equal F[:, :, r, c] for
    # idx = r*10 + c. If this drifts, every move gathers the wrong square.
    model = _model()
    pos = encode_position_v2(CHAIN_FEN).unsqueeze(0)
    with torch.no_grad():
        spatial = model._spatial(pos)            # (1, C, 10, 10)
    flat = spatial.reshape(1, model.c_filters, 100)
    idx = _sq10("d4")
    r10, f10 = divmod(idx, 10)
    assert torch.equal(flat[0, :, idx], spatial[0, :, r10, f10])
