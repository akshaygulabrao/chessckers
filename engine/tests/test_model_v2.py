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
from chessckers_engine.model import (
    ChesskersScorer,
    ChesskersScorerV2,
    ResidualBlock,
    TransformerBlock2d,
    _AddSpatialPosEmb,
)
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


# --------------------------------------------------------------------------- #
# Transformer trunk (ResTNet residual-first interleave; gather head unchanged)
# --------------------------------------------------------------------------- #
def test_zero_transformer_blocks_is_byte_identical_resnet():
    # n_tf_blocks=0 (the default) must keep the trunk a pure ResNet: no
    # positional-embedding or attention params, so pre-transformer V2 checkpoints
    # and the committed module stay loadable byte-for-byte.
    keys = set(ChesskersScorerV2(d_hidden=64, c_filters=32, n_blocks=2).state_dict())
    assert not any(k.endswith(".pos") or "attn" in k or ".ff." in k for k in keys)


def test_transformer_trunk_adds_attention_and_is_residual_first():
    model = ChesskersScorerV2(d_hidden=64, c_filters=32, n_blocks=2,
                              n_tf_blocks=2, n_heads=4)
    keys = list(model.state_dict())
    assert any(k.endswith(".pos") for k in keys)      # learned positional embedding
    assert any("attn" in k for k in keys)             # multi-head attention params
    # Trunk body (after the 3-module conv stem) must OPEN with the positional
    # embedding then a residual block — never a transformer first (ResTNet: the
    # transformer-first ordering was the worst).
    body = list(model.position_trunk.children())[3:]
    assert isinstance(body[0], _AddSpatialPosEmb)
    first_res = next(i for i, m in enumerate(body) if isinstance(m, ResidualBlock))
    first_tf = next(i for i, m in enumerate(body) if isinstance(m, TransformerBlock2d))
    assert first_res < first_tf


def test_transformer_config_is_param_matched_to_v1():
    # The headline scale-up: 9 residual + 7 transformer @ default 96 filters lands
    # at ~2.52M params — matched to V1's ~2.51M (the delta is the 9.6K pos-emb), so
    # the A/B is a fair "same budget, better architecture" comparison.
    v2t = ChesskersScorerV2(n_blocks=9, n_tf_blocks=7, n_heads=4, tf_ff_mult=4)
    n = sum(p.numel() for p in v2t.parameters())
    assert n == 2_522_597, n
    n_v1 = sum(p.numel() for p in ChesskersScorer().parameters())
    assert abs(n - n_v1) < 25_000  # within ~1% of V1 — param-matched


def test_transformer_invalid_heads_rejected():
    import pytest
    with pytest.raises(ValueError):
        ChesskersScorerV2(c_filters=32, n_tf_blocks=1, n_heads=5)  # 5 ∤ 32


def test_transformer_batch_eval_has_no_cross_position_contamination():
    # Attention runs over ONE position's 100 tokens; batching two positions must
    # not leak features between them, so batched priors == per-position softmax.
    torch.manual_seed(0)
    model = ChesskersScorerV2(d_hidden=64, c_filters=32, n_blocks=2,
                              n_tf_blocks=2, n_heads=4).eval()
    fen_a, moves_a = _client_moves()
    fen_b, moves_b = _client_moves(CHAIN_FEN)
    pos_a = encode_position_v2(fen_a).unsqueeze(0)
    pos_b = encode_position_v2(fen_b).unsqueeze(0)
    mv_a = torch.stack([encode_move_v2(m) for m in moves_a])
    mv_b = torch.stack([encode_move_v2(m) for m in moves_b])
    with torch.no_grad():
        log_a = model(pos_a, mv_a)
        assert torch.isfinite(log_a).all()
        ref_a = torch.softmax(log_a, dim=0)
        ref_b = torch.softmax(model(pos_b, mv_b), dim=0)
        _, priors = model.batch_eval(torch.cat([pos_a, pos_b]), [mv_a, mv_b])
    assert torch.allclose(priors[0], ref_a, atol=1e-4)
    assert torch.allclose(priors[1], ref_b, atol=1e-4)


def test_arch_sidecar_roundtrip_rebuilds_transformer(tmp_path):
    # save_checkpoint drops a `.arch.json`; load_scorer rebuilds the EXACT trunk
    # (transformer included) from a bare .pt and reproduces the same logits —
    # the property the gauntlet relies on to reload a V2T net at its true shape.
    from chessckers_engine.checkpoints import load_scorer
    from chessckers_engine.train_az import save_checkpoint

    torch.manual_seed(0)
    model = ChesskersScorerV2(d_hidden=64, c_filters=32, n_blocks=2,
                              n_tf_blocks=2, n_heads=4).eval()
    path = tmp_path / "v2t.pt"
    save_checkpoint(model, path)
    assert (tmp_path / "v2t.pt.arch.json").exists()
    reloaded = load_scorer(path).eval()
    assert reloaded.arch == model.arch
    fen, moves = _client_moves(CHAIN_FEN)
    pos = encode_position_v2(fen).unsqueeze(0)
    mv = torch.stack([encode_move_v2(m) for m in moves])
    with torch.no_grad():
        assert torch.allclose(model(pos, mv), reloaded(pos, mv), atol=1e-6)
