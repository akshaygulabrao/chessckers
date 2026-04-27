import torch

from chessckers_engine.encoding import MOVE_D, POS_C, encode_move, encode_position
from chessckers_engine.model import ChesskersScorer

INITIAL_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)


def _candidate_moves():
    return [
        {"from": "e2", "to": "e4", "uci": "e2e4"},
        {"from": "g1", "to": "f3", "uci": "g1f3"},
        {"from": "d2", "to": "d3", "uci": "d2d3"},
    ]


def test_forward_returns_one_logit_per_move():
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    pos = encode_position(INITIAL_FEN).unsqueeze(0)
    moves = torch.stack([encode_move(m) for m in _candidate_moves()])
    with torch.no_grad():
        logits = model(pos, moves)
    assert logits.shape == (3,)
    assert logits.dtype == torch.float32


def test_forward_output_is_finite_with_random_init():
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    pos = encode_position(INITIAL_FEN).unsqueeze(0)
    moves = torch.stack([encode_move(m) for m in _candidate_moves()])
    with torch.no_grad():
        logits = model(pos, moves)
    assert torch.isfinite(logits).all().item()


def test_forward_is_deterministic_for_same_input_and_seed():
    pos = encode_position(INITIAL_FEN).unsqueeze(0)
    moves = torch.stack([encode_move(m) for m in _candidate_moves()])

    torch.manual_seed(42)
    model_a = ChesskersScorer().eval()
    with torch.no_grad():
        out_a = model_a(pos, moves)

    torch.manual_seed(42)
    model_b = ChesskersScorer().eval()
    with torch.no_grad():
        out_b = model_b(pos, moves)

    assert torch.allclose(out_a, out_b)


def test_different_moves_get_different_logits():
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    pos = encode_position(INITIAL_FEN).unsqueeze(0)
    moves = torch.stack([encode_move(m) for m in _candidate_moves()])
    with torch.no_grad():
        logits = model(pos, moves)
    # Random init + distinct move features should not collapse to identical scores.
    assert logits.unique().numel() == logits.numel()


def test_forward_handles_single_candidate_move():
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    pos = encode_position(INITIAL_FEN).unsqueeze(0)
    moves = encode_move(_candidate_moves()[0]).unsqueeze(0)
    with torch.no_grad():
        logits = model(pos, moves)
    assert logits.shape == (1,)


def test_forward_rejects_unbatched_position():
    model = ChesskersScorer().eval()
    pos = encode_position(INITIAL_FEN)  # (C, 8, 8) — missing batch dim
    moves = torch.zeros((1, MOVE_D))
    try:
        model(pos, moves)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unbatched position tensor")


def test_default_dimensions_match_encoding_module():
    model = ChesskersScorer()
    # First Conv2d should accept POS_C input channels.
    first_conv = next(m for m in model.position_trunk if isinstance(m, torch.nn.Conv2d))
    assert first_conv.in_channels == POS_C
    # First Linear in move_encoder should accept MOVE_D features.
    first_move_linear = next(m for m in model.move_encoder if isinstance(m, torch.nn.Linear))
    assert first_move_linear.in_features == MOVE_D
