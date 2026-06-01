"""`ChesskersScorer.batch_eval` batches BOTH heads — trunk+value and the
policy head (ragged move lists padded to n_max, padded logits masked to -inf
before the softmax). This must reproduce, position-for-position, the
per-position reference (`policy_and_value` + softmax), including:

- genuinely ragged batches (different N per position),
- value-only entries (no moves) mixed with real ones — the all-`-inf` row
  softmaxes to NaN internally but is sliced to empty and must never leak into
  a neighbour's priors,
- a fully value-only batch (n_max == 0).
"""
from __future__ import annotations

import torch

from chessckers_engine.encoding import encode_move, encode_position
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.variant_py import PyVariantClient

# Positions chosen for DIFFERENT legal-move counts → a ragged batch.
FEN_KING_ALONE = "8/8/8/8/8/8/8/4K3 w - - 0 1"
FEN_D4E4 = "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1"
FEN_START = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s] w KQkq - 0 1"
)


def _legal(fen: str) -> list[dict]:
    return PyVariantClient().new_game(fen)["legalMoves"]


def _ref(model: ChesskersScorer, fen: str, legal: list[dict]) -> tuple[float, list[float]]:
    """Per-position reference — the path `InferenceServer` callers expect."""
    pos = encode_position(fen).unsqueeze(0)
    if not legal:
        with torch.no_grad():
            return float(model.value(pos).item()), []
    moves = torch.stack([encode_move(m) for m in legal])
    with torch.no_grad():
        logits, v = model.policy_and_value(pos, moves)
        priors = torch.softmax(logits, dim=0)
    return float(v.item()), priors.tolist()


def _batched(model, fens, legals):
    positions = torch.stack([encode_position(f) for f in fens])
    moves_list = [
        (torch.stack([encode_move(m) for m in lm]) if lm else None) for lm in legals
    ]
    with torch.no_grad():
        return model.batch_eval(positions, moves_list)


def _assert_matches(model, fens, legals):
    values, priors_list = _batched(model, fens, legals)
    assert values.shape == (len(fens),)
    for i, (fen, legal) in enumerate(zip(fens, legals)):
        ref_v, ref_p = _ref(model, fen, legal)
        assert abs(float(values[i]) - ref_v) < 1e-5, f"value mismatch @{i} ({fen})"
        got = priors_list[i]
        assert got.numel() == len(ref_p), f"prior count mismatch @{i}: {got.numel()} vs {len(ref_p)}"
        assert not torch.isnan(got).any(), f"NaN leaked into priors @{i} ({fen})"
        for p, ep in zip(got.tolist(), ref_p):
            assert abs(p - ep) < 1e-5, f"prior mismatch @{i} ({fen})"
        if got.numel():
            assert abs(float(got.sum()) - 1.0) < 1e-5, f"priors don't sum to 1 @{i}"


def test_ragged_batch_matches_per_position():
    torch.manual_seed(0)
    model = ChesskersScorer().eval()
    fens = [FEN_KING_ALONE, FEN_D4E4, FEN_START]
    legals = [_legal(f) for f in fens]

    counts = sorted({len(lm) for lm in legals})
    assert len(counts) >= 2, f"expected a ragged batch (distinct N); got counts {counts}"

    _assert_matches(model, fens, legals)


def test_empty_moves_mixed_with_real_no_nan_leak():
    """A value-only entry next to real ones: its all-`-inf` row must not
    corrupt the neighbours' softmaxes (the masking is per-row)."""
    torch.manual_seed(1)
    model = ChesskersScorer().eval()
    fens = [FEN_D4E4, FEN_KING_ALONE, FEN_START]
    legals = [_legal(FEN_D4E4), [], _legal(FEN_START)]  # middle = value-only

    values, priors_list = _batched(model, fens, legals)
    assert priors_list[1].numel() == 0, "value-only entry must yield empty priors"
    _assert_matches(model, fens, legals)


def test_all_value_only_batch():
    """n_max == 0 fast path: values still computed, every priors entry empty."""
    torch.manual_seed(2)
    model = ChesskersScorer().eval()
    fens = [FEN_KING_ALONE, FEN_KING_ALONE]
    values, priors_list = _batched(model, fens, [[], []])
    assert values.shape == (2,)
    assert all(p.numel() == 0 for p in priors_list)
    for i, fen in enumerate(fens):
        ref_v, _ = _ref(model, fen, [])
        assert abs(float(values[i]) - ref_v) < 1e-5
