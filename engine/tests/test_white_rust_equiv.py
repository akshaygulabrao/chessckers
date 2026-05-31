"""Equivalence harness for the White-movegen Rust port.

This is the correctness contract for porting White move generation to the
`chessckers_movegen` Rust crate. For a corpus of White-to-move positions
(endgame seeds, the standard start, random rollouts, and hand-built
castling / en-passant / promotion / in-check edge cases) it asserts that the
native Rust `white_legal_moves` returns the *exact same* set of LegalMove
dicts as the pure-Python python-chess path (the established ground truth).

Until the Rust function exists, the comparison cases skip with a clear reason;
the corpus builder + Python-truth side still run, so the scaffold is exercised.
When Rust lands (task #2), these activate and lock equivalence.

Ground-truth side forces `_rs_movegen = None` so the comparison is always
Python-truth vs. Rust, never Rust vs. Rust.
"""
from __future__ import annotations

import random

import chess
import pytest

import chessckers_engine.variant_py.client as _cl
import chessckers_engine.variant_py.moves_black as _mb
import chessckers_engine.variant_py.moves_white as _mw
from chessckers_engine.variant_py.state import STARTING_FEN, State, parse_fen, serialize_fen

# The reverse-curriculum endgame seeds (Black to move) — we roll out from these
# to reach White-to-move positions with the king near the rim.
_ENDGAME_SEEDS = [
    "8/8/8/8/8/3kk3/8/4K3[d3:kk,e3:kk] b - - 0 1",
    "8/8/8/8/4k3/4k3/8/4K3[e3:kk,e4:kk] b - - 0 1",
    "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1",
]

# Hand-built FENs exercising the error-prone White rules. All White to move.
# Board encodes Black Stones as `p` and Kings as `k`; the `s`/`k`/`S` letters
# live only in the bracketed overlay, with one entry per Black square.
_EDGE_CASES = [
    # Castling available both sides, nothing in the way.
    "4k3/8/8/8/8/8/8/R3K2R[e8:k] w KQ - 0 1",
    # Kingside castle only.
    "6k1/8/8/8/8/8/8/4K2R[g8:k] w K - 0 1",
    # Queenside castle only.
    "1k6/8/8/8/8/8/8/R3K3[b8:k] w Q - 0 1",
    # En passant available (black stone just advanced two squares to d5).
    "4k3/8/8/3pP3/8/8/8/4K3[d5:s,e8:k] w - d6 0 1",
    # White pawn one step from promotion (push to empty d8 = e8 occupied? no:
    # pawn on e7, black king on d8 → push e8 or capture exd8, each ×4 promos).
    "3k4/4P3/8/8/8/8/8/4K3[d8:k] w - - 0 1",
    # White pawn on 7th with a promotion capture to either side (d7 → c8/d8/e8).
    "2k1k3/3P4/8/8/8/8/8/4K3[c8:k,e8:k] w - - 0 1",
    # White king adjacent to a Black king-top tower (the false-FIDE-check trap).
    "8/8/8/8/8/3k4/8/3K4[d3:kk] w - - 0 1",
    # Black stone diagonally attacking a square next to the white king.
    "4k3/8/8/8/8/2p5/8/3K4[c3:s,e8:k] w - - 0 1",
]


def _canon(move: dict) -> tuple:
    """Canonical comparable form of a White LegalMove dict: the 13 stable keys,
    missing → None. Ignores any extra/internal keys."""
    keys = ("uci", "from", "to", "piece", "color", "capture", "waypoints",
            "chainHops", "promotion", "demotedKings", "demotionsRequired",
            "sourceKingPositions", "deployCount")
    out = []
    for k in keys:
        v = move.get(k)
        out.append((k, tuple(v) if isinstance(v, list) else v))
    return tuple(out)


def _python_truth(state: State) -> set[tuple]:
    """White legal moves via the pure-Python python-chess path (rust bypassed)."""
    saved_mw, saved_mb = _mw._rs_movegen, _mb._rs_movegen
    _mw._rs_movegen = None
    _mb._rs_movegen = None
    try:
        return {_canon(m) for m in _mw.white_legal_moves(state)}
    finally:
        _mw._rs_movegen = saved_mw
        _mb._rs_movegen = saved_mb


def _rust_white(state: State) -> set[tuple] | None:
    """White legal moves via the native Rust function, or None if not built."""
    rs = _mw._rs_movegen
    if rs is None or not hasattr(rs, "white_legal_moves"):
        return None
    b = state.board
    moves = rs.white_legal_moves(
        b.occupied,
        b.occupied_co[chess.WHITE],
        b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
        b.castling_rights,
        -1 if b.ep_square is None else b.ep_square,
        state.stacks,
    )
    return {_canon(m) for m in moves}


def _random_rollout_white_positions(start_fen: str, n_plies: int, rng: random.Random):
    """Yield serialized FENs of every White-to-move position reached by playing
    random legal moves from `start_fen` for up to n_plies."""
    from chessckers_engine.variant_py import PyVariantClient
    client = PyVariantClient()
    fen = start_fen
    for _ in range(n_plies):
        g = client.new_game(fen)
        if g.get("status"):
            return
        legal = g["legalMoves"]
        if not legal:
            return
        if g["turn"] == "white":
            yield fen
        mv = rng.choice(legal)
        fen = client.make_move(fen, mv["uci"])["fen"]


def _corpus() -> list[str]:
    """White-to-move FENs: start, edge cases, and random rollouts."""
    rng = random.Random(20260530)
    fens: list[str] = list(_EDGE_CASES)
    fens.append(STARTING_FEN)
    for seed in [STARTING_FEN, *_ENDGAME_SEEDS]:
        for _ in range(6):
            fens.extend(_random_rollout_white_positions(seed, 50, rng))
    # De-dup while preserving order.
    seen, uniq = set(), []
    for f in fens:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


_CORPUS = _corpus()


def test_corpus_is_nonempty_and_white_to_move():
    """Scaffold sanity: the corpus builds and every position is White to move."""
    assert len(_CORPUS) >= 50, f"corpus too small: {len(_CORPUS)}"
    for fen in _CORPUS:
        st = parse_fen(fen)
        assert st.board.turn == chess.WHITE, f"not white-to-move: {fen}"


@pytest.mark.parametrize("fen", _CORPUS)
def test_white_legal_moves_rust_matches_python(fen: str):
    """Rust white_legal_moves == python-chess truth, exact set equality."""
    st = parse_fen(fen)
    rust = _rust_white(st)
    if rust is None:
        pytest.skip("Rust white_legal_moves not built yet (task #2)")
    truth = _python_truth(st)
    missing = truth - rust
    extra = rust - truth
    assert not missing and not extra, (
        f"\nFEN: {fen}\n"
        f"  in python-truth but MISSING from rust: {sorted(m[0][1] for m in missing)}\n"
        f"  in rust but EXTRA (not legal):         {sorted(m[0][1] for m in extra)}"
    )
