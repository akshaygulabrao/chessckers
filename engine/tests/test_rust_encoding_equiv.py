"""Byte-for-byte equivalence: Rust encode_position/encode_move vs the Python
reference (_encode_position_py / _encode_move_py) in chessckers_engine.encoding.

The public encoding.encode_position / encode_move dispatch to the Rust extension
when encoding._rs is not None; the Python `_*_py` functions are the spec. Every
assertion is torch.equal (exact bit equality), not allclose.

Skipped entirely when the Rust extension is unavailable.
"""
from __future__ import annotations

import pytest
import torch

from chessckers_engine import encoding
from chessckers_engine.variant_py.client import PyVariantClient

if encoding._rs is None:
    pytest.skip("Rust extension not built; nothing to compare", allow_module_level=True)


# --- Diverse FENs ---------------------------------------------------------

CHESS_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

CURRICULUM_SEEDS = [
    "8/8/8/8/8/3kk3/8/4K3[d3:kk,e3:kk] b - - 0 1",
    "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1",
    "8/8/8/3kk3/8/8/8/4K3[d5:kk,e5:kk] b - - 0 1",
    "8/8/3kk3/8/8/8/8/4K3[d6:kk,e6:kk] b - - 0 1",
    "8/8/8/8/8/2k2k2/8/4K3[c3:kk,f3:kk] b - - 0 1",
]

# Chess start board with crafted overlays exercising every position channel.
# Squares chosen on Black's home ranks (6/7/8) so the bitboard pawn 'p' top
# matches the overlay top (Stone), and king-top squares carry 'k' on the board.
# The bitboard glyph (p vs k) must agree with the overlay top for a *valid*
# Chessckers position, but encode_position reads them independently, so the
# overlay drives channels 8-12 regardless. We exercise each overlay shape:
#   a6: height-1 unmoved stone (top_is_unmoved_stone)
#   b6: height-1 moved stone 'S'
#   c6: height-2 stone-over-king (second_is_king, top moved stone)
#   d6: height-2 unmoved-stone-over-king (top_is_unmoved_stone AND second_is_king)
#   e6: tall mixed tower height-4
#   f6: all-King tower height-3 (top is king -> neither flag)
#   g6: king-top over stone (second is stone, not king)
#   h6: height-2 unmoved stone over moved stone
OVERLAY_BOARD = (
    "pppppppp/8/8/8/8/4K3"  # ranks 8..3 region; we only need a parseable 8-rank board
)
OVERLAY_FENS = [
    # full 8-rank board with Black pawns on rank 6 so 'p' tops line up with overlay
    "8/8/pppppppp/8/8/8/8/4K3"
    "[a6:s,b6:S,c6:Sk,d6:sk,e6:Sksk,f6:kkk,g6:sk,h6:sS] w - - 0 1",
    "8/8/pppppppp/8/8/8/8/4K3"
    "[a6:s,b6:S,c6:Sk,d6:sk,e6:Sksk,f6:kkk,g6:sk,h6:sS] b - - 0 1",
    # An all-King overlay across a wide tower
    "8/8/8/8/8/3kkk2/8/4K3[d3:kkkk,e3:k,f3:kk] w - - 0 1",
    "8/8/8/8/8/3kkk2/8/4K3[d3:kkkk,e3:k,f3:kk] b - - 0 1",
]

# Positions whose legal moves carry capture-chain waypoints (rim + on-board).
WAYPOINT_FENS = [
    "8/8/8/8/2P1P3/8/3kk3/4K3[d2:kk,e2:kk] b - - 0 1",
    "8/8/2P1P1P1/8/4k3/8/8/4K3[e4:kkkk] b - - 0 1",
    # Long multi-hop chain c3->d8 wp=['d4','e5','f6','e7','d8','c9'] (rim c9)
    "8/2P1P3/8/2P1P3/8/2kk4/8/4K3[c3:kk] b - - 0 1",
    "8/3P1P2/8/1P1P4/8/3kkkk1/8/4K3[d3:kkkk] b - - 0 1",
]

ALL_FENS = (
    CURRICULUM_SEEDS
    + [f"{CHESS_START} w - - 0 1", f"{CHESS_START} b - - 0 1"]
    + OVERLAY_FENS
    + WAYPOINT_FENS
)


def _assert_pos_equal(fen: str) -> None:
    rust = encoding.encode_position(fen)
    py = encoding._encode_position_py(fen)
    assert rust.shape == py.shape == (encoding.POS_C, 8, 8)
    assert torch.equal(rust, py), f"position encoding mismatch for {fen!r}"


def _assert_move_equal(mv: dict, ctx: str) -> None:
    rust = encoding.encode_move(mv)
    py = encoding._encode_move_py(mv)
    assert rust.shape == py.shape == (encoding.MOVE_D,)
    assert torch.equal(rust, py), f"move encoding mismatch ({ctx}): {mv}"


@pytest.mark.parametrize("fen", ALL_FENS)
def test_position_equivalence(fen: str) -> None:
    _assert_pos_equal(fen)


@pytest.mark.parametrize("fen", ALL_FENS)
def test_legal_moves_equivalence(fen: str) -> None:
    # Not every synthetic FEN has legal moves for the side to move (e.g. an
    # artificial White-to-move endgame, or a Black side with no real towers);
    # those just contribute zero move comparisons. The curriculum + waypoint
    # FENs that MUST produce moves are asserted in their own tests.
    client = PyVariantClient()
    res = client.new_game(fen)
    for mv in res["legalMoves"]:
        _assert_move_equal(mv, ctx=fen)


@pytest.mark.parametrize("fen", CURRICULUM_SEEDS)
def test_curriculum_seeds_have_moves(fen: str) -> None:
    res = PyVariantClient().new_game(fen)
    assert res["legalMoves"], f"curriculum seed produced no legal moves: {fen!r}"


def test_waypoint_chains_present_and_equal() -> None:
    """Ensure the waypoint-FEN positions actually emit moves with waypoints
    (rim + on-board) so the 100-bit 10x10 mask is genuinely exercised, and
    those moves encode equivalently."""
    client = PyVariantClient()
    rim_seen = set()
    onboard_seen = set()
    multi_hop = 0
    for fen in WAYPOINT_FENS:
        res = client.new_game(fen)
        wp_moves = [m for m in res["legalMoves"] if m.get("waypoints")]
        assert wp_moves, f"expected waypoint-carrying moves for {fen!r}"
        for mv in wp_moves:
            _assert_move_equal(mv, ctx=f"waypoint:{fen}")
            wps = mv["waypoints"]
            if len(wps) > 1:
                multi_hop += 1
            for w in wps:
                # rim files z/i or rim ranks 0/9
                if w[0] in "zi" or w[1] in "09":
                    rim_seen.add(w)
                else:
                    onboard_seen.add(w)
    assert rim_seen, "no rim waypoints exercised"
    assert onboard_seen, "no on-board waypoints exercised"
    assert multi_hop > 0, "no multi-hop chain exercised"


# --- Synthetic move dicts: every feature field independently --------------

ALL_RIM_AND_BOARD_WAYPOINTS = (
    # rim corners/edges
    ["z0", "i9", "a0", "h9", "z5", "i5", "e0", "e9"]
    # full on-board file/rank sweep
    + [f"{f}{r}" for f in "abcdefgh" for r in "1234"]
)


def _base_move(**over) -> dict:
    mv = {"from": "a1", "to": "h8"}
    mv.update(over)
    return mv


SYNTHETIC_MOVES = [
    # from/to extremes across the board
    _base_move(**{"from": "a1", "to": "a1"}),
    _base_move(**{"from": "h8", "to": "a8"}),
    _base_move(**{"from": "d4", "to": "e5"}),
    # capture set / unset
    _base_move(capture="e5"),
    _base_move(capture=None),
    # waypoints: a single rim square
    _base_move(waypoints=["z0"]),
    _base_move(waypoints=["i9"]),
    _base_move(waypoints=["a0"]),
    _base_move(waypoints=["h9"]),
    # waypoints: empty list (should NOT set is_chain)
    _base_move(waypoints=[]),
    # waypoints: large mixed set spanning rim + board
    _base_move(waypoints=ALL_RIM_AND_BOARD_WAYPOINTS),
    # deployCount
    _base_move(deployCount=1),
    _base_move(deployCount=24),
    _base_move(deployCount=0),
    # demotionsRequired
    _base_move(demotionsRequired=1),
    _base_move(demotionsRequired=8),
    _base_move(demotionsRequired=0),
    # promotions: all 5 values
    _base_move(promotion=None),
    _base_move(promotion="q"),
    _base_move(promotion="r"),
    _base_move(promotion="b"),
    _base_move(promotion="n"),
    # everything at once
    _base_move(
        capture="d4",
        waypoints=["z0", "c3", "i9", "e5"],
        deployCount=7,
        demotionsRequired=3,
        promotion="q",
    ),
    # optional keys entirely missing (only from/to) — must match Python's .get
    {"from": "c3", "to": "f6"},
    # optional keys present but explicitly None
    {
        "from": "c3",
        "to": "f6",
        "capture": None,
        "waypoints": None,
        "deployCount": None,
        "demotionsRequired": None,
        "promotion": None,
    },
    # a degenerate waypoint glyph that Python skips (len != 2 / unknown char)
    _base_move(waypoints=["abc", "x1", "z0", "9z"]),
]


@pytest.mark.parametrize("mv", SYNTHETIC_MOVES)
def test_synthetic_move_equivalence(mv: dict) -> None:
    _assert_move_equal(dict(mv), ctx="synthetic")


# --- Documented boundary: malformed / unreachable inputs ------------------
#
# The equivalence contract holds for every VALID Chessckers FEN (overlay squares
# are always on-board a1..h8 via chess.square_name; board ranks always 8 files)
# and every move dict the engine emits (UCI squares a1..h8; deployCount /
# demotionsRequired are int|None). On such inputs Rust == Python bit-for-bit,
# verified above plus a 14k-position / 335k-move self-play fuzz.
#
# On MALFORMED, ENGINE-UNREACHABLE inputs the two intentionally differ, and the
# Python reference's own behavior there is accidental (torch tolerates negative
# index wrap but raises on positive overflow). Rust's guards are deliberately
# more defensive (skip / no-op). These tests pin that boundary so a future change
# that breaks VALID-input equivalence can't hide behind "it's just an edge case".


def test_offboard_overlay_square_is_unreachable_and_rust_skips() -> None:
    """An overlay on a rim square (z9/a9/i1) can never be produced by the
    engine. Python raises IndexError (no bounds check before tensor write);
    Rust skips the entry. Documented difference, not a regression."""
    fen = "8/8/8/8/8/8/8/4K3[z9:k] w - - 0 1"
    with pytest.raises(IndexError):
        encoding._encode_position_py(fen)
    # Rust does not raise; it silently ignores the off-board entry.
    rust = encoding.encode_position(fen)
    # Result equals the same board with NO overlay (the bad entry dropped).
    assert torch.equal(rust, encoding._encode_position_py("8/8/8/8/8/8/8/4K3 w - - 0 1"))


def test_overlong_rank_is_unreachable_and_rust_skips() -> None:
    """A board rank with >8 files (10 pawns) is never produced by python-chess.
    Python raises IndexError on the 9th file write; Rust drops the overflow."""
    fen = "pppppppppp/8/8/8/8/8/8/8 b - - 0 1"
    with pytest.raises(IndexError):
        encoding._encode_position_py(fen)
    encoding.encode_position(fen)  # Rust returns a tensor without raising


def test_malformed_uci_square_is_unreachable() -> None:
    """A move with an off-board UCI square (z9) can't come from the engine.
    Python's square_index writes a wrong/out-of-range bit; Rust leaves the
    one-hot unset. Documented difference."""
    mv = {"from": "z9", "to": "h8"}
    rust = encoding.encode_move(mv)
    py = encoding._encode_move_py(mv)
    assert not torch.equal(rust, py)


def test_reachable_overlay_squares_never_offboard() -> None:
    """Guard the unreachability assumption: the engine serializes overlay
    squares via chess.square_name -> always a1..h8, files a-h, ranks 1-8."""
    import chess

    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        assert name[0] in "abcdefgh" and name[1] in "12345678"


def test_encode_position_state_matches_python():
    """Per-leaf hot-path encoder (from State piece bitboards) must equal the
    Python reference AND the FEN encoder for the same position."""
    c = PyVariantClient()
    fens = CURRICULUM_SEEDS + [
        "8/3kk3/8/8/8/8/8/4K3[d7:kk,e7:kk] b - - 0 1",
        "3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1",
        "7k/8/8/8/8/8/8/4K3[h8:kkkk] b - - 0 1",
        "7p/8/8/8/8/8/8/4K3[h8:ssss] b - - 0 1",
        "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR[a6:s,b6:S,c6:k] w KQkq - 0 1",
        "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR[a6:ssk,b6:kS,c6:kkk,h8:kkkk] w KQkq - 0 1",
    ]
    for fen in fens:
        st = c.parse(fen)
        r = encoding.encode_position_state(st)
        assert torch.equal(r, encoding._encode_position_state_py(st)), f"state-enc {fen}"
        assert torch.equal(r, encoding._encode_position_py(fen)), f"state-vs-fen {fen}"
