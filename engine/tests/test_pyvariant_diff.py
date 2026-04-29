"""Differential tests: compare `PyVariantClient` against scalachess on the
same FENs. Every Black-side feature (chain capture, mandatory rule, suicide,
deploy, demote, T-boundary) gets a paired test here as it lands in the port.

These tests are skipped when scalachess isn't reachable on localhost:8080,
so the suite still runs in CI even without the JVM.
"""
from __future__ import annotations

from typing import Any

import pytest

from chessckers_engine.server_client import ServerClient
from chessckers_engine.variant_py import PyVariantClient


@pytest.fixture(scope="module")
def scalachess() -> ServerClient:
    """A live ServerClient if scalachess is up; otherwise skip the tests
    that depend on it. Tests that don't need scalachess can ignore this
    fixture."""
    client = ServerClient()
    try:
        client.new_game()
    except Exception:  # noqa: BLE001
        pytest.skip("scalachess not reachable on localhost:8080")
    yield client
    client.close()


# ----- helpers -----


def _normalize_legal_moves(moves: list[dict[str, Any]]) -> list[tuple]:
    """Order-independent canonical form so we can compare two engines'
    legalMoves lists. Sort by uci. Coerce list fields to tuples so the
    rows are hashable (needed for set-diff in failure messages)."""
    keep = ("uci", "from", "to", "piece", "color", "capture",
            "waypoints", "chainHops", "promotion", "demotedKings",
            "demotionsRequired", "sourceKingPositions", "deployCount")

    def _coerce(v):
        return tuple(v) if isinstance(v, list) else v

    rows = [tuple((k, _coerce(m.get(k))) for k in keep) for m in moves]
    return sorted(rows)


def assert_legal_moves_match(py: PyVariantClient, scala: ServerClient, fen: str) -> None:
    py_state = py.new_game(fen)
    scala_state = scala.new_game(fen)
    py_moves = _normalize_legal_moves(py_state.get("legalMoves") or [])
    scala_moves = _normalize_legal_moves(scala_state.get("legalMoves") or [])
    assert py_moves == scala_moves, (
        f"legalMoves diverge for FEN {fen!r}\n"
        f"  py only:    {set(py_moves) - set(scala_moves)}\n"
        f"  scala only: {set(scala_moves) - set(py_moves)}"
    )


def assert_make_move_matches(
    py: PyVariantClient, scala: ServerClient, fen: str, uci: str
) -> None:
    py_after = py.make_move(fen, uci)
    scala_after = scala.make_move(fen, uci)
    for k in ("fen", "turn", "status", "winner", "check"):
        assert py_after.get(k) == scala_after.get(k), (
            f"{k} diverges for FEN={fen!r} uci={uci!r}: "
            f"py={py_after.get(k)!r} scala={scala_after.get(k)!r}"
        )


# ----- tests -----


def test_pyvariant_constructs():
    """Sanity check that the package imports and instantiates."""
    client = PyVariantClient()
    client.close()


def test_unported_stubs_raise_not_implemented():
    """Black-side functionality and chain-stepping are not yet ported.
    Removed entries here as each piece lands."""
    client = PyVariantClient()
    with pytest.raises(NotImplementedError):
        client.chain_step("fen", "f6", [])
    with pytest.raises(NotImplementedError):
        client.chain_end("fen", "f6", [])


# ---- new_game parity ----

INITIAL_FEN_KQ = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQ - 0 1"
)


def test_new_game_default_returns_starting_position():
    state = PyVariantClient().new_game()
    assert state["turn"] == "white"
    assert state["check"] is False
    assert state["status"] is None
    assert state["winner"] is None
    assert "RNBQKBNR" in state["fen"]
    # 24 stacks × ~2 chars each, so the bracket overlay should have a healthy size.
    assert state["fen"].count(":") == 24


def test_new_game_with_fen_echoes_input():
    """For non-default input FENs, the returned `fen` should be the input
    verbatim (matches scalachess's parse-time behavior)."""
    state = PyVariantClient().new_game(INITIAL_FEN_KQ)
    assert state["fen"] == INITIAL_FEN_KQ


def test_new_game_diff_against_scala_white_to_move(scalachess):
    """new_game on a White-to-move position should match scalachess on
    fen/turn/check/status/winner."""
    py = PyVariantClient()
    fen = INITIAL_FEN_KQ
    py_state = py.new_game(fen)
    sc_state = scalachess.new_game(fen)
    for k in ("fen", "turn", "check", "status", "winner"):
        assert py_state.get(k) == sc_state.get(k), (
            f"{k} diverges: py={py_state.get(k)!r} scala={sc_state.get(k)!r}"
        )


def test_new_game_diff_against_scala_black_to_move(scalachess):
    """Same parity check on a Black-to-move position."""
    py = PyVariantClient()
    fen = (
        "pppppppp/kkkkkkkk/pppppppp/8/8/P7/1PPPPPPP/RNBQKBNR"
        "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
        "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
        "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] b KQ - 0 1"
    )
    py_state = py.new_game(fen)
    sc_state = scalachess.new_game(fen)
    for k in ("fen", "turn", "check", "status", "winner"):
        assert py_state.get(k) == sc_state.get(k), (
            f"{k} diverges: py={py_state.get(k)!r} scala={sc_state.get(k)!r}"
        )


# ---- placeholders for upcoming differential tests ----
# Uncomment / un-skip as each piece lands.

def test_diff_starting_position_white_moves(scalachess):
    py = PyVariantClient()
    START_FEN = (
        "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
        "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
        "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
        "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
    )
    assert_legal_moves_match(py, scalachess, START_FEN)


@pytest.mark.parametrize("fen", [
    # Pawn captures available against a Black "stack" on rank 6.
    "8/8/p7/1P6/8/8/8/4K3[a6:s] w - - 0 1",
    # Castling-eligible: White king + both rooks on home squares with clear lanes.
    "8/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    # Pawn ready to promote on rank 7 (will produce 4 promotion moves).
    "8/3P4/8/8/8/8/8/4K3 w - - 0 1",
    # Mid-game position from saved games — captures + non-captures mixed.
    "ppp1p1pp/k1kk1k2/kpppppkp/8/8/PPPPPP2/R5PP/1NBQKBNR"
    "[a6:sk,b6:s,c6:s,d6:skS,e6:s,f6:skS,g6:sk,h6:s,a7:k,c7:k,d7:k,f7:k,a8:s,b8:s,c8:s,e8:s,g8:s,h8:s] w K - 0 1",
])
def test_diff_white_moves_various_positions(scalachess, fen):
    """python-chess vs scalachess on a variety of White-to-move positions
    (captures, castling, promotions, mid-game)."""
    assert_legal_moves_match(PyVariantClient(), scalachess, fen)


@pytest.mark.parametrize("fen,uci", [
    # Simple opening pawn move (no capture).
    (
        "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
        "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
        "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
        "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1",
        "e2e4",
    ),
    # White pawn captures Black "stack" on diagonal — captured square's
    # entire stack should be removed from the overlay.
    ("8/8/p7/1P6/8/8/8/4K3[a6:s] w - - 0 1", "b5a6"),
    # Kingside castling, standard UCI.
    ("8/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1"),
    # Kingside castling, king-to-rook UCI (Chess960-style; PyVariantClient
    # should translate this to standard before applying).
    ("8/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1h1"),
    # Pawn promotion to queen.
    ("8/3P4/8/8/8/8/8/4K3 w - - 0 1", "d7d8q"),
])
def test_diff_make_move_white(scalachess, fen, uci):
    """White make_move output should match scalachess on fen/turn/check/status/winner."""
    py = PyVariantClient()
    py_after = py.make_move(fen, uci)
    sc_after = scalachess.make_move(fen, uci)
    for k in ("fen", "turn", "check", "status", "winner"):
        assert py_after.get(k) == sc_after.get(k), (
            f"{k} diverges for fen={fen!r} uci={uci!r}: "
            f"py={py_after.get(k)!r} scala={sc_after.get(k)!r}"
        )


def test_diff_make_move_black_elimination(scalachess):
    """Capturing the last Black stack should yield variantEnd / winner=white."""
    py = PyVariantClient()
    fen = "8/8/p7/1P6/8/8/8/4K3[a6:s] w - - 0 1"
    py_after = py.make_move(fen, "b5a6")
    sc_after = scalachess.make_move(fen, "b5a6")
    assert py_after["status"] == "variantEnd"
    assert py_after["winner"] == "white"
    assert py_after["status"] == sc_after["status"]
    assert py_after["winner"] == sc_after["winner"]


@pytest.mark.parametrize("fen", [
    # Lone Stone-top tower at e6 → 2 forward diagonals (d5, f5).
    "8/8/4p3/8/8/8/8/4K3[e6:s] b - - 0 1",
    # Stone-top at corner a6 → only forward-right (b5).
    "8/8/p7/8/8/8/8/4K3[a6:s] b - - 0 1",
    # Stone-top at corner h6 → only forward-left (g5).
    "8/8/7p/8/8/8/8/4K3[h6:s] b - - 0 1",
    # Two Stone-tops far from White: e6 and d5. e6 can merge onto d5
    # (e6→d5), or move to f5; d5 has c4/e4 forward.
    "8/8/4p3/3p4/8/8/8/4K3[d5:s,e6:s] b - - 0 1",
])
def test_diff_black_diagonal_quiet_stone_top(scalachess, fen):
    """Phase 2A — height-1 Stone-top diagonals only. Positions are
    constructed so scalachess emits exclusively diagonal-quiet moves
    (no captures available, no rank-8 sprint, height=1 so no deploy,
    no King so no charge)."""
    assert_legal_moves_match(PyVariantClient(), scalachess, fen)


@pytest.mark.parametrize("fen", [
    # Height-2 Stone-top tower at d6 (sS = unmoved-bottom, moved-top stone).
    # Should produce 4 full-tower diag moves (range 2 forward) + 2 deploys
    # (s=1, range 1 forward).
    "8/8/3p4/8/8/8/8/4K3[d6:sS] b - - 0 1",
    # Height-3 Stone-top in middle — 6 diag moves (range 3) × forward only,
    # plus deploys with s=1 and s=2.
    "8/8/3p4/8/8/8/8/4K3[d6:ssS] b - - 0 1",
])
def test_diff_black_quiet_plus_deploy_stone_top(scalachess, fen):
    """Phase 2B — height ≥ 2 Stone-top stacks emit both full-tower diagonal
    quiets and deploy sub-moves. No captures available, no mandate, no
    sprint, no charge (Stone-top can't charge)."""
    assert_legal_moves_match(PyVariantClient(), scalachess, fen)


@pytest.mark.parametrize("fen", [
    # Lone unmoved Stone at e8 → 2 normal diags (d7, f7) + 2 sprints (c6, g6).
    "4p3/8/8/8/8/8/8/4K3[e8:s] b - - 0 1",
    # MOVED Stone at e8 → no sprint, just 2 diag moves.
    "4p3/8/8/8/8/8/8/4K3[e8:S] b - - 0 1",
    # Sprint at corner a8: only one forward diagonal exists (b7), so only one
    # normal diag and one sprint (b7, c6).
    "p7/8/8/8/8/8/8/4K3[a8:s] b - - 0 1",
    # Sprint blocked: friendly tower on intervening square (d7) blocks e8 sprint to c6.
    "4p3/3p4/8/8/8/8/8/4K3[d7:s,e8:s] b - - 0 1",
])
def test_diff_black_back_rank_sprint(scalachess, fen):
    """Phase 2C — back-rank sprint. Height-1 unmoved Stone-top on rank 8
    can move 2 squares forward-diagonal when path is clear."""
    assert_legal_moves_match(PyVariantClient(), scalachess, fen)


@pytest.mark.parametrize("fen", [
    # Lone height-1 King-top: 4 diags + 4 charges (1-square each, all demote
    # the only King, demotion fields null).
    "8/8/4p3/8/8/8/8/4K3[e6:k] b - - 0 1",
    # Height-2 King-top: 8 diags + 4 deploys + 12 charges.
    "8/8/4p3/8/8/8/8/4K3[e6:kk] b - - 0 1",
    # Height-2 King-top with a path-blocking friendly tower.
    "8/8/4p3/8/4p3/8/8/4K3[e6:kk,e4:k] b - - 0 1",
    # Charge with path captures + ram landing combo.
    "8/8/4p3/8/4P3/8/4P3/4K3[e6:kkk] b - - 0 1",
])
def test_diff_black_charge(scalachess, fen):
    """Phase 2E — King-top charges: distance, demotion choice, path
    captures, friendly merge, rams."""
    assert_legal_moves_match(PyVariantClient(), scalachess, fen)


@pytest.mark.skip(reason="King-top emits charges too — handled in 2E/2F")
def test_diff_starting_position_black_moves(scalachess):
    py = PyVariantClient()
    BLACK_TO_MOVE = (
        "pppppppp/kkkkkkkk/pppppppp/8/8/P7/1PPPPPPP/RNBQKBNR"
        "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
        "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
        "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] b KQ - 0 1"
    )
    assert_legal_moves_match(py, scalachess, BLACK_TO_MOVE)


@pytest.mark.skip(reason="black king-capture path not yet ported")
def test_diff_black_king_capture(scalachess):
    """The classic regression — Black jumps over white king and captures."""
    py = PyVariantClient()
    KING_CAPTURE_FEN = "8/8/8/1p6/2K5/8/8/8[b5:s] b - - 0 1"
    assert_legal_moves_match(py, scalachess, KING_CAPTURE_FEN)
    assert_make_move_matches(py, scalachess, KING_CAPTURE_FEN, "b5d3")
