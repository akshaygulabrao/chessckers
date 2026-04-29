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
    legalMoves lists. Sort by uci. Drop fields scalachess includes that we
    don't yet emit."""
    keep = ("uci", "from", "to", "piece", "color", "capture",
            "waypoints", "chainHops", "promotion", "demotedKings",
            "demotionsRequired", "sourceKingPositions", "deployCount")
    rows = [tuple((k, m.get(k)) for k in keep) for m in moves]
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


def test_stubs_raise_not_implemented():
    """Every API method should currently raise NotImplementedError. As we
    port each one, the corresponding test below should start passing and
    these stub assertions should be removed."""
    client = PyVariantClient()
    with pytest.raises(NotImplementedError):
        client.new_game()
    with pytest.raises(NotImplementedError):
        client.make_move("fen", "e2e4")
    with pytest.raises(NotImplementedError):
        client.moves_at("fen", "e2")


# ---- placeholders for upcoming differential tests ----
# Uncomment / un-skip as each piece lands.

@pytest.mark.skip(reason="white-side move-gen not yet ported")
def test_diff_starting_position_white_moves(scalachess):
    py = PyVariantClient()
    START_FEN = (
        "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
        "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
        "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
        "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
    )
    assert_legal_moves_match(py, scalachess, START_FEN)


@pytest.mark.skip(reason="black-side move-gen not yet ported")
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
