"""Smoke test against a live server. Skipped if server isn't running."""

import httpx
import pytest

from chessckers_engine import ServerClient


def _server_up() -> bool:
    try:
        httpx.post("http://localhost:8080/api/game/new", json={}, timeout=1.0)
        return True
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason="server not running on :8080")


def test_new_game_returns_state() -> None:
    with ServerClient() as c:
        state = c.new_game()
    assert "fen" in state
    assert "legalMoves" in state
    assert "stacks" in state


def test_moves_at_returns_list() -> None:
    with ServerClient() as c:
        state = c.new_game()
        moves = c.moves_at(state["fen"], "e2")
    assert isinstance(moves, list)
