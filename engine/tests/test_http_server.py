import threading
from typing import Any

import httpx

from chessckers_engine.http_server import Picker, make_server
from chessckers_engine.random_player import pick_random


class FakeClient:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self.calls: list[str] = []

    def new_game(self, fen: str | None = None) -> dict[str, Any]:
        self.calls.append(fen or "")
        return self._state


def _random_picker(state: dict[str, Any]) -> dict[str, Any] | None:
    return pick_random(state.get("legalMoves") or [])


def _serve(
    state: dict[str, Any],
    pickers: dict[str, Picker] | None = None,
    default: str = "random",
) -> tuple[str, FakeClient, threading.Thread, Any]:
    client = FakeClient(state)
    server = make_server(
        "127.0.0.1",
        0,
        client,  # type: ignore[arg-type]
        pickers if pickers is not None else {"random": _random_picker},
        default_picker=default,
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}", client, thread, server


def test_move_uses_default_picker_when_request_has_no_picker_field() -> None:
    state = {"legalMoves": [{"uci": "f6g5"}, {"uci": "f6h4"}]}
    url, client, _t, server = _serve(state)
    try:
        r = httpx.post(f"{url}/move", json={"fen": "FEN"}, timeout=2.0)
        assert r.status_code == 200
        assert r.json()["uci"] in {"f6g5", "f6h4"}
        assert client.calls == ["FEN"]
    finally:
        server.shutdown()


def test_move_returns_null_when_no_legal_moves() -> None:
    url, _c, _t, server = _serve({"legalMoves": []})
    try:
        r = httpx.post(f"{url}/move", json={"fen": "FEN"}, timeout=2.0)
        assert r.json() == {"uci": None}
    finally:
        server.shutdown()


def test_missing_fen_400() -> None:
    url, _c, _t, server = _serve({"legalMoves": []})
    try:
        r = httpx.post(f"{url}/move", json={}, timeout=2.0)
        assert r.status_code == 400
    finally:
        server.shutdown()


def test_options_preflight() -> None:
    url, _c, _t, server = _serve({"legalMoves": []})
    try:
        r = httpx.options(f"{url}/move", timeout=2.0)
        assert r.status_code == 204
        assert r.headers["access-control-allow-origin"] == "*"
    finally:
        server.shutdown()


def test_picker_field_routes_to_named_picker() -> None:
    state = {"fen": "FEN", "legalMoves": [{"uci": "e2e4"}, {"uci": "d2d4"}]}

    def picker_a(_s):
        return {"uci": "FROM_A"}

    def picker_b(_s):
        return {"uci": "FROM_B"}

    url, _c, _t, server = _serve(state, pickers={"a": picker_a, "b": picker_b}, default="a")
    try:
        r = httpx.post(f"{url}/move", json={"fen": "FEN", "picker": "b"}, timeout=2.0)
        assert r.json() == {"uci": "FROM_B"}
        # Default kicks in when picker is omitted
        r = httpx.post(f"{url}/move", json={"fen": "FEN"}, timeout=2.0)
        assert r.json() == {"uci": "FROM_A"}
    finally:
        server.shutdown()


def test_unknown_picker_returns_400() -> None:
    state = {"fen": "FEN", "legalMoves": [{"uci": "e2e4"}]}
    url, _c, _t, server = _serve(state, pickers={"random": _random_picker}, default="random")
    try:
        r = httpx.post(f"{url}/move", json={"fen": "FEN", "picker": "no-such"}, timeout=2.0)
        assert r.status_code == 400
        assert "no-such" in r.json()["error"]
    finally:
        server.shutdown()


def test_make_server_rejects_default_picker_not_in_pickers() -> None:
    import pytest

    client = FakeClient({"legalMoves": []})
    with pytest.raises(ValueError):
        make_server("127.0.0.1", 0, client, {"a": _random_picker}, default_picker="b")  # type: ignore[arg-type]
