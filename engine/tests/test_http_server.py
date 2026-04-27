import threading
from typing import Any

import httpx

from chessckers_engine.http_server import make_server


class FakeClient:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self.calls: list[str] = []

    def new_game(self, fen: str | None = None) -> dict[str, Any]:
        self.calls.append(fen or "")
        return self._state


def _serve(state: dict[str, Any]) -> tuple[str, FakeClient, threading.Thread, Any]:
    client = FakeClient(state)
    server = make_server("127.0.0.1", 0, client)  # type: ignore[arg-type]
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}", client, thread, server


def test_move_picks_from_legal_moves() -> None:
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
