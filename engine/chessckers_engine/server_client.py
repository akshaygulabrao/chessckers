from typing import Any

import httpx

GameState = dict[str, Any]
HopDTO = dict[str, Any]
ChainStepResponse = dict[str, Any]


class ServerClient:
    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 5.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ServerClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def new_game(self, fen: str | None = None) -> GameState:
        body: dict[str, Any] = {"fen": fen} if fen is not None else {}
        return self._post("/api/game/new", body)

    def make_move(self, fen: str, uci: str) -> GameState:
        return self._post("/api/game/move", {"fen": fen, "uci": uci})

    def moves_at(self, fen: str, square: str) -> list[dict[str, Any]]:
        resp = self._post("/api/game/moves-at", {"fen": fen, "square": square})
        return resp["moves"]

    def chain_step(
        self, fen: str, chain_start: str, hops_so_far: list[HopDTO]
    ) -> ChainStepResponse:
        return self._post(
            "/api/game/chain-step",
            {"fen": fen, "chainStart": chain_start, "hopsSoFar": hops_so_far},
        )

    def chain_end(
        self, fen: str, chain_start: str, hops_so_far: list[HopDTO]
    ) -> GameState:
        return self._post(
            "/api/game/chain-end",
            {"fen": fen, "chainStart": chain_start, "hopsSoFar": hops_so_far},
        )

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        r = self._client.post(path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"{path} -> {r.status_code}: {r.text}")
        return r.json()
