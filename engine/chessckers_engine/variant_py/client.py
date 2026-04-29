"""`PyVariantClient` — drop-in replacement for `ServerClient`.

Surface mirrors `chessckers_engine.server_client.ServerClient`: every public
method takes the same arguments and returns the same JSON-shaped dicts that
scalachess returns over HTTP. This lets the engine swap between scalachess
(via `ServerClient`) and pure-Python (via `PyVariantClient`) without any
caller changes.

All methods raise NotImplementedError until the corresponding piece of the
variant is ported. The differential test harness in
`tests/test_pyvariant_diff.py` exercises each method against scalachess on
identical FENs and asserts identical outputs — so we'll know immediately
when we get something wrong.
"""
from __future__ import annotations

from typing import Any

GameState = dict[str, Any]
HopDTO = dict[str, Any]
ChainStepResponse = dict[str, Any]


class PyVariantClient:
    """Same surface as `ServerClient` but evaluates positions in-process.

    Constructor takes optional kwargs for parity with `ServerClient` so the
    same construction sites work; the `base_url`/`timeout` args are accepted
    and ignored."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        # Accepted for ServerClient parity; this implementation has no
        # network surface.
        del base_url, timeout

    def close(self) -> None:
        pass

    def __enter__(self) -> "PyVariantClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ----- API methods (stubbed) -----

    def new_game(self, fen: str | None = None) -> GameState:
        raise NotImplementedError("PyVariantClient.new_game: not yet ported")

    def make_move(self, fen: str, uci: str) -> GameState:
        raise NotImplementedError("PyVariantClient.make_move: not yet ported")

    def moves_at(self, fen: str, square: str) -> list[dict[str, Any]]:
        raise NotImplementedError("PyVariantClient.moves_at: not yet ported")

    def chain_step(
        self, fen: str, chain_start: str, hops_so_far: list[HopDTO]
    ) -> ChainStepResponse:
        raise NotImplementedError("PyVariantClient.chain_step: not yet ported")

    def chain_end(
        self, fen: str, chain_start: str, hops_so_far: list[HopDTO]
    ) -> GameState:
        raise NotImplementedError("PyVariantClient.chain_end: not yet ported")
