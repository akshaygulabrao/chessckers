"""`PyVariantClient` — drop-in replacement for `ServerClient`.

Surface mirrors `chessckers_engine.server_client.ServerClient`: every public
method takes the same arguments and returns the same JSON-shaped dicts that
scalachess returns over HTTP. This lets the engine swap between scalachess
(via `ServerClient`) and pure-Python (via `PyVariantClient`) without any
caller changes.

Implementation is incremental:
- new_game(fen) returns the right shape (fen/turn/check/status/winner) but
  with an empty `legalMoves` list until the move generators land.
- make_move / moves_at / chain_step / chain_end still raise NotImplementedError.

The differential test harness in `tests/test_pyvariant_diff.py` exercises
each method against scalachess on identical FENs and asserts identical
outputs.
"""
from __future__ import annotations

from typing import Any

import chess

from chessckers_engine.variant_py.moves_black import (
    black_charge_moves,
    black_deploy_moves,
    black_diagonal_quiet_moves,
)
from chessckers_engine.variant_py.moves_white import (
    apply_white_move,
    white_legal_moves,
)
from chessckers_engine.variant_py.state import STARTING_FEN, State, parse_fen, serialize_fen

GameState = dict[str, Any]
HopDTO = dict[str, Any]
ChainStepResponse = dict[str, Any]


def _detect_status(state: State) -> tuple[str | None, str | None]:
    """Return (status, winner) for the given state. Currently only handles
    the Black-elimination → variantEnd/winner=white path. mate / stalemate
    / Black-king-capture detection lands when Black move-gen is ported."""
    if not state.stacks:
        return ("variantEnd", "white")
    return (None, None)


def _state_to_dict(state: State, fen_override: str | None = None) -> GameState:
    """Render a State to the dict shape scalachess returns. `fen_override`
    is used by new_game to echo the input FEN verbatim (preserving e.g. the
    `KQkq` quirk on the initial position); for post-move outputs we let
    python-chess canonicalize."""
    fen = fen_override if fen_override is not None else serialize_fen(state)
    turn = "white" if state.board.turn == chess.WHITE else "black"
    try:
        check = bool(state.board.is_check())
    except Exception:  # noqa: BLE001
        # python-chess can throw on positions without a king of one color;
        # Black-to-move positions in Chessckers don't have a chess king at
        # all, so check is meaningless there.
        check = False
    status, winner = _detect_status(state)
    if state.board.turn == chess.WHITE:
        legal_moves = white_legal_moves(state)
    else:
        # Black: incremental — quiet diagonals + deploys + sprint + charges.
        # Diagonal capture chains (Phase 2D) and mandate filter (2F) pending.
        legal_moves = (
            black_diagonal_quiet_moves(state)
            + black_deploy_moves(state)
            + black_charge_moves(state)
        )
    return {
        "fen": fen,
        "turn": turn,
        "check": check,
        "status": status,
        "winner": winner,
        "legalMoves": legal_moves,
    }


class PyVariantClient:
    """Same surface as `ServerClient` but evaluates positions in-process.

    Constructor takes optional kwargs for parity with `ServerClient` so the
    same construction sites work; `base_url`/`timeout` are accepted and
    ignored."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        del base_url, timeout

    def close(self) -> None:
        pass

    def __enter__(self) -> "PyVariantClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ----- API methods -----

    def new_game(self, fen: str | None = None) -> GameState:
        """Same as scalachess `/api/game/new`. With no `fen`, returns the
        canonical Chessckers starting position. With a `fen`, echoes it
        verbatim (matching scalachess's parse-time behavior — useful for
        positions where canonicalization would diverge, e.g. `KQkq` →
        `KQ`)."""
        input_fen = fen if fen is not None else STARTING_FEN
        state = parse_fen(input_fen)
        return _state_to_dict(state, fen_override=input_fen)

    def make_move(self, fen: str, uci: str) -> GameState:
        """Apply a UCI move to the position. Currently White-only — Black
        moves will land with the Black move-gen port."""
        state = parse_fen(fen)
        if state.board.turn == chess.WHITE:
            new_state = apply_white_move(state, uci)
        else:
            raise NotImplementedError(
                "PyVariantClient.make_move: Black-side moves not yet ported"
            )
        return _state_to_dict(new_state)

    def moves_at(self, fen: str, square: str) -> list[dict[str, Any]]:
        """Legal moves originating from `square` (UI helper). White-only
        until Black move-gen lands."""
        state = parse_fen(fen)
        if state.board.turn == chess.WHITE:
            return [m for m in white_legal_moves(state) if m["from"] == square]
        raise NotImplementedError(
            "PyVariantClient.moves_at: Black-side move-gen not yet ported"
        )

    def chain_step(
        self, fen: str, chain_start: str, hops_so_far: list[HopDTO]
    ) -> ChainStepResponse:
        raise NotImplementedError("PyVariantClient.chain_step: not yet ported")

    def chain_end(
        self, fen: str, chain_start: str, hops_so_far: list[HopDTO]
    ) -> GameState:
        raise NotImplementedError("PyVariantClient.chain_end: not yet ported")
