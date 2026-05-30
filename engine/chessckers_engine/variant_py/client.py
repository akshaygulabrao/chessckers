"""`PyVariantClient` — the engine's in-process game API.

Every method returns the JSON-shaped dicts the rest of the engine is written
against (`fen`/`turn`/`check`/`status`/`winner`/`legalMoves`/`stacks`); that
dict shape is the stable contract for MCTS, self-play, and rendering. The
engine is stateless — each call carries the full FEN.
"""
from __future__ import annotations

import os
from typing import Any

import chess

from chessckers_engine.variant_py.moves_black import (
    apply_black_move,
    apply_black_move_known,
    black_charge_moves,
    black_deploy_moves,
    black_diagonal_capture_moves,
    black_diagonal_quiet_moves,
    filter_for_mandate,
)
from chessckers_engine.variant_py.moves_white import (
    apply_white_move,
    white_legal_moves,
)

try:
    import chessckers_movegen as _rs_movegen  # type: ignore[import-not-found]
except ImportError:
    _rs_movegen = None
# See moves_black.py: bypass the native extension during the §3B rule work.
if __import__("os").environ.get("CHESSCKERS_NO_RUST"):
    _rs_movegen = None
from chessckers_engine.variant_py.state import STARTING_FEN, State, parse_fen, serialize_fen


def _default_start_fen() -> str:
    """Start FEN used by `new_game()` when no `fen` is passed. Overridable via
    the `CHESSCKERS_START_FEN` env var so experiments can pin a custom start
    position (e.g. an endgame) without threading a start_fen through every call
    site (self-play, eval, workers all go through `new_game()`)."""
    return os.environ.get("CHESSCKERS_START_FEN", STARTING_FEN)

GameState = dict[str, Any]
HopDTO = dict[str, Any]
ChainStepResponse = dict[str, Any]


def _detect_status(state: State) -> tuple[str | None, str | None]:
    """Return (status, winner) for the given state.

    Detection paths handled here (cheap, no move-gen):
    - Black eliminated → variantEnd / winner=white.
    - White's king captured during a Black chain → variantEnd / winner=black.
    White checkmate/stalemate and Black stalemate need the full legal-move
    list, so they're detected in _state_to_dict after move generation. (Mate
    uses the Chessckers check predicate, NOT python-chess's FIDE is_checkmate,
    which wrongly treats an adjacent Black King as giving check.)"""
    if not state.stacks:
        return ("variantEnd", "white")
    if state.board.king(chess.WHITE) is None:
        return ("variantEnd", "black")
    return (None, None)


def _state_to_dict(state: State, fen_override: str | None = None) -> GameState:
    """Render a State to the dict shape scalachess returns. `fen_override`
    is used by new_game to echo the input FEN verbatim (preserving e.g. the
    `KQkq` quirk on the initial position); for post-move outputs we let
    python-chess canonicalize."""
    fen = fen_override if fen_override is not None else serialize_fen(state)
    turn = "white" if state.board.turn == chess.WHITE else "black"
    # `check` = is the side-to-move in check. White-to-move uses the Chessckers
    # check (NOT python-chess is_check(), which can't see Black's diagonal/chain/
    # charge king-captures). Black-to-move has no chess king to be checked.
    if state.board.turn == chess.WHITE:
        from chessckers_engine.variant_py.moves_white import _is_white_in_chessckers_check
        check = _is_white_in_chessckers_check(state)
    else:
        check = False
    status, winner = _detect_status(state)
    if state.board.turn == chess.WHITE:
        legal_moves = white_legal_moves(state)
        # White checkmate / stalemate (Chessckers, not FIDE): no legal move →
        # Black wins. "mate" if the king is in Chessckers-check, else White is
        # simply stuck — being stuck loses, mirroring the Black rule below.
        if not legal_moves and status is None:
            status, winner = ("mate" if check else "variantEnd"), "black"
    else:
        all_moves = (
            black_diagonal_quiet_moves(state)
            + black_deploy_moves(state)
            + black_charge_moves(state)
            + black_diagonal_capture_moves(state)
        )
        legal_moves = filter_for_mandate(state, all_moves)
        # Black stalemate: no legal moves and no other terminal condition.
        # scalachess fires specialEnd → variantEnd/white in this case.
        if not legal_moves and status is None:
            status, winner = "variantEnd", "white"
    return {
        "fen": fen,
        "turn": turn,
        "check": check,
        "status": status,
        "winner": winner,
        "legalMoves": legal_moves,
    }


class PyVariantClient:
    """Evaluates positions in-process; the engine is stateless (each call
    carries the full FEN). `base_url`/`timeout` are vestigial kwargs, accepted
    and ignored so older construction sites still work."""

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
        input_fen = fen if fen is not None else _default_start_fen()
        state = parse_fen(input_fen)
        return _state_to_dict(state, fen_override=input_fen)

    def make_move(self, fen: str, uci: str) -> GameState:
        """Apply a UCI move to the position."""
        state = parse_fen(fen)
        if state.board.turn == chess.WHITE:
            new_state = apply_white_move(state, uci)
        else:
            new_state = apply_black_move(state, uci)
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

    # ----- Fast-path API for the MCTS hot loop -----
    #
    # The dict-based API above forces parse/serialize round-trips on every
    # call: `make_move(fen, uci)` parses the FEN, applies the move, then
    # serializes the result and ALSO runs a full Black move-gen pass to fill
    # `legalMoves` in the returned dict — even though MCTS never reads that
    # field (it has its own legal-move cache). The methods below let MCTS
    # parse a position once, then apply known-legal moves to the in-memory
    # State directly. Profiling showed this path is ~50% of pre-fix MCTS time.

    def parse(self, fen: str) -> State:
        """Parse a FEN once; return the in-memory State for downstream
        `apply_known` calls. Cheaper than going through `new_game(fen)` since
        it skips status detection + legal-move enumeration."""
        return parse_fen(fen)

    def state_to_fen(self, state: State) -> str:
        return serialize_fen(state)

    def apply_known(self, state: State, move: dict[str, Any]) -> State:
        """Apply a move dict that the caller knows is legal in `state`. Skips
        the redundant move-gen-for-validation pass that `make_move(fen, uci)`
        runs inside `apply_black_move` to look up the UCI."""
        if state.board.turn == chess.WHITE:
            return apply_white_move(state, move["uci"])
        return apply_black_move_known(state, move)

    def status_and_legal(
        self, state: State
    ) -> tuple[str | None, str | None, list[dict[str, Any]] | None]:
        """Detect status; if non-terminal-by-cheap-checks, also return the
        legal-move list (which the caller should cache for future MCTS lookups
        on this state). When `status` is set via a cheap check (no stacks,
        no king, white checkmate), `legal_moves` is None — those positions
        are terminal and never need expansion.

        (No transposition cache here: an instrumented run showed only ~0.6%
        hit rate, since MCTS already caches the legal-moves list on each
        PuctNode — true cross-subtree transpositions are rare at our depth
        and branching factor, so a per-call cache costs more in key-building
        than it saves.)"""
        if not state.stacks:
            return ("variantEnd", "white", None)
        if state.board.king(chess.WHITE) is None:
            return ("variantEnd", "black", None)
        if state.board.turn == chess.WHITE:
            # Don't use python-chess's `is_checkmate` — it treats Black-King
            # bitboard entries (= Chessckers king-top stacks) as standard
            # 8-direction chess kings, which over-reports check and can
            # turn a non-mate into a false-mate (see
            # tests/test_screenshot_false_mate.py). Instead: compute white
            # legal moves under Chessckers attack rules; mate iff zero legal
            # moves AND king is in Chessckers-check.
            from chessckers_engine.variant_py.moves_white import (
                _is_white_in_chessckers_check,
            )
            moves = white_legal_moves(state)
            if not moves:
                if _is_white_in_chessckers_check(state):
                    return ("mate", "black", None)
                # No legal moves and not in check → stalemate (chess rule;
                # scalachess returns "stalemate" / draw).
                return ("stalemate", None, None)
            return (None, None, moves)
        # Black to move.
        if _rs_movegen is not None:
            # Native fast path: one Rust call returns the post-mandate-filter
            # legal-move list. ~75% of the pre-Rust `status_and_legal` cost
            # collapses into this single call.
            wk = state.board.king(chess.WHITE)
            legal_moves = _rs_movegen.all_black_legal_moves(
                state.board.occupied,
                state.board.occupied_co[chess.WHITE],
                -1 if wk is None else wk,
                state.stacks,
            )
        else:
            all_moves = (
                black_diagonal_quiet_moves(state)
                + black_deploy_moves(state)
                + black_charge_moves(state)
                + black_diagonal_capture_moves(state)
            )
            legal_moves = filter_for_mandate(state, all_moves)
        if not legal_moves:
            # Black stalemate → variantEnd / white wins (matches scalachess).
            return ("variantEnd", "white", legal_moves)
        return (None, None, legal_moves)

    def state_check(self, state: State) -> bool:
        try:
            return bool(state.board.is_check())
        except Exception:  # noqa: BLE001
            return False
