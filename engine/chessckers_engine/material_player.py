"""1-ply material picker for the Chessckers engine.

For each legal move at the current position, asks the API to apply the move
and returns the FEN of the resulting position; scores that resulting FEN with
`material_for_side_to_move`; picks the move whose resulting FEN has the
highest score from the side that just moved (i.e. the *current* side to move
maximizes its own material, which is what we want).

`material_for_side_to_move(post_fen)` flips the sign based on whose turn it
*now* is, which is the opponent of the player we just had move. So we negate
that score to get the score from the perspective of the player who moved.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from chessckers_engine.material import material_for_side_to_move

log = logging.getLogger("chessckers_engine.material_player")


class _Mover(Protocol):
    """Subset of ServerClient's API that pick_material needs (testable)."""

    def make_move(self, fen: str, uci: str) -> dict[str, Any]: ...


GameState = dict[str, Any]
LegalMove = dict[str, Any]


def pick_material(state: GameState, client: _Mover) -> LegalMove | None:
    legal_moves = state.get("legalMoves") or []
    if not legal_moves:
        return None

    fen = state["fen"]
    best_score = None
    best_move: LegalMove | None = None
    for move in legal_moves:
        try:
            new_state = client.make_move(fen, move["uci"])
        except Exception as e:  # noqa: BLE001
            # Server can return moves in legalMoves that its own validator then
            # rejects (known scalachess bug around e.p.-shaped captures against
            # Chessckers towers). Skip those candidates rather than crash; if
            # every candidate is rejected, fall back to the first legal move
            # so the game can still progress.
            log.debug("skipping unreachable candidate uci=%s: %s", move["uci"], e)
            continue
        post_fen = new_state["fen"]
        # material_for_side_to_move(post_fen) is from the next-mover's perspective.
        # Negate to get the score from the perspective of the player who just moved.
        score = -material_for_side_to_move(post_fen)
        if best_score is None or score > best_score:
            best_score = score
            best_move = move
    return best_move if best_move is not None else legal_moves[0]
