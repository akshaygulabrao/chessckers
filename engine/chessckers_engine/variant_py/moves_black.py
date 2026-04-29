"""Black-side move generation. Implemented incrementally; each phase has
matching differential tests against scalachess in tests/test_pyvariant_diff.py.

Phase status:
- [x] 2A: Diagonal quiet moves (full-tower, no deploy, no capture)
- [x] 2B: Deploy moves
- [x] 2C: Back-rank sprint
- [ ] 2D: Diagonal capture chains
- [ ] 2E: Charge (orthogonal capture)
- [ ] 2F: Mandatory rule filter
- [ ] 2G: State transition (apply Black move)
"""

from __future__ import annotations

from typing import Any

import chess

from chessckers_engine.variant_py.state import State

# Diagonal direction vectors as (file_delta, rank_delta).
# Black "forward" is toward rank 1 → rank decreases.
_FORWARD_DIAGS = [(-1, -1), (1, -1)]
_BACKWARD_DIAGS = [(-1, 1), (1, 1)]
_ALL_DIAGS = _FORWARD_DIAGS + _BACKWARD_DIAGS


def _on_board(file: int, rank: int) -> bool:
    return 0 <= file <= 7 and 0 <= rank <= 7


def _piece_name(top_char: str) -> str:
    """Top-piece character → scalachess piece-name field for the move dict."""
    return "king" if top_char == "k" else "pawn"


def _is_black_top_at(state: State, sq: chess.Square) -> bool:
    """True iff `sq` has a Black tower (Stone-top or King-top)."""
    return sq in state.stacks


def black_diagonal_quiet_moves(state: State) -> list[dict[str, Any]]:
    """Phase 2A — full-tower diagonal moves with no captures and no deploys.

    For each Black tower of height n with top piece p:
      directions = forward-only if Stone-top, all four if King-top
      for each direction, walk 1..n squares; emit a move when the target
      square is empty or contains a friendly Black tower (merge). Stop
      scanning a direction when a non-empty square is reached (friendly
      → emit and stop; White → stop without emitting; off-board → stop).
    """
    if state.board.turn != chess.BLACK:
        return []

    moves: list[dict[str, Any]] = []
    for from_sq, pieces in state.stacks.items():
        if not pieces:
            continue
        height = len(pieces)
        top = pieces[-1]
        directions = _ALL_DIAGS if top == "k" else _FORWARD_DIAGS
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        from_name = chess.square_name(from_sq)

        for df, dr in directions:
            for k in range(1, height + 1):
                tf = from_file + k * df
                tr = from_rank + k * dr
                if not _on_board(tf, tr):
                    break
                to_sq = chess.square(tf, tr)
                target = state.board.piece_at(to_sq)
                if target is None:
                    moves.append(_quiet_move(from_name, to_sq, top))
                    continue  # empty square — keep scanning further along this diagonal
                # Non-empty square: classify and stop.
                if target.color == chess.BLACK and _is_black_top_at(state, to_sq):
                    # Friendly Black tower → merge is legal here, but blocks further walk.
                    moves.append(_quiet_move(from_name, to_sq, top))
                # White piece (or any non-friendly): stop, no emit (captures handled separately).
                break

        # Phase 2C: Back-rank sprint. A height-1 unmoved Stone-top at rank 8
        # may move 2 squares forward-diagonal. Intervening square must be
        # empty; destination empty or friendly (merge).
        if height == 1 and top == "s" and from_rank == 7:
            for df, dr in _FORWARD_DIAGS:
                int_f = from_file + df
                int_r = from_rank + dr
                if not _on_board(int_f, int_r):
                    continue
                int_sq = chess.square(int_f, int_r)
                if state.board.piece_at(int_sq) is not None:
                    continue  # path blocked at the intervening square
                tf = from_file + 2 * df
                tr = from_rank + 2 * dr
                if not _on_board(tf, tr):
                    continue
                to_sq = chess.square(tf, tr)
                target = state.board.piece_at(to_sq)
                if target is None:
                    moves.append(_quiet_move(from_name, to_sq, top))
                elif target.color == chess.BLACK and _is_black_top_at(state, to_sq):
                    moves.append(_quiet_move(from_name, to_sq, top))

    return moves


def _quiet_move(from_name: str, to_sq: chess.Square, top: str) -> dict[str, Any]:
    to_name = chess.square_name(to_sq)
    return {
        "uci": f"{from_name}{to_name}",
        "from": from_name,
        "to": to_name,
        "piece": _piece_name(top),
        "color": "black",
        "capture": None,
        "waypoints": None,
        "chainHops": None,
        "promotion": None,
        "demotedKings": None,
        "demotionsRequired": None,
        "sourceKingPositions": None,
        "deployCount": None,
    }


def black_deploy_moves(state: State) -> list[dict[str, Any]]:
    """Phase 2B — deploy moves.

    For a tower of height n, take the top s pieces (1 ≤ s < n) and move
    them as a sub-tower up to s squares along a diagonal. The sub-tower's
    top piece is the original tower's top piece, so the same Stone-vs-King
    direction rule applies. The path-clearance rules match diagonal quiet
    moves (friendly merges, White stops without emit). The remainder
    `n - s` pieces stay put."""
    if state.board.turn != chess.BLACK:
        return []

    moves: list[dict[str, Any]] = []
    for from_sq, pieces in state.stacks.items():
        n = len(pieces)
        if n < 2:
            continue  # height-1 towers cannot deploy (s would have to be 1, but s < n requires s < 1)
        top = pieces[-1]
        directions = _ALL_DIAGS if top == "k" else _FORWARD_DIAGS
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        from_name = chess.square_name(from_sq)

        for s in range(1, n):  # 1..n-1
            for df, dr in directions:
                for k in range(1, s + 1):
                    tf = from_file + k * df
                    tr = from_rank + k * dr
                    if not _on_board(tf, tr):
                        break
                    to_sq = chess.square(tf, tr)
                    target = state.board.piece_at(to_sq)
                    if target is None:
                        moves.append(_deploy_move(from_name, to_sq, top, s))
                        continue
                    if target.color == chess.BLACK and _is_black_top_at(state, to_sq):
                        moves.append(_deploy_move(from_name, to_sq, top, s))
                    break
    return moves


def _deploy_move(from_name: str, to_sq: chess.Square, top: str, s: int) -> dict[str, Any]:
    to_name = chess.square_name(to_sq)
    return {
        "uci": f"{from_name}{to_name}[{s}]",
        "from": from_name,
        "to": to_name,
        "piece": _piece_name(top),
        "color": "black",
        "capture": None,
        "waypoints": None,
        "chainHops": None,
        "promotion": None,
        "demotedKings": None,
        "demotionsRequired": None,
        "sourceKingPositions": None,
        "deployCount": s,
    }
