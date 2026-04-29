"""Black-side move generation. Implemented incrementally; each phase has
matching differential tests against scalachess in tests/test_pyvariant_diff.py.

Phase status:
- [x] 2A: Diagonal quiet moves (full-tower, no deploy, no capture)
- [x] 2B: Deploy moves
- [x] 2C: Back-rank sprint
- [~] 2D: Diagonal capture chains (single hop, no rim, no chain, no promote)
- [x] 2E: Charge (orthogonal capture)
- [x] 2F: Mandatory rule filter
- [ ] 2G: State transition (apply Black move)
"""

from __future__ import annotations

from itertools import combinations
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


_ORTHO_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def black_diagonal_capture_moves(state: State) -> list[dict[str, Any]]:
    """Phase 2D (simplified) — single-hop diagonal captures.

    Limitations of this initial cut:
    - Only single hops (no chains). A chain landing position with a follow-up
      capture in another direction is not yet emitted as a multi-hop move.
    - Only board landings. Rim landings and rim-bounces are not yet handled.
    - No rank-1 promotion. Hops touching rank 1 don't promote yet.

    Per §3B: walk up to n steps along a diagonal, find the first White at
    distance d ∈ [1, n], pick a landing k ∈ [d, n+1]. Capture every
    on-board White on path squares 1..k-1. Land at k:
        empty board square → normal hop
        White piece → ram (no landing capture; tower destroyed at landing)
        friendly tower → that k is illegal
    Ram-at-k=d additional reachability: k=d+1 must be on the 10x10 grid
    and not a friendly Black tower."""
    if state.board.turn != chess.BLACK:
        return []
    moves: list[dict[str, Any]] = []
    for from_sq, pieces in state.stacks.items():
        if not pieces:
            continue
        n = len(pieces)
        top = pieces[-1]
        directions = _ALL_DIAGS if top == "k" else _FORWARD_DIAGS
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        from_name = chess.square_name(from_sq)

        for df, dr in directions:
            # Find first enemy at d ∈ [1, n]. Friendly intervening tower
            # blocks the scan and means no first enemy in this direction.
            d = None
            for step in range(1, n + 1):
                pf = from_file + step * df
                pr = from_rank + step * dr
                if not _on_board(pf, pr):
                    break
                psq = chess.square(pf, pr)
                p = state.board.piece_at(psq)
                if p is None:
                    continue
                if p.color == chess.BLACK and _is_black_top_at(state, psq):
                    break  # friendly blocks scan
                d = step  # White piece — first enemy
                break
            if d is None:
                continue

            for k in range(d, n + 2):
                tf = from_file + k * df
                tr = from_rank + k * dr
                if not _on_board(tf, tr):
                    continue  # rim handling deferred
                to_sq = chess.square(tf, tr)
                to_name = chess.square_name(to_sq)
                target = state.board.piece_at(to_sq)

                # Path captures (steps 1..k-1, board squares only — rim
                # squares capture nothing per spec, deferred).
                path_captures: list[str] = []
                path_off_board = False
                for step in range(1, k):
                    pf2 = from_file + step * df
                    pr2 = from_rank + step * dr
                    if not _on_board(pf2, pr2):
                        path_off_board = True
                        break
                    psq2 = chess.square(pf2, pr2)
                    p2 = state.board.piece_at(psq2)
                    if p2 is not None and p2.color == chess.WHITE:
                        path_captures.append(chess.square_name(psq2))
                if path_off_board:
                    continue  # rim path deferred

                # Landing classification.
                if target is None:
                    is_ram = False
                elif target.color == chess.WHITE:
                    is_ram = True
                else:
                    # Friendly Black tower at landing → that k is illegal.
                    continue

                # Ram reachability (only applies when k == d).
                if is_ram and k == d:
                    next_f = from_file + (d + 1) * df
                    next_r = from_rank + (d + 1) * dr
                    # Off the 10×10 grid?
                    if not (-1 <= next_f <= 8 and -1 <= next_r <= 8):
                        continue
                    # On 8×8: friendly tower disqualifies the ram.
                    if _on_board(next_f, next_r):
                        next_sq = chess.square(next_f, next_r)
                        next_p = state.board.piece_at(next_sq)
                        if (
                            next_p is not None
                            and next_p.color == chess.BLACK
                            and _is_black_top_at(state, next_sq)
                        ):
                            continue

                # `capture` field: closest path-captured White, or ram-destination.
                if path_captures:
                    capture_field: str | None = path_captures[0]
                elif is_ram:
                    capture_field = to_name
                else:
                    capture_field = None

                moves.append({
                    "uci": f"{from_name}{to_name}",
                    "from": from_name,
                    "to": to_name,
                    "piece": _piece_name(top),
                    "color": "black",
                    "capture": capture_field,
                    "waypoints": None,
                    "chainHops": [to_name],
                    "promotion": None,
                    "demotedKings": None,
                    "demotionsRequired": None,
                    "sourceKingPositions": None,
                    "deployCount": None,
                })
    return moves


def black_mandatory_capture_active(state: State) -> bool:
    """§4 — mandate fires when at least one diagonal capture has a normal
    (empty-board) landing. Rams alone do not trigger; rim-only landings
    do not trigger (we don't emit them yet, but if/when we do, they
    should not count). Charge captures don't trigger either — only
    diagonal hops with normal landings."""
    for cap in black_diagonal_capture_moves(state):
        # Normal landing = empty board square.
        to_sq = chess.parse_square(cap["to"])
        if state.board.piece_at(to_sq) is None:
            return True
    return False


def filter_for_mandate(state: State, all_moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """If mandate is active, suppress non-capturing moves (quiet diagonals,
    deploys, sprints, non-capturing charges) per §4. Capturing moves
    (path-capture or ram) all satisfy the mandate."""
    if not black_mandatory_capture_active(state):
        return all_moves
    return [m for m in all_moves if m.get("capture") is not None]


def black_charge_moves(state: State) -> list[dict[str, Any]]:
    """Phase 2E — Charge (orthogonal capture).

    A King-top tower may move along a rank or file. Cost is one King
    demoted per square moved. Path Whites are captured for free; landing
    on a White piece is a ram (tower destroyed at landing, no landing
    capture). Friendly Black towers in the path block; landing on a
    friendly is a merge.

    For non-ram landings with `n_kings > d` (charge distance), each
    combinatorial choice of which Kings to demote produces a separate
    move (UCI suffix `{a,b,c}`). When `n_kings == d` only one combination
    exists and demotion fields are emitted as null. Rams always emit a
    single move with null demotion fields (the tower dies at landing,
    so the choice is moot).

    The `capture` field — when there are path captures, scalachess emits
    the closest-to-source captured square; for a ram with no path
    captures, it emits the destination; otherwise null. We mirror this
    quirk for byte-for-byte parity.
    """
    if state.board.turn != chess.BLACK:
        return []
    moves: list[dict[str, Any]] = []
    for from_sq, pieces in state.stacks.items():
        if not pieces or pieces[-1] != "k":
            continue
        n_kings = sum(1 for p in pieces if p == "k")
        if n_kings == 0:
            continue
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        from_name = chess.square_name(from_sq)
        king_positions = [i + 1 for i, p in enumerate(pieces) if p == "k"]

        for df, dr in _ORTHO_DIRS:
            stop_after = False  # set when a friendly tower mid-scan blocks further walk
            for d in range(1, n_kings + 1):
                if stop_after:
                    break
                # Scan path 1..d-1 for blockers / collect path captures fresh.
                blocked = False
                path_captures: list[str] = []
                for k in range(1, d):
                    pf = from_file + k * df
                    pr = from_rank + k * dr
                    psq = chess.square(pf, pr)
                    p = state.board.piece_at(psq)
                    if p is not None:
                        if p.color == chess.BLACK and _is_black_top_at(state, psq):
                            blocked = True
                            break
                        if p.color == chess.WHITE:
                            path_captures.append(chess.square_name(psq))
                if blocked:
                    break
                tf = from_file + d * df
                tr = from_rank + d * dr
                if not _on_board(tf, tr):
                    break
                to_sq = chess.square(tf, tr)
                to_name = chess.square_name(to_sq)
                target = state.board.piece_at(to_sq)
                is_ram = target is not None and target.color == chess.WHITE
                is_friendly_merge = (
                    target is not None
                    and target.color == chess.BLACK
                    and _is_black_top_at(state, to_sq)
                )

                # Capture-field convention (scalachess parity).
                if path_captures:
                    capture_field: str | None = path_captures[0]
                elif is_ram:
                    capture_field = to_name
                else:
                    capture_field = None

                if is_ram:
                    # Ram: tower destroyed at landing; demotion choice is moot.
                    moves.append({
                        "uci": f"{from_name}{to_name}",
                        "from": from_name,
                        "to": to_name,
                        "piece": "king",
                        "color": "black",
                        "capture": capture_field,
                        "waypoints": None,
                        "chainHops": None,
                        "promotion": None,
                        "demotedKings": None,
                        "demotionsRequired": None,
                        "sourceKingPositions": None,
                        "deployCount": None,
                    })
                    # The white at d remains on the board for OUR move-gen
                    # purposes; longer charges pass over it as a path capture
                    # (not a blocker). Continue scanning.
                    continue

                # Non-ram landing (empty or friendly merge).
                if n_kings == d:
                    # Forced demotion — null fields.
                    new_pieces = list(pieces)
                    for pos in king_positions:
                        new_pieces[pos - 1] = "S"
                    resulting_top = new_pieces[-1]
                    moves.append({
                        "uci": f"{from_name}{to_name}",
                        "from": from_name,
                        "to": to_name,
                        "piece": _piece_name(resulting_top),
                        "color": "black",
                        "capture": capture_field,
                        "waypoints": None,
                        "chainHops": None,
                        "promotion": None,
                        "demotedKings": None,
                        "demotionsRequired": None,
                        "sourceKingPositions": None,
                        "deployCount": None,
                    })
                else:
                    # Multiple demotion choices.
                    for choice in combinations(king_positions, d):
                        choice_sorted = list(choice)  # combinations yields sorted
                        new_pieces = list(pieces)
                        for pos in choice_sorted:
                            new_pieces[pos - 1] = "S"
                        resulting_top = new_pieces[-1]
                        choice_str = ",".join(str(x) for x in choice_sorted)
                        moves.append({
                            "uci": f"{from_name}{to_name}{{{choice_str}}}",
                            "from": from_name,
                            "to": to_name,
                            "piece": _piece_name(resulting_top),
                            "color": "black",
                            "capture": capture_field,
                            "waypoints": None,
                            "chainHops": None,
                            "promotion": None,
                            "demotedKings": choice_sorted,
                            "demotionsRequired": d,
                            "sourceKingPositions": list(king_positions),
                            "deployCount": None,
                        })

                if is_friendly_merge:
                    # Friendly landing blocks further scanning beyond this d.
                    stop_after = True

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
