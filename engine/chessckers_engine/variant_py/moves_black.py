"""Black-side move generation. Implemented incrementally; each phase has
matching differential tests against scalachess in tests/test_pyvariant_diff.py.

Phase status:
- [x] 2A: Diagonal quiet moves (full-tower, no deploy, no capture)
- [x] 2B: Deploy moves
- [x] 2C: Back-rank sprint
- [~] 2D: Diagonal capture chains (single hop, no rim, no chain, no promote)
- [x] 2E: Charge (orthogonal capture)
- [x] 2F: Mandatory rule filter
- [~] 2G: State transition (quiet, deploy, charge, simple capture)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chess

from chessckers_engine.variant_py.state import State

# Diagonal direction vectors as (file_delta, rank_delta).
# Black "forward" is toward rank 1 → rank decreases.
_FORWARD_DIAGS = [(-1, -1), (1, -1)]
_BACKWARD_DIAGS = [(-1, 1), (1, 1)]
_ALL_DIAGS = _FORWARD_DIAGS + _BACKWARD_DIAGS

# Precomputed string tables — module-load is the only place these are built.
# `chess.square_name` was the second-most-called function in the MCTS profile
# (557k calls per 100 sims) and `_coord_to_key` was building 573k strings via
# f-format. Both reduce to constant-time list/dict lookups now.
_SQ_NAME: tuple[str, ...] = tuple(chess.square_name(i) for i in range(64))


def _on_board(file: int, rank: int) -> bool:
    return 0 <= file <= 7 and 0 <= rank <= 7


def _on_grid(file: int, rank: int) -> bool:
    """The 10×10 grid (board + 1-square rim)."""
    return -1 <= file <= 8 and -1 <= rank <= 8


def _build_coord_key_table() -> dict[tuple[int, int], str]:
    """Materialize the (file, rank) → 2-char key map for the entire 10×10
    grid. Files -1/8 use 'z'/'i'; ranks -1/8 use '0'/'9'."""
    table: dict[tuple[int, int], str] = {}
    for f in range(-1, 9):
        for r in range(-1, 9):
            if f == -1:
                fc = "z"
            elif 0 <= f <= 7:
                fc = chr(ord("a") + f)
            elif f == 8:
                fc = "i"
            else:
                fc = "?"
            table[(f, r)] = f"{fc}{r + 1}"
    return table


_COORD_KEY: dict[tuple[int, int], str] = _build_coord_key_table()


# Maximum tower height. Stacks are capped at 5 pieces.
from chessckers_engine.variant_py.state import MAX_TOWER_HEIGHT  # noqa: E402

# Maximum trace length for any capture-walk. The largest tower on the board
# has MAX_TOWER_HEIGHT pieces; n+1 ≤ 6. We cap at 10 to keep all paths in-table
# with headroom.
_MAX_HOP_STEPS = 10


def _build_capture_paths() -> dict[tuple[int, int, int, int], list[tuple]]:
    """Precompute every straight-diagonal trajectory the capture-hop walker
    can take. Keyed by (start_file, start_rank, df, dr); value is a list of
    step records:
        (f, r, sq_or_minus1, key, df_after, dr_after, did_bounce)
    Per spec §3B step 3 (no-bounce rule), the path is a pure straight diagonal
    walk. If a step would go off the 10×10 grid the trace just terminates —
    no reflection, no salvaging the landing. `df_after`/`dr_after` always
    equal the original direction; `did_bounce` is always False (kept in the
    tuple for shape compatibility with the consumer)."""
    paths: dict[tuple[int, int, int, int], list[tuple]] = {}
    coord_key = _COORD_KEY
    for f0 in range(-1, 9):
        for r0 in range(-1, 9):
            for df0 in (-1, 1):
                for dr0 in (-1, 1):
                    steps: list[tuple] = []
                    f, r = f0, r0
                    for _ in range(_MAX_HOP_STEPS):
                        nf = f + df0
                        nr = r + dr0
                        if nf < -1 or nf > 8 or nr < -1 or nr > 8:
                            break
                        f, r = nf, nr
                        on_board_now = 0 <= nf <= 7 and 0 <= nr <= 7
                        sq = ((r << 3) | f) if on_board_now else -1
                        steps.append(
                            (f, r, sq, coord_key.get((f, r), "??"), df0, dr0, False)
                        )
                    paths[(f0, r0, df0, dr0)] = steps
    return paths


_CAPTURE_PATHS: dict[tuple[int, int, int, int], list[tuple]] = _build_capture_paths()


def _coord_to_key(file: int, rank: int) -> str:
    """(file, rank) on the 10×10 grid → 2-char key. Rim coords use 'z'/'i'
    for files -1/8 and '0'/'9' for ranks -1/8."""
    cached = _COORD_KEY.get((file, rank))
    if cached is not None:
        return cached
    # Fallback for out-of-grid coords (shouldn't occur in normal play).
    return f"?{rank + 1}"


def _parse_waypoint_key(s: str) -> tuple[int, int] | None:
    if len(s) != 2:
        return None
    f_char, r_char = s[0], s[1]
    if f_char == "z":
        f = -1
    elif "a" <= f_char <= "h":
        f = ord(f_char) - ord("a")
    elif f_char == "i":
        f = 8
    else:
        return None
    if "0" <= r_char <= "9":
        r = int(r_char) - 1
    else:
        return None
    return (f, r)


@dataclass
class CaptureHop:
    """A single hop's trace + outcome. Mirrors scalachess's CaptureHop.

    `landing_square` is None when the hop lands on the rim (T). `captures`
    holds chess.Square ints of every White captured along the path (board
    squares only — rim squares hold nothing). `waypoints` is the list of
    every traced step's key (10×10 coords), excluding the start; for a
    k-step hop, len(waypoints) == k. `direction` is the (df, dr) at the
    landing step (post-bounce if any). `crossed_rank1` flips True if any
    step in this hop's path is on rank 1 (idx 0)."""
    direction: tuple[int, int]
    landing_key: str
    landing_square: int | None
    captures: list[int] = field(default_factory=list)
    waypoints: list[str] = field(default_factory=list)
    is_suicide: bool = False
    crossed_rank1: bool = False
    # `cadence` is the hop's landing distance k. For an on-grid landing it
    # equals len(waypoints); for an off-grid overshoot it is one past the last
    # on-grid step, so it can exceed len(waypoints) — hence stored explicitly.
    cadence: int = 0
    # True when the cadence landing fell off the 10×10 grid: the hop captured
    # its path Whites but cannot land, so it settles on the last on-board
    # square (computed in _build_final_move) and ENDS the chain.
    is_overshoot: bool = False


def _piece_name(top_char: str) -> str:
    """Top-piece character → scalachess piece-name field for the move dict."""
    return "king" if top_char == "k" else "pawn"


def _is_black_top_at(state: State, sq: chess.Square) -> bool:
    """True iff `sq` has a Black tower (Stone-top or King-top)."""
    return sq in state.stacks


# Owner-of-square codes for the bitboard fast path. board.piece_at() builds a
# Piece object every call (~500ns); this returns a small int, ~50ns.
SQ_EMPTY = 0
SQ_WHITE = 1
SQ_BLACK = 2


def _owner(board: chess.Board, sq: int) -> int:
    """Return SQ_EMPTY/SQ_WHITE/SQ_BLACK by reading the bitboards directly.
    Avoids `board.piece_at(sq)`'s `Piece(...)` allocation, which dominated
    the move-gen inner loops at ~544k calls per 100 sims."""
    mask = chess.BB_SQUARES[sq]
    if not (board.occupied & mask):
        return SQ_EMPTY
    if board.occupied_co[chess.WHITE] & mask:
        return SQ_WHITE
    return SQ_BLACK


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
        from_name = _SQ_NAME[from_sq]

        board = state.board
        stacks = state.stacks
        for df, dr in directions:
            for k in range(1, height + 1):
                tf = from_file + k * df
                tr = from_rank + k * dr
                if not _on_board(tf, tr):
                    break
                to_sq = chess.square(tf, tr)
                owner = _owner(board, to_sq)
                if owner == SQ_EMPTY:
                    moves.append(_quiet_move(from_name, to_sq, top))
                    continue  # empty square — keep scanning further along this diagonal
                # Non-empty square: classify and stop.
                if owner == SQ_BLACK and to_sq in stacks:
                    # Friendly Black tower → merge is legal here, but blocks further walk.
                    if len(stacks[to_sq]) + height <= MAX_TOWER_HEIGHT:
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
                if _owner(board, int_sq) != SQ_EMPTY:
                    continue  # path blocked at the intervening square
                tf = from_file + 2 * df
                tr = from_rank + 2 * dr
                if not _on_board(tf, tr):
                    continue
                to_sq = chess.square(tf, tr)
                owner = _owner(board, to_sq)
                if owner == SQ_EMPTY:
                    moves.append(_quiet_move(from_name, to_sq, top))
                elif owner == SQ_BLACK and to_sq in stacks:
                    if len(stacks[to_sq]) + 1 <= MAX_TOWER_HEIGHT:
                        moves.append(_quiet_move(from_name, to_sq, top))

    return moves


def _quiet_move(from_name: str, to_sq: chess.Square, top: str) -> dict[str, Any]:
    to_name = _SQ_NAME[to_sq]
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
        from_name = _SQ_NAME[from_sq]

        board = state.board
        stacks = state.stacks
        for s in range(1, n):  # 1..n-1
            for df, dr in directions:
                for k in range(1, s + 1):
                    tf = from_file + k * df
                    tr = from_rank + k * dr
                    if not _on_board(tf, tr):
                        break
                    to_sq = chess.square(tf, tr)
                    owner = _owner(board, to_sq)
                    if owner == SQ_EMPTY:
                        moves.append(_deploy_move(from_name, to_sq, top, s))
                        continue
                    if owner == SQ_BLACK and to_sq in stacks:
                        if len(stacks[to_sq]) + s <= MAX_TOWER_HEIGHT:
                            moves.append(_deploy_move(from_name, to_sq, top, s))
                    break
    return moves


_ORTHO_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def _find_capture_hops(
    board: chess.Board,
    f0: int,
    r0: int,
    df0: int,
    dr0: int,
    n: int,
    stacks: dict[chess.Square, str],
) -> list[CaptureHop]:
    """Port of scalachess.Chessckers.findSlideCaptureOptionsFrom.

    Walk along (df0, dr0) up to n+1 steps from (f0, r0) using the
    precomputed straight-diagonal path (`_CAPTURE_PATHS`). Emit a CaptureHop
    for every legal landing (every k where the hop terminates: empty board
    revisit-after-capture, White ram (k > d only), empty board with prior
    captures, or rim-T after captures). Per §3B step 2, rams require k > d
    — the first-enemy ram (k=d, no path captures) is no longer emitted.
    Per spec §3B step 3 there is no bouncing — if the precomputed path
    runs out before reaching cadence, that direction is unavailable."""
    options: list[CaptureHop] = []
    captures_so_far: list[int] = []
    captured_set: set[int] = set()
    waypoints_so_far: list[str] = []

    # Hoist hot bitboard attrs to locals — Python LOAD_FAST is ~3× faster than
    # the LOAD_ATTR that `board.occupied` would do every loop iteration.
    BB_SQUARES = chess.BB_SQUARES
    occupied = board.occupied
    occupied_white = board.occupied_co[chess.WHITE]

    crossed_rank1 = False

    path = _CAPTURE_PATHS[(f0, r0, df0, dr0)]
    # Walk only the steps we actually need: max n+1 steps; precomputed path
    # may end earlier when it would have left the grid.
    max_step = n + 1
    friendly_blocked = False
    for step_idx, step_data in enumerate(path):
        if step_idx >= max_step:
            break
        f, r, sq, cur_key, df, dr, did_bounce = step_data
        waypoints_so_far.append(cur_key)
        if r == 0:
            crossed_rank1 = True
        step = step_idx + 1  # 1-based step number = the landing distance k

        if sq != -1:
            if sq in captured_set:
                # Revisit of an already-captured (now empty) square.
                if captures_so_far:
                    options.append(CaptureHop(
                        direction=(df, dr),
                        landing_key=cur_key,
                        landing_square=sq,
                        captures=list(captures_so_far),
                        waypoints=list(waypoints_so_far),
                        crossed_rank1=crossed_rank1,
                        cadence=step,
                    ))
            else:
                # Bitboard fast path — board.piece_at would allocate a Piece
                # object per call (this loop runs ~205k times per 100 sims).
                mask = BB_SQUARES[sq]
                if not (occupied & mask):
                    owner = SQ_EMPTY
                elif occupied_white & mask:
                    owner = SQ_WHITE
                else:
                    owner = SQ_BLACK
                if owner == SQ_EMPTY:
                    if captures_so_far:
                        options.append(CaptureHop(
                            direction=(df, dr),
                            landing_key=cur_key,
                            landing_square=sq,
                            captures=list(captures_so_far),
                            waypoints=list(waypoints_so_far),
                            crossed_rank1=crossed_rank1,
                            cadence=step,
                        ))
                elif owner == SQ_BLACK and sq in stacks:
                    # Friendly tower terminates the trace (a block, NOT an
                    # off-grid exit — so no overshoot is emitted past it).
                    friendly_blocked = True
                    break
                else:
                    # White piece (or Black-piece-without-stack, defensive).
                    # Per §3B step 2: rams require k > d (an intermediate path
                    # capture must have already happened). Landing on the first
                    # enemy (captures_so_far empty) is no longer a legal hop.
                    if captures_so_far:
                        ram_hop = CaptureHop(
                            direction=(df, dr),
                            landing_key=cur_key,
                            landing_square=sq,
                            captures=list(captures_so_far),
                            waypoints=list(waypoints_so_far),
                            is_suicide=True,
                            crossed_rank1=crossed_rank1,
                            cadence=step,
                        )
                        options.append(ram_hop)
                    captures_so_far.append(sq)
                    captured_set.add(sq)
        else:
            # Rim square (T) — never friendly. Emit T-landing if captures.
            if captures_so_far:
                options.append(CaptureHop(
                    direction=(df, dr),
                    landing_key=cur_key,
                    landing_square=None,
                    captures=list(captures_so_far),
                    waypoints=list(waypoints_so_far),
                    crossed_rank1=crossed_rank1,
                    cadence=step,
                ))

    # §3B off-grid overshoot. The straight path left the 10×10 grid before
    # reaching the cadence limit (the precomputed path is shorter than n+1,
    # and we weren't stopped by a friendly tower) AND at least one White was
    # captured on the way. The hop cannot land off-grid, so it settles on the
    # last on-board square (resolved in _build_final_move) and ends the turn.
    # Cadence is one step past the last on-grid square. It is a candidate
    # DISTINCT from the rim landing at the same key (different cadence), so
    # the dedup in _next_capture_options must keep both.
    if (not friendly_blocked) and captures_so_far and len(path) < max_step:
        options.append(CaptureHop(
            direction=(df0, dr0),
            landing_key=waypoints_so_far[-1],
            landing_square=None,
            captures=list(captures_so_far),
            waypoints=list(waypoints_so_far),
            crossed_rank1=crossed_rank1,
            cadence=len(path) + 1,
            is_overshoot=True,
        ))

    return options


def _hop_promotes(hop: CaptureHop) -> bool:
    """A hop promotes the moving stack if its path touched rank 1 — either
    by crossing rank 1 mid-trace en route to T, or by landing on rank 1."""
    if hop.crossed_rank1:
        return True
    if hop.landing_square is not None and chess.square_rank(hop.landing_square) == 0:
        return True
    return False


def _promote_all_stones(stack: str) -> str:
    """Every Stone (s/S) → King. Used on rank-1 promotion."""
    return "".join("k" if c in ("s", "S") else c for c in stack)


def _next_capture_options(
    board: chess.Board,
    stacks: dict[chess.Square, str],
    cf: int,
    cr: int,
    cur_stack: str,
    last_dir: tuple[int, int] | None,
    n: int,
    cadence: int | None,
    include_suicide: bool = False,
) -> list[CaptureHop]:
    """Hops available from (cf, cr) for the moving stack. Filters by:
    - direction != 180° reverse of last_dir,
    - suicide hops removed unless include_suicide=True (default matches
      scalachess's validMoves chain enumeration; first-hop suicides are
      emitted separately by `_first_hop_suicides`),
    - len(waypoints) == cadence when cadence is locked,
    - dedup by (direction, landing_key, captures, is_suicide)."""
    if not cur_stack:
        return []
    is_king_top = cur_stack[-1] == "k"
    dirs = _ALL_DIAGS if is_king_top else _FORWARD_DIAGS
    if last_dir is not None:
        ldf, ldr = last_dir
        dirs = [(df, dr) for df, dr in dirs if not (df == -ldf and dr == -ldr)]
    options: list[CaptureHop] = []
    for df, dr in dirs:
        options.extend(_find_capture_hops(board, cf, cr, df, dr, n, stacks))
    if not include_suicide:
        options = [h for h in options if not h.is_suicide]
    if cadence is not None:
        options = [h for h in options if h.cadence == cadence]
    seen: set[tuple[Any, ...]] = set()
    deduped: list[CaptureHop] = []
    for h in options:
        # is_overshoot + cadence are part of identity: a rim landing and an
        # off-grid overshoot can share direction/landing_key/captures but are
        # distinct moves (§3B), so they must not dedup into one.
        key = (h.direction, h.landing_key, tuple(h.captures), h.is_suicide,
               h.is_overshoot, h.cadence)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    return deduped


def _enumerate_chains(state: State, chain_start: chess.Square) -> list[dict[str, Any]]:
    """DFS-enumerate every complete chain starting from `chain_start`. A
    chain leaf is a state with no further legal hops; that leaf becomes one
    move via `_build_final_move`. Single-hop captures are length-1 chains
    (emitted with 4-char UCI). Multi-hop chains share a cadence (the first
    hop's k) and forbid the 180°-reverse direction at each continuation."""
    orig_stack = state.stacks.get(chain_start, "")
    if not orig_stack:
        return []
    n = len(orig_stack)
    results: list[dict[str, Any]] = []

    def explore(
        board: chess.Board,
        stacks: dict[chess.Square, str],
        cf: int,
        cr: int,
        cur_stack: str,
        last_dir: tuple[int, int] | None,
        hops_so_far: list[CaptureHop],
        cadence: int | None,
    ) -> None:
        # If the white king has been captured the game ends; terminate the chain.
        white_king_captured = board.king(chess.WHITE) is None
        if white_king_captured:
            options: list[CaptureHop] = []
        else:
            options = _next_capture_options(
                board, stacks, cf, cr, cur_stack, last_dir, n, cadence
            )
        if not options:
            # No further capture available. Whatever chain reached here was
            # already emitted as a "stop" by the parent loop below (every hop
            # emits its stop-move), so there's nothing to add — just unwind.
            return
        for hop in options:
            if hop.is_overshoot:
                # Off-grid overshoot: captured its path Whites, can't land, so
                # it settles on the last on-board square (computed in
                # _build_final_move) and ENDS the chain — no continuation.
                results.append(_build_final_move(state, chain_start, hops_so_far + [hop]))
                continue
            # §3B: continuing is optional — stopping after this capture is a
            # legal move. Emit the chain ending here, then ALSO recurse to
            # extend it (the recursion emits the longer variants). Each chain
            # prefix is emitted exactly once, by the hop that produces it.
            results.append(_build_final_move(state, chain_start, hops_so_far + [hop]))
            new_board = board.copy(stack=False)
            new_stacks = dict(stacks)
            # Remove moving tower from current square if on B (mid-chain rim has none).
            if _on_board(cf, cr):
                cur_sq = chess.square(cf, cr)
                new_board.remove_piece_at(cur_sq)
                new_stacks.pop(cur_sq, None)
            # Capture path Whites (board squares only).
            for cap_sq in hop.captures:
                new_board.remove_piece_at(cap_sq)
            # Promote in-transit stack if path touched rank 1.
            should_promote = _hop_promotes(hop)
            land_stack = _promote_all_stones(cur_stack) if should_promote else cur_stack
            # Place tower at landing (if on board). Suicide hops are filtered
            # upstream in `_next_capture_options`, so we never see is_suicide
            # here in the chain DFS — first-hop suicides go via `_first_hop_suicides`.
            if hop.landing_square is not None:
                new_board.remove_piece_at(hop.landing_square)
                top = land_stack[-1]
                if top == "k":
                    new_board.set_piece_at(hop.landing_square, chess.Piece(chess.KING, chess.BLACK))
                else:
                    new_board.set_piece_at(hop.landing_square, chess.Piece(chess.PAWN, chess.BLACK))
                new_stacks[hop.landing_square] = land_stack
            # Compute next position (board or rim coords).
            if hop.landing_square is not None:
                nf = chess.square_file(hop.landing_square)
                nr = chess.square_rank(hop.landing_square)
            else:
                parsed = _parse_waypoint_key(hop.landing_key)
                nf, nr = parsed if parsed else (cf, cr)
            next_cadence = cadence if cadence is not None else hop.cadence
            explore(
                new_board, new_stacks, nf, nr, land_stack,
                hop.direction, hops_so_far + [hop], next_cadence,
            )

    cf0 = chess.square_file(chain_start)
    cr0 = chess.square_rank(chain_start)
    explore(state.board, dict(state.stacks), cf0, cr0, orig_stack, None, [], None)
    return results


def _build_final_move(
    orig_state: State,
    chain_start: chess.Square,
    hops: list[CaptureHop],
) -> dict[str, Any]:
    """Construct the LegalMove dict for a complete chain. Computes:
    - final landing square (rim → fall back to last on-board waypoint, else
      chain_start; on-board → that square),
    - resulting top piece (after promotions, or original top for suicides),
    - capture field (first path capture; or final landing for suicide-with-no-path-captures),
    - UCI: 4-char for single-hop, `orig~allWaypoints~dest` for multi-hop."""
    orig_stack = orig_state.stacks[chain_start]
    is_suicide_chain = bool(hops) and hops[-1].is_suicide
    all_captures: list[int] = []
    all_waypoints: list[str] = []
    hop_keys: list[str] = []
    for h in hops:
        all_captures.extend(h.captures)
        all_waypoints.extend(h.waypoints)
        hop_keys.append(h.landing_key)

    # Final landing.
    last_landing = hops[-1].landing_square
    if last_landing is not None:
        final_landing: chess.Square = last_landing
    else:
        # End-of-turn fallback: walk waypoints backwards for last on-board key.
        final_landing = chain_start
        for wp in reversed(all_waypoints):
            parsed = _parse_waypoint_key(wp)
            if parsed is None:
                continue
            f, r = parsed
            if _on_board(f, r):
                final_landing = chess.square(f, r)
                break

    # Determine final top piece (post-promotion accumulated through chain).
    if is_suicide_chain:
        final_top = orig_stack[-1]
    else:
        stack_thru = orig_stack
        for h in hops:
            if _hop_promotes(h):
                stack_thru = _promote_all_stones(stack_thru)
        final_top = stack_thru[-1]

    # capture field: first path-captured White, else final-landing for naked suicide.
    if all_captures:
        capture_field: str | None = _SQ_NAME[all_captures[0]]
    elif is_suicide_chain:
        capture_field = _SQ_NAME[final_landing]
    else:
        capture_field = None

    from_name = _SQ_NAME[chain_start]
    dest_name = _SQ_NAME[final_landing]
    # §3B notation: c<N>:<from>~<hop landings>-><rest>. The cadence (the first
    # hop's k) leads; <rest> is always shown and always on-board. The hop keys
    # are on-grid landing keys — for an overshoot the final key is the last
    # on-grid square that hop reached, never an off-grid coordinate. Cadence is
    # the discriminator: a rim landing and an off-grid overshoot can share the
    # same keys and rest, and only the leading c<N> tells them apart.
    cadence = hops[0].cadence
    hop_key_list = [hop.landing_key for hop in hops]
    uci = f"c{cadence}:{from_name}~{'~'.join(hop_key_list)}->{dest_name}"
    waypoints_field: list[str] | None = list(all_waypoints) if len(hops) > 1 else None

    # Internal fields used by _apply_chain_move; not exposed to callers.
    all_cap_names = [_SQ_NAME[sq] for sq in all_captures]
    # Promotion: did any hop's path touch rank 1?
    chain_promotes = any(_hop_promotes(h) for h in hops)

    return {
        "uci": uci,
        "from": from_name,
        "to": dest_name,
        "piece": "king" if final_top == "k" else "pawn",
        "color": "black",
        "capture": capture_field,
        "waypoints": waypoints_field,
        "chainHops": list(hop_keys),
        "cadence": cadence,
        "promotion": None,
        "demotedKings": None,
        "demotionsRequired": None,
        "sourceKingPositions": None,
        "deployCount": None,
        "_chain_all_captures": all_cap_names,
        "_is_suicide": is_suicide_chain,
        "_chain_promotes": chain_promotes,
    }


def _first_hop_suicides(state: State, chain_start: chess.Square) -> list[dict[str, Any]]:
    """Single-hop suicide (ram) captures from a tower's starting square.
    scalachess's `genBlackSuicideJumps`: enumerate every directional ram
    available immediately, regardless of whether the position has any
    non-suicide chain available. Each emitted as a length-1 chain."""
    pieces = state.stacks.get(chain_start, "")
    if not pieces:
        return []
    n = len(pieces)
    is_king_top = pieces[-1] == "k"
    dirs = _ALL_DIAGS if is_king_top else _FORWARD_DIAGS
    cf = chess.square_file(chain_start)
    cr = chess.square_rank(chain_start)
    moves: list[dict[str, Any]] = []
    for df, dr in dirs:
        for hop in _find_capture_hops(state.board, cf, cr, df, dr, n, state.stacks):
            if hop.is_suicide:
                moves.append(_build_final_move(state, chain_start, [hop]))
    return moves


def black_diagonal_capture_moves(state: State) -> list[dict[str, Any]]:
    """Phase 2D — diagonal captures (single hops + chains).

    Combines:
    - `_enumerate_chains`: non-suicide DFS-enumerated chains (single-hop
      and multi-hop). Final landings are board squares, or rim-fallback to
      the last on-board waypoint.
    - `_first_hop_suicides`: single-hop ram captures (each tower, each
      direction with a ram available).

    Tested against scalachess in `tests/test_pyvariant_diff.py`."""
    if state.board.turn != chess.BLACK:
        return []
    moves: list[dict[str, Any]] = []
    for from_sq, pieces in list(state.stacks.items()):
        if not pieces:
            continue
        moves.extend(_enumerate_chains(state, from_sq))
        moves.extend(_first_hop_suicides(state, from_sq))
    return moves


def black_mandatory_capture_active(state: State) -> bool:
    """§4 mandate trigger. Mirrors scalachess hasMandatoryCapture exactly:

    For each Black tower, for each diagonal direction it can move:
      - Check if the immediately adjacent square (Chebyshev distance 1)
        in that direction contains a White piece.
      - If yes, scan from that tower with _find_capture_hops. If any
        resulting hop is non-suicide AND lands on a board square
        (landingSquare is not None), mandate fires.

    This directly uses the raw hop scan, NOT chain leaves, so it correctly
    detects 'f2' as a triggering board landing even when f2 isn't a chain
    leaf (because the DFS continues past it)."""
    for from_sq, pieces in state.stacks.items():
        if not pieces:
            continue
        n = len(pieces)
        is_king_top = pieces[-1] == "k"
        dirs = _ALL_DIAGS if is_king_top else _FORWARD_DIAGS
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        for df, dr in dirs:
            adj_f, adj_r = from_file + df, from_rank + dr
            if not _on_board(adj_f, adj_r):
                continue
            adj_sq = chess.square(adj_f, adj_r)
            if _owner(state.board, adj_sq) != SQ_WHITE:
                continue
            # Adjacent White found. Check for a non-suicide hop that lands on board.
            hops = _find_capture_hops(state.board, from_file, from_rank, df, dr, n, state.stacks)
            if any(not h.is_suicide and h.landing_square is not None for h in hops):
                return True
    return False


def filter_for_mandate(state: State, all_moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """If mandate is active, suppress non-capturing moves (quiet diagonals,
    deploys, sprints, non-capturing charges) per §4. Capturing moves
    (path-capture or ram) all satisfy the mandate."""
    if not black_mandatory_capture_active(state):
        return all_moves
    return [m for m in all_moves if m.get("capture") is not None]


# ---------------- Phase 2G: applying Black moves ----------------

def _all_black_legal(state: State) -> list[dict[str, Any]]:
    """Aggregate all currently-legal Black moves with mandate applied."""
    return filter_for_mandate(
        state,
        black_diagonal_quiet_moves(state)
        + black_deploy_moves(state)
        + black_charge_moves(state)
        + black_diagonal_capture_moves(state),
    )


def _set_top_piece_on_board(state: State, sq: chess.Square, top_char: str) -> None:
    """Sync the bitboard's piece at `sq` with the stack's top character.
    Stone-top (s/S) → black pawn; King-top (k) → black king.
    Empty stack → remove any piece at sq."""
    state.board.remove_piece_at(sq)
    if top_char == "k":
        state.board.set_piece_at(sq, chess.Piece(chess.KING, chess.BLACK))
    elif top_char in ("s", "S"):
        state.board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))


def _move_full_tower(state: State, from_sq: chess.Square, to_sq: chess.Square,
                     pieces_override: str | None = None) -> None:
    """Move the entire stack from `from_sq` to `to_sq` (merging onto a
    friendly destination if present). `pieces_override` lets the caller
    substitute different stack contents (e.g. after sprinting a Stone:
    's' → 'S')."""
    moving = pieces_override if pieces_override is not None else state.stacks[from_sq]
    state.stacks.pop(from_sq, None)
    state.board.remove_piece_at(from_sq)
    existing = state.stacks.get(to_sq, "")
    new_stack = existing + moving  # incoming on top
    assert len(new_stack) <= MAX_TOWER_HEIGHT, (
        f"move would create tower height {len(new_stack)} > {MAX_TOWER_HEIGHT}"
    )
    state.stacks[to_sq] = new_stack
    _set_top_piece_on_board(state, to_sq, new_stack[-1])


def _apply_quiet_or_sprint(state: State, move: dict[str, Any]) -> None:
    from_sq = chess.parse_square(move["from"])
    to_sq = chess.parse_square(move["to"])
    pieces = state.stacks[from_sq]
    # Sprint: a height-1 unmoved Stone moving 2 squares from rank 8 marks the Stone as moved.
    is_sprint = (
        len(pieces) == 1 and pieces == "s"
        and chess.square_rank(from_sq) == 7
        and abs(chess.square_rank(to_sq) - chess.square_rank(from_sq)) == 2
    )
    pieces_override = "S" if is_sprint else None
    _move_full_tower(state, from_sq, to_sq, pieces_override=pieces_override)
    # Per spec §5, quiet diagonals + sprints promote when the destination is on
    # rank 1. Promote AFTER the merge so existing stones at dest also promote.
    if chess.square_rank(to_sq) == 0:
        promoted = _promote_all_stones(state.stacks[to_sq])
        state.stacks[to_sq] = promoted
        _set_top_piece_on_board(state, to_sq, promoted[-1])


def _apply_deploy(state: State, move: dict[str, Any]) -> None:
    from_sq = chess.parse_square(move["from"])
    to_sq = chess.parse_square(move["to"])
    s = move["deployCount"]
    pieces = state.stacks[from_sq]
    sub = pieces[-s:]
    remainder = pieces[:-s]
    state.stacks[from_sq] = remainder
    _set_top_piece_on_board(state, from_sq, remainder[-1])
    existing = state.stacks.get(to_sq, "")
    new_stack = existing + sub
    # Per spec §5, deploys promote when the destination is on rank 1.
    if chess.square_rank(to_sq) == 0:
        new_stack = _promote_all_stones(new_stack)
    state.stacks[to_sq] = new_stack
    _set_top_piece_on_board(state, to_sq, new_stack[-1])


def _apply_charge(state: State, move: dict[str, Any]) -> None:
    from_sq = chess.parse_square(move["from"])
    to_sq = chess.parse_square(move["to"])
    pieces = state.stacks[from_sq]
    n = len(pieces)
    df = (chess.square_file(to_sq) - chess.square_file(from_sq))
    dr = (chess.square_rank(to_sq) - chess.square_rank(from_sq))
    d = max(abs(df), abs(dr))
    df_sign = (df // d) if d else 0
    dr_sign = (dr // d) if d else 0

    # An overshoot charge (`waypoints` carries the rim landing key) actually
    # travels one square past `to`: it lands on the rim and falls back to `to`.
    # Its true distance is d+1, and `to` (step d) was itself a path square — so
    # any White there (e.g. a king on a board edge directly ahead) is a path
    # capture, NOT a ram. A plain on-board charge captures steps 1..d-1 and may
    # ram a White sitting on `to`.
    is_rim_overshoot = bool(move.get("waypoints"))
    last_step = d if is_rim_overshoot else d - 1

    # Capture path Whites.
    for k in range(1, last_step + 1):
        f = chess.square_file(from_sq) + k * df_sign
        r = chess.square_rank(from_sq) + k * dr_sign
        sq = chess.square(f, r)
        if _owner(state.board, sq) == SQ_WHITE:
            state.board.remove_piece_at(sq)

    is_ram = (not is_rim_overshoot) and _owner(state.board, to_sq) == SQ_WHITE
    if is_ram:
        # Tower destroyed at landing; landing White stays.
        state.stacks.pop(from_sq, None)
        state.board.remove_piece_at(from_sq)
        return

    # Compute demotion choice. With explicit `demotedKings`, use it; otherwise
    # the forced-choice path demotes ALL Kings (for n_kings == d).
    king_positions = [i + 1 for i, p in enumerate(pieces) if p == "k"]
    chosen = move.get("demotedKings") or king_positions
    new_pieces = list(pieces)
    for pos in chosen:
        new_pieces[pos - 1] = "S"
    new_stack = "".join(new_pieces)
    _move_full_tower(state, from_sq, to_sq, pieces_override=new_stack)


def _apply_diagonal_capture(state: State, move: dict[str, Any]) -> None:
    from_sq = chess.parse_square(move["from"])
    to_sq = chess.parse_square(move["to"])
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    d = max(abs(df), abs(dr))
    df_sign = df // d if d else 0
    dr_sign = dr // d if d else 0

    # Capture path Whites at steps 1..d-1.
    for k in range(1, d):
        f = chess.square_file(from_sq) + k * df_sign
        r = chess.square_rank(from_sq) + k * dr_sign
        sq = chess.square(f, r)
        if _owner(state.board, sq) == SQ_WHITE:
            state.board.remove_piece_at(sq)

    is_ram = _owner(state.board, to_sq) == SQ_WHITE
    if is_ram:
        state.stacks.pop(from_sq, None)
        state.board.remove_piece_at(from_sq)
        return

    _move_full_tower(state, from_sq, to_sq)


def _apply_chain_move(state: State, move: dict[str, Any]) -> None:
    """Apply a diagonal capture chain (single-hop or multi-hop).

    Uses the `_chain_all_captures`, `_is_suicide`, and `_chain_promotes`
    internal fields stored by `_build_final_move`. Effect:
      1. Remove all path-captured Whites from board.
      2. Remove the tower from origin.
      3. Promote the moving stack if chain crossed rank 1.
      4. Suicide: tower destroyed, done.
      5. Normal: place final stack at dest."""
    from_sq = chess.parse_square(move["from"])
    to_sq = chess.parse_square(move["to"])
    orig_stack = state.stacks[from_sq]

    # Remove path captures.
    for cap_name in move.get("_chain_all_captures", []):
        sq = chess.parse_square(cap_name)
        state.board.remove_piece_at(sq)

    # Remove tower from origin.
    state.stacks.pop(from_sq, None)
    state.board.remove_piece_at(from_sq)

    if move.get("_is_suicide"):
        # Tower destroyed at the ram landing — done.
        return

    final_stack = _promote_all_stones(orig_stack) if move.get("_chain_promotes") else orig_stack
    state.stacks[to_sq] = final_stack
    top = final_stack[-1]
    if top == "k":
        state.board.set_piece_at(to_sq, chess.Piece(chess.KING, chess.BLACK))
    else:
        state.board.set_piece_at(to_sq, chess.Piece(chess.PAWN, chess.BLACK))


def apply_black_move(state: State, uci: str) -> State:
    """Parse and apply a Black move. Looks up the UCI in the current legal
    move list (with mandate applied) and dispatches based on move type.

    Chain UCIs (containing `~`) are matched against chain moves. Plain
    UCIs that match a chain leaf (e.g. single-hop T-fallback like `h4e1`)
    are also handled. Raises ValueError if no match found."""
    legal = _all_black_legal(state)

    # Resolve UCI: exact match first; for chain-form UCIs that don't appear
    # verbatim, try matching by orig+dest, then by orig+chainHops-contains-dest
    # (handles rim-dest UCIs like 'h4~d0' where d0 is a rim key stored in
    # chainHops, not in "to" which holds the final board fallback).
    matches = [m for m in legal if m["uci"] == uci]
    if not matches and "~" in uci:
        parts = uci.split("~")
        orig_key, dest_key = parts[0], parts[-1]
        matches = [m for m in legal if m["from"] == orig_key and m["to"] == dest_key]
        if not matches:
            # dest_key may be a rim coordinate (like 'd0') stored in chainHops.
            matches = [
                m for m in legal
                if m["from"] == orig_key and m.get("chainHops") and dest_key in m["chainHops"]
            ]

    if not matches:
        raise ValueError(f"illegal or unrecognized Black move: {uci!r}")
    return apply_black_move_known(state, matches[0])


def apply_black_move_known(state: State, move: dict[str, Any]) -> State:
    """Apply a Black move dict that the caller already validated as legal in
    `state` — skips the `_all_black_legal` re-derivation that the UCI-string
    `apply_black_move` does to look the move up.

    Used by the MCTS hot path: each expansion already has the parent's full
    legal-move list (it just used those dicts to seed children), so re-running
    move-gen inside the apply step is pure waste. Roughly half of MCTS CPU
    cost was that redundancy."""
    new_state = state.copy()
    saved_castling = state.board.castling_rights

    if move.get("deployCount") is not None:
        _apply_deploy(new_state, move)
    elif move.get("chainHops") is not None:
        # A diagonal capture chain is identified by chainHops and MUST dispatch
        # here even when its net origin->landing displacement is orthogonal (a
        # chain can start and end on the same rank/file, e.g. c3~e1~g3). Checking
        # _is_orthogonal_move first would misroute it to _apply_charge, which
        # walks the straight origin->dest line and captures whatever sits on it
        # (e.g. a king two files away that the chain never touched).
        _apply_chain_move(new_state, move)
    elif _is_orthogonal_move(move):
        _apply_charge(new_state, move)
    elif move.get("capture") is not None:
        _apply_diagonal_capture(new_state, move)
    else:
        _apply_quiet_or_sprint(new_state, move)

    # scalachess Chessckers doesn't update castling rights during Black's turn
    # (they're a White-only concept in this variant). python-chess strips them
    # when pieces are removed, so restore the original rights explicitly.
    new_state.board.castling_rights = saved_castling
    new_state.board.turn = chess.WHITE
    # Rank-8 counter (#3): a check from Black at any point resets it to 0. Only
    # worth probing when the counter is actually running (skips the cost normally).
    if new_state.rank8_count:
        from chessckers_engine.variant_py.moves_white import _is_white_in_chessckers_check
        if _is_white_in_chessckers_check(new_state):
            new_state.rank8_count = 0
    return new_state


def _is_orthogonal_move(move: dict[str, Any]) -> bool:
    fr = chess.parse_square(move["from"])
    to = chess.parse_square(move["to"])
    return (
        chess.square_file(fr) == chess.square_file(to)
        or chess.square_rank(fr) == chess.square_rank(to)
    ) and fr != to


def black_charge_moves(state: State) -> list[dict[str, Any]]:
    """Phase 2E — Charge (orthogonal capture).

    A King-top tower may move along a rank or file. Cost is one King
    demoted per square moved. Path Whites are captured for free; landing
    on a White piece is a ram (tower destroyed at landing, no landing
    capture). Friendly Black towers in the path block; landing on a
    friendly is a merge.

    A charge of distance `d` demotes the **bottom `d` Kings** (no player
    choice; v6 rule change). Each (from -> landing) charge is therefore a
    single move (no `{a,b,c}` suffix); `demotedKings` carries the forced
    bottom-`d` positions and `demotionsRequired` = `d`. Rams always emit a
    single move with null demotion fields (the tower dies at landing, so
    the choice is moot).

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
        from_name = _SQ_NAME[from_sq]
        king_positions = [i + 1 for i, p in enumerate(pieces) if p == "k"]

        for df, dr in _ORTHO_DIRS:
            stop_after = False  # set when a friendly tower mid-scan blocks further walk
            for d in range(1, n_kings + 1):
                if stop_after:
                    break
                # Scan path 1..d-1 for blockers / collect path captures fresh.
                # Per §3C revised: rim squares are allowed mid-path (no
                # pieces there, no captures); only off-grid (file/rank
                # outside [-1, 8] in board coords) invalidates the charge.
                blocked = False
                off_grid = False
                path_captures: list[str] = []
                last_on_board_sq: int | None = None
                for k in range(1, d):
                    pf = from_file + k * df
                    pr = from_rank + k * dr
                    if pf < -1 or pf > 8 or pr < -1 or pr > 8:
                        off_grid = True
                        break
                    if 0 <= pf <= 7 and 0 <= pr <= 7:
                        psq = chess.square(pf, pr)
                        powner = _owner(state.board, psq)
                        if powner == SQ_BLACK and psq in state.stacks:
                            blocked = True
                            break
                        if powner == SQ_WHITE:
                            path_captures.append(_SQ_NAME[psq])
                        last_on_board_sq = psq
                    # else: rim square, no action
                if off_grid:
                    # Higher d would also be off-grid in this direction.
                    break
                if blocked:
                    break
                tf = from_file + d * df
                tr = from_rank + d * dr
                if tf < -1 or tf > 8 or tr < -1 or tr > 8:
                    break  # off-grid landing; higher d also off-grid
                # Determine landing classification: board / rim-with-fallback.
                # `rim_landing_key` is the on-grid key of the actual rim
                # landing for an overshoot charge (None for an on-board land);
                # it disambiguates the notation (`e2e0->e1` vs a ram `e2e1`)
                # and flags the move for apply.
                rim_landing_key: str | None = None
                if 0 <= tf <= 7 and 0 <= tr <= 7:
                    to_sq = chess.square(tf, tr)
                    to_name = _SQ_NAME[to_sq]
                    towner = _owner(state.board, to_sq)
                    is_ram = towner == SQ_WHITE
                    is_friendly_merge = (
                        towner == SQ_BLACK and to_sq in state.stacks
                    )
                else:
                    # Rim landing → fallback to last on-board square (per §3C
                    # revised). If no path step landed on the board (i.e. d=1
                    # rim, impossible since origin is on board so d=1 lands
                    # adjacent), there's no fallback target — skip.
                    if last_on_board_sq is None:
                        # d=1 charge with rim landing would imply origin is
                        # one square from rim AND the only step IS rim;
                        # fallback square == origin, which is a no-op.
                        continue
                    rim_landing_key = _coord_to_key(tf, tr)
                    to_sq = last_on_board_sq
                    to_name = _SQ_NAME[to_sq]
                    # The fallback square is always empty: either it was
                    # empty originally, or it was a white captured during
                    # the path traversal. Treat as a normal empty landing.
                    towner = SQ_EMPTY
                    is_ram = False
                    is_friendly_merge = False

                # Notation: an overshoot charge spells out the rim landing it
                # aimed at, then `->` its on-board resting square, so the
                # intent (charge to the rim, capturing in transit) is explicit
                # and never reads as a ram. `waypoints` carries the rim key —
                # it is both the apply flag and the policy-encoding key.
                landing_repr = (
                    to_name if rim_landing_key is None
                    else f"{rim_landing_key}->{to_name}"
                )
                charge_waypoints = None if rim_landing_key is None else [rim_landing_key]

                # Capture-field convention (scalachess parity).
                if path_captures:
                    capture_field: str | None = path_captures[0]
                elif is_ram:
                    capture_field = to_name
                else:
                    capture_field = None

                if is_ram:
                    # Per §3C (revised): rams require at least one path
                    # capture — the charge must overshoot at least one enemy
                    # before crashing. A distance-1 ram (no intermediate
                    # squares) has zero path captures and is illegal.
                    if path_captures:
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
                if is_friendly_merge and len(state.stacks.get(to_sq, "")) + len(pieces) > MAX_TOWER_HEIGHT:
                    stop_after = True
                    continue

                # v6 rule change: a charge of distance d demotes the BOTTOM d
                # Kings — no player choice. king_positions is ascending from the
                # bottom, so king_positions[:d] is the bottom d. This preserves
                # the upper Kings (keeps the tower King-top) and collapses each
                # (from -> landing) charge to ONE move (no {choice} suffix),
                # removing the old C(n_kings, d) demotion fan-out. When n_kings==d
                # this is exactly the old forced-demote-all case.
                chosen = king_positions[:d]
                new_pieces = list(pieces)
                for pos in chosen:
                    new_pieces[pos - 1] = "S"
                resulting_top = new_pieces[-1]
                moves.append({
                    "uci": f"{from_name}{landing_repr}",
                    "from": from_name,
                    "to": to_name,
                    "piece": _piece_name(resulting_top),
                    "color": "black",
                    "capture": capture_field,
                    "waypoints": charge_waypoints,
                    "chainHops": None,
                    "promotion": None,
                    "demotedKings": chosen,
                    "demotionsRequired": d,
                    "sourceKingPositions": list(king_positions),
                    "deployCount": None,
                })

                if is_friendly_merge:
                    # Friendly landing blocks further scanning beyond this d.
                    stop_after = True

    return moves


def _deploy_move(from_name: str, to_sq: chess.Square, top: str, s: int) -> dict[str, Any]:
    to_name = _SQ_NAME[to_sq]
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
