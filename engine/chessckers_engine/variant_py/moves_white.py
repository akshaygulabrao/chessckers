"""White-side move generation.

The variant spec says White plays standard FIDE chess: pawn, knight,
bishop, rook, queen, king moves with all the usual rules (castling, en
passant, promotion, can't-leave-own-king-in-check). The bitboard already
encodes Black "stacks" as black pawns / kings, so python-chess's
`Board.legal_moves` correctly handles them as blockers and capture
targets — we just convert each python-chess Move into scalachess's
LegalMove dict format.
"""

from __future__ import annotations

from typing import Any

import chess

from chessckers_engine.variant_py.state import State

_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

_PROMOTION_NAMES = {
    chess.QUEEN: "queen",
    chess.ROOK: "rook",
    chess.BISHOP: "bishop",
    chess.KNIGHT: "knight",
}

# Precomputed square-name table — replaces 5 `chess.square_name(sq)` calls in
# the white-move emit path.
_SQ_NAME: tuple[str, ...] = tuple(chess.square_name(i) for i in range(64))


def white_legal_moves(state: State) -> list[dict[str, Any]]:
    """All legal White chess moves at the current position, in scalachess's
    LegalMove dict format. Empty list if Black is to move.

    Castling is emitted twice: once as the standard UCI (`e1g1`/`e1c1`)
    and once as the king-to-rook form (`e1h1`/`e1a1`) — scalachess does
    this and the differential tests require parity.

    `state.board.legal_moves` would over-filter: python-chess treats the
    Chessckers Black-King encoding (which is really a 4-direction-only
    king-top stack) as a standard 8-direction chess king. So it rejects
    white moves that "leave the king in check" even when the Black king-top
    can't actually attack those squares per Chessckers rules. We use
    `pseudo_legal_moves` instead and filter via the correct Chessckers-aware
    check predicate (defined below)."""
    if state.board.turn != chess.WHITE:
        return []
    if _rs_movegen is not None:
        # Native fast path: one Rust call generates + check-filters all White
        # moves on its own bitboards, eliminating the per-pseudo-move
        # python-chess apply/copy churn that dominated self-play (see the
        # equivalence harness tests/test_white_rust_equiv.py for parity).
        b = state.board
        return _rs_movegen.white_legal_moves(
            b.occupied,
            b.occupied_co[chess.WHITE],
            b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
            b.castling_rights,
            -1 if b.ep_square is None else b.ep_square,
            state.stacks,
        )
    out: list[dict[str, Any]] = []
    for move in state.board.pseudo_legal_moves:
        # Apply tentatively and reject if it leaves white-king in
        # Chessckers-check. Castling additionally requires that none of the
        # squares the king passes through is attacked (per chess.is_castling
        # rules); python-chess already enforces that on pseudo_legal output,
        # but we need to re-verify under the Chessckers attack model.
        new_state = apply_white_move(state, move.uci())
        if _is_white_in_chessckers_check(new_state):
            continue
        if state.board.is_castling(move):
            # Castling: also reject if any square the king *crosses* is
            # under Chessckers attack (e1->f1->g1 for kingside; e1->d1->c1
            # for queenside). Pseudo-legal castles already require the rook
            # path is clear; we add the king-path-attack check here.
            if _castling_path_attacked_chessckers(state, move):
                continue
        out.append(_to_scala_move(state.board, move))
        if state.board.is_castling(move):
            out.append(_castling_alt_form(state.board, move))
    return out


# ---- Chessckers-correct check / attack detection ----
#
# The Black-King encoding on the bitboard is a king-top stack; it attacks
# only diagonally (1..n where n=stack_height). Charge (orthogonal) attacks
# work via path-capture only — landing on a White piece is a "ram" that
# destroys the tower without capturing the white piece (per the spec).
# So a 1-king king-top can NEVER capture an adjacent orthogonal white piece,
# despite python-chess thinking otherwise.

# Direction tables (mirror moves_black to avoid a circular import).
_FORWARD_DIAGS_BLACK = [(-1, -1), (1, -1)]
_ALL_DIAGS = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
_ORTHO_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

try:
    import chessckers_movegen as _rs_movegen  # type: ignore[import-not-found]
except ImportError:
    _rs_movegen = None


def _on_board(f: int, r: int) -> bool:
    return 0 <= f <= 7 and 0 <= r <= 7


def _square_owner(board: chess.Board, sq: int) -> int:
    """0 = empty, 1 = white, 2 = black. Inline-equivalent of moves_black._owner."""
    mask = chess.BB_SQUARES[sq]
    if not (board.occupied & mask):
        return 0
    if board.occupied_co[chess.WHITE] & mask:
        return 1
    return 2


def _square_attacked_by_black_chessckers(state: State, target_sq: int) -> bool:
    if _rs_movegen is not None:
        return _rs_movegen.square_attacked_by_black_chessckers(
            state.board.occupied,
            state.board.occupied_co[chess.WHITE],
            state.stacks,
            target_sq,
        )
    return _square_attacked_by_black_chessckers_py(state, target_sq)


def _square_attacked_by_black_chessckers_py(state: State, target_sq: int) -> bool:
    """True iff any Black stack can capture a White piece at `target_sq` next
    turn under Chessckers rules.

    Attack mechanics:
      - Diagonal: stone-top forward 1..n, king-top all 4 dirs 1..n. Path
        through `target_sq` (or landing on it as suicide) = capture. White
        pieces in path are free path-captures and DO NOT block. Friendly
        Black DOES block.
      - Orthogonal charge (king-top only, n_kings ≥ 2): path squares
        1..n_kings-1 in any orthogonal direction = capture if the charge
        can extend at least one square past the target (so landing is
        on-board). White path = free path-cap, doesn't block. Friendly
        Black blocks.

    Doesn't model rim-bounce diagonals — those produce additional attack
    squares for tall towers in corner positions. Can be added later if
    needed; for the canonical-piece-near-king check pattern this is
    sufficient.
    """
    board = state.board
    stacks = state.stacks
    for from_sq, pieces in stacks.items():
        if not pieces:
            continue
        n = len(pieces)
        is_king_top = pieces[-1] == "k"
        n_kings = sum(1 for p in pieces if p == "k") if is_king_top else 0
        sf = chess.square_file(from_sq)
        sr = chess.square_rank(from_sq)

        # Diagonal walk: target reachable within n squares with no friendly
        # Black blocking the path.
        diag_dirs = _ALL_DIAGS if is_king_top else _FORWARD_DIAGS_BLACK
        for df, dr in diag_dirs:
            for k in range(1, n + 1):
                nf = sf + k * df
                nr = sr + k * dr
                if not _on_board(nf, nr):
                    break
                nsq = chess.square(nf, nr)
                if nsq == target_sq:
                    return True
                o = _square_owner(board, nsq)
                if o == 2 and nsq in stacks:
                    break  # friendly black tower blocks further reach
                # White or empty: continue walking (white = free path-cap).

        # Charge: orthogonal, path squares 1..n_kings-1 (landing = ram).
        if is_king_top and n_kings >= 2:
            for df, dr in _ORTHO_DIRS:
                for k in range(1, n_kings):
                    nf = sf + k * df
                    nr = sr + k * dr
                    if not _on_board(nf, nr):
                        break
                    nsq = chess.square(nf, nr)
                    if nsq == target_sq:
                        # Need a legal landing past the target (charge to k+1
                        # must reach an on-board square).
                        nf2 = sf + (k + 1) * df
                        nr2 = sr + (k + 1) * dr
                        if _on_board(nf2, nr2):
                            return True
                        break
                    o = _square_owner(board, nsq)
                    if o == 2 and nsq in stacks:
                        break
    return False


def _is_white_in_chessckers_check(state: State) -> bool:
    """White king is in Chessckers check iff Black, to move, has a capture that
    captures the white king's square.

    The targeted attack model (`_square_attacked_by_black_chessckers`) only
    covers single diagonals and orthogonal charges — NOT multi-hop diagonal
    chains or off-grid overshoots (§3B), where Black turns a chain around to
    reach the king. So we additionally scan the real diagonal capture
    generator, which models chains/overshoots correctly. A hit means the king
    is among a move's *path* captures — a ram landing on the king does NOT
    capture it, so path captures are exactly the right test. Defining check via
    the generator keeps it consistent with move-gen by construction; the
    charge case stays on the targeted model (charges don't chain, so it's
    complete there)."""
    king_sq = state.board.king(chess.WHITE)
    if king_sq is None:
        # King already captured — game-over status handled elsewhere.
        return False
    # Diagonal hops / chains / overshoots — reuse the correct generator. Probe
    # as if it were Black to move (this is asked on both white- and black-to-
    # move states).
    from chessckers_engine.variant_py import moves_black as _mb
    if state.board.turn == chess.BLACK:
        probe = state
    else:
        probe = state.copy()
        probe.board.turn = chess.BLACK
    # Hot path: a native bool early-exit that runs the same chain search but
    # stops at the first king-capturing hop and builds no move dicts. This is
    # called once per White candidate move in white_legal_moves, so avoiding the
    # full black_diagonal_capture_moves list (and its PyO3 marshalling) is the
    # difference between ~92µs and a few µs per call. Falls back to scanning the
    # generated list when the native extension is bypassed (gated on the same
    # _mb._rs_movegen the tests monkeypatch).
    if _mb._rs_movegen is not None:
        if _mb._rs_movegen.black_can_capture_white_king(
            probe.board.occupied,
            probe.board.occupied_co[chess.WHITE],
            king_sq,
            probe.stacks,
        ):
            return True
    else:
        king_name = _SQ_NAME[king_sq]
        for m in _mb.black_diagonal_capture_moves(probe):
            if king_name in (m.get("_chain_all_captures") or ()):
                return True
    # Single diagonals + orthogonal charges (charges don't chain).
    return _square_attacked_by_black_chessckers(state, king_sq)


def _castling_path_attacked_chessckers(state: State, move: chess.Move) -> bool:
    """True if any square the king crosses during castling is attacked under
    Chessckers rules. Excludes the king's destination (caught separately by
    the post-move check filter); checks origin + intermediate."""
    if not state.board.is_castling(move):
        return False
    rank = chess.square_rank(move.from_square)
    is_kingside = state.board.is_kingside_castling(move)
    # King passes from e -> f -> g (kingside) or e -> d -> c (queenside).
    file_path = (4, 5) if is_kingside else (4, 3)
    for f in file_path:
        sq = chess.square(f, rank)
        if _square_attacked_by_black_chessckers(state, sq):
            return True
    return False


def _castling_alt_form(board: chess.Board, move: chess.Move) -> dict[str, Any]:
    """Build the king-to-rook form of a castling move (`e1h1` for kingside,
    `e1a1` for queenside), with the same other fields as the standard form."""
    rank = chess.square_rank(move.from_square)
    rook_file = 7 if board.is_kingside_castling(move) else 0
    rook_sq = chess.square(rook_file, rank)
    from_name = _SQ_NAME[move.from_square]
    rook_name = _SQ_NAME[rook_sq]
    return {
        "uci": f"{from_name}{rook_name}",
        "from": from_name,
        "to": rook_name,
        "piece": "king",
        "color": "white",
        "capture": None,
        "waypoints": None,
        "chainHops": None,
        "promotion": None,
        "demotedKings": None,
        "demotionsRequired": None,
        "sourceKingPositions": None,
        "deployCount": None,
    }


def parse_white_uci(board: chess.Board, uci: str) -> chess.Move:
    """Parse a UCI string from scalachess into a python-chess Move.

    Translates king-to-rook castling notation (`e1h1`/`e1a1`) into the
    standard UCI form (`e1g1`/`e1c1`) so python-chess accepts it as a
    castling move."""
    move = chess.Move.from_uci(uci)
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.KING or piece.color != chess.WHITE:
        return move
    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    rank = chess.square_rank(move.from_square)
    # Only translate when king starts on e-file and target is the rook square (a/h).
    if from_file == 4 and to_file in (0, 7) and chess.square_rank(move.to_square) == rank:
        # Only treat as castling if the side has matching castling rights.
        if to_file == 7 and board.has_kingside_castling_rights(chess.WHITE):
            return chess.Move(move.from_square, chess.square(6, rank))
        if to_file == 0 and board.has_queenside_castling_rights(chess.WHITE):
            return chess.Move(move.from_square, chess.square(2, rank))
    return move


def apply_white_move(state: State, uci: str) -> State:
    """Apply a White move to the state and return the resulting State (a
    new instance — `state` is not mutated). Updates the stack overlay if
    the move captures a Black piece (the entire stack at the captured
    square is removed, per the Chessckers rule that capturing a Black
    stack eliminates the whole tower)."""
    new_state = state.copy()
    move = parse_white_uci(new_state.board, uci)

    if new_state.board.is_capture(move):
        if new_state.board.is_en_passant(move):
            ep_file = chess.square_file(move.to_square)
            ep_rank = chess.square_rank(move.from_square)
            captured_sq = chess.square(ep_file, ep_rank)
        else:
            captured_sq = move.to_square
        new_state.stacks.pop(captured_sq, None)

    new_state.board.push(move)
    return new_state


def _to_scala_move(board: chess.Board, move: chess.Move) -> dict[str, Any]:
    from_sq = _SQ_NAME[move.from_square]
    to_sq = _SQ_NAME[move.to_square]

    piece = board.piece_at(move.from_square)
    piece_name = _PIECE_NAMES.get(piece.piece_type, "unknown") if piece else "unknown"

    capture_sq: str | None = None
    if board.is_capture(move):
        if board.is_en_passant(move):
            ep_file = chess.square_file(move.to_square)
            ep_rank = chess.square_rank(move.from_square)
            capture_sq = _SQ_NAME[chess.square(ep_file, ep_rank)]
        else:
            capture_sq = to_sq

    promotion = _PROMOTION_NAMES.get(move.promotion) if move.promotion is not None else None

    return {
        "uci": move.uci(),
        "from": from_sq,
        "to": to_sq,
        "piece": piece_name,
        "color": "white",
        "capture": capture_sq,
        "waypoints": None,
        "chainHops": None,
        "promotion": promotion,
        "demotedKings": None,
        "demotionsRequired": None,
        "sourceKingPositions": None,
        "deployCount": None,
    }
