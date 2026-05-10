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
    this and the differential tests require parity."""
    if state.board.turn != chess.WHITE:
        return []
    out: list[dict[str, Any]] = []
    for move in state.board.legal_moves:
        out.append(_to_scala_move(state.board, move))
        if state.board.is_castling(move):
            out.append(_castling_alt_form(state.board, move))
    return out


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
