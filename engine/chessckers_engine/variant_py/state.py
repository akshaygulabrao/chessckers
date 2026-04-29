"""State representation: `chess.Board` for the bitboard, plus a Chessckers
stack overlay.

Per the variant spec, the bitboard encodes Stone-top stacks as Black-Pawn
and King-top stacks as Black-King — meaning python-chess already treats
Black squares correctly as blockers/captures during White's standard
chess move generation. The overlay carries the rest of each stack
(everything below the top piece) plus the moved-state bit on Stones.

Stack piece characters (bottom-to-top):
  's' = Stone, unmoved (eligible for back-rank sprint from rank 8)
  'S' = Stone, moved
  'k' = King

Invariant: for every square present in `stacks`, `stacks[sq][-1]` matches
the bitboard's piece on `sq`:
  's' / 'S' → black pawn (Stone top)
  'k'       → black king (King top)
The bitboard is truth for the top piece; the overlay is truth for
everything below it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import chess


# Canonical initial Chessckers FEN, preserving `KQkq` as scalachess emits at
# game start (after any move both engines strip `kq` since Chessckers Black
# has no chess king to castle).
STARTING_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:k,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)


# Captures: <board>[<overlay>] <rest>  |  <board> <rest>  |  <board>[<overlay>]  |  <board>
_FEN_HEAD_RE = re.compile(r"^([^\s\[]+)(?:\[([^\]]*)\])?(\s.*)?$")


@dataclass
class State:
    board: chess.Board
    stacks: dict[chess.Square, str] = field(default_factory=dict)

    def copy(self) -> "State":
        return State(board=self.board.copy(stack=False), stacks=dict(self.stacks))


def parse_fen(fen: str) -> State:
    """Parse a Chessckers FEN — standard 6-field FEN with an optional
    bracketed `[sq:pieces,sq:pieces,...]` overlay between the board and
    turn fields. Empty overlay is valid (`pos[] w - - 0 1`)."""
    head_match = _FEN_HEAD_RE.match(fen.strip())
    if not head_match:
        raise ValueError(f"unparseable Chessckers FEN: {fen!r}")
    board_str, overlay_str, rest_str = head_match.groups()
    rest_str = (rest_str or " w - - 0 1").strip()
    standard_fen = f"{board_str} {rest_str}"
    try:
        board = chess.Board(standard_fen)
    except ValueError as e:
        raise ValueError(f"chess.Board rejected board portion: {standard_fen!r} ({e})") from e

    stacks: dict[chess.Square, str] = {}
    if overlay_str:
        for entry in overlay_str.split(","):
            entry = entry.strip()
            if not entry:
                continue
            sq_name, _, pieces = entry.partition(":")
            try:
                sq = chess.parse_square(sq_name)
            except ValueError as e:
                raise ValueError(f"invalid square in overlay: {entry!r}") from e
            stacks[sq] = pieces

    return State(board=board, stacks=stacks)


def serialize_fen(state: State) -> str:
    """Inverse of parse_fen. Stacks are emitted in ascending square index
    order to match scalachess's canonical form."""
    chess_fen = state.board.fen()
    board_part, _, rest = chess_fen.partition(" ")
    if state.stacks:
        overlay = ",".join(
            f"{chess.square_name(sq)}:{pieces}"
            for sq, pieces in sorted(state.stacks.items())
        )
        return f"{board_part}[{overlay}] {rest}"
    return f"{board_part} {rest}"
