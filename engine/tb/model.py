"""Position representation and classification for the endgame tablebase.

A position's *value* is `(wdl, dtm)` from the side-to-move perspective:
  wdl in {+1 win, 0 draw, -1 loss}; dtm = plies to mate (None for draws).

Positions are grouped into *material classes* keyed by total Black piece count
(Phase 1 fixes White = lone King). Total piece count is monotonically
non-increasing under legal play — only captures/rams reduce it — so classes are
solved bottom-up, lowest total first.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

from chessckers_engine.variant_py.state import parse_fen

# Value: (wdl, dtm). wdl: +1 win, 0 draw, -1 loss. dtm: plies (None for draw).
Value = tuple[int, int | None]


@dataclass(frozen=True, slots=True)
class MaterialClass:
    """A solvable class of positions. Phase 1: White is always a lone King, so
    a class is fully identified by the total Black piece count."""

    black_total: int

    def key(self) -> str:
        return f"N{self.black_total}"


def side_to_move(fen: str) -> str:
    return "white" if parse_fen(fen).board.turn == chess.WHITE else "black"


def black_total(fen: str) -> int:
    """Total Black pieces (sum of tower heights) in a FEN."""
    return sum(len(t) for t in parse_fen(fen).stacks.values())


def terminal_value(status: str, winner: str | None, mover: str) -> Value:
    """Map a terminal status/winner to (wdl, dtm) from the `mover`'s view."""
    if status == "stalemate" or winner is None:
        return (0, None)
    if winner == mover:
        return (1, 0)
    return (-1, 0)
