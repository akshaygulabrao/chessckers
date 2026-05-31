"""Chessckers endgame tablebase package.

Phase 1 scope: White = lone King, Black = a bounded total of pieces (Stones +
Kings) organized as towers. Positions are solved exactly bottom-up by total
Black piece count via a retrograde fixpoint (see `tablebase.py` for the
reference FEN-keyed solver; this package holds the reusable representation).
"""
from tb.model import (
    Value,
    MaterialClass,
    side_to_move,
    black_total,
    terminal_value,
)
from tb.enumerate import (
    tower_compositions,
    integer_partitions,
    tower_partitions,
    build_fen,
    enumerate_level,
    mirror_fen,
    canonical_fen,
)

__all__ = [
    "Value",
    "MaterialClass",
    "side_to_move",
    "black_total",
    "terminal_value",
    "tower_compositions",
    "integer_partitions",
    "tower_partitions",
    "build_fen",
    "enumerate_level",
    "mirror_fen",
    "canonical_fen",
]
