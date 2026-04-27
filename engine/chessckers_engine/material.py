"""Material counting for Chessckers positions.

Piece values:
  White:  P=1, N=3, B=3, R=5, Q=9, K=1000
  Black:  Stone (s or S) = 1, King (k) = 2

`material(fen)` returns `(sum of White) - (sum of Black)`. Positive favors White.

`material_for_side_to_move(fen)` returns the same value flipped if Black is to move,
so both colors can maximize during training and inference.

Black piece counts come from the FEN bracket overlay (the bitboard only encodes
the top of each tower). Use this module on FENs returned by the API after a
move has been applied; the M4 training target is the material score of the
resulting position from the side-to-move's perspective.
"""

from __future__ import annotations

import re

WHITE_VALUES: dict[str, int] = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 1000}
BLACK_STONE_VALUE = 1
BLACK_KING_VALUE = 2

_FEN_HEAD = re.compile(r"^([^\s\[]+)(?:\[([^\]]*)\])?\s+([wb])\b")


def material(fen: str) -> int:
    m = _FEN_HEAD.match(fen)
    if not m:
        raise ValueError(f"unrecognized FEN: {fen!r}")
    board, overlay = m.group(1), m.group(2)

    white = sum(WHITE_VALUES[ch] for ch in board if ch in WHITE_VALUES)

    black = 0
    if overlay:
        for entry in overlay.split(","):
            if ":" not in entry:
                continue
            _sq, pieces = entry.split(":", 1)
            for p in pieces:
                if p in "sS":
                    black += BLACK_STONE_VALUE
                elif p == "k":
                    black += BLACK_KING_VALUE

    return white - black


def material_for_side_to_move(fen: str) -> int:
    m = _FEN_HEAD.match(fen)
    if not m:
        raise ValueError(f"unrecognized FEN: {fen!r}")
    turn = m.group(3)
    raw = material(fen)
    return raw if turn == "w" else -raw
