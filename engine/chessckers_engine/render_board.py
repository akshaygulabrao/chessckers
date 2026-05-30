"""Render a Chessckers position as a terminal board.

Unlike the python-chess ASCII board (which drops the stack overlay), this
renders the Chessckers-specific state we actually need to verify by eye:

  - Black stacks shown as their literal stack string (`s`=stone, `k`=king),
    read left-to-right = bottom-to-top, so the RIGHTMOST char is the piece
    on top (the one that moves). Moved/unmoved stones are NOT distinguished.
  - White pieces shown as standard uppercase letters (P N B R Q K).
  - The full 10x10 grid: the inner 8x8 board PLUS the 1-square rim ring used
    by diagonal capture chains (reflections / corner retroreflect / rim
    landings). Rim files are `z` (left) and `i` (right); rim ranks are `0`
    (bottom) and `9` (top). This matches the waypoint coordinate convention
    in encoding.py (`_FILE10`/`_RANK10`).

Optionally overlays a chain/move path as numbered steps on any cell,
including rim cells, so a bounce off the boundary is visible.
"""
from __future__ import annotations

import chess

from .variant_py.state import parse_fen

# 10x10 grid coordinate convention, mirroring encoding.py.
# File char -> column 0..9; 'z' and 'i' are the rim, 'a'..'h' are the board.
_FILE10 = {"z": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9}
_FILE_LABELS = ["z", "a", "b", "c", "d", "e", "f", "g", "h", "i"]

RIM_EMPTY = "·"   # rim cell, no piece (rim is transient — never holds a resting piece)
BOARD_EMPTY = "."  # board cell, no piece


def _board_cell(state, file: int, rank: int) -> str:
    """Content for an inner board square (file/rank are 0..7)."""
    sq = chess.square(file, rank)
    stack = state.stacks.get(sq)
    if stack:
        # Overlay string is bottom-to-top; keep order so rightmost = top.
        # Drop the moved/unmoved distinction (S -> s) per display settings.
        return stack.lower()
    piece = state.board.piece_at(sq)
    if piece is not None:
        # White pieces only (every Black square carries a stack overlay).
        return piece.symbol().upper()
    return BOARD_EMPTY


def _parse_path_cell(cell: str) -> tuple[int, int] | None:
    """Map a 2-char grid cell name like 'c5' or 'z6' to (col 0..9, row 0..9)."""
    if len(cell) != 2:
        return None
    col = _FILE10.get(cell[0])
    if col is None or not cell[1].isdigit():
        return None
    row = int(cell[1])
    if not 0 <= row <= 9:
        return None
    return col, row


def render_board(fen: str, path: list[str] | None = None) -> str:
    """Return a terminal-printable 10x10 board for `fen`.

    `path` is an optional ordered list of grid cells (e.g.
    `["c5", "d6", "z7", "b6"]`) overlaid as numbered steps 1..n — used to
    visualize a capture chain, including where it touches the rim.
    """
    state = parse_fen(fen)

    # grid[row][col], row/col in 0..9. row 0 = rank 0 (bottom rim), row 9 = top rim.
    grid: list[list[str]] = []
    for row in range(10):
        line: list[str] = []
        for col in range(10):
            is_board = 1 <= col <= 8 and 1 <= row <= 8
            if is_board:
                line.append(_board_cell(state, col - 1, row - 1))
            else:
                line.append(RIM_EMPTY)
        grid.append(line)

    # Overlay the numbered path (steps win over whatever was in the cell).
    if path:
        for step, cell in enumerate(path, start=1):
            pos = _parse_path_cell(cell)
            if pos is None:
                continue
            col, row = pos
            grid[row][col] = str(step)

    width = max(2, max(len(c) for line in grid for c in line))

    def fmt_row(label: str, cells: list[str]) -> str:
        body = " ".join(c.center(width) for c in cells)
        return f"{label:>2} {body}"

    header = "   " + " ".join(lbl.center(width) for lbl in _FILE_LABELS)
    out = [header]
    for row in range(9, -1, -1):  # top (rank 9) down to bottom (rank 0)
        out.append(fmt_row(str(row), grid[row]))
    out.append(header)
    return "\n".join(out)


def main() -> int:
    import argparse

    from .variant_py.state import STARTING_FEN

    p = argparse.ArgumentParser(description="Render a Chessckers FEN to the terminal.")
    p.add_argument("fen", nargs="?", default=STARTING_FEN, help="Chessckers FEN (default: start)")
    p.add_argument("--path", default=None,
                   help="comma-separated grid cells to overlay as a numbered path, e.g. c5,d6,z7,b6")
    args = p.parse_args()
    path = args.path.split(",") if args.path else None
    print(render_board(args.fen, path=path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
