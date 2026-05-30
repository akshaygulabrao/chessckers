"""Golden-string tests for the terminal board renderer.

These pin the exact text output so the renderer can be verified without a
human eyeballing it — what we assert here is character-for-character what
prints to the terminal.
"""
from __future__ import annotations

from chessckers_engine.render_board import render_board
from chessckers_engine.variant_py.state import STARTING_FEN

# A tall king-top stack at a7 (ssk = stone, stone, king-on-top) to exercise
# variable-width alignment.
TALL_FEN = (
    "pppppppp/kkkkkkkk/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
    "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,"
    "a7:ssk,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:k,"
    "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1"
)


def test_starting_position_renders_exactly():
    expected = "\n".join([
        "   z  a  b  c  d  e  f  g  h  i ",
        " 9 ·  ·  ·  ·  ·  ·  ·  ·  ·  · ",
        " 8 ·  s  s  s  s  s  s  s  s  · ",
        " 7 ·  k  k  k  k  k  k  k  k  · ",
        " 6 ·  s  s  s  s  s  s  s  s  · ",
        " 5 ·  .  .  .  .  .  .  .  .  · ",
        " 4 ·  .  .  .  .  .  .  .  .  · ",
        " 3 ·  .  .  .  .  .  .  .  .  · ",
        " 2 ·  P  P  P  P  P  P  P  P  · ",
        " 1 ·  R  N  B  Q  K  B  N  R  · ",
        " 0 ·  ·  ·  ·  ·  ·  ·  ·  ·  · ",
        "   z  a  b  c  d  e  f  g  h  i ",
    ])
    assert render_board(STARTING_FEN) == expected


def test_tall_stack_widens_grid_and_keeps_king_on_top():
    out = render_board(TALL_FEN)
    rank7 = [ln for ln in out.splitlines() if ln.startswith(" 7 ")][0]
    # rightmost char of the stack string is the top piece -> "ssk" = king on top
    assert "ssk" in rank7
    # every column is padded to width 3 once a 3-char stack is present
    header = out.splitlines()[0]
    assert " z   a   b " in header


def test_rim_bounce_path_numbers_steps_including_rim_cell():
    # c3 -> b4 -> a5 -> z6 (LEFT RIM) -> a7 (reflected back onto board)
    out = render_board(STARTING_FEN, path=["c3", "b4", "a5", "z6", "a7"])
    lines = {ln[:2].strip(): ln for ln in out.splitlines() if ln[:2].strip().isdigit()}
    # step 4 lands on the left rim (column z) at rank 6
    rank6 = lines["6"]
    assert rank6.split()[1] == "4"  # first cell after the rank label is column z
    # step 5 reflected back onto the board at a7
    assert "5" in lines["7"]


def test_unknown_path_cells_are_ignored():
    # malformed cells (steps 1-3) must not crash or shift the grid; only the
    # valid cell c3 (step 4) gets numbered. Inspect the c3 cell directly —
    # substring checks are confounded by the numeric rank labels.
    out = render_board(STARTING_FEN, path=["zz", "a", "q9", "c3"])
    rank3 = [ln for ln in out.splitlines() if ln[:2].strip() == "3"][0]
    cells = rank3.split()  # [rank_label, z, a, b, c, d, ...]
    assert cells[4] == "4"  # column c (z,a,b,c -> offset 4) holds step 4
    # a clean render of the same position has a board-empty '.' at c3
    assert render_board(STARTING_FEN).splitlines()[7].split()[4] == "."
