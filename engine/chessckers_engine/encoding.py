"""FEN and LegalMove tensor encodings for the Chessckers neural-net player.

Position tensor: shape (14, 8, 8), dtype float32. Channels:
   0  White Pawn          (one-hot per square from board bitboard)
   1  White Knight
   2  White Bishop
   3  White Rook
   4  White Queen
   5  White King
   6  Stone-top           (Chessckers convention: Black-Pawn bitboard = Stone top)
   7  King-top            (Black-King bitboard = King top)
   8  tower_height        (len(stack) / 24)
   9  stone_count         (count(Stone) / 24)
  10  king_count          (count(King) / 24)
  11  top_is_unmoved_stone (1 iff stack top is unmoved Stone 's')
  12  second_is_king      (1 iff stack[-2] is a King)
  13  side_to_move        (all-1 if Black to move, else all-0)

Squares use (file, rank) → tensor index (rank, file). Internal y axis runs
0 (rank 1) to 7 (rank 8); FEN ranks are read top-to-bottom, so FEN's first
rank string corresponds to y = 7.

Move feature vector: shape (240,), dtype float32:
   bits   0..63   from_square one-hot
   bits  64..127  to_square one-hot
   bit  128       is_capture
   bit  129       is_chain         (waypoints non-empty)
   bit  130       is_deploy        (deployCount set)
   bit  131       is_ortho         (demotionsRequired set)
   bit  132       chain_length / 8
   bit  133       deploy_count / 24
   bit  134       demotions_required / 8
   bits 135..139  promotion one-hot over (none, q, r, b, n)
   bits 140..239  waypoint mask over the 10×10 grid (board + 1-square rim).
                  Index = rank10 * 10 + file10, where file10 maps
                  'z'→0, 'a'→1 ... 'h'→8, 'i'→9 and rank10 maps
                  '0'→0, '1'→1 ... '9'→9. Lets the policy distinguish two
                  chains with identical from→to but different intermediate
                  landings (including rim-square hops, which are common
                  diagonal-capture waypoints).
"""

from __future__ import annotations

import re
from typing import Any

import torch

POS_C = 14
MOVE_D = 240

# Position channel indices
CH_W_PAWN, CH_W_KNIGHT, CH_W_BISHOP, CH_W_ROOK, CH_W_QUEEN, CH_W_KING = range(6)
CH_STONE_TOP, CH_KING_TOP = 6, 7
CH_TOWER_HEIGHT, CH_STONE_COUNT, CH_KING_COUNT = 8, 9, 10
CH_TOP_IS_UNMOVED_STONE, CH_SECOND_IS_KING = 11, 12
CH_SIDE_TO_MOVE = 13

_WHITE_PIECE_CH = {
    "P": CH_W_PAWN,
    "N": CH_W_KNIGHT,
    "B": CH_W_BISHOP,
    "R": CH_W_ROOK,
    "Q": CH_W_QUEEN,
    "K": CH_W_KING,
}
_BLACK_BITBOARD_CH = {"p": CH_STONE_TOP, "k": CH_KING_TOP}

# Move feature offsets
MV_CAPTURE = 128
MV_CHAIN = 129
MV_DEPLOY = 130
MV_ORTHO = 131
MV_CHAIN_LEN = 132
MV_DEPLOY_COUNT = 133
MV_DEMOTIONS_REQ = 134
MV_PROMO_BASE = 135  # 5 entries: none, q, r, b, n
MV_WAYPOINT_BASE = 140  # 100 bits over the 10×10 grid (board + rim)
_PROMO_INDEX = {None: 0, "q": 1, "r": 2, "b": 3, "n": 4}

# 10×10 grid char → 0..9 maps. Files: 'z' (rim) = 0, 'a'..'h' = 1..8, 'i' (rim) = 9.
# Ranks: '0' (rim) = 0, '1'..'8' = 1..8, '9' (rim) = 9. See moves_black.py
# `_coord_to_key` for the producing convention.
_FILE10 = {"z": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9}
_RANK10 = {str(i): i for i in range(10)}

_FEN_HEAD = re.compile(r"^([^\s\[]+)(?:\[([^\]]*)\])?\s+([wb])\b")


def square_xy(square: str) -> tuple[int, int]:
    """Return (file 0..7, rank 0..7) for a square name like 'a1' or 'h8'."""
    return ord(square[0]) - ord("a"), int(square[1]) - 1


def square_index(square: str) -> int:
    """Return 0..63 with a1=0, b1=1, ..., h8=63."""
    f, r = square_xy(square)
    return r * 8 + f


def encode_position(fen: str) -> torch.Tensor:
    """Encode a Chessckers FEN as a (14, 8, 8) float32 tensor."""
    m = _FEN_HEAD.match(fen)
    if not m:
        raise ValueError(f"unrecognized Chessckers FEN: {fen!r}")
    board, overlay, turn = m.group(1), m.group(2), m.group(3)

    out = torch.zeros((POS_C, 8, 8), dtype=torch.float32)

    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN board must have 8 ranks: {board!r}")
    for fen_rank_idx, rank_str in enumerate(ranks):
        y = 7 - fen_rank_idx
        x = 0
        for ch in rank_str:
            if ch.isdigit():
                x += int(ch)
                continue
            if ch in _WHITE_PIECE_CH:
                out[_WHITE_PIECE_CH[ch], y, x] = 1.0
            elif ch in _BLACK_BITBOARD_CH:
                out[_BLACK_BITBOARD_CH[ch], y, x] = 1.0
            # Other Black piece glyphs (n/b/r/q) shouldn't appear in Chessckers; ignored.
            x += 1

    if overlay:
        for entry in overlay.split(","):
            if ":" not in entry:
                continue
            sq, pieces = entry.split(":", 1)
            x, y = square_xy(sq)
            height = len(pieces)
            if height == 0:
                continue
            stones = sum(1 for p in pieces if p in "sS")
            kings = sum(1 for p in pieces if p == "k")
            out[CH_TOWER_HEIGHT, y, x] = height / 24.0
            out[CH_STONE_COUNT, y, x] = stones / 24.0
            out[CH_KING_COUNT, y, x] = kings / 24.0
            if pieces[-1] == "s":
                out[CH_TOP_IS_UNMOVED_STONE, y, x] = 1.0
            if height >= 2 and pieces[-2] == "k":
                out[CH_SECOND_IS_KING, y, x] = 1.0

    if turn == "b":
        out[CH_SIDE_TO_MOVE].fill_(1.0)

    return out


# Map python-chess piece (color, piece_type) to position channel index.
# Built lazily so this module doesn't hard-import `chess` at top level.
_BB_TO_CH: dict[tuple[bool, int], int] | None = None


def _bb_to_ch_table() -> dict[tuple[bool, int], int]:
    global _BB_TO_CH
    if _BB_TO_CH is None:
        import chess
        _BB_TO_CH = {
            (chess.WHITE, chess.PAWN): CH_W_PAWN,
            (chess.WHITE, chess.KNIGHT): CH_W_KNIGHT,
            (chess.WHITE, chess.BISHOP): CH_W_BISHOP,
            (chess.WHITE, chess.ROOK): CH_W_ROOK,
            (chess.WHITE, chess.QUEEN): CH_W_QUEEN,
            (chess.WHITE, chess.KING): CH_W_KING,
            (chess.BLACK, chess.PAWN): CH_STONE_TOP,
            (chess.BLACK, chess.KING): CH_KING_TOP,
        }
    return _BB_TO_CH


def encode_position_state(state: Any) -> torch.Tensor:
    """Encode a `variant_py.State` directly to the (14, 8, 8) tensor without
    going through a FEN serialize+parse round-trip. ~3-5x faster per leaf
    eval, and avoids re-parsing what the engine already has in memory.

    Reads python-chess bitboards via `piece_map()` (a single dict iteration
    over occupied squares) instead of walking a FEN string."""
    import chess  # local import keeps top-level light
    out = torch.zeros((POS_C, 8, 8), dtype=torch.float32)
    bb_to_ch = _bb_to_ch_table()
    board = state.board

    for sq, piece in board.piece_map().items():
        ch = bb_to_ch.get((piece.color, piece.piece_type))
        if ch is None:
            continue  # silently ignore stray Black non-king/pawn glyphs
        y = sq >> 3      # chess.square_rank(sq)
        x = sq & 7       # chess.square_file(sq)
        out[ch, y, x] = 1.0

    for sq, pieces in state.stacks.items():
        height = len(pieces)
        if height == 0:
            continue
        y = sq >> 3
        x = sq & 7
        # Iterate once for both stones and kings rather than two list-comps.
        stones = 0
        kings = 0
        for p in pieces:
            if p == "k":
                kings += 1
            else:
                # 's' or 'S'
                stones += 1
        out[CH_TOWER_HEIGHT, y, x] = height / 24.0
        out[CH_STONE_COUNT, y, x] = stones / 24.0
        out[CH_KING_COUNT, y, x] = kings / 24.0
        if pieces[-1] == "s":
            out[CH_TOP_IS_UNMOVED_STONE, y, x] = 1.0
        if height >= 2 and pieces[-2] == "k":
            out[CH_SECOND_IS_KING, y, x] = 1.0

    if board.turn == chess.BLACK:
        out[CH_SIDE_TO_MOVE].fill_(1.0)

    return out


def encode_move(move: dict[str, Any]) -> torch.Tensor:
    """Encode a LegalMove dict (as returned by the API) as a (MOVE_D,) float32 vector."""
    out = torch.zeros(MOVE_D, dtype=torch.float32)
    out[square_index(move["from"])] = 1.0
    out[64 + square_index(move["to"])] = 1.0

    if move.get("capture") is not None:
        out[MV_CAPTURE] = 1.0
    waypoints = move.get("waypoints") or []
    if waypoints:
        out[MV_CHAIN] = 1.0
    if move.get("deployCount") is not None:
        out[MV_DEPLOY] = 1.0
    if move.get("demotionsRequired") is not None:
        out[MV_ORTHO] = 1.0

    out[MV_CHAIN_LEN] = len(waypoints) / 8.0
    out[MV_DEPLOY_COUNT] = (move.get("deployCount") or 0) / 24.0
    out[MV_DEMOTIONS_REQ] = (move.get("demotionsRequired") or 0) / 8.0

    promo = move.get("promotion")
    out[MV_PROMO_BASE + _PROMO_INDEX.get(promo, 0)] = 1.0

    # Waypoint mask over the 10×10 grid. Two distinct chain paths from the
    # same from→to pair have different waypoint sets (whether on-board or
    # rim), so they encode to different vectors — without this, the policy
    # head can't tell them apart and gets contradictory training labels.
    for w in waypoints:
        if len(w) != 2:
            continue
        f10 = _FILE10.get(w[0])
        r10 = _RANK10.get(w[1])
        if f10 is None or r10 is None:
            continue
        out[MV_WAYPOINT_BASE + r10 * 10 + f10] = 1.0
    return out
