"""FEN and LegalMove tensor encodings for the Chessckers neural-net player.

Position tensor: shape (15, 8, 8), dtype float32. Channels:
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
  14  rank8_progress      (rank8_count / 3, constant plane — White's rank-8 win counter)

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

POS_C = 15
MOVE_D = 240

# --- V2 (square-grounded / gather head) shapes ------------------------------
# V2 keeps the board feature map SPATIAL on a 10x10 grid (the 8x8 board + the
# 1-square rim) so per-move logits can gather trunk features at the squares a
# move touches (Leela's source x target dot-product head, generalized to
# variable-length capture chains via path-pooling over the rim-aware waypoints).
# See ChesskersScorerV2 + the `project-policy-head-redesign` memory.
#   Position: the 15 V1 channels (written into the 10x10 interior) + 1 on-board
#   mask plane (1 on the 8x8 interior, 0 on the rim) so a pure conv stack can
#   tell an empty interior square from a rim square. 16 channels, 10x10.
POS_C_V2 = 16
CH_V2_ONBOARD = 15  # the new on-board mask plane (indices 0..14 = the V1 channels)
# Move: from_idx, to_idx (10x10 flat indices, gathered against the spatial
# trunk), a 100-wide waypoint path mask (rim-aware, endpoints excluded), and
# MV2_K type scalars. NO from/to one-hots (the gather indexes by construction)
# and NO 100-bit waypoint one-hot fed to an MLP — the mask here drives a
# mean-pool over the spatial features, not a learned lookup.
MV2_FROM = 0
MV2_TO = 1
MV2_PATH_BASE = 2          # 100 waypoint-mask cells over the 10x10 grid
MV2_SCALAR_BASE = 102      # MV2_K type scalars
MV2_K = 12
MOVE_D_V2 = MV2_SCALAR_BASE + MV2_K  # 114
# Type-scalar layout (offsets from MV2_SCALAR_BASE):
#   0 is_capture  1 is_chain  2 is_deploy  3 is_ortho
#   4 chain_len/8  5 deploy_count/24  6 demotions/8
#   7..11 promotion one-hot (none, q, r, b, n)

# Position channel indices
CH_W_PAWN, CH_W_KNIGHT, CH_W_BISHOP, CH_W_ROOK, CH_W_QUEEN, CH_W_KING = range(6)
CH_STONE_TOP, CH_KING_TOP = 6, 7
CH_TOWER_HEIGHT, CH_STONE_COUNT, CH_KING_COUNT = 8, 9, 10
CH_TOP_IS_UNMOVED_STONE, CH_SECOND_IS_KING = 11, 12
CH_SIDE_TO_MOVE = 13
CH_RANK8 = 14

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
# White's rank-8 win counter, read from the FEN's trailing {..,r8:N} block.
# "r8:" occurs only in that block (no board square is named r8), so a plain
# search is unambiguous.
_FEN_R8 = re.compile(r"\br8:(\d+)")

def square_xy(square: str) -> tuple[int, int]:
    """Return (file 0..7, rank 0..7) for a square name like 'a1' or 'h8'."""
    return ord(square[0]) - ord("a"), int(square[1]) - 1


def square_index(square: str) -> int:
    """Return 0..63 with a1=0, b1=1, ..., h8=63."""
    f, r = square_xy(square)
    return r * 8 + f


def encode_position(fen: str) -> torch.Tensor:
    """Encode a Chessckers FEN as a (15, 8, 8) float32 tensor."""
    return _encode_position_py(fen)


def _encode_position_py(fen: str) -> torch.Tensor:
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

    m_r8 = _FEN_R8.search(fen)
    if m_r8:
        out[CH_RANK8].fill_(int(m_r8.group(1)) / 3.0)

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
    """Encode a `variant_py.State` directly to the (15, 8, 8) tensor — the
    per-leaf hot-path encoder."""
    return _encode_position_state_py(state)


def _encode_position_state_py(state: Any) -> torch.Tensor:
    """Pure-Python reference: reads python-chess bitboards via `piece_map()`
    (a single dict iteration over occupied squares)."""
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
    if state.rank8_count:
        out[CH_RANK8].fill_(state.rank8_count / 3.0)

    return out


def encode_move(move: dict[str, Any]) -> torch.Tensor:
    """Encode a LegalMove dict as a (MOVE_D,) float32 vector."""
    return _encode_move_py(move)


def _encode_move_py(move: dict[str, Any]) -> torch.Tensor:
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


# ===========================================================================
# V2 encoders: 10x10 spatial position + gather-indexed moves. The
# (rank10, file10) convention is shared with the V1 waypoint mask, so a board
# square 'e4' and a rim coord 'z3' map through the SAME _FILE10/_RANK10 tables
# to the SAME 10x10 flat index the position tensor uses — the gather and the
# board fill are coordinate-aligned by construction.
# ===========================================================================


def _sq10(s: str) -> int | None:
    """10x10 flat index (rank10*10 + file10) for a square/coord string, or None
    if either char is off the known grid. Works for board squares ('a1'..'h8')
    and rim waypoint coords ('z0'..'i9') alike."""
    if len(s) != 2:
        return None
    f10 = _FILE10.get(s[0])
    r10 = _RANK10.get(s[1])
    if f10 is None or r10 is None:
        return None
    return r10 * 10 + f10


def _board_cell(x: int, y: int) -> tuple[int, int]:
    """(file 0..7, rank 0..7) -> (rank10, file10) interior cell of the 10x10."""
    return y + 1, x + 1


def encode_position_v2(fen: str) -> torch.Tensor:
    """Encode a Chessckers FEN as a (16, 10, 10) float32 tensor for V2: the 15
    V1 channels written into the 8x8 interior of a 10x10 grid + an on-board
    mask plane (1 interior, 0 rim)."""
    m = _FEN_HEAD.match(fen)
    if not m:
        raise ValueError(f"unrecognized Chessckers FEN: {fen!r}")
    board, overlay, turn = m.group(1), m.group(2), m.group(3)

    out = torch.zeros((POS_C_V2, 10, 10), dtype=torch.float32)
    out[CH_V2_ONBOARD, 1:9, 1:9] = 1.0

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
            r10, f10 = _board_cell(x, y)
            if ch in _WHITE_PIECE_CH:
                out[_WHITE_PIECE_CH[ch], r10, f10] = 1.0
            elif ch in _BLACK_BITBOARD_CH:
                out[_BLACK_BITBOARD_CH[ch], r10, f10] = 1.0
            x += 1

    if overlay:
        for entry in overlay.split(","):
            if ":" not in entry:
                continue
            sq, pieces = entry.split(":", 1)
            x, y = square_xy(sq)
            r10, f10 = _board_cell(x, y)
            height = len(pieces)
            if height == 0:
                continue
            stones = sum(1 for p in pieces if p in "sS")
            kings = sum(1 for p in pieces if p == "k")
            out[CH_TOWER_HEIGHT, r10, f10] = height / 24.0
            out[CH_STONE_COUNT, r10, f10] = stones / 24.0
            out[CH_KING_COUNT, r10, f10] = kings / 24.0
            if pieces[-1] == "s":
                out[CH_TOP_IS_UNMOVED_STONE, r10, f10] = 1.0
            if height >= 2 and pieces[-2] == "k":
                out[CH_SECOND_IS_KING, r10, f10] = 1.0

    if turn == "b":
        out[CH_SIDE_TO_MOVE, 1:9, 1:9] = 1.0

    m_r8 = _FEN_R8.search(fen)
    if m_r8:
        out[CH_RANK8, 1:9, 1:9] = int(m_r8.group(1)) / 3.0

    return out


def encode_position_state_v2(state: Any) -> torch.Tensor:
    """V2 per-leaf encoder from a `variant_py.State` (mirrors encode_position_v2
    but reads python-chess bitboards + the stacks overlay directly)."""
    import chess

    out = torch.zeros((POS_C_V2, 10, 10), dtype=torch.float32)
    out[CH_V2_ONBOARD, 1:9, 1:9] = 1.0
    bb_to_ch = _bb_to_ch_table()
    board = state.board

    for sq, piece in board.piece_map().items():
        ch = bb_to_ch.get((piece.color, piece.piece_type))
        if ch is None:
            continue
        r10, f10 = (sq >> 3) + 1, (sq & 7) + 1
        out[ch, r10, f10] = 1.0

    for sq, pieces in state.stacks.items():
        height = len(pieces)
        if height == 0:
            continue
        r10, f10 = (sq >> 3) + 1, (sq & 7) + 1
        stones = kings = 0
        for p in pieces:
            if p == "k":
                kings += 1
            else:
                stones += 1
        out[CH_TOWER_HEIGHT, r10, f10] = height / 24.0
        out[CH_STONE_COUNT, r10, f10] = stones / 24.0
        out[CH_KING_COUNT, r10, f10] = kings / 24.0
        if pieces[-1] == "s":
            out[CH_TOP_IS_UNMOVED_STONE, r10, f10] = 1.0
        if height >= 2 and pieces[-2] == "k":
            out[CH_SECOND_IS_KING, r10, f10] = 1.0

    if board.turn == chess.BLACK:
        out[CH_SIDE_TO_MOVE, 1:9, 1:9] = 1.0
    if state.rank8_count:
        out[CH_RANK8, 1:9, 1:9] = state.rank8_count / 3.0

    return out


def encode_move_v2(move: dict[str, Any]) -> torch.Tensor:
    """Encode a LegalMove as a (114,) V2 vector: [from_idx, to_idx, path_mask(100),
    type_scalars(12)]. from/to are 10x10 flat indices the model GATHERS against
    the spatial trunk (not learned one-hots); the path mask drives a mean-pool
    over the intermediate waypoint squares (endpoints excluded)."""
    out = torch.zeros(MOVE_D_V2, dtype=torch.float32)
    fi = _sq10(move["from"])
    ti = _sq10(move["to"])
    # from/to are always real board squares; fall back to interior origin if a
    # malformed dict slips through rather than crashing the encoder.
    out[MV2_FROM] = float(fi if fi is not None else 11)
    out[MV2_TO] = float(ti if ti is not None else 11)

    waypoints = move.get("waypoints") or []
    for w in waypoints:
        wi = _sq10(w)
        if wi is not None:
            out[MV2_PATH_BASE + wi] = 1.0
    # Endpoints are gathered separately as source/target — never count them as
    # intermediate path cells (the final waypoint == `to`).
    if fi is not None:
        out[MV2_PATH_BASE + fi] = 0.0
    if ti is not None:
        out[MV2_PATH_BASE + ti] = 0.0

    s = MV2_SCALAR_BASE
    if move.get("capture") is not None:
        out[s + 0] = 1.0
    if waypoints:
        out[s + 1] = 1.0
    if move.get("deployCount") is not None:
        out[s + 2] = 1.0
    if move.get("demotionsRequired") is not None:
        out[s + 3] = 1.0
    out[s + 4] = len(waypoints) / 8.0
    out[s + 5] = (move.get("deployCount") or 0) / 24.0
    out[s + 6] = (move.get("demotionsRequired") or 0) / 8.0
    out[s + 7 + _PROMO_INDEX.get(move.get("promotion"), 0)] = 1.0
    return out


def encoders_for(version: str = "v1"):
    """Return the (position-from-FEN, position-from-State, move) encoder trio for
    an arch version, so the play/train hot paths can pick the right encoding from
    a model's VERSION tag. Default 'v1' is the existing encoding."""
    if version == "v2":
        return encode_position_v2, encode_position_state_v2, encode_move_v2
    return encode_position, encode_position_state, encode_move
