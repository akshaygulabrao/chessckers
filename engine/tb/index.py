"""Compact bijective index for tablebase positions (Phase 1: White lone King).

Each material class N (= total Black pieces) gets a dense integer index space.
A canonical position encodes to an int in `[0, class_size(N))` and back:

    index = stm_bit + 2 * (wk_square + 64 * black_rank)

`black_rank` ranks the Black configuration (a multiset of tower compositions
placed on distinct squares) within class N:

    black_rank = shape_offset(shape)
               + combination_rank(tower_squares) * msperm_count(shape)
               + msperm_rank(compositions_in_square_order)

where `shape` is the sorted tuple of composition strings. Positions are
**mirror-canonicalized** before encoding (see `tb.enumerate.canonical_fen`), so
a position and its file mirror share one index; the non-canonical twin's slot
is VOID. White-king/tower overlaps and rank-1-stone slots are also VOID.

`decode` returns the canonical FEN for a live slot, or None for a VOID slot.
Round-trip contract: `decode(*encode(fen)) == canonical_fen(fen)`.
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

import chess

from chessckers_engine.variant_py.state import parse_fen
from tb.enumerate import build_fen, canonical_fen, tower_partitions
from tb.model import MaterialClass

NSQ = 64


# --------------------------------------------------------------------------- #
# Combination ranking (combinadics) — rank a sorted k-subset of {0..NSQ-1}
# --------------------------------------------------------------------------- #

def combination_rank(squares_sorted: list[int]) -> int:
    """Lexicographic rank of a sorted subset within all C(NSQ, k) subsets."""
    rank = 0
    for i, sq in enumerate(squares_sorted):
        # count subsets that come before: those whose i-th element is < sq.
        prev = squares_sorted[i - 1] + 1 if i > 0 else 0
        for c in range(prev, sq):
            rank += math.comb(NSQ - 1 - c, len(squares_sorted) - 1 - i)
    return rank


def combination_unrank(rank: int, k: int) -> list[int]:
    """Inverse of `combination_rank`: the k-subset at lexicographic `rank`."""
    squares: list[int] = []
    start = 0
    for i in range(k):
        for c in range(start, NSQ):
            cnt = math.comb(NSQ - 1 - c, k - 1 - i)
            if rank < cnt:
                squares.append(c)
                start = c + 1
                break
            rank -= cnt
    return squares


# --------------------------------------------------------------------------- #
# Multiset-permutation ranking — order compositions over the sorted squares
# --------------------------------------------------------------------------- #

def _msperm_count(counts: dict[str, int]) -> int:
    """Number of distinct permutations of a multiset with the given counts."""
    total = sum(counts.values())
    denom = 1
    for c in counts.values():
        denom *= math.factorial(c)
    return math.factorial(total) // denom


def msperm_count(shape: tuple[str, ...]) -> int:
    return _msperm_count(dict(Counter(shape)))


def msperm_rank(seq: list[str]) -> int:
    """Lexicographic rank of `seq` among permutations of its own multiset."""
    counts = dict(Counter(seq))
    rank = 0
    n = len(seq)
    for i, sym in enumerate(seq):
        for s in sorted(counts):
            if s >= sym:
                break
            if counts[s] == 0:
                continue
            counts[s] -= 1
            rank += _msperm_count(counts)
            counts[s] += 1
        counts[sym] -= 1
    return rank


def msperm_unrank(rank: int, shape: tuple[str, ...]) -> list[str]:
    """Inverse of `msperm_rank` for the multiset described by `shape`."""
    counts = dict(Counter(shape))
    seq: list[str] = []
    for _ in range(len(shape)):
        for s in sorted(counts):
            if counts[s] == 0:
                continue
            counts[s] -= 1
            block = _msperm_count(counts)
            if rank < block:
                seq.append(s)
                break
            rank -= block
            counts[s] += 1
    return seq


# --------------------------------------------------------------------------- #
# Shapes per class
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def shapes(total: int) -> tuple[tuple[str, ...], ...]:
    """All distinct composition-multisets (as sorted tuples) summing to `total`
    Black pieces, in deterministic order."""
    seen: set[tuple[str, ...]] = set()
    for combo in tower_partitions(total):
        seen.add(tuple(sorted(combo)))
    return tuple(sorted(seen))


@lru_cache(maxsize=None)
def _shape_offsets(total: int) -> tuple[dict[tuple[str, ...], int], int]:
    """Prefix offsets of each shape's black-rank block, and the total black
    space size for the class."""
    offsets: dict[tuple[str, ...], int] = {}
    acc = 0
    for shape in shapes(total):
        offsets[shape] = acc
        k = len(shape)
        acc += math.comb(NSQ, k) * msperm_count(shape)
    return offsets, acc


def black_space_size(total: int) -> int:
    return _shape_offsets(total)[1]


def class_size(total: int) -> int:
    """Total index slots for material class `total` (includes VOID slots)."""
    return 2 * NSQ * black_space_size(total)


# --------------------------------------------------------------------------- #
# Encode / decode
# --------------------------------------------------------------------------- #

def _decompose(fen: str) -> tuple[int, int, list[tuple[int, str]]]:
    """(stm_bit, wk_square, sorted [(square, composition)]) from a FEN.
    stm_bit: 0 = White to move, 1 = Black to move."""
    st = parse_fen(fen)
    wk = st.board.king(chess.WHITE)
    if wk is None:
        raise ValueError(f"no White king in {fen!r}")
    stm = 0 if st.board.turn == chess.WHITE else 1
    placements = sorted(st.stacks.items())
    return stm, wk, placements


def encode(fen: str) -> tuple[MaterialClass, int]:
    """Encode a position to (MaterialClass, index). The position is
    mirror-canonicalized first, so mirror twins collapse to one index."""
    cf = canonical_fen(fen)
    stm, wk, placements = _decompose(cf)
    total = sum(len(t) for _, t in placements)
    mc = MaterialClass(total)

    squares = [sq for sq, _ in placements]
    comps_in_order = [t for _, t in placements]  # already in square-sorted order
    shape = tuple(sorted(comps_in_order))

    offsets, _ = _shape_offsets(total)
    shape_off = offsets[shape]
    comb_r = combination_rank(squares)
    perm_r = msperm_rank(comps_in_order)
    black_rank = shape_off + comb_r * msperm_count(shape) + perm_r

    index = stm + 2 * (wk + NSQ * black_rank)
    return mc, index


def decode(mc: MaterialClass, index: int) -> str | None:
    """Decode (MaterialClass, index) back to a canonical FEN, or None if the
    slot is VOID (illegal placement, wk/tower overlap, or a non-canonical mirror
    twin)."""
    total = mc.black_total
    stm = index & 1
    rest = index >> 1
    wk = rest % NSQ
    black_rank = rest // NSQ

    # Locate the shape block.
    sel_shape: tuple[str, ...] | None = None
    sel_off = 0
    for shape in shapes(total):
        off = _shape_offsets(total)[0][shape]
        size = math.comb(NSQ, len(shape)) * msperm_count(shape)
        if off <= black_rank < off + size:
            sel_shape = shape
            sel_off = off
            break
    if sel_shape is None:
        return None  # out of range

    within = black_rank - sel_off
    mspc = msperm_count(sel_shape)
    comb_r = within // mspc
    perm_r = within % mspc

    squares = combination_unrank(comb_r, len(sel_shape))
    comps = msperm_unrank(perm_r, sel_shape)
    placements = list(zip(squares, comps))

    if wk in set(squares):
        return None  # White king overlaps a tower -> VOID

    turn = "w" if stm == 0 else "b"
    fen = build_fen(wk, placements, turn)
    if fen is None:
        return None  # illegal (e.g. rank-1 stone)
    # Only canonical representatives are live; the mirror twin's slot is VOID.
    if canonical_fen(fen) != fen:
        return None
    return fen
