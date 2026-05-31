"""Enumeration of tablebase positions and the file-mirror symmetry.

Phase 1 enumerates every legal position with White = lone King and a fixed
total Black piece count, in canonical FEN form. The only board symmetry that
survives Chessckers' rules is the **left-right file mirror** — Black moves
toward rank 1, so vertical (rank) and diagonal flips are NOT symmetries.
"""
from __future__ import annotations

import itertools

import chess

from chessckers_engine.variant_py.state import State, parse_fen, serialize_fen


# --------------------------------------------------------------------------- #
# Tower compositions / partitions
# --------------------------------------------------------------------------- #

def tower_compositions(height: int) -> list[str]:
    """All bottom-to-top tower strings of the given height over {s, S, k}."""
    return ["".join(p) for p in itertools.product("sSk", repeat=height)]


def integer_partitions(n: int) -> list[list[int]]:
    """Partitions of n into positive parts; each part is one tower's height."""
    if n == 0:
        return [[]]
    result: list[list[int]] = []

    def rec(remaining: int, max_part: int, acc: list[int]) -> None:
        if remaining == 0:
            result.append(list(acc))
            return
        for part in range(min(remaining, max_part), 0, -1):
            acc.append(part)
            rec(remaining - part, part, acc)
            acc.pop()

    rec(n, n, [])
    return result


def tower_partitions(total: int) -> list[list[str]]:
    """All multisets of towers whose piece counts sum to `total`.

    Each element is a list of tower-strings (one per tower): partition `total`
    into tower heights, then take the cartesian product of compositions per
    height. Identical towers can sit on different squares, so we keep a list.
    """
    out: list[list[str]] = []
    for heights in integer_partitions(total):
        comp_choices = [tower_compositions(h) for h in heights]
        for combo in itertools.product(*comp_choices):
            out.append(list(combo))
    return out


# --------------------------------------------------------------------------- #
# FEN construction
# --------------------------------------------------------------------------- #

def build_fen(
    wk: int, placements: list[tuple[int, str]], turn: str
) -> str | None:
    """Canonical Chessckers FEN for the placement, or None if illegal.

    `placements` is a list of (square, tower-string); White king on `wk`.
    Returns the `serialize_fen` canonical key, or None if `parse_fen` rejects
    the position.

    There is intentionally no rank-1 filter: a Black stone CAN sit on rank 1
    (a King that charges onto rank 1 demotes to a Stone there — charges never
    promote, §3D/§5). Excluding such placements would leave the enumerated set
    *not closed* under legal moves. Over-including a few unreachable placements
    (e.g. an unmoved stone on rank 1) is harmless: they are never probed via
    legal play, and `parse_fen` is the only real legality gate.
    """
    # Board grid: index by square. Top piece -> 'k' (king-top) or 'p' (stone-top).
    top: dict[int, str] = {wk: "K"}
    overlay: dict[int, str] = {}
    for sq, tower in placements:
        top[sq] = "k" if tower[-1] == "k" else "p"
        overlay[sq] = tower

    rank_strs: list[str] = []
    for rank in range(7, -1, -1):
        run = 0
        cells: list[str] = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = top.get(sq)
            if piece is None:
                run += 1
            else:
                if run:
                    cells.append(str(run))
                    run = 0
                cells.append(piece)
        if run:
            cells.append(str(run))
        rank_strs.append("".join(cells) or "8")
    board_field = "/".join(rank_strs)

    overlay_field = ",".join(
        f"{chess.square_name(sq)}:{overlay[sq]}" for sq in sorted(overlay)
    )
    if overlay_field:
        fen = f"{board_field}[{overlay_field}] {turn} - - 0 1"
    else:
        fen = f"{board_field} {turn} - - 0 1"

    try:
        state = parse_fen(fen)
    except ValueError:
        return None
    return serialize_fen(state)


def enumerate_level(total: int) -> set[str]:
    """All canonical FENs (both turns) with White = lone King and Black total
    pieces == `total`."""
    fens: set[str] = set()
    squares = list(range(64))

    if total == 0:
        # No Black towers: terminal (White wins). Place WK anywhere.
        for wk in squares:
            for turn in ("w", "b"):
                key = build_fen(wk, [], turn)
                if key is not None:
                    fens.add(key)
        return fens

    for partition in tower_partitions(total):
        n_towers = len(partition)
        for tower_sqs in itertools.permutations(squares, n_towers):
            placements = list(zip(tower_sqs, partition))
            occupied = set(tower_sqs)
            for wk in squares:
                if wk in occupied:
                    continue
                for turn in ("w", "b"):
                    key = build_fen(wk, placements, turn)
                    if key is not None:
                        fens.add(key)
    return fens


# --------------------------------------------------------------------------- #
# File-mirror symmetry
# --------------------------------------------------------------------------- #

def _mirror_square(sq: int) -> int:
    """Reflect a square across the central vertical axis (file f -> 7 - f)."""
    return chess.square(7 - chess.square_file(sq), chess.square_rank(sq))


def _clockless(state: State) -> str:
    """Serialize a state with the move clocks zeroed (halfmove 0, fullmove 1).

    A position's tablebase value is clock-independent — Chessckers' win
    conditions have no 50-move / repetition rule — so the clocks must NOT be
    part of the canonical key. Successors from `make_move` carry a bumped clock;
    without this, equal positions would key differently."""
    state.board.halfmove_clock = 0
    state.board.fullmove_number = 1
    return serialize_fen(state)


def mirror_fen(fen: str) -> str:
    """The left-right file mirror of a position, as a clock-normalized canonical
    FEN.

    `Board.transform(flip_horizontal)` mirrors the bitboards (and castling/ep);
    we mirror the stack overlay squares to match. Turn is preserved.
    """
    st = parse_fen(fen)
    mb = st.board.transform(chess.flip_horizontal)
    ms = {_mirror_square(sq): pieces for sq, pieces in st.stacks.items()}
    return _clockless(State(board=mb, stacks=ms))


def canonical_fen(fen: str) -> str:
    """The mirror-canonical, clock-normalized representative of a position: the
    lexicographically smaller of the position and its file mirror. Mirror twins
    and clock variants collapse to one key."""
    a = _clockless(parse_fen(fen))
    b = mirror_fen(fen)
    return a if a <= b else b
