"""Query a generated tablebase: exact value and best move for a position.

`probe(root, fen)` returns the position's `(wdl, dtm)` value from the side-to-
move perspective, or None if the position's class is not solved (or the slot is
VOID). `best_move(root, fen)` returns the UCI of an optimal move — toward the
fastest win, the slowest loss, or a value-preserving drawing move — by probing
the children (whose values are already in the table or terminal).
"""
from __future__ import annotations

import os

from chessckers_engine.variant_py import PyVariantClient
from tb import store
from tb.index import encode
from tb.model import MaterialClass, Value, black_total, terminal_value

_client = PyVariantClient()
_mmap_cache: dict[tuple[str, int], object] = {}


def classify(fen: str) -> MaterialClass:
    return MaterialClass(black_total(fen))


def _class_array(root, total: int):
    """Memmap for a class, cached; None if the class file is absent."""
    key = (str(root), total)
    if key not in _mmap_cache:
        try:
            _mmap_cache[key] = store.open_class(root, total, "r")
        except FileNotFoundError:
            _mmap_cache[key] = None
    return _mmap_cache[key]


def probe(root: str | os.PathLike, fen: str) -> Value | None:
    """Exact `(wdl, dtm)` for the side to move, or None if unavailable
    (class not solved, or VOID slot)."""
    g = _client.new_game(fen)
    if g.get("status") is not None:
        return terminal_value(g["status"], g.get("winner"), g["turn"])
    mc = classify(fen)
    arr = _class_array(root, mc.black_total)
    if arr is None:
        return None
    _, idx = encode(fen)
    return store.byte_decode(int(arr[idx]))


def _child_value(root, fen: str, uci: str) -> Value | None:
    s2 = _client.make_move(fen, uci)
    status = s2.get("status")
    if status is not None:
        return terminal_value(status, s2.get("winner"), s2["turn"])
    mc = MaterialClass(black_total(s2["fen"]))
    arr = _class_array(root, mc.black_total)
    if arr is None:
        return None
    _, idx = encode(s2["fen"])
    return store.byte_decode(int(arr[idx]))


def best_move(root: str | os.PathLike, fen: str) -> str | None:
    """An optimal UCI move per the tablebase, or None if unavailable / terminal.

    A child value is from the *opponent's* perspective, so the mover wins by
    moving to the opponent's fastest loss, must accept the opponent's slowest
    win when losing, and preserves a draw otherwise."""
    val = probe(root, fen)
    if val is None:
        return None
    g = _client.new_game(fen)
    if g.get("status") is not None:
        return None
    wdl, _ = val

    best: tuple | None = None
    best_uci: str | None = None
    for m in g["legalMoves"]:
        uci = m["uci"]
        cv = _child_value(root, fen, uci)
        if cv is None:
            continue
        c_wdl, c_dtm = cv
        cd = c_dtm or 0
        if wdl > 0:  # we win: pick a child that is a loss for the opponent, min dtm
            if c_wdl < 0 and (best is None or cd < best[0]):
                best, best_uci = (cd,), uci
        elif wdl < 0:  # we lose: delay — pick the child with the largest dtm
            if c_wdl > 0 and (best is None or cd > best[0]):
                best, best_uci = (cd,), uci
        else:  # draw: pick any move to a drawn child
            if c_wdl == 0 and best is None:
                best, best_uci = (0,), uci
    return best_uci
