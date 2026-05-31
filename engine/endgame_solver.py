"""Exact distance-to-mate oracle for tiny Chessckers endgames.

Minimax over the real move generator: Black tries to force the shortest mate,
White tries to avoid/delay it. `distance_to_mate(fen, max_plies)` returns the
number of plies to a forced Black mate under optimal play, or None if Black
cannot force mate within `max_plies` (white escapes — e.g. captures a tower).

From a Black-to-move position the distance is always odd (1, 3, 5, …).

Uses: (1) verify curriculum seed FENs are genuinely forced wins of a known
depth, (2) as a ground-truth detector — compare the net's chosen move / win
rate against optimal play.

CLI:
    python endgame_solver.py "<fen>"            # print dtm
    python endgame_solver.py "<fen>" --depth 8
"""
from __future__ import annotations

import sys

from chessckers_engine.variant_py import PyVariantClient

_client = PyVariantClient()
_memo: dict[tuple, int | None] = {}


def _legal(fen: str) -> list[dict]:
    return _client.new_game(fen).get("legalMoves") or []


def _dtm_black(fen: str, depth: int) -> int | None:
    """Black to move: plies to forced mate (>=1), or None if not forced within depth."""
    if depth <= 0:
        return None
    key = ("B", fen, depth)
    if key in _memo:
        return _memo[key]
    best: int | None = None
    for m in _legal(fen):
        s2 = _client.make_move(fen, m["uci"])
        st, win = s2.get("status"), s2.get("winner")
        if st is not None:          # terminal right after Black's move
            if win == "black":      # checkmate OR White stuck → Black wins now
                best = 1
                break
            continue                # Black lost/drew this line; skip it
        d = _dtm_white(s2["fen"], depth - 1)
        if d is not None and (best is None or 1 + d < best):
            best = 1 + d
    _memo[key] = best
    return best


def _dtm_white(fen: str, depth: int) -> int | None:
    """White to move: plies to forced (black) mate, or None if white escapes within depth."""
    if depth <= 0:
        return None
    key = ("W", fen, depth)
    if key in _memo:
        return _memo[key]
    moves = _legal(fen)
    if not moves:
        _memo[key] = None
        return None
    worst = 0
    for m in moves:
        s2 = _client.make_move(fen, m["uci"])
        if s2.get("status") is not None:  # terminal after White's move = White
            _memo[key] = None             # escaped (Black stuck/eliminated)
            return None
        d = _dtm_black(s2["fen"], depth - 1)
        if d is None:  # this white move avoids forced mate within depth
            _memo[key] = None
            return None
        worst = max(worst, d)
    _memo[key] = 1 + worst
    return 1 + worst


def distance_to_mate(fen: str, max_plies: int = 9) -> int | None:
    """Plies to forced Black mate under optimal play, or None if not forced
    within `max_plies`. Assumes Black to move (the curriculum convention)."""
    _memo.clear()
    return _dtm_black(fen, max_plies)


def best_black_moves(fen: str, max_plies: int = 9) -> list[str]:
    """UCIs of the optimal (shortest-mate) Black moves from `fen`."""
    _memo.clear()
    target = _dtm_black(fen, max_plies)
    if target is None:
        return []
    out = []
    for m in _legal(fen):
        s2 = _client.make_move(fen, m["uci"])
        st, win = s2.get("status"), s2.get("winner")
        if st is not None:          # terminal right after Black's move
            if win == "black" and target == 1:  # mate OR variantEnd (stuck) win
                out.append(m["uci"])
            continue                # Black lost/drew this line; skip it
        d = _dtm_white(s2["fen"], max_plies - 1)
        if d is not None and 1 + d == target:
            out.append(m["uci"])
    return out


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fen")
    p.add_argument("--depth", type=int, default=9)
    args = p.parse_args()
    d = distance_to_mate(args.fen, args.depth)
    print(f"distance-to-mate: {d}" + ("" if d is None else f"  ({(d + 1) // 2} move(s))"))
    if d is not None:
        print("optimal Black move(s):", best_black_moves(args.fen, args.depth))
    return 0


if __name__ == "__main__":
    sys.exit(main())
