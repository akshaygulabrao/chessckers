"""Exact distance-to-mate oracle for tiny Chessckers endgames.

Black tries to force the shortest mate; White tries to avoid/delay it.
`distance_to_mate(fen, max_plies)` returns the number of plies to a forced Black
mate under optimal play, or None if Black cannot force mate within `max_plies`
(White escapes — e.g. captures a tower). From a Black-to-move position the
distance is always odd (1, 3, 5, …).

Search design (all three matter, and compound):
  * **Iterative deepening** — ask "mate in <= 1?", "<= 3?", … and stop at the
    first depth that succeeds, so a dtm-d mate only expands a depth-d tree
    instead of the full `max_plies` horizon.
  * **Boolean forced-mate test with pruning** — each depth is decided by
    `_black_mates_within` / `_white_mated_within`, which return as soon as the
    answer is known: Black cuts off on the FIRST forcing move, White cuts off on
    the FIRST escape. (A plain min/max dtm search can't cut off — it must value
    every move — which is why it was slow.)
  * **King-mobility move ordering** — Black tries the moves that leave the White
    king the FEWEST replies first (driving it toward the edge/corner is how you
    force mate), and White tries captures first (thinning Black is often an
    instant escape). Good ordering makes the cutoffs above fire immediately.

Uses: (1) verify curriculum seed FENs are genuinely forced wins of a known
depth, (2) as a ground-truth detector — compare the net's chosen move / win
rate against optimal play.

CLI:
    python endgame_solver.py "<fen>"            # print dtm
    python endgame_solver.py "<fen>" --depth 8
"""
from __future__ import annotations

import os
import sys
import time

from chessckers_engine.variant_py import PyVariantClient

_client = PyVariantClient()
# Memo of boolean forced-mate results, keyed by (side, fen, depth). Each entry
# is an exact answer (not a bound), so it is always safe to reuse.
_memo: dict[tuple, bool] = {}

# Optional live trace of the search (set CHESSCKERS_SOLVER_TRACE=1). Off by
# default; when off the per-node hook is a single bool check, so the hot path is
# unaffected. When on, it streams a throttled snapshot of the current line.
_TRACE = os.environ.get("CHESSCKERS_SOLVER_TRACE") == "1"
_trace_path: list[str] = []
_trace_nodes = [0]
_trace_last = [0.0]
_trace_t0 = [0.0]


def _trace_tick() -> None:
    if not _TRACE:
        return
    _trace_nodes[0] += 1
    now = time.perf_counter()
    if now - _trace_last[0] >= 0.25:           # ~4 snapshots/sec
        _trace_last[0] = now
        line = " ".join(_trace_path) if _trace_path else "(root)"
        print(f"[{now - _trace_t0[0]:7.1f}s n={_trace_nodes[0]:>9}] {line}", flush=True)


def _legal(fen: str) -> list[dict]:
    return _client.new_game(fen).get("legalMoves") or []


def _order_white(moves: list[dict]) -> list[dict]:
    """White move ordering for escape detection: captures first (thinning Black
    is often an immediate escape), otherwise natural order."""
    return sorted(moves, key=lambda m: m.get("capture") is not None, reverse=True)


def _black_mates_within(fen: str, depth: int) -> bool:
    """Black to move: can Black force mate within `depth` plies?"""
    _trace_tick()
    if depth <= 0:
        return False
    key = ("B", fen, depth)
    cached = _memo.get(key)
    if cached is not None:
        return cached

    # Build the candidate replies, then order by White's mobility — fewest White
    # replies first, since restricting the king is what forces mate, so the
    # forcing move (if any) tends to be found first and cut the search off.
    candidates: list[tuple[int, str, str, list[dict]]] = []
    for m in _legal(fen):
        s2 = _client.make_move(fen, m["uci"])
        if s2.get("status") is not None:
            if s2.get("winner") == "black":   # mate (or White stuck) right now
                _memo[key] = True
                return True
            continue                          # Black lost/drew this line
        cf = s2["fen"]
        white_moves = _legal(cf)              # also reused by _white_mated_within
        candidates.append((len(white_moves), m["uci"], cf, white_moves))
    candidates.sort(key=lambda c: c[0])

    result = False
    for _, uci, cf, white_moves in candidates:
        _trace_path.append(uci)
        sub = _white_mated_within(cf, depth - 1, white_moves)
        _trace_path.pop()
        if sub:
            result = True                     # a forcing Black move → cut off
            break
    _memo[key] = result
    return result


def _white_mated_within(fen: str, depth: int, moves: list[dict] | None = None) -> bool:
    """White to move: is Black's mate forced within `depth` plies (i.e. every
    White reply still loses)?"""
    _trace_tick()
    if depth <= 0:
        return False
    key = ("W", fen, depth)
    cached = _memo.get(key)
    if cached is not None:
        return cached

    if moves is None:
        moves = _legal(fen)
    if not moves:
        # A White-with-no-moves position is terminal (mate if in check, else a
        # stalemate DRAW) and is resolved by the caller's status check before we
        # recurse here — so reaching this is not a confirmed Black mate.
        _memo[key] = False
        return False

    result = True
    for m in _order_white(moves):
        s2 = _client.make_move(fen, m["uci"])
        if s2.get("status") is not None:      # White escaped (eliminated Black / stalemate)
            result = False
            break
        _trace_path.append(m["uci"])
        sub = _black_mates_within(s2["fen"], depth - 1)
        _trace_path.pop()
        if not sub:
            result = False                    # this White move avoids mate → cut off
            break
    _memo[key] = result
    return result


def distance_to_mate(fen: str, max_plies: int = 9) -> int | None:
    """Plies to forced Black mate under optimal play, or None if not forced
    within `max_plies`. Assumes Black to move (the curriculum convention)."""
    _memo.clear()
    if _TRACE:
        _trace_t0[0] = time.perf_counter()
        _trace_nodes[0] = 0
        _trace_last[0] = 0.0
    for depth in range(1, max_plies + 1, 2):  # Black-to-move mate distance is odd
        if _TRACE:
            print(f"==== searching forced capture-mate in <= {depth}? ====", flush=True)
        if _black_mates_within(fen, depth):
            if _TRACE:
                print(f">>> FORCED CAPTURE-MATE IN {depth}", flush=True)
            return depth
        if _TRACE:
            print(f"---- none within {depth}  (n={_trace_nodes[0]}, "
                  f"{time.perf_counter() - _trace_t0[0]:.1f}s) ----", flush=True)
    return None


def best_black_moves(fen: str, max_plies: int = 9, target: int | None = None) -> list[str]:
    """UCIs of the optimal (shortest-mate) Black moves from `fen`.

    Pass `target` (a known distance-to-mate, e.g. from a prior
    `distance_to_mate(fen, …)` call) to skip the internal iterative-deepening
    re-search and reuse the warm `_memo` the caller already built — otherwise
    the mate is solved twice. With `target=None` the distance is computed here
    (which clears `_memo` first)."""
    if target is None:
        target = distance_to_mate(fen, max_plies)  # iterative deepening; clears _memo
    if target is None:
        return []
    out: list[str] = []
    for m in _legal(fen):
        s2 = _client.make_move(fen, m["uci"])
        if s2.get("status") is not None:
            if s2.get("winner") == "black" and target == 1:
                out.append(m["uci"])
            continue
        # `target` is the global minimum, so no move mates faster; a move is
        # optimal exactly when it forces mate within `target` plies.
        if _white_mated_within(s2["fen"], target - 1):
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
        # Reuse the distance + warm _memo from the search above (don't re-solve).
        print("optimal Black move(s):", best_black_moves(args.fen, args.depth, target=d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
