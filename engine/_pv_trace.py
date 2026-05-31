"""Trace the full forced-mate principal variation for a Black-to-move endgame.

Black picks a shortest-mate move; White picks the move that maximizes dtm
(longest defense). Prints the ply-by-ply line until Black wins.
"""
import sys
from chessckers_engine.variant_py import PyVariantClient
from endgame_solver import distance_to_mate, _dtm_white, _legal

client = PyVariantClient()


def best_black(fen, depth):
    target = distance_to_mate(fen, depth)
    if target is None:
        return None, None
    for m in _legal(fen):
        s2 = client.make_move(fen, m["uci"])
        st, win = s2.get("status"), s2.get("winner")
        if st is not None:
            if win == "black" and target == 1:
                return m["uci"], target
            continue
        d = _dtm_white(s2["fen"], depth - 1)
        if d is not None and 1 + d == target:
            return m["uci"], target
    return None, target


def worst_white(fen, depth):
    """White move that maximizes remaining dtm (best defense)."""
    best_uci, best_d = None, -1
    for m in _legal(fen):
        s2 = client.make_move(fen, m["uci"])
        if s2.get("status") is not None:
            continue  # White escaped/won — shouldn't happen on a forced line
        from endgame_solver import _dtm_black
        d = _dtm_black(s2["fen"], depth - 1)
        if d is not None and d > best_d:
            best_d, best_uci = d, m["uci"]
    return best_uci


def main():
    fen = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    dtm = distance_to_mate(fen, depth)
    print(f"start: {fen}")
    print(f"dtm = {dtm}  ({None if dtm is None else (dtm+1)//2} move(s))\n")
    if dtm is None:
        print("NO forced mate within depth.")
        return
    ply = 0
    cur = fen
    while True:
        ply += 1
        remaining = depth - (ply - 1)
        buci, _ = best_black(cur, remaining)
        if buci is None:
            print(f"  (no black mate move found at ply {ply})")
            break
        s = client.make_move(cur, buci)
        st, win = s.get("status"), s.get("winner")
        print(f"{ply}. BLACK {buci}   -> status={st} winner={win}")
        print(f"     fen: {s['fen']}")
        if st is not None:
            print(f"\n*** Black wins in {ply} plies ***")
            break
        cur = s["fen"]
        ply += 1
        remaining = depth - (ply - 1)
        wuci = worst_white(cur, remaining)
        if wuci is None:
            print(f"  (white has no move / stuck at ply {ply})")
            break
        s = client.make_move(cur, wuci)
        print(f"{ply}. WHITE {wuci}   (best defense)")
        print(f"     fen: {s['fen']}")
        cur = s["fen"]


if __name__ == "__main__":
    main()
