"""Heuristic-guided forced-mate solver for the e8/d8 Chessckers endgame.

AND-OR search over PyVariant's exact rules:
  * Black to move = OR node: needs ONE winning move. Moves are ordered by the
    "restrain the White king" heuristic (minimize White's reply mobility), so the
    winning move surfaces early despite Black's huge branching -> we stop at the
    first win, taming the OR fan-out.
  * White to move = AND node: EVERY legal reply is expanded, so a returned win is
    a real forced mate against all defenses (a proof, not a line vs one defender).
    White moves are ordered by the user's heuristic (capture undefended towers,
    then race toward rank 8) -- this orders the AND expansion and is the defense
    shown in the printed line.

Iterative deepening finds the exact forced-mate distance D (smallest D with a
proof). Then a line is extracted with White playing its heuristic defense.

  cd engine && .venv/bin/python scripts/solve_endgame.py [--max-depth 26] [--budget 8000000]
"""
from __future__ import annotations
import argparse, sys, time
from chessckers_engine.variant_py.client import PyVariantClient

START_FEN = "3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] b - - 0 1"
client = PyVariantClient()

WIN, LOSS, UNKNOWN = "win", "loss", "unknown"  # from Black's POV
nodes = 0


def expand(fen):
    """(side, winner, moves) for `fen`. side in {'black','white'}; winner in
    {'black','white',None}; moves = legal move dicts (None if terminal)."""
    st = client.parse(fen)
    status, winner, moves = client.status_and_legal(st)
    side = "white" if st.board.turn else "black"  # python-chess: WHITE=True
    return side, status, winner, moves, st


def step(fen, move):
    """Child FEN after applying `move` (re-parse parent for aliasing safety)."""
    st = client.parse(fen)
    child = client.apply_known(st, move)
    return client.state_to_fen(child)


def wking_sq(st):
    import chess
    return st.board.king(chess.WHITE)


def order_white(moves, parent_st):
    """User heuristic: capture undefended Black towers first, then race to rank 8."""
    import chess
    black_sq = set(parent_st.stacks.keys())

    def key(m):
        to = m["to"]
        is_cap = chess.parse_square(to) in black_sq
        rank = int(to[1])
        return (not is_cap, -rank, -(rank == 8))  # captures first, then higher rank
    return sorted(moves, key=key)


def order_black(fen, moves):
    """User heuristic: restrain the White king. Score each Black move by the
    White king's resulting mobility (fewer = more restrained); king-capture and
    checks float to the top. Returns [(move, child_fen, child_winner)] ordered."""
    scored = []
    for m in moves:
        cf = step(fen, m)
        cside, cstatus, cwinner, cmoves, cst = expand(cf)
        if cwinner == "black":
            mob = -1  # captured the king / mate — best
        elif cwinner == "white":
            mob = 999  # losing move — worst
        else:
            mob = len(cmoves) if cmoves else 0
        scored.append((mob, m, cf, cwinner))
    scored.sort(key=lambda t: t[0])
    return [(m, cf, cw) for _, m, cf, cw in scored]


def prove(fen, depth, memo):
    """True iff Black (to move somewhere below) forces mate within `depth` ply."""
    global nodes
    nodes += 1
    side, status, winner, moves, st = expand(fen)
    if winner == "black":
        return True            # White king captured / mated
    if winner == "white" or (status and winner is None):
        return False           # Black eliminated / rank-8 camp / stalemate
    if depth <= 0:
        return False
    key = (fen, depth)
    if key in memo:
        return memo[key]
    if side == "black":        # OR: one winning move suffices
        res = False
        for m, cf, cw in order_black(fen, moves):
            if cw == "white":
                continue       # never play into a loss
            if prove(cf, depth - 1, memo):
                res = True
                break
    else:                      # AND: every White reply must lose
        res = True
        for m in order_white(moves, st):
            if not prove(step(fen, m), depth - 1, memo):
                res = False
                break
    memo[key] = res
    return res


def extract_line(fen, depth, memo):
    """Walk a proven win: Black plays its first proving move; White plays its
    heuristic defense (capture/race). Returns list of (side, uci, fen)."""
    line, cur, d = [], fen, depth
    while d > 0:
        side, status, winner, moves, st = expand(cur)
        if winner == "black":
            break
        if side == "black":
            played = None
            for m, cf, cw in order_black(cur, moves):
                if cw == "black":
                    played = (m, cf); break
                if cw != "white" and prove(cf, d - 1, memo):
                    played = (m, cf); break
            if played is None:
                break
            m, cf = played
            line.append(("B", m["uci"], cf))
        else:
            m = order_white(moves, st)[0]      # White's heuristic defense
            cf = step(cur, m)
            line.append(("W", m["uci"], cf))
        cur = cf
        d -= 1
    return line


def black_can_capture_king(fen):
    """Does Black (to move) have a move that captures the White king?"""
    side, status, winner, moves, st = expand(fen)
    if side != "black" or not moves:
        return False
    for m in moves:
        _, _, w2, _, _ = expand(step(fen, m))
        if w2 == "black":
            return True
    return False


def white_play(fen, moves, st):
    """Heuristic defender: never hang the king; capture an UNDEFENDED tower
    (a capture after which Black can't take the king); else race to rank 8."""
    import chess
    safe = [m for m in moves if not black_can_capture_king(step(fen, m))]
    pool = safe or moves
    black_sq = set(st.stacks.keys())
    caps = [m for m in pool if chess.parse_square(m["to"]) in black_sq]   # safe => undefended
    if caps:
        return max(caps, key=lambda m: int(m["to"][1]))
    return max(pool, key=lambda m: int(m["to"][1]))   # toward rank 8


def black_play(fen, moves):
    """Heuristic attacker: capture the king if possible; else RESTRAIN — pick the
    move minimizing the White king's reply mobility (tie: prefer check, then drive
    the king toward a corner)."""
    import chess
    best, best_key = None, None
    for m in moves:
        cf = step(fen, m)
        side, status, winner, cmoves, cst = expand(cf)
        if winner == "black":
            return m, cf, 0            # king captured now
        if winner == "white":
            continue                   # don't play into a loss
        mob = len(cmoves) if cmoves else 0
        wk = wking_sq(cst)
        corner = min(chess.square_distance(wk, c) for c in (0, 7, 56, 63)) if wk is not None else 9
        from chessckers_engine.variant_py.moves_white import _is_white_in_chessckers_check
        chk = _is_white_in_chessckers_check(cst)
        key = (mob, 0 if chk else 1, corner)
        if best_key is None or key < best_key:
            best, best_key, best_cf = m, key, cf
    return (best, best_cf, best_key[0]) if best else (moves[0], step(fen, moves[0]), -1)


# ---- Box-shrink Black policy (the KRK "drive to the edge with a coordinated wall") ----
import chess  # noqa: E402

CORNERS = (chess.A1, chess.H1, chess.A8, chess.H8)
W = dict(edge=6.0, corner=3.0, mob=3.0, tk=1.5, hang=25.0)  # tunable via CLI


def _edge_dist(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return min(f, 7 - f, r, 7 - r)


def _corner_dist(sq):
    return min(chess.square_distance(sq, c) for c in CORNERS)


def box_eval(cur, m):
    """Score a Black move by the box-shrink objective (LOWER = better):
    drive the White king to the rim/corner, keep confining without stalemating,
    keep both towers near the king, and don't hang a tower. Returns (score, child_fen, wmob)."""
    cf = step(cur, m)
    _, status, winner, wmoves, cst = expand(cf)
    if winner == "black":
        return (-1e9, cf, -1)                 # mate / king captured -> take it
    if status:
        return (1e9, cf, 0)                   # any other terminal (White win / stalemate-draw) -> avoid
    mob = len(wmoves) if wmoves else 0
    wk = wking_sq(cst)
    towers = list(cst.stacks.keys())          # int squares
    tk = sum(chess.square_distance(t, wk) for t in towers) if wk is not None else 0
    hang = 0
    for wm in wmoves:                         # White grabbing a tower it can SAFELY take = a real hang
        if chess.parse_square(wm["to"]) in cst.stacks and not black_can_capture_king(step(cf, wm)):
            hang += 1
    score = (W["edge"] * _edge_dist(wk) + W["corner"] * _corner_dist(wk)
             + W["mob"] * mob + W["tk"] * tk + W["hang"] * hang)
    return (score, cf, mob)


def black_play_box(cur, moves):
    best = None
    for m in moves:
        sc, cf, mob = box_eval(cur, m)
        if best is None or sc < best[0]:
            best = (sc, m, cf, mob)
    return best[1], best[2], best[3], best[0]   # move, child_fen, wmob, score


def playout(fen, max_plies):
    line, cur, seen = [], fen, {}
    for ply in range(max_plies):
        side, status, winner, moves, st = expand(cur)
        if winner:
            break
        if side == "black":
            m, cf, mob, sc = black_play_box(cur, moves)
            tag = f"wmob={mob} box={sc:.0f}"
        else:
            m = white_play(cur, moves, st)
            cf = step(cur, m)
            tag = ""
        line.append((side, m["uci"], cf, tag))
        cur = cf
        seen[cur] = seen.get(cur, 0) + 1
        if seen[cur] >= 3:
            return line, "draw(repetition)", cur
    _, _, winner, _, _ = expand(cur)
    return line, (winner or "draw(move-cap)"), cur


def human_play(fen, max_plies):
    """You play Black (pick from a numbered, restraint-sorted move list) vs the
    heuristic White (capture undefended towers / race to rank 8)."""
    from chessckers_engine.render_board import render_board
    hist, cur, ply = [fen], fen, 0
    while ply < max_plies:
        side, status, winner, moves, st = expand(cur)
        print("\n" + "=" * 54)
        print(render_board(cur))
        print(f"camp counter r8={st.rank8_count}/3   ply {ply}   to move: {side}")
        print(f"fen: {cur}")
        if winner:
            who = "YOU (Black)" if winner == "black" else "White (heuristic)"
            print(f"\n#### {who} WINS ({status or 'king captured'}) ####")
            return
        if status:
            print(f"\n#### DRAW ({status}) ####")
            return
        if side == "white":
            m = white_play(cur, moves, st)
            cur = step(cur, m); hist.append(cur); ply += 1
            print(f"\nWhite plays: {m['uci']}")
            continue
        scored = []
        for m in moves:
            sc, cf, mob = box_eval(cur, m)
            scored.append((sc, mob, m, cf))
        scored.sort(key=lambda t: t[0])
        print("\nYour moves (sorted by box-shrink score; [0] = heuristic's pick):")
        for i, (sc, mob, m, _) in enumerate(scored):
            note = ("KING CAPTURE / MATE — WINS!" if sc <= -1e9
                    else "(loses or stalemates — avoid)" if sc >= 1e9
                    else f"box={sc:>5.0f}  wmob={mob}")
            print(f"  [{i:>2}] {m['uci']:<15} {note}")
        while True:
            sel = input("\npick # (u=undo, q=quit): ").strip().lower()
            if sel in ("q", "quit"):
                print("bye"); return
            if sel in ("u", "undo"):
                if len(hist) >= 3:
                    hist = hist[:-2]; cur = hist[-1]; ply = max(0, ply - 2)
                    print("(undid last full move)")
                else:
                    print("(nothing to undo)")
                break
            try:
                _, _, m, cf = scored[int(sel)]
            except (ValueError, IndexError):
                print("  invalid — enter a number from the list"); continue
            cur = cf; hist.append(cur); ply += 1
            print(f"  you played {m['uci']}")
            break
    print("\n(move cap reached)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default=START_FEN)
    ap.add_argument("--max-depth", type=int, default=26)
    ap.add_argument("--budget", type=int, default=8_000_000, help="node cap")
    ap.add_argument("--play", action="store_true", help="heuristic-policy playout (Black restrain vs White capture/race)")
    ap.add_argument("--human", action="store_true", help="YOU play Black vs the heuristic White")
    ap.add_argument("--validate", action="store_true",
                    help="pre-flight: is this start winnable for Black (box-shrink mates)? how long?")
    ap.add_argument("--max-plies", type=int, default=80)
    for k in W:
        ap.add_argument(f"--w-{k}", type=float, default=W[k], help=f"box-shrink weight: {k}")
    a = ap.parse_args()
    W.update({k: getattr(a, f"w_{k}") for k in W})

    if a.validate:
        line, result, final = playout(a.fen, a.max_plies)
        print(f"start: {a.fen}")
        if result == "black":
            print(f"WINNABLE ✓  box-shrink Black mates in {len(line)} plies vs the heuristic White.")
            print("  line: " + " ".join(u for _, u, _, _ in line))
        else:
            print(f"NOT a clean win under the box-shrink heuristic (result={result}, {len(line)} plies).")
            print("  -> needs a stronger Black policy, or it isn't a forced Black win — inspect with --human.")
        print(f"  final: {final}")
        return

    if a.human:
        human_play(a.fen, a.max_plies)
        return

    if a.play:
        print(f"heuristic playout from: {a.fen}\n")
        line, result, final = playout(a.fen, a.max_plies)
        for i, (sd, uci, _, tag) in enumerate(line):
            print(f"  {i+1:>3} {sd:<5} {uci:<10} {tag}")
        print(f"\nresult: {result}   ({len(line)} plies)")
        print(f"final:  {final}")
        return
    global nodes
    print(f"solving from: {a.fen}\n")
    memo = {}
    t0 = time.time()
    for D in range(1, a.max_depth + 1):
        nodes = 0
        memo.clear()
        ok = prove(a.fen, D, memo)
        dt = time.time() - t0
        print(f"  depth {D:>2} ply: {'FORCED MATE' if ok else 'no'}   "
              f"({nodes} nodes, {dt:.1f}s)")
        if ok:
            print(f"\n>>> Black forces mate in {D} ply (proven vs ALL White defenses).\n")
            line = extract_line(a.fen, D, memo)
            ms = []
            for i, (sd, uci, _) in enumerate(line):
                ms.append(f"{i//2+1}.{'' if sd=='B' else '..'}{uci}" if sd == "B"
                          else f"{uci}")
            print("Line (White plays the capture/race heuristic):")
            print("  " + "  ".join(u for _, u, _ in line))
            print(f"\nfinal FEN: {line[-1][2] if line else a.fen}")
            return
        if nodes >= a.budget:
            print(f"\n(stopped: node budget {a.budget} exceeded at depth {D})")
            return
    print("\nno forced mate within max-depth")


if __name__ == "__main__":
    main()
