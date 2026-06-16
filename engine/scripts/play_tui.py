#!/usr/bin/env python
"""Interactive Chessckers TUI — play a side against a roster of bots.

You never hand-type a move. On your turn you **pick a from-square**, then **pick
a move from that square's menu** (so capture cadence/deploy notation is never
typed); multi-hop capture chains render their path on the board before you
commit. PyVariant is the rules/render/legal-move authority for both sides; the
opponent is a swappable bot.

  cd engine
  .venv/bin/python scripts/play_tui.py                      # instant: you=Black vs Random
  .venv/bin/python scripts/play_tui.py --side white --bot random
  .venv/bin/python scripts/play_tui.py --menu               # pick side + opponent interactively
  .venv/bin/python scripts/play_tui.py --bot engine --nodes 200   # vs the lc0 fork
  .venv/bin/python scripts/play_tui.py "<FEN>" --bot greedy        # from a custom position

Bots: random (uniform legal) · greedy (1-ply, grabs the most captures) ·
heuristic (1-ply hand-eval: material + elimination/backrank/check) ·
engine (the akshay-chessckers-0 lc0 fork over UCI; strength = --nodes).
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENG = os.path.dirname(_HERE)


# ----------------------------------------------------------------------------- bots

class EngineParityError(RuntimeError):
    """The engine returned a move PyVariant doesn't consider legal — a fork↔
    PyVariant divergence worth surfacing rather than silently swallowing."""


class RandomBot:
    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.last: dict = {}

    def choose(self, state: dict) -> str | None:
        legal = state["legalMoves"]
        return self.rng.choice(legal)["uci"] if legal else None

    def close(self) -> None:
        pass


class GreedyBot:
    name = "Greedy 1-ply"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.last: dict = {}

    @staticmethod
    def _captures(m: dict) -> int:
        caps = m.get("_chain_all_captures")
        if caps:
            return len(caps)
        return 1 if m.get("capture") else 0

    def choose(self, state: dict) -> str | None:
        legal = state["legalMoves"]
        if not legal:
            return None
        best = max(self._captures(m) for m in legal)
        cands = [m for m in legal if self._captures(m) == best]
        return self.rng.choice(cands)["uci"]

    def close(self) -> None:
        pass


class HeuristicBot:
    """1-ply hand-eval bot (no net, no MCTS) — the template for a custom heuristic.

    Scores a position from WHITE's POV with Chessckers' win conditions in mind:
    White material, minus Black tower-material (the elimination win is black-
    pieces→0), plus White's rank-8 backrank win (king on rank 8 + the `r8` hold
    counter), minus a penalty for White being in Chessckers-check (Black
    threatening the king-capture win). The mover maximizes `sign * eval`.
    Naive (no opponent-reply lookahead) — a baseline, not strong; deepen by
    recursing into the opponent's best reply (negamax) if you want more."""

    name = "Heuristic 1-ply"

    def __init__(self, seed: int = 0) -> None:
        import chess
        from chessckers_engine.variant_py import PyVariantClient
        self._chess = chess
        self.client = PyVariantClient()
        self.rng = random.Random(seed)
        self.last: dict = {}
        self._pval = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
                      chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}

    def _eval_white(self, ns: dict) -> float:
        """Position score, White POV (positive = White better). `ns` is a state
        dict (carries fen + terminal status/winner)."""
        if ns.get("status"):
            w = ns.get("winner")
            return 1e4 if w == "white" else (-1e4 if w == "black" else 0.0)
        chess = self._chess
        st = self.client.parse(ns["fen"])
        white_mat = sum(self._pval.get(p.piece_type, 0.0)
                        for p in st.board.piece_map().values() if p.color == chess.WHITE)
        black_mat = sum(2.0 if ch == "k" else 1.0
                        for stack in st.stacks.values() for ch in stack)
        wk = st.board.king(chess.WHITE)
        on_rank8 = 1.0 if (wk is not None and chess.square_rank(wk) == 7) else 0.0
        r8 = float(getattr(st, "rank8_count", 0) or 0)
        try:
            from chessckers_engine.variant_py.moves_white import _is_white_in_chessckers_check
            in_check = 1.0 if _is_white_in_chessckers_check(st) else 0.0
        except Exception:  # noqa: BLE001 — eval must never crash the game
            in_check = 0.0
        return white_mat - black_mat + 0.7 * on_rank8 + 1.2 * r8 - 1.0 * in_check

    def choose(self, state: dict) -> str | None:
        legal = state["legalMoves"]
        if not legal:
            return None
        sign = 1.0 if state["turn"] == "white" else -1.0  # mover maximizes its own side
        best_uci, best = None, None
        for m in legal:
            ns = self.client.make_move(state["fen"], m["uci"])
            score = sign * self._eval_white(ns) + self.rng.uniform(-1e-3, 1e-3)
            if best is None or score > best:
                best, best_uci = score, m["uci"]
        return best_uci

    def close(self) -> None:
        pass


class SearchBot:
    """Alpha-beta (minimax) search over PyVariant's fast path (parse-once /
    apply-known — no FEN round-trips per node). Leaf eval is material from
    White's POV: chess pieces standard (P1 N3 B3 R5 Q9), Black Kings 3 / Stones
    1. Iterative-deepens to `depth` under a wall-clock cap so it stays
    interactive, and reports the depth it actually reached (Chessckers' large
    branching means deep middlegames may not hit full depth in the time budget;
    endgames will)."""

    _MATE = 1e6

    def __init__(self, depth: int = 5, time_limit: float = 10.0, beam: int = 6,
                 eval_mode: str = "positional", seed: int = 0) -> None:
        import chess
        import time
        from chessckers_engine.variant_py import PyVariantClient
        self._chess = chess
        self._time = time
        self.client = PyVariantClient()
        self.depth = max(1, depth)
        self.time_limit = time_limit
        self.beam = max(0, beam)  # internal-node move cap (0 = full width); root is never pruned
        self.rng = random.Random(seed)
        self.eval_mode = eval_mode
        self.name = (f"Search depth-{self.depth}" + (f"/beam{self.beam}" if self.beam else "")
                     + (f" ({eval_mode})" if eval_mode != "positional" else ""))
        self.last: dict = {}
        self._pval = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
                      chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}
        self._static = (self._eval_white_material if eval_mode == "material"
                        else self._eval_white_positional)

    def _eval_white_material(self, state) -> float:
        """Pure material, White POV (+ = White better): chess standard, Black King 3 / Stone 1."""
        white = self._chess.WHITE
        wmat = sum(self._pval.get(p.piece_type, 0.0)
                   for p in state.board.piece_map().values() if p.color == white)
        bmat = sum(3.0 if ch == "k" else 1.0 for stk in state.stacks.values() for ch in stk)
        return wmat - bmat

    def _eval_white_positional(self, state) -> float:
        """Strategic eval, White POV — scores what the win conditions actually turn on:

          * material (chess std; Black King 3, Stone 1 rising toward 3 as it nears
            rank 1, since a Stone PROMOTES the whole tower there);
          * minus White-king DANGER from bearing Black towers (Black's only win is
            capturing the king — a tower with the king on a diagonal within its reach,
            or on a rank/file within charge reach, is a live threat);
          * plus White's RANK-8 race (r8 counter + king on the far ranks) and Black
            IMMOBILIZATION (forward-trapped Stones — Black with no move LOSES);
          * minus a CONCENTRATION penalty per excess tower height, because one White
            capture removes the ENTIRE tower (§2) — stacking is a shared-fate risk.

        Cheap: O(pieces) + O(towers), no move-gen, no check predicate (a geometric
        proxy stands in; the search detects real mates at internal nodes)."""
        chess = self._chess
        board, stacks = state.board, state.stacks
        KDANGER, IMMOB, CONC, PROMO = 1.2, 0.3, 0.10, 2.0
        wk = board.king(chess.WHITE)
        white_mat = sum(self._pval.get(p.piece_type, 0.0)
                        for p in board.piece_map().values() if p.color == chess.WHITE)
        wkf = chess.square_file(wk) if wk is not None else -9
        wkr = chess.square_rank(wk) if wk is not None else -9
        black_mat = danger = immob = 0.0
        for sq, stk in stacks.items():
            h = len(stk)
            tf, tr = chess.square_file(sq), chess.square_rank(sq)
            nk = stk.count("k")
            ns = h - nk
            top_king = stk[-1] == "k"
            black_mat += 3.0 * nk + (1.0 + PROMO * (7 - tr) / 7.0) * ns
            black_mat -= CONC * (h - 1)                   # whole-tower-capture risk
            if wk is not None:                             # does this tower bear on the King?
                adf, adr = abs(wkf - tf), abs(wkr - tr)
                cheb = adf if adf > adr else adr
                if adf == adr and 1 <= cheb <= h and (top_king or wkr < tr):
                    danger += (h / cheb) * (1.3 if top_king else 1.0)   # diagonal hop reach
                elif top_king and (adf == 0 or adr == 0) and 1 <= cheb <= max(1, nk):
                    danger += nk / cheb                    # charge reach (King-top only)
            if not top_king and h == 1:                    # forward-trapped Stone (zugzwang fuel)
                blocked = 0
                for ddf in (-1, 1):
                    ff, rr = tf + ddf, tr - 1
                    if not (0 <= ff <= 7 and 0 <= rr <= 7):
                        blocked += 1
                    elif chess.square(ff, rr) in stacks or board.piece_at(chess.square(ff, rr)):
                        blocked += 1
                if blocked == 2:
                    immob += 1.0
        race = 0.0
        if wk is not None:
            r8 = float(getattr(state, "rank8_count", 0) or 0)
            race = 8.0 * r8 + (1.5 if wkr == 7 else 0.5 if wkr == 6 else 0.0)
        danger = 8.0 if danger > 8.0 else danger           # cap a swarm at ~losing
        return white_mat - black_mat - KDANGER * danger + race + IMMOB * immob

    def _leaf(self, state) -> float:
        """Depth-0 value: cheap terminal checks (no move-gen) then the static eval."""
        if not state.stacks:
            return self._MATE                       # Black eliminated → White wins
        if getattr(state, "rank8_count", 0) >= 3:
            return self._MATE                       # White held rank 8 → White wins
        if state.board.king(self._chess.WHITE) is None:
            return -self._MATE                      # White king captured → Black wins
        return self._static(state)

    def _order(self, legal: list[dict]) -> list[dict]:
        """Captures first (by # captured) so alpha-beta prunes more."""
        def caps(m: dict) -> int:
            c = m.get("_chain_all_captures")
            return len(c) if c else (1 if m.get("capture") else 0)
        return sorted(legal, key=lambda m: -caps(m))

    def _ab(self, state, depth: int, alpha: float, beta: float, deadline: float) -> float:
        if depth <= 0:
            return self._leaf(state)
        status, winner, legal = self.client.status_and_legal(state)
        if status is not None:
            if winner == "white":
                return self._MATE + depth           # prefer faster wins
            if winner == "black":
                return -self._MATE - depth
            return 0.0                               # draw / stalemate
        if not legal or self._time.time() > deadline:
            return self._leaf(state)
        moves = self._order(legal)
        if self.beam:
            moves = moves[:self.beam]               # only search the best few
        if state.board.turn == self._chess.WHITE:    # maximizer
            v = -1e18
            for m in moves:
                v = max(v, self._ab(self.client.apply_known(state, m), depth - 1, alpha, beta, deadline))
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v
        v = 1e18                                     # Black minimizes White's score
        for m in moves:
            v = min(v, self._ab(self.client.apply_known(state, m), depth - 1, alpha, beta, deadline))
            beta = min(beta, v)
            if beta <= alpha:
                break
        return v

    def choose(self, state_dict: dict) -> str | None:
        time = self._time
        root = self.client.parse(state_dict["fen"])
        _, _, legal = self.client.status_and_legal(root)
        if not legal:
            return None
        white = root.board.turn == self._chess.WHITE
        deadline = time.time() + self.time_limit
        ordered = self._order(legal)
        best_uci, best_v, reached = ordered[0]["uci"], 0.0, 0
        for d in range(1, self.depth + 1):           # iterative deepening
            alpha, beta = -1e18, 1e18
            local_best, local_v = None, (-1e18 if white else 1e18)
            completed = True
            for m in ordered:
                if time.time() > deadline:
                    completed = False
                    break
                v = self._ab(self.client.apply_known(root, m), d - 1, alpha, beta, deadline)
                if white:
                    if v > local_v:
                        local_v, local_best = v, m["uci"]
                    alpha = max(alpha, local_v)
                else:
                    if v < local_v:
                        local_v, local_best = v, m["uci"]
                    beta = min(beta, local_v)
            if completed and local_best is not None:
                best_uci, best_v, reached = local_best, local_v, d
                ordered.sort(key=lambda m: m["uci"] != best_uci)  # PV-first next iter
            if not completed or abs(best_v) > self._MATE / 2:
                break                                # out of time, or a forced result
        pov = best_v if white else -best_v           # report from the bot's POV
        self.last = {"uci": best_uci, "depth": reached, "pv": [best_uci],
                     "score_cp": int(pov * 100) if abs(best_v) < self._MATE / 2 else None}
        return best_uci

    def close(self) -> None:
        pass


class EngineBot:
    def __init__(self, engine, nodes: int, label: str) -> None:
        self.engine = engine
        self.nodes = nodes
        self.name = label
        self.last: dict = {}

    def choose(self, state: dict) -> str | None:
        res = self.engine.bestmove(state["fen"], self.nodes)
        self.last = res
        uci = res.get("uci")
        if uci is None:
            raise EngineParityError("engine returned no bestmove (search aborted?)")
        legal = {m["uci"] for m in state["legalMoves"]}
        if uci not in legal:
            raise EngineParityError(
                f"engine played {uci!r}, not in PyVariant's legal set "
                f"(e.g. {sorted(legal)[:8]})"
            )
        return uci

    def close(self) -> None:
        self.engine.close()


# ----------------------------------------------------------------------------- moves

def _preview_path(m: dict) -> list[str]:
    """from → (waypoints / hop landings) → final landing, for the path overlay."""
    mid = m.get("waypoints") or m.get("chainHops") or []
    path = list(mid)
    if not path or path[0] != m["from"]:
        path = [m["from"], *path]
    if path[-1] != m["to"]:
        path = [*path, m["to"]]
    return path


def _group_by_from(legal: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}
    for m in legal:
        g.setdefault(m["from"], []).append(m)
    return g


def _dedupe_moves(client, fen: str, moves: list[dict]) -> list[dict]:
    """Drop moves that reach the SAME resulting position (true duplicates — e.g.
    two capture cadences that both overshoot to the same landing). Order
    preserved, first occurrence kept. Raw UCI distinguishes every
    genuinely-distinct move, so nothing else is collapsed."""
    out: list[dict] = []
    seen: set[str] = set()
    for m in moves:
        try:
            rf = client.make_move(fen, m["uci"])["fen"]
        except Exception:  # noqa: BLE001 — keep an un-applicable move visible
            rf = "ERR:" + m["uci"]
        if rf in seen:
            continue
        seen.add(rf)
        out.append(m)
    return out


# ----------------------------------------------------------------------------- render

def _clear() -> None:
    print("\033[2J\033[H", end="")


def _render(render_board, state: dict, you: str, opp_label: str, ply: int,
            eval_info: dict, log: list[str]) -> None:
    _clear()
    mover = state["turn"]
    who = "YOU" if mover == you else "opp"
    print(f"  Chessckers   you:{you.upper()}  vs  {opp_label}   |   ply {ply}  ({mover} to move — {who})")
    print(render_board(state["fen"]))
    if state.get("check"):
        print("  ** CHECK **")
    if eval_info and eval_info.get("uci") is not None:
        cp, pv = eval_info.get("score_cp"), eval_info.get("pv") or []
        shown = f"{cp / 100:+.2f}" if cp is not None else "mate"
        meta = ", ".join(s for s in (
            f"d{eval_info['depth']}" if eval_info.get("depth") else "",
            f"{eval_info['nodes']}n" if eval_info.get("nodes") else "",
        ) if s)
        print(f"  opp eval {shown} (opp POV{f', {meta}' if meta else ''})   pv: {' '.join(pv[:6])}")
    if log:
        print("  log: " + "  ".join(log[-8:]))


# ----------------------------------------------------------------------------- input

def _two_step_pick(render_board, client, state: dict) -> str | None:
    """Square-first move picker. Returns a UCI, 'UNDO', or None (quit)."""
    groups = _group_by_from(state["legalMoves"])
    while True:
        squares = sorted(groups)
        print(f"\n  your pieces with moves: {' '.join(squares)}")
        sel = input("  from-square (u=undo, q=quit): ").strip()
        if sel in ("q", "Q", ""):
            return None
        if sel in ("u", "U"):
            return "UNDO"
        if sel not in groups:
            print(f"  -- no legal move from {sel!r}")
            continue
        # Drop true duplicates (same resulting position); show RAW uci.
        moves = _dedupe_moves(client, state["fen"], groups[sel])
        while True:
            print(f"\n  moves from {sel}:")
            for i, m in enumerate(moves):
                print(f"    [{i:2}] {m['uci']}")
            pick = input("  pick # (b=back, u=undo, q=quit): ").strip()
            if pick in ("q", "Q", ""):
                return None
            if pick in ("u", "U"):
                return "UNDO"
            if pick in ("b", "B"):
                break
            if pick.isdigit() and 0 <= int(pick) < len(moves):
                m = moves[int(pick)]
                if m.get("waypoints"):  # chain / overshoot: show the path, confirm
                    print(render_board(state["fen"], path=_preview_path(m)))
                    if input("  confirm this move? (Y/n): ").strip().lower() in ("n", "no"):
                        continue
                return m["uci"]
            print("  -- out of range")


def _undo(history: list[dict], you: str) -> tuple[dict, int]:
    """Roll back to YOUR previous turn (drops the opponent's reply + your move).
    FEN turn field is the truth, so this is correct under White's double-move."""
    if len(history) <= 1:
        print("  (nothing to undo)")
        return history[-1], len(history) - 1
    history.pop()
    while len(history) > 1 and history[-1]["turn"] != you:
        history.pop()
    return history[-1], len(history) - 1


# ----------------------------------------------------------------------------- roster

def _discover_nets() -> list[tuple[str, str]]:
    """[(short-label, path)] for lc0-loadable nets: engine/net-*.bin plus the
    live fleet champion if it's been published locally."""
    out: list[tuple[str, str]] = []
    for p in sorted(glob.glob(os.path.join(_ENG, "net-*.bin"))):
        h = os.path.basename(p)[4:12]  # short hash
        out.append((h, p))
    fleet = os.path.join(_ENG, "..", "lczero-server", "trainer", "run1", "weights.bin")
    if os.path.exists(fleet):
        out.insert(0, ("fleet", fleet))
    return out


def _resolve_net(arg: str | None) -> tuple[str, str]:
    """Map --net (a path, a short hash, or None→first discovered) to (label, path)."""
    nets = _discover_nets()
    if arg and os.path.exists(arg):
        return (os.path.basename(arg)[4:12] or "net", arg)
    if arg:
        for lab, p in nets:
            if arg in (lab, os.path.basename(p)):
                return lab, p
        raise SystemExit(f"--net {arg!r} not found among {[l for l, _ in nets]}")
    if not nets:
        raise SystemExit("no nets found (engine/net-*.bin); pass --net PATH")
    return nets[0]


def _make_engine_bot(net_arg, nodes, engine_bin, backend) -> EngineBot:
    from chessckers_engine.engine_uci import UciEngine
    label, path = _resolve_net(net_arg)
    engine = UciEngine(path, binary=engine_bin, backend=backend)
    return EngineBot(engine, nodes, label=f"engine {label}@{nodes}n")


def _make_bot(kind: str, args):
    if kind == "random":
        return RandomBot(args.seed)
    if kind == "greedy":
        return GreedyBot(args.seed)
    if kind == "heuristic":
        return HeuristicBot(args.seed)
    if kind == "search":
        return SearchBot(depth=args.depth, time_limit=args.time_limit, beam=args.beam,
                         eval_mode=args.eval, seed=args.seed)
    if kind == "engine":
        return _make_engine_bot(args.net, args.nodes, args.engine_bin, args.backend)
    raise SystemExit(f"unknown bot {kind!r}")


def _startup_menu(args) -> tuple[str, object]:
    """Interactive side + opponent selection (only on --menu)."""
    s = input("play as (W)hite or (B)lack? [B]: ").strip().lower()
    side = "white" if s.startswith("w") else "black"
    nets = _discover_nets()
    print("\nopponents:")
    print("  [1] Random           (instant)")
    print("  [2] Greedy 1-ply     (instant)")
    print("  [3] Heuristic 1-ply  (instant)")
    print(f"  [4] Search depth-{args.depth}     (alpha-beta beam-{args.beam}, ~{args.time_limit:.0f}s/move)")
    for i, (lab, _) in enumerate(nets, start=5):
        print(f"  [{i}] engine net {lab}")
    pick = (input(f"pick [1-{4 + len(nets)}] [1]: ").strip() or "1")
    if pick == "2":
        return side, GreedyBot(args.seed)
    if pick == "3":
        return side, HeuristicBot(args.seed)
    if pick == "4":
        return side, SearchBot(depth=args.depth, time_limit=args.time_limit, beam=args.beam,
                               eval_mode=args.eval, seed=args.seed)
    if pick.isdigit() and 5 <= int(pick) <= 4 + len(nets):
        lab, path = nets[int(pick) - 5]
        nodes_raw = input(f"nodes/move [{args.nodes}]: ").strip()
        nodes = int(nodes_raw) if nodes_raw.isdigit() else args.nodes
        from chessckers_engine.engine_uci import UciEngine
        engine = UciEngine(path, binary=args.engine_bin, backend=args.backend)
        return side, EngineBot(engine, nodes, label=f"engine {lab}@{nodes}n")
    return side, RandomBot(args.seed)


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive Chessckers TUI vs a roster of bots.")
    ap.add_argument("fen", nargs="?", default=None, help="start FEN (default: standard start)")
    ap.add_argument("--side", choices=["white", "black"], default="black",
                    help="the side YOU play (default black = the towers)")
    ap.add_argument("--bot", choices=["random", "greedy", "heuristic", "search", "engine"], default=None,
                    help="opponent (default: random). Use --menu to pick interactively.")
    ap.add_argument("--depth", type=int, default=5, help="search bot: alpha-beta target depth (plies)")
    ap.add_argument("--time-limit", type=float, default=8.0,
                    help="search bot: wall-clock cap per move (seconds); it deepens until this")
    ap.add_argument("--beam", type=int, default=6,
                    help="search bot: internal-node move cap (top-N by capture ordering; 0 = full width)")
    ap.add_argument("--eval", choices=["positional", "material"], default="positional",
                    help="search bot leaf eval: positional (king-danger/race/promo/etc.) or pure material")
    ap.add_argument("--net", default=None, help="engine net: a .bin path or short hash (default: first found)")
    ap.add_argument("--nodes", type=int, default=200, help="engine search nodes/move (strength dial)")
    ap.add_argument("--backend", default=None, help="engine backend (e.g. metal) — faster thinking")
    ap.add_argument("--engine-bin", default=None, help="path to the akshay-chessckers-0 binary")
    ap.add_argument("--menu", action="store_true", help="interactive side + opponent picker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=400)
    args = ap.parse_args()

    from chessckers_engine.render_board import render_board
    from chessckers_engine.selfplay_az import _outcome_from_state
    from chessckers_engine.variant_py import PyVariantClient
    from chessckers_engine.variant_py.state import STARTING_FEN

    if args.menu and args.bot is None:
        you, bot = _startup_menu(args)
    else:
        you = args.side
        bot = _make_bot(args.bot or "random", args)

    client = PyVariantClient()
    state = client.new_game(fen=args.fen or STARTING_FEN)
    print(f"you play {you} | opponent: {bot.name} | start: {(args.fen or 'standard')[:40]}")

    history = [state]
    log: list[str] = []
    ply = 0
    try:
        while not state.get("status") and ply < args.max_plies:
            legal = state.get("legalMoves") or []
            if not legal:
                break
            _render(render_board, state, you, bot.name, ply + 1, getattr(bot, "last", {}), log)

            if state["turn"] == you:
                mv = _two_step_pick(render_board, client, state)
                if mv is None:
                    print("bye.")
                    return 0
                if mv == "UNDO":
                    state, ply = _undo(history, you)
                    del log[ply:]
                    continue
            else:
                print(f"\n  {bot.name} thinking…", flush=True)
                try:
                    mv = bot.choose(state)
                except EngineParityError as e:
                    print(f"\n!! engine/PyVariant divergence: {e}\n   (ending game — this is a parity bug to investigate)")
                    return 1
                if mv is None:
                    break

            state = client.make_move(state["fen"], mv)
            log.append(f"{('W' if history[-1]['turn'] == 'white' else 'B')}:{mv if len(mv) <= 12 else mv[:11] + '…'}")
            history.append(state)
            ply += 1

        _render(render_board, state, you, bot.name, ply, getattr(bot, "last", {}), log)
        status = state.get("status")
        if status:
            outcome = _outcome_from_state(state)
            if outcome == "draw":
                print(f"\n######## DRAW ({status}) in {ply} plies ########")
            else:
                tag = "YOU WIN" if outcome == you else "you lose"
                print(f"\n######## {tag} — {outcome.upper()} ({status}) in {ply} plies ########")
        else:
            print(f"\n######## stopped at {ply} plies (max-plies={args.max_plies}) ########")
        return 0
    finally:
        bot.close()


if __name__ == "__main__":
    sys.exit(main())
