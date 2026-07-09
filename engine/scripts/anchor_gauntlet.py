#!/usr/bin/env python3
"""Anchored gauntlet: the current net vs a FIXED anchor pool → absolute strength trajectory.

`cc gauntlet` and the fleet's promote-always match series both chain comparisons
against MOVING references (older snapshots / the previous net), so their Elo
numbers accumulate noise and can't certify gradual improvement. This script
measures each net against anchors that never change, so score-vs-anchor is
directly comparable across the whole run — and across runs, for on-disk nets.

Anchors (--anchors, comma list):
  random     seed-0 random init of the CURRENT net's arch (the cold-start floor;
             fixed within a run, changes only if you change arch)
  search:D   net-free alpha-beta bot, fixed depth D (default 3) — an absolute
             anchor no training run can move (SearchBot, resurrected from the
             removed play_tui.py)
  seed13     the run-13 warm-start seed net (auto-resolves the box/Mac backup
             paths) — puts nets from runs 14/15/17 on one scale
  <path>.pt  any explicit checkpoint

Run it every ~10 published nets with the SAME --games/--sims/--temperature, and
the appended JSONL history (--out, on by default) becomes the run's strength
trajectory. Each row records the operating point so mixed histories are auditable.

  cc anchor                                    # current vs random + search:3 + seed13
  cc anchor --games 40                         # tighter error bars
  cc anchor --current trainer/run1/iter-async-000123.pt   # any snapshot
options: --run-dir DIR  --current PATH  --anchors LIST  --games G  --sims S
         --temperature 1.0  --temp-plies 20  --search-time 1.0  --c-puct 1.5
         --max-plies 160  --start-fen FEN  --device auto|cuda|mps|cpu  --seed 0
         --out FILE ('' to disable)

Games are diversified by visit-sampling at --temperature for the first
--temp-plies plies (both nets; the search bot is deterministic, so diversity
comes from the net side). The printed 95% CI is the real precision — 20 games
resolve Elo only to roughly ±150; raise --games for tighter bars.

SLOW: pure-Python PyVariant MCTS + a CPU alpha-beta bot, same per-game cost
class as `cc gauntlet` — background long runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

# Default to the live fleet run dir. lczero-server is a SIBLING of engine on the
# box (/workspace/chessckers/{engine,lczero-server}) but two levels up on the Mac
# (engine nested in chessckers/, lczero-server its sibling). Pick whichever exists.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENG = os.path.dirname(_HERE)
_SERVER_DIR = next(
    (p for p in (os.path.join(_ENG, "..", "lczero-server"),
                 os.path.join(_ENG, "..", "..", "lczero-server"))
     if os.path.isdir(p)),
    os.path.join(_ENG, "..", "lczero-server"),
)
_DEFAULT_RUN_DIR = os.path.join(_SERVER_DIR, "trainer", "run1")
sys.path.insert(0, _HERE)  # so `import watch_game` resolves regardless of cwd
from watch_game import DEFAULT_START_FEN  # noqa: E402  (the training start FEN, read from the fork)

# Well-known homes of the run-13 warm-start seed (box, Mac backup).
_SEED13_PATHS = (
    "/workspace/run13_seed/weights.pt",
    os.path.expanduser("~/chessckers-backups/run13-army-d6e6f6-c64b6-20260702/weights.pt"),
)


def _label(path: str) -> str:
    """Short label: 'i<iter>' for a snapshot, 'best' for weights.pt, else basename."""
    b = os.path.basename(path)
    m = re.search(r"iter-async-0*(\d+)\.pt$", b)
    if m:
        return f"i{m.group(1)}"
    if b == "weights.pt":
        return "best"
    return b.replace(".pt", "")[:8]


def _elo(score: float) -> float:
    """Elo lead implied by a score fraction in [0,1] (capped at ±800)."""
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return max(-800.0, min(800.0, -400.0 * math.log10(1.0 / score - 1.0)))


def _wilson(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval on a score fraction (well-behaved at 0/1 and small n)."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    center = (score + z * z / (2 * n)) / denom
    half = z * math.sqrt(score * (1.0 - score) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ----------------------------------------------------------------------------- players

class NetPlayer:
    """A net moving via PUCT MCTS, with opening-ply temperature for game diversity."""

    def __init__(self, name, model, sims, cpuct, temperature, temp_plies):
        self.name = name
        self.model = model
        self.sims = sims
        self.cpuct = cpuct
        self.temperature = temperature
        self.temp_plies = temp_plies

    def choose(self, state: dict, client, ply: int) -> str | None:
        from chessckers_engine.mcts_puct import pick_puct
        temp = self.temperature if ply < self.temp_plies else 0.0
        m = pick_puct(state, client, self.model, n_sims=self.sims, c_puct=self.cpuct,
                      temperature=temp)
        return m["uci"] if m else None


class SearchBot:
    """Alpha-beta (minimax) over PyVariant's fast path (parse-once / apply-known —
    no FEN round-trips per node), ported from the removed play_tui.py. Leaf eval
    is the hand-built positional score from White's POV. Iterative-deepens to
    `depth` under a wall-clock cap. Deterministic — a fixed absolute anchor."""

    _MATE = 1e6

    def __init__(self, depth: int = 3, time_limit: float = 1.0, beam: int = 6) -> None:
        import chess
        from chessckers_engine.variant_py import PyVariantClient
        self._chess = chess
        self.client = PyVariantClient()
        self.depth = max(1, depth)
        self.time_limit = time_limit
        self.beam = max(0, beam)  # internal-node move cap (0 = full width); root is never pruned
        self.name = f"search:{self.depth}"
        self._pval = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
                      chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}

    def _eval_white_positional(self, state) -> float:
        """Strategic eval, White POV — scores what the win conditions actually turn on:
        material (Black Stones rise toward King-value as they near rank 1 = promotion),
        White-king DANGER from bearing towers (diagonal-hop / charge reach), White's
        RANK-8 race (r8 counter), Black IMMOBILIZATION, and a CONCENTRATION penalty
        per excess tower height (one capture removes the whole tower)."""
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
        return self._eval_white_positional(state)

    @staticmethod
    def _order(legal: list[dict]) -> list[dict]:
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
        if not legal or time.time() > deadline:
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

    def choose(self, state: dict, client=None, ply: int = 0) -> str | None:
        root = self.client.parse(state["fen"])
        _, _, legal = self.client.status_and_legal(root)
        if not legal:
            return None
        white = root.board.turn == self._chess.WHITE
        deadline = time.time() + self.time_limit
        ordered = self._order(legal)
        best_uci, best_v = ordered[0]["uci"], 0.0
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
                best_uci, best_v = local_best, local_v
                ordered.sort(key=lambda m: m["uci"] != best_uci)  # PV-first next iter
            if not completed or abs(best_v) > self._MATE / 2:
                break                                # out of time, or a forced result
        return best_uci


# ----------------------------------------------------------------------------- game loop

def play_game(white, black, client, max_plies, start_fen) -> tuple[str, bool]:
    """One game from start_fen; returns ('white'|'black'|'draw', truncated), where
    truncated=True means it hit the ply cap with no win condition."""
    from chessckers_engine.selfplay_az import _outcome_from_state
    state = client.new_game(fen=start_fen)
    ply = 0
    while not state.get("status") and ply < max_plies:
        player = white if state["turn"] == "white" else black
        uci = player.choose(state, client, ply)
        if uci is None:
            break
        state = client.make_move(state["fen"], uci)
        ply += 1
    truncated = not state.get("status")
    return _outcome_from_state(state), truncated


def resolve_anchors(specs, current_path, dev, args):
    """Build the anchor player list from comma-separated specs. Unresolvable
    anchors (e.g. seed13 with no seed on disk) are skipped with a warning."""
    import torch
    from chessckers_engine.checkpoints import load_scorer
    from chessckers_engine.model import build_model

    def net_player(name, model):
        return NetPlayer(name, model.to(dev).eval(), args.sims, args.c_puct,
                         args.temperature, args.temp_plies)

    players = []
    for spec in (s.strip() for s in specs.split(",") if s.strip()):
        if spec == "random":
            arch_path = current_path + ".arch.json"
            torch.manual_seed(0)  # the anchor is DEFINED as the seed-0 init of this arch
            model = (build_model(**json.loads(open(arch_path).read()))
                     if os.path.exists(arch_path) else build_model())
            torch.manual_seed(args.seed)  # restore the game-sampling seed
            players.append(net_player("random", model))
        elif spec == "search" or spec.startswith("search:"):
            depth = int(spec.split(":", 1)[1]) if ":" in spec else 3
            players.append(SearchBot(depth=depth, time_limit=args.search_time))
        elif spec == "seed13":
            path = next((p for p in _SEED13_PATHS if os.path.exists(p)), None)
            if path is None:
                print(f"  ⚠ anchor 'seed13' skipped: no seed at {' or '.join(_SEED13_PATHS)}")
                continue
            players.append(net_player("seed13", load_scorer(path)))
        elif spec.endswith(".pt"):
            if not os.path.exists(spec):
                print(f"  ⚠ anchor '{spec}' skipped: file not found")
                continue
            players.append(net_player(_label(spec), load_scorer(spec)))
        else:
            raise SystemExit(f"anchor_gauntlet: unknown anchor spec '{spec}' "
                             f"(expected random | search[:D] | seed13 | <path>.pt)")
    if not players:
        raise SystemExit("anchor_gauntlet: no usable anchors")
    return players


def main() -> int:
    ap = argparse.ArgumentParser(description="Current net vs fixed anchors (absolute strength trajectory).")
    ap.add_argument("--run-dir", default=_DEFAULT_RUN_DIR)
    ap.add_argument("--current", default="", help="net to measure (default: <run-dir>/weights.pt)")
    ap.add_argument("--anchors", default="random,search:3,seed13",
                    help="comma list: random | search[:D] | seed13 | <path>.pt")
    ap.add_argument("--games", type=int, default=20, help="games per anchor (colors split)")
    ap.add_argument("--sims", type=int, default=100, help="MCTS sims/move — keep FIXED across the run for comparability")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="opening visit-sampling temperature (0 = deterministic; games would repeat)")
    ap.add_argument("--temp-plies", type=int, default=20, help="plies of temperature before argmax")
    ap.add_argument("--search-time", type=float, default=1.0, help="SearchBot wall-clock per move (s)")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--max-plies", type=int, default=160, help="ply cap; capped games score as draws")
    ap.add_argument("--start-fen", default=DEFAULT_START_FEN, help="start FEN (default: the training start)")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="append one JSONL history row (default <run-dir>/anchor_gauntlet.jsonl; '' disables)")
    args = ap.parse_args()

    import torch
    from chessckers_engine.checkpoints import load_scorer
    from chessckers_engine.variant_py import PyVariantClient

    dev = args.device
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    current = args.current or os.path.join(args.run_dir, "weights.pt")
    if not os.path.exists(current):
        raise SystemExit(f"anchor_gauntlet: net not found: {current} (pass --current)")
    cur_label = _label(current)
    anchors = resolve_anchors(args.anchors, current, dev, args)

    print(f"anchor gauntlet: '{cur_label}' vs {len(anchors)} fixed anchors on {dev} | "
          f"{args.games} games/anchor | {args.sims} sims | temp {args.temperature} for {args.temp_plies} plies"
          f"\n  net: {current}", flush=True)

    cur_model = load_scorer(current).to(dev).eval()
    cur_player_name = cur_label
    client = PyVariantClient()

    rows = []
    n_trunc = 0
    for anchor in anchors:
        cur_player = NetPlayer(cur_player_name, cur_model, args.sims, args.c_puct,
                               args.temperature, args.temp_plies)
        w = d = l = 0
        for gi in range(args.games):
            cur_white = gi % 2 == 0
            pw, pb = (cur_player, anchor) if cur_white else (anchor, cur_player)
            out, trunc = play_game(pw, pb, client, args.max_plies, args.start_fen)
            n_trunc += trunc
            if out == "draw":
                d += 1
            elif (out == "white") == cur_white:
                w += 1
            else:
                l += 1
        ng = w + d + l
        sc = (w + 0.5 * d) / ng if ng else 0.0
        lo, hi = _wilson(sc, ng)
        rows.append((anchor.name, w, d, l, sc, lo, hi))
        print(f"  vs {anchor.name:>8}: {w}-{d}-{l}  ({100 * sc:.0f}%)  "
              f"Elo {_elo(sc):+.0f} [{_elo(lo):+.0f}, {_elo(hi):+.0f}] 95%", flush=True)

    wlbl = max(6, max(len(r[0]) for r in rows))
    print(f"\n  {'anchor':>{wlbl}}   W-D-L    cur%   Elo±  (95% CI)")
    print("  " + "─" * (wlbl + 40))
    for lbl, w, d, l, sc, lo, hi in rows:
        print(f"  {lbl:>{wlbl}}  {w:>2}-{d}-{l:<2}  {100 * sc:>4.0f}%  {_elo(sc):>+5.0f}  "
              f"[{_elo(lo):+.0f}, {_elo(hi):+.0f}]")
    total_g = sum(w + d + l for _, w, d, l, *_ in rows)
    if n_trunc:
        frac = 100 * n_trunc / total_g if total_g else 0
        print(f"  \033[33m⚠ {n_trunc}/{total_g} games ({frac:.0f}%) hit the {args.max_plies}-ply cap "
              f"→ scored DRAW.\033[0m")

    out_path = args.out if args.out is not None else os.path.join(args.run_dir, "anchor_gauntlet.jsonl")
    if out_path:
        row = {
            "ts": int(time.time()),
            "current": cur_label,
            "current_path": os.path.abspath(current),
            "games": args.games, "sims": args.sims,
            "temperature": args.temperature, "temp_plies": args.temp_plies,
            "search_time": args.search_time,
            "anchors": [
                {"anchor": lbl, "w": w, "d": d, "l": l, "score": round(sc, 4),
                 "elo": round(_elo(sc), 1),
                 "elo_lo": round(_elo(lo), 1), "elo_hi": round(_elo(hi), 1)}
                for lbl, w, d, l, sc, lo, hi in rows
            ],
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  appended history row → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
