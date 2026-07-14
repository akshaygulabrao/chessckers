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

The history WATCHES ITSELF (the 8-hourly cron must fire alarms, not grow a file
nobody reads): each row records the fleet's best net number (best_net, from the
server DB); an anchor scoring ≥0.9 in the last 3 rows is SATURATED → it drops to
a --saturated-games tripwire budget and its saved games reallocate onto the last
unsaturated anchor, the discriminative rung (--no-realloc disables); when the
last rung itself saturates, the CURRENT net is pinned as a new anchor
'pin:net<N>' (state: anchor_pins.json next to the jsonl) for future invocations;
and a plateau on the discriminative anchor (<+40 Elo across 3 rows spanning
≥16h) appends an alert to <server>/../ALERTS.log — with a gate stall-floor
screen for context — and pushes via ntfy.sh when NTFY_TOPIC is set.

  cc anchor                                    # current vs random + search:3 + seed13
  cc anchor --games 40                         # tighter error bars
  cc anchor --current trainer/run1/iter-async-000123.pt   # any snapshot
options: --run-dir DIR  --current PATH  --anchors LIST  --games G  --sims S
         --temperature 1.0  --temp-plies 20  --search-time 1.0  --c-puct 1.5
         --max-plies 160  --start-fen FEN  --device auto|cuda|mps|cpu  --seed 0
         --out FILE ('' to disable)  --saturated-games 6  --no-realloc

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
import shutil
import sqlite3
import subprocess
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
import _run_ident  # noqa: E402  (RUN_NAME for the header + JSONL rows)
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


# ----------------------------------------------------------------------------- monitoring
#
# The jsonl history watches itself: saturation → tripwire budgets + pin rotation,
# plateau on the discriminative anchor → ALERTS.log. Pure/testable helpers, shared
# with run_doctor's strength-trend section so the report can never drift from the
# alarm. EVERY path here is non-fatal — monitoring must never break the measurement.

SATURATION_SCORE = 0.9    # score ≥ this in each of the last SATURATION_ROWS rows → saturated
SATURATION_ROWS = 3
PLATEAU_MIN_ROWS = 3      # plateau alarm: the anchor's last N rows ...
PLATEAU_MIN_HOURS = 16.0  # ... spanning at least this many hours ...
PLATEAU_MIN_GAIN = 40.0   # ... gained less than this much Elo → fire
FLOOR_SCORE = 0.05        # score ≤ this in ALL those rows → anchor FLOORED (net too weak
                          # to measure against it yet — e.g. seed13 on a cold run's day 1);
                          # a floored anchor is unmeasurable, not plateaued: no alarm
_DB_PATH = os.path.join(_SERVER_DIR, "chessckers.db")
# <server>/../ALERTS.log = /workspace/chessckers/ALERTS.log on the box; works on the Mac too.
_ALERTS_PATH = os.path.abspath(os.path.join(_SERVER_DIR, "..", "ALERTS.log"))


def load_history(path: str) -> list[dict]:
    """Parse the appended JSONL history (oldest→newest). Non-fatal: a missing file
    or corrupt lines just mean fewer rows."""
    rows: list[dict] = []
    try:
        with open(path) as f:
            for ln in f:
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def anchor_series(rows: list[dict], name: str) -> list[dict]:
    """This anchor's per-row entries (oldest→newest), each with the row's 'ts' attached."""
    out = []
    for row in rows:
        for a in row.get("anchors", []):
            if a.get("anchor") == name:
                out.append({**a, "ts": row.get("ts", 0)})
                break
    return out


def saturated_anchors(rows: list[dict], names: list[str]) -> set[str]:
    """Anchors that scored ≥ SATURATION_SCORE in ALL of the file's last SATURATION_ROWS
    rows. An anchor missing from any of those rows (e.g. a fresh pin) is NOT saturated;
    with fewer than SATURATION_ROWS rows nothing is."""
    last = rows[-SATURATION_ROWS:]
    if len(last) < SATURATION_ROWS:
        return set()
    sat = set()
    for name in names:
        scores = [a.get("score") for row in last for a in row.get("anchors", [])
                  if a.get("anchor") == name]
        if len(scores) == SATURATION_ROWS and all(s is not None and s >= SATURATION_SCORE
                                                  for s in scores):
            sat.add(name)
    return sat


def plan_budgets(names: list[str], saturated: set[str], games: int,
                 saturated_games: int) -> tuple[dict[str, int], list[str]]:
    """Per-anchor game budgets: each saturated anchor drops to the tripwire budget and
    the saved games move onto the LAST unsaturated anchor (the discriminative rung), so
    the total stays ≈ the same while the CI shrinks where the signal is. Returns
    (budgets, log lines)."""
    budgets = {n: games for n in names}
    log: list[str] = []
    tripwire = max(0, min(saturated_games, games))
    saved = 0
    for n in names:
        if n in saturated:
            budgets[n] = tripwire
            saved += games - tripwire
    if not saved:
        return budgets, log
    unsat = [n for n in names if n not in saturated]
    if unsat:
        budgets[unsat[-1]] += saved
        log.append(f"budget: {', '.join(n for n in names if n in saturated)} saturated "
                   f"(score ≥ {SATURATION_SCORE} in last {SATURATION_ROWS} rows) → "
                   f"{tripwire} tripwire games each; +{saved} games reallocated → "
                   f"{unsat[-1]} ({budgets[unsat[-1]]} games)")
    else:
        log.append(f"budget: ALL anchors saturated → {tripwire} tripwire games each "
                   f"({saved} games saved; no unsaturated anchor to reallocate onto)")
    return budgets, log


def plateau_check(rows: list[dict], name: str) -> tuple[bool, float, float] | None:
    """The plateau condition on `name`'s Elo series: over the anchor's last
    PLATEAU_MIN_ROWS rows, returns (fired, delta_elo, span_hours) — fired when the
    span is ≥ PLATEAU_MIN_HOURS and the Elo gain is < PLATEAU_MIN_GAIN. None when
    the anchor has fewer than PLATEAU_MIN_ROWS rows."""
    series = anchor_series(rows, name)[-PLATEAU_MIN_ROWS:]
    if len(series) < PLATEAU_MIN_ROWS:
        return None
    if all(float(s.get("score", 0.0)) <= FLOOR_SCORE for s in series):
        return None  # floored: the net can't score against this anchor yet — unmeasurable
    hours = (series[-1]["ts"] - series[0]["ts"]) / 3600.0
    delta = float(series[-1].get("elo", 0.0)) - float(series[0].get("elo", 0.0))
    return (hours >= PLATEAU_MIN_HOURS and delta < PLATEAU_MIN_GAIN), delta, hours


def best_net_number(db: str = "") -> int | None:
    """networks.network_number of training_runs.best_network_id (newest run) — the
    fleet-visible number of the crowned champion. None on ANY failure (no db, empty
    stub, no best yet)."""
    try:
        con = sqlite3.connect(f"file:{db or _DB_PATH}?mode=ro", uri=True, timeout=2)
        row = con.execute(
            "SELECT n.network_number FROM training_runs t "
            "JOIN networks n ON n.id = t.best_network_id "
            "ORDER BY t.id DESC LIMIT 1").fetchone()
        con.close()
        return int(row[0]) if row and row[0] is not None else None
    except Exception:  # noqa: BLE001 — identity is decoration, never fatal
        return None


def gate_stall_screen(db: str = "", last: int = 15, need: int = 10) -> tuple[bool, str] | None:
    """Screen the last `last` done gate matches for the stall-floor signature: the
    lenient gate still promoting at a healthy-looking rate but each promotion tiny —
    the cum-Elo ratchet idling, not climbing (see the gate-elo-inflation math).
    Returns (stalled, 'pass N/M, mean +X') or None (<`need` matches / no db)."""
    try:
        con = sqlite3.connect(f"file:{db or _DB_PATH}?mode=ro", uri=True, timeout=2)
        rows = con.execute(
            "SELECT wins, losses, draws, passed FROM matches "
            "WHERE done = 1 AND test_only = 0 AND deleted_at IS NULL "
            "ORDER BY id DESC LIMIT ?", (last,)).fetchall()
        con.close()
    except Exception:  # noqa: BLE001
        return None
    if len(rows) < need:
        return None
    npass = sum(1 for r in rows if r[3])
    rate = npass / len(rows)
    elos = [_elo((w + 0.5 * d) / (w + l + d)) for w, l, d, p in rows if p and (w + l + d)]
    mean = sum(elos) / len(elos) if elos else 0.0
    stalled = 0.55 <= rate <= 0.95 and mean <= 45.0
    return stalled, f"pass {npass}/{len(rows)}, mean {mean:+.0f}"


def _pins_state_path(out_dir: str) -> str:
    return os.path.join(out_dir, "anchor_pins.json")


def load_pins(out_dir: str) -> list[dict]:
    """Pinned rotation anchors registered next to the jsonl: [{name, path, ...}],
    oldest first (so the NEWEST pin lands last in the anchor list)."""
    try:
        with open(_pins_state_path(out_dir)) as f:
            return list(json.load(f).get("pins", []))
    except Exception:  # noqa: BLE001 — no/corrupt state = no pins
        return []


def rotate_pin(current_path: str, out_dir: str, best_net: int | None) -> dict | None:
    """Pin the CURRENT net as a new fixed rung: copy the .pt (+ its .arch.json
    sidecar — load_scorer needs it to rebuild the exact arch) to a stable path next
    to the jsonl and register it in anchor_pins.json, so subsequent invocations
    include anchor 'pin:net<N>'. Returns the pin dict, or None (already pinned /
    copy failed — non-fatal either way)."""
    tag = str(best_net) if best_net is not None else time.strftime("%Y%m%d%H%M")
    name = f"pin:net{tag}"
    pins = load_pins(out_dir)
    if any(p.get("name") == name for p in pins):
        print(f"  saturation rotation: '{name}' already pinned — skipping (net unchanged?)")
        return None
    dst = os.path.join(out_dir, f"pinned-anchor-net{tag}.pt")
    try:
        shutil.copy2(current_path, dst)
        arch = current_path + ".arch.json"
        if os.path.exists(arch):
            shutil.copy2(arch, dst + ".arch.json")
        pins.append({"name": name, "path": os.path.abspath(dst), "ts": int(time.time()),
                     "best_net": best_net, "src": os.path.abspath(current_path)})
        with open(_pins_state_path(out_dir), "w") as f:
            json.dump({"pins": pins}, f, indent=1)
    except Exception as e:  # noqa: BLE001 — rotation failing must not break the gauntlet
        print(f"  ⚠ saturation rotation FAILED (non-fatal): {e}")
        return None
    print(f"\033[35m  ★ SATURATION ROTATION: anchor pool exhausted → pinned the current net "
          f"as '{name}'\n    {dst} (+ sidecar) — subsequent gauntlets measure against it\033[0m",
          flush=True)
    return pins[-1]


def plateau_alert_line(anchor: str, delta: float, hours: float, run: str,
                       best_net: int | None, db: str = "") -> str:
    """The ALERTS.log line for a fired plateau, with the gate stall-floor screen
    appended as context when it corroborates."""
    line = (f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} [plateau] {anchor} {delta:+.0f} Elo "
            f"over last {PLATEAU_MIN_ROWS} rows (~{hours:.0f}h) run={run or '?'} "
            f"best_net={best_net if best_net is not None else '?'} "
            f"— LR-drop trigger per run policy")
    screen = gate_stall_screen(db)
    if screen and screen[0]:
        line += f" | gate≈stall-floor ({screen[1]})"
    return line


def emit_alert(line: str, alerts_path: str = "") -> None:
    """Append to ALERTS.log and best-effort push via ntfy.sh when NTFY_TOPIC is set.
    Never fatal."""
    path = alerts_path or _ALERTS_PATH
    try:
        with open(path, "a") as f:
            f.write(line + "\n")
        print(f"\033[31m  ⚠ ALERT appended → {path}\n    {line}\033[0m", flush=True)
    except OSError as e:
        print(f"  ⚠ alert write FAILED (non-fatal): {e}\n    {line}")
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        try:
            subprocess.run(["curl", "-s", "-m", "5", f"https://ntfy.sh/{topic}", "-d", line],
                           capture_output=True, timeout=10, check=False)
        except Exception:  # noqa: BLE001 — the push is best-effort
            pass


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


def resolve_anchors(specs, current_path, dev, args, pins=()):
    """Build the anchor player list from comma-separated specs, then append the
    registered pin rotations (so the newest pin is the LAST, discriminative rung).
    Unresolvable anchors (e.g. seed13 with no seed on disk, a deleted pin) are
    skipped with a warning."""
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
    for pin in pins:
        try:
            players.append(net_player(pin["name"], load_scorer(pin["path"])))
        except Exception as e:  # noqa: BLE001 — a bad pin must not break the gauntlet
            print(f"  ⚠ pinned anchor '{pin.get('name')}' skipped: {e}")
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
    ap.add_argument("--saturated-games", type=int, default=6,
                    help="tripwire budget for a SATURATED anchor (score ≥ 0.9 in the last 3 "
                         "jsonl rows); the saved games reallocate onto the last unsaturated anchor")
    ap.add_argument("--no-realloc", action="store_true",
                    help="disable saturation budget reallocation (every anchor plays --games)")
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

    out_path = args.out if args.out is not None else os.path.join(args.run_dir, "anchor_gauntlet.jsonl")
    out_dir = os.path.dirname(os.path.abspath(out_path)) if out_path else args.run_dir
    best_net = best_net_number()  # (a) fleet net identity for the row; None = unknown, field absent
    pins = load_pins(out_dir) if out_path else []
    anchors = resolve_anchors(args.anchors, current, dev, args, pins=pins)
    names = [a.name for a in anchors]

    history = load_history(out_path) if out_path else []
    sat = saturated_anchors(history, names)
    # (c) rotation: the last (discriminative) rung — or everything — saturated → pin the
    # current net as a NEW rung for future invocations (playing it now would be the net
    # vs a copy of itself, 50% by construction, so this invocation doesn't include it).
    if out_path and (names[-1] in sat or all(n in sat for n in names)):
        rotate_pin(current, out_dir, best_net)
    # (b) budgets: saturated anchors → tripwire games, savings onto the discriminator.
    if args.no_realloc:
        budgets = {n: args.games for n in names}
    else:
        budgets, blog = plan_budgets(names, sat, args.games, args.saturated_games)
        for ln in blog:
            print(f"  {ln}", flush=True)

    rn = _run_ident.run_name()
    print(f"anchor gauntlet{f' [{rn}]' if rn else ''}: '{cur_label}' vs {len(anchors)} fixed anchors on {dev} | "
          f"{args.games} games/anchor | {args.sims} sims | temp {args.temperature} for {args.temp_plies} plies"
          f"\n  net: {current}"
          + (f"  |  fleet best: net #{best_net}" if best_net is not None else ""), flush=True)

    cur_model = load_scorer(current).to(dev).eval()
    cur_player_name = cur_label
    client = PyVariantClient()

    rows = []
    n_trunc = 0
    for anchor in anchors:
        cur_player = NetPlayer(cur_player_name, cur_model, args.sims, args.c_puct,
                               args.temperature, args.temp_plies)
        w = d = l = 0
        for gi in range(budgets.get(anchor.name, args.games)):
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

    if out_path:
        row = {
            "ts": int(time.time()),
            "run": rn,
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
                if w + d + l > 0  # a 0-game anchor would poison the score history
            ],
        }
        if best_net is not None:
            row["best_net"] = best_net
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  appended history row → {out_path}")

        # (d) plateau alarm on the discriminative anchor, recomputed WITH the new row
        # (an anchor that just saturated stops being the one we alarm on). Non-fatal.
        try:
            hist = load_history(out_path)
            sat = saturated_anchors(hist, names)
            disc = next((n for n in reversed(names) if n not in sat), None)
            if disc is None and (pins := load_pins(out_dir)):
                disc = pins[-1]["name"]  # newest pin: no history yet → no alarm, by design
            if disc is None:
                print("  plateau check: no discriminative anchor (all saturated, no pin)")
            else:
                check = plateau_check(hist, disc)
                if check is None:
                    print(f"  plateau check [{disc}]: <{PLATEAU_MIN_ROWS} rows — skipped")
                elif not check[0]:
                    print(f"  plateau check [{disc}]: {check[1]:+.0f} Elo over last "
                          f"{PLATEAU_MIN_ROWS} rows (~{check[2]:.0f}h) — ok")
                else:
                    emit_alert(plateau_alert_line(disc, check[1], check[2], rn, best_net))
        except Exception as e:  # noqa: BLE001 — monitoring never breaks the measurement
            print(f"  ⚠ plateau check FAILED (non-fatal): {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
