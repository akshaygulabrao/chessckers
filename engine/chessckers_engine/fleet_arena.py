"""Fleet arena — lc0/AGZ-style keep-best gating for the continuous trainer.

Runs on the trainer box, sharing the trainer's run-dir. `train_continuous` keeps
publishing candidate checkpoints (`iter-async-*.pt`); this arena decides which
candidate becomes the gated champion `best.pt` that self-play actually uses (the
server versions on `best.pt`, so workers reload once per PROMOTION, not per
iteration). It also archives every champion by timestamp and logs a per-side
capacity signal, so we can read empirically when the network stops improving.

The promotion gate is played by the FLEET, lc0-style: the arena opens a gate
(writes match.json) and the server round-robins the candidate-vs-panel games to
the self-play clients, which POST each result back; the arena only TALLIES them. The
trainer/server host never plays a gate game itself (mirrors lc0's server/client
responsibility split). With no client connected a gate simply never completes —
intentional; the local box always runs a loopback self-play client. The host's own
native/Python search is used only for the cheap favored-side labelling pass and the
on-promotion benches (vs-anchor capacity + the regression ladder).

Imbalance-aware promotion rule
------------------------------
Chessckers is structurally lopsided and the curriculum seeds are one-sided by
construction (each seed is a position one side is *supposed* to win). A naive
win-rate gate would reward a net for playing the favored side, not for being
better. So the gate is built around the imbalance rather than ignoring it:

  1. COLOR-SWAPPED SEED PAIRS. Each seed is played as pairs (candidate-as-White /
     best-as-Black, then swapped). The seed's structural advantage is handed to
     BOTH nets equally, so it cancels and the per-seed pair-score is bias-free
     skill, not "who drew the winning side".
  2. PER-SIDE CLASS BALANCE. Seeds are partitioned by which side they favor
     (auto-labelled by playing best-vs-random both ways — the side that beats
     random is the favored side; logged each cycle, self-correcting as best
     strengthens). The promotion score is the BALANCED MEAN of the candidate's
     score on White-favored vs Black-favored seeds — equal weight to each side's
     task regardless of how many seeds fall in each class.
  3. PROMOTE iff  balanced_score >= --threshold  AND  every populated class score
     >= --side-floor. The floor is the no-side-regression guard: it refuses a net
     that won overall by mastering one side while going backwards on the other.

On promotion: best.pt <- candidate, archive nets/net-<unix_ts>.pt, bump the
cumulative Elo by the gate margin, and benchmark the new best vs a FROZEN ANCHOR
(the run's first champion) per side. Comparing to a *fixed* anchor (rather than
the moving best, against which per-side advantage is not separable) gives two
independent per-side strength curves over time — their plateaus are the per-side
capacity answer. A REGRESSION LADDER additionally benches the new best against
several PAST champions (-1/-4/-16 back by default): the gate alone only ever
compares to the *immediate* predecessor, so a rung score < 0.5 is the alarm that
strength cycled backwards even as the gate kept promoting. Archived champions are
capped at the newest --keep-nets (the ladder's rungs are always retained).
Everything is appended to gate_log.jsonl.

Run on the trainer box, sharing its run-dir:

    python -m chessckers_engine.fleet_arena --run-dir weights/run \\
      --seed-mix-file ../scripts/seed_mix.txt --d-hidden 256 --c-filters 96 --n-blocks 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.device import pick_device
from chessckers_engine.evaluate import _state_to_outcome
from chessckers_engine.mcts_puct import run_mcts
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.random_player import pick_random
from chessckers_engine.runtime import setup_logging
from chessckers_engine.variant_py import PyVariantClient

log = logging.getLogger("chessckers_engine.fleet_arena")

# Native C++ search for gate games (~5x the Python MCTS). Optional — falls back
# to the Python search if the extension isn't built on this box.
try:
    import chessckers_cpp as _cpp
    from chessckers_engine.native_net import export_state_dict as _export_state_dict
    NATIVE_OK = True
except Exception:  # noqa: BLE001
    _cpp = None
    NATIVE_OK = False


# --- game runner ---------------------------------------------------------------

def _play_from(white_picker, black_picker, client, start_fen: str, max_plies: int) -> str:
    """Play one game from `start_fen`, returning 'white' | 'black' | 'draw'
    (winner perspective). A picker raising / returning None ends the game as a
    draw — gating never crashes on a single bad move."""
    state = client.new_game(fen=start_fen)
    ply = 0
    while not state.get("status") and ply < max_plies:
        picker = white_picker if state["turn"] == "white" else black_picker
        chosen = picker(state)
        if chosen is None:
            break
        try:
            state = client.make_move(state["fen"], chosen["uci"])
        except Exception as e:  # noqa: BLE001
            log.warning("gate game: make_move failed at ply %d (%s) — scoring draw", ply, e)
            return "draw"
        ply += 1
    return _state_to_outcome(state)


def _native_picker(net, sims: int, c_puct: float, dir_alpha: float | None, dir_eps: float,
                   counter):
    """A PUCT picker driven by the native C++ search. A per-move incrementing
    `counter` seeds the root Dirichlet so repeated gate games diverge (light
    noise — play stays strong). Fresh tree per move (no reuse): the 2-net gate
    descends two plies between a side's turns, so single-ply reuse doesn't apply.
    """
    def pick(state):
        chosen, _vd, _val, _tree = _cpp.run_mcts_native(
            _cpp.parse_fen(state["fen"]), net, int(sims), float(c_puct),
            float(dir_alpha or 0.0), float(dir_eps), next(counter))
        if not chosen:
            return None
        for m in (state.get("legalMoves") or []):
            if m["uci"] == chosen:
                return m
        return None
    return pick


def _model_picker(model: ChesskersScorer, client, sims: int, c_puct: float,
                  dir_alpha: float | None, dir_eps: float):
    """Python-MCTS picker (fallback when the native extension isn't available)."""
    def pick(state):
        return run_mcts(state, client, model, n_sims=sims, c_puct=c_puct,
                        dirichlet_alpha=dir_alpha, dirichlet_eps=dir_eps).chosen
    return pick


def _random_picker():
    def pick(state):
        return pick_random(state.get("legalMoves") or [])
    return pick


# --- favored-side labelling ----------------------------------------------------

def _label_seeds(best_pick, rand_pick, client, seeds: list[str], max_plies: int) -> dict[str, str]:
    """Label each seed by favored side: play best-as-White/random-Black and
    random-White/best-as-Black. The side on which best beats random is the
    favored side. 'balanced' if best wins on both sides (or neither) — a
    balanced seed contributes equally to both class scores downstream."""
    labels: dict[str, str] = {}
    for seed in seeds:
        w = _play_from(best_pick, rand_pick, client, seed, max_plies) == "white"
        b = _play_from(rand_pick, best_pick, client, seed, max_plies) == "black"
        lab = "white" if (w and not b) else "black" if (b and not w) else "balanced"
        labels[seed] = lab
        log.info("  label %-16s = %-8s (best wins as W:%s as B:%s)", _seed_tag(seed), lab, w, b)
    return labels


# --- the gate ------------------------------------------------------------------

def _score(outcome: str, cand_is_white: bool) -> float:
    """Candidate's points for a single game (1 win / 0.5 draw / 0 loss)."""
    if outcome == "draw":
        return 0.5
    cand_won = (outcome == "white") if cand_is_white else (outcome == "black")
    return 1.0 if cand_won else 0.0


def _score_opp(collected: dict, seeds: list[str], labels: dict[str, str], pairs: int) -> dict:
    """Imbalance-aware score of ONE opponent's color-swapped seed-paired games. `collected`
    maps (seed, cand_white) -> list of outcomes ('white'|'black'|'draw'); the first `pairs`
    per (seed, side) are scored (exactly 2*pairs games/seed). The single definition of the
    promotion math, shared by the local benches (_gate) and the fleet-played panel gate."""
    per_seed: dict[str, float] = {}
    rec = [0, 0, 0]  # candidate's aggregate [W, L, D] vs opponent across all games
    for seed in seeds:
        pts = 0.0
        wld = [0, 0, 0]  # this seed's [W, L, D] from the candidate's perspective
        for cand_white in (True, False):
            for outcome in collected.get((seed, cand_white), [])[:pairs]:
                s = _score(outcome, cand_white)
                pts += s
                wld[0 if s == 1.0 else 1 if s == 0.0 else 2] += 1
        per_seed[seed] = pts / (2 * pairs)
        for i in range(3):
            rec[i] += wld[i]

    # Class scores: a 'balanced' seed counts toward BOTH classes.
    w_scores = [per_seed[s] for s in seeds if labels.get(s) in ("white", "balanced")]
    b_scores = [per_seed[s] for s in seeds if labels.get(s) in ("black", "balanced")]
    class_w = sum(w_scores) / len(w_scores) if w_scores else None
    class_b = sum(b_scores) / len(b_scores) if b_scores else None
    populated = [c for c in (class_w, class_b) if c is not None]
    balanced = sum(populated) / len(populated) if populated else 0.0  # equal weight per side
    return {
        "per_seed": {s: round(v, 3) for s, v in per_seed.items()},
        "class_white": None if class_w is None else round(class_w, 3),
        "class_black": None if class_b is None else round(class_b, 3),
        "balanced_score": round(balanced, 3),
        "min_class": round(min(populated), 3) if populated else 0.0,
        "record": {"w": rec[0], "l": rec[1], "d": rec[2]},  # candidate vs opponent, whole gate
    }


def _gate(cand_pick, opp_pick, client, seeds: list[str], labels: dict[str, str],
          pairs: int, max_plies: int, tag: str = "gate") -> dict:
    """Color-swapped seed-paired match played LOCALLY on this host. Used ONLY for the
    on-promotion benches (vs-anchor capacity + the regression ladder) — NOT the promotion
    gate, which the fleet plays and the arena merely tallies (see _GateCollector). Logs a
    line per seed so a bench is visibly progressing; returns the imbalance-aware score."""
    collected: dict[tuple[str, bool], list[str]] = {}
    for si, seed in enumerate(seeds, 1):
        pts = 0.0
        wld = [0, 0, 0]  # this seed's [W, L, D] from the candidate's perspective
        for _ in range(pairs):
            for cand_white in (True, False):
                wp, bp = (cand_pick, opp_pick) if cand_white else (opp_pick, cand_pick)
                outcome = _play_from(wp, bp, client, seed, max_plies)
                collected.setdefault((seed, cand_white), []).append(outcome)
                s = _score(outcome, cand_white)
                pts += s
                wld[0 if s == 1.0 else 1 if s == 0.0 else 2] += 1
        log.info("  [%s] seed %d/%d %-16s -> %.3f  (cand W%d L%d D%d of %d)",
                 tag, si, len(seeds), _seed_tag(seed), pts / (2 * pairs), wld[0], wld[1], wld[2], 2 * pairs)
    return _score_opp(collected, seeds, labels, pairs)


class _GateCollector:
    """Tally of FLEET-played gate outcomes for the open gate. Self-play clients POST each
    result to the server, which writes it as match_results/<match_id>_<n>.json tagged with
    its opponent; this drains those files into an in-memory bucket keyed by (opponent, seed,
    cand_white). The arena plays NONE of the gate (lc0: the trainer/server host dispatches +
    tallies; the clients play every game) — it just waits here until every panel unit has
    --pairs results. One shared bucket is required because clients contribute every opponent's
    games CONCURRENTLY (the server round-robins opponent x seed x side), so a drain must never
    discard a result for an opponent the scoring loop hasn't reached yet."""

    def __init__(self, results_dir: Path, match_id: int) -> None:
        self._dir = results_dir
        self._mid = match_id
        self._buf: dict[tuple[str, str, bool], list[str]] = {}

    def _drain(self) -> None:
        for p in sorted(self._dir.glob(f"{self._mid}_*.json")):
            try:
                r = json.loads(p.read_text())
            except (OSError, ValueError):
                r = None
            if r and r.get("outcome") in ("white", "black", "draw"):
                key = (r.get("opp") or "best", r.get("seed"), bool(r.get("cand_white")))
                self._buf.setdefault(key, []).append(r["outcome"])
            try:
                p.unlink()  # consumed into memory (or unreadable) — don't re-read it
            except OSError:
                pass

    def have(self, opps: list[str], seeds: list[str], pairs: int) -> int:
        """Drain pending results, then count how many of the panel's units are satisfied
        (each unit counts up to `pairs`). == len(opps)*len(seeds)*2*pairs once the gate is
        complete; the gate loop polls this until it reaches that total."""
        self._drain()
        return sum(min(len(self._buf.get((o, s, cw), [])), pairs)
                   for o in opps for s in seeds for cw in (True, False))

    def collected_for(self, opp: str, seeds: list[str]) -> dict:
        """One opponent's outcomes as {(seed, cand_white): [outcomes]} — the shape
        _score_opp consumes (it scores the first `pairs` per side)."""
        return {(s, cw): self._buf.get((opp, s, cw), [])
                for s in seeds for cw in (True, False)}


def _elo_delta(score: float, cap: float = 400.0) -> float:
    """Elo points implied by a head-to-head score (logistic), clamped."""
    s = min(max(score, 1e-3), 1 - 1e-3)
    return max(-cap, min(cap, 400.0 * math.log10(s / (1 - s))))


def _fmt_dur(seconds: float) -> str:
    """Human-readable elapsed span, e.g. '2h05m', '12m34s', '45s'."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _clock(t: float) -> str:
    """Local wall-clock 'YYYY-MM-DD HH:MM:SS' for a unix timestamp."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


# --- net loading ---------------------------------------------------------------

def _load_model(path: Path, arch: dict, device) -> ChesskersScorer:
    m = ChesskersScorer(**arch).to(device)
    load_checkpoint(m, str(path))
    m.eval()
    return m


def _make_net(path: Path, arch: dict, bin_path: Path, device):
    """Load a checkpoint into a gate-playable net. Native: export the state_dict
    to a flat .bin and return a cc::ChesskersNet (CPU C++ search). Fallback: the
    torch model for the Python search."""
    if not NATIVE_OK:
        return _load_model(path, arch, device)
    m = ChesskersScorer(**arch)
    load_checkpoint(m, str(path))
    m.eval()
    tmp = str(bin_path) + ".tmp"
    _export_state_dict(m.state_dict(), tmp)
    os.replace(tmp, bin_path)
    return _cpp.ChesskersNet(str(bin_path))


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _newest_candidate(run_dir: Path, settle_s: float) -> Path | None:
    """Newest fully-written iter checkpoint (older than settle_s so we never read
    a half-flushed torch.save)."""
    now = time.time()
    cands = []
    for p in run_dir.glob("iter-async-*.pt"):
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if now - mt >= settle_s:
            cands.append((mt, p))
    return max(cands)[1] if cands else None


def _last_elo(log_path: Path) -> float:
    """Resume the cumulative Elo from the last promoted line of the gate log."""
    if not log_path.exists():
        return 0.0
    elo = 0.0
    try:
        for line in log_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("promoted") and "best_elo" in rec:
                elo = float(rec["best_elo"])
    except OSError:
        pass
    return elo


# --- main ----------------------------------------------------------------------

def main() -> int:
    setup_logging()
    # Tag every line with [arena] so the arena can share the trainer's unified log
    # stream (both append to /tmp/cc_train.log) and still be told apart from the
    # trainer's per-game lines — one log to watch.
    for _h in logging.getLogger().handlers:
        _h.setFormatter(logging.Formatter("%(asctime)s [arena] %(message)s"))
    p = argparse.ArgumentParser(description="Keep-best gating arena (imbalance-aware).")
    p.add_argument("--run-dir", required=True, type=Path, help="trainer's run-dir (shared)")
    p.add_argument("--seed-mix-file", required=True, type=Path,
                   help="curriculum seed mix (one FEN/line, '#'/blank ignored) — same file self-play uses")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=96)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--sims", type=int, default=160, help="MCTS sims per move in gate games")
    p.add_argument("--pairs", type=int, default=4, help="color-swapped pairs per seed (2x games/seed)")
    p.add_argument("--threshold", type=float, default=0.55, help="balanced-score bar to promote")
    p.add_argument("--side-floor", type=float, default=0.45, help="min per-class score (no-side-regression guard)")
    p.add_argument("--gate-opponents", type=int, default=3,
                   help="gate the candidate against the last N champions (newest first; fewer if "
                        "<N exist). N=1 restores the single-best gate.")
    p.add_argument("--no-regress", type=float, default=0.50,
                   help="min balanced-score vs each OLDER champion (beyond the immediate best) to "
                        "promote — the across-time no-regression guard.")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--dirichlet-eps", type=float, default=0.15, help="root noise for gate-game diversity (light)")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--gate-seconds", type=float, default=60.0, help="poll cadence between gate cycles")
    p.add_argument("--device", default="cpu", help="device for the Python fallback (native runs on CPU C++)")
    p.add_argument("--anchor-pairs", type=int, default=0, help="pairs for the vs-anchor per-side capacity bench (0 = use --pairs)")
    p.add_argument("--ladder-rungs", default="1,4,16",
                   help="comma offsets of PAST champions to bench the new best against on "
                        "promotion (regression ladder; empty disables)")
    p.add_argument("--ladder-pairs", type=int, default=2,
                   help="color-swapped pairs per seed for each ladder rung (kept small — it runs only on promotion)")
    p.add_argument("--keep-nets", type=int, default=32,
                   help="retain only the newest N archived champions in nets/ (0 = keep all); "
                        "floored above the deepest ladder rung so rungs always survive GC")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir: Path = args.run_dir.resolve()
    weights_path = run_dir / "weights.pt"
    best_path = run_dir / "best.pt"
    nets_dir = run_dir / "nets"
    anchor_path = nets_dir / "anchor.pt"
    arena_dir = run_dir / "_arena"          # working .bin exports for the native gate
    log_path = run_dir / "gate_log.jsonl"
    stop_path = run_dir / "STOP"
    match_path = run_dir / "match.json"           # open-gate manifest the server hands to clients
    results_dir = run_dir / "match_results"       # client gate outcomes (server writes them here)
    cand_served = run_dir / "cand.pt"             # candidate net (server serves it by sha via /get_network)
    served_dir = run_dir / "match_nets"           # per-opponent gate nets (served by sha via /get_network)
    nets_dir.mkdir(parents=True, exist_ok=True)
    arena_dir.mkdir(parents=True, exist_ok=True)
    if match_path.exists():
        match_path.unlink()                       # no gate open at startup; drop any stale manifest

    arch = {"d_hidden": args.d_hidden, "c_filters": args.c_filters, "n_blocks": args.n_blocks}
    anchor_pairs = args.anchor_pairs or args.pairs
    ladder_offsets = [int(x) for x in args.ladder_rungs.split(",") if x.strip()]
    # Keep enough champions that every ladder rung still exists after GC.
    keep_floor = max(args.keep_nets, max(ladder_offsets) + 1) if (args.keep_nets and ladder_offsets) else args.keep_nets
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    client = PyVariantClient()
    rand_pick = _random_picker()
    seed_counter = itertools.count(1)       # per-move Dirichlet seed (native diversity)

    seeds = [ln.strip() for ln in args.seed_mix_file.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    if not seeds:
        log.error("no seeds in %s", args.seed_mix_file)
        return 2
    backend = "native(cpp)" if NATIVE_OK else "python(mcts)"
    log.info("arena up: backend=%s seeds=%d sims=%d pairs=%d tau=%.2f floor=%.2f max_plies=%d",
             backend, len(seeds), args.sims, args.pairs, args.threshold, args.side_floor, args.max_plies)
    log.info("regression ladder rungs=%s pairs=%d | keep-nets=%s",
             ladder_offsets or "off", args.ladder_pairs, keep_floor or "all")

    # Establish best v0 (the gated champion) + the frozen anchor. Adopt the
    # trainer's current weights as the first champion so self-play has something
    # to pull immediately, before any gate has run.
    while not best_path.exists():
        if stop_path.exists():
            return 0
        if weights_path.exists():
            ts0 = int(time.time())
            _atomic_copy(weights_path, best_path)
            _atomic_copy(weights_path, nets_dir / f"net-{ts0}.pt")
            if not anchor_path.exists():
                _atomic_copy(weights_path, anchor_path)
            log.info("best v0 seeded from %s (anchor frozen) @ %s -> %s",
                     weights_path.name, _clock(ts0), best_path.name)
            break
        log.info("waiting for trainer to publish weights.pt ...")
        time.sleep(5.0)

    # Wall-clock of the current champion: best.pt's mtime (the v0 seed just now,
    # or a pre-existing best on resume). Drives the "since last best" readouts.
    last_best_time = best_path.stat().st_mtime
    best_net = _make_net(best_path, arch, arena_dir / "best.bin", device)
    anchor_net = _make_net(anchor_path if anchor_path.exists() else best_path,
                           arch, arena_dir / "anchor.bin", device)
    best_elo = _last_elo(log_path)
    last_cand: str | None = None
    promotions = 0
    idle_logged = False

    def _picker(net):
        if NATIVE_OK:
            return _native_picker(net, args.sims, args.c_puct, args.dirichlet_alpha,
                                  args.dirichlet_eps, seed_counter)
        return _model_picker(net, client, args.sims, args.c_puct, args.dirichlet_alpha, args.dirichlet_eps)

    while not stop_path.exists():
        cand_path = _newest_candidate(run_dir, settle_s=3.0)
        if cand_path is None or cand_path.name == last_cand:
            if not idle_logged:
                log.info("caught up (newest gated: %s) — best is %s old (last promoted %s) — waiting for the next checkpoint",
                         last_cand or "none", _fmt_dur(time.time() - last_best_time), _clock(last_best_time))
                idle_logged = True
            time.sleep(args.gate_seconds)
            continue
        idle_logged = False

        log.info("new candidate %s — labelling %d seeds (best vs random)...", cand_path.name, len(seeds))
        cand_net = _make_net(cand_path, arch, arena_dir / "cand.bin", device)
        labels = _label_seeds(_picker(best_net), rand_pick, client, seeds, args.max_plies)
        glyph = {"white": "W", "black": "B", "balanced": "~"}
        # Opponent panel: the last --gate-opponents champions, newest first. panel[0] is the
        # current best; the rest are older champions from nets/ (fewer than N early in a run).
        # Promotion needs the full bar vs the immediate best AND no regression (>= --no-regress)
        # vs each older champion — the regression ladder folded into the gate, so strength can't
        # cycle backwards. The arena does NOT play the panel: it serves the candidate + every
        # champion net (clients fetch both by sha via /get_network) and the FLEET plays every
        # opponent's games, so the panel is just (id, file) to serve — no net is loaded here.
        champ_paths = list(reversed(sorted(nets_dir.glob("net-*.pt"), key=_net_ts)))  # newest first
        panel = [("best", best_path)] + [(cp.stem, cp) for cp in champ_paths[1:args.gate_opponents]]
        panel_oppids = [oppid for oppid, _src in panel]
        per_opp = len(seeds) * args.pairs * 2
        need = len(panel) * per_opp
        log.info("GATE START %s vs last %d champion(s) | favored: %s | %d x %d games = %d total (the fleet plays them; the arena tallies)",
                 cand_path.name, len(panel),
                 ", ".join(f"{_seed_tag(s)}={glyph[labels[s]]}" for s in seeds),
                 len(panel), per_opp, need)

        # Open ONE gate covering the whole panel: clear stale results, serve the candidate and
        # every opponent net, write the manifest (opponent ids newest-first — index 0 is the
        # primary vs-best). The server then round-robins (opponent x seed x side) as `match`
        # jobs to every client; the arena waits below until the fleet has played the whole panel.
        match_id = int(time.time())
        for d in (results_dir, served_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        _atomic_copy(cand_path, cand_served)
        for oppid, src in panel:
            _atomic_copy(src, served_dir / f"{oppid}.pt")
        _write_json(match_path, {
            "match_id": match_id,
            "seeds": seeds,
            "arch": arch,
            "opponents": panel_oppids,
            "params": {"sims": args.sims, "c_puct": args.c_puct,
                       "dir_alpha": args.dirichlet_alpha, "dir_eps": args.dirichlet_eps,
                       "max_plies": args.max_plies},
        })

        # Wait for the FLEET to play the whole panel — the arena plays NO gate game (lc0: the
        # server host dispatches + tallies). Drain client results until every (opponent, seed,
        # side) has --pairs games; STOP abandons the open gate. With no client connected this
        # never completes (intentional — the local box always runs a loopback self-play client).
        collector = _GateCollector(results_dir, match_id)
        last_progress = 0.0
        while not stop_path.exists():
            have = collector.have(panel_oppids, seeds, args.pairs)
            if have >= need:
                break
            now = time.time()
            if now - last_progress >= 30.0:
                log.info("  GATE %s: %d/%d fleet results in — waiting...", cand_path.name, have, need)
                last_progress = now
            time.sleep(min(args.gate_seconds, 5.0))
        try:
            match_path.unlink()                         # close the gate: clients fall back to self-play
        except OSError:
            pass
        if stop_path.exists():
            break                                       # STOP during the gate — abandon this candidate

        opp_results = []
        for oi, oppid in enumerate(panel_oppids):
            r = _score_opp(collector.collected_for(oppid, seeds), seeds, labels, args.pairs)
            opp_results.append((oppid, r))
            log.info("  panel %d/%d vs %-14s -> balanced=%.3f min_class=%.3f (cand W%d-L%d-D%d of %d)",
                     oi + 1, len(panel), oppid, r["balanced_score"], r["min_class"],
                     r["record"]["w"], r["record"]["l"], r["record"]["d"], per_opp)
        log.info("  gate complete: %d fleet games tallied across %d opponent(s)", need, len(panel))

        res = opp_results[0][1]                         # vs immediate best — drives elo + the promote bar
        primary_ok = res["balanced_score"] >= args.threshold and res["min_class"] >= args.side_floor
        regress = [(n, rr["balanced_score"]) for n, rr in opp_results[1:]
                   if rr["balanced_score"] < args.no_regress]
        promoted = primary_ok and not regress

        rec = {
            "ts": int(time.time()),
            "candidate": cand_path.name,
            "labels": {_seed_tag(s): labels[s] for s in seeds},
            "panel": [{"net": n, "balanced": rr["balanced_score"], "min_class": rr["min_class"],
                       "record": rr["record"]} for n, rr in opp_results],
            "promoted": promoted,
            **res,
        }

        if promoted:
            promotions += 1
            ts = int(time.time())
            past = sorted(nets_dir.glob("net-*.pt"), key=_net_ts)  # champions BEFORE this one (ladder rungs)
            _atomic_copy(cand_path, best_path)             # server versions on best.pt -> clients pull
            _atomic_copy(cand_path, nets_dir / f"net-{ts}.pt")
            best_net = cand_net                            # in-memory net; cand.bin may be overwritten later
            best_elo += _elo_delta(res["balanced_score"])
            rec["best_elo"] = round(best_elo, 1)
            log.info("PROMOTED %s -> best #%d @ %s | %s since previous best | record W%d-L%d-D%d (score=%.3f) W=%s B=%s | elo=%.1f | benchmarking vs anchor...",
                     cand_path.name, promotions, _clock(ts), _fmt_dur(ts - last_best_time),
                     res["record"]["w"], res["record"]["l"], res["record"]["d"], res["balanced_score"],
                     res["class_white"], res["class_black"], best_elo)
            last_best_time = ts
            # Per-side capacity: new best vs the FROZEN anchor (separable, unlike vs best).
            cap = _gate(_picker(best_net), _picker(anchor_net), client, seeds, labels,
                        anchor_pairs, args.max_plies, tag="anchor")
            rec["anchor_white"] = cap["class_white"]
            rec["anchor_black"] = cap["class_black"]
            log.info("  vs-anchor per-side: W=%s B=%s (capacity curve)", cap["class_white"], cap["class_black"])
            # Regression ladder: new best vs several PAST champions (-k back). The gate
            # only ever compares to the *immediate* best, so it can't see strength
            # cycling — a net beating its predecessor while weaker than one from N
            # promotions ago. A rung score < 0.5 means we regressed vs that older net.
            if ladder_offsets and past:
                ladder = []
                for k in ladder_offsets:
                    if k > len(past):
                        continue
                    rung = past[-k]
                    rung_net = _make_net(rung, arch, arena_dir / "rung.bin", device)
                    lr = _gate(_picker(best_net), _picker(rung_net), client, seeds, labels,
                               args.ladder_pairs, args.max_plies, tag=f"ladder-{k}")
                    gap = _elo_delta(lr["balanced_score"])
                    ladder.append({"back": k, "net": rung.name,
                                   "score": lr["balanced_score"], "elo_gap": round(gap, 1)})
                    log.info("  ladder -%d vs %s: score=%.3f elo_gap=%+.1f%s", k, rung.name,
                             lr["balanced_score"], gap,
                             "  <-- REGRESSION" if lr["balanced_score"] < 0.5 else "")
                rec["ladder"] = ladder
                if ladder:
                    summ = "  ".join(
                        f"-{d['back']}={d['score']:.3f}({d['elo_gap']:+.0f}elo"
                        + (" REGRESSED" if d['score'] < 0.5 else "") + ")"
                        for d in ladder)
                    log.info("  regression ladder vs past champions (now best #%d): %s", promotions, summ)
            # Retention: cap nets/ at the newest keep_floor champions (>= the deepest
            # ladder rung, so rungs survive). anchor.pt is a separate filename, untouched.
            if keep_floor:
                gone = _gc_nets(nets_dir, keep_floor)
                if gone:
                    log.info("  pruned %d old champion net(s) (kept newest %d)", gone, keep_floor)
        else:
            why = []
            if not primary_ok:
                why.append(f"vs-best score={res['balanced_score']:.3f} (need {args.threshold:.2f}) "
                           f"min_class={res['min_class']:.3f} (need {args.side_floor:.2f})")
            if regress:
                why.append("regressed vs " + ", ".join(f"{n}={s:.3f}<{args.no_regress:.2f}" for n, s in regress))
            log.info("rejected %s | %s | record W%d-L%d-D%d W=%s B=%s",
                     cand_path.name, " ; ".join(why),
                     res["record"]["w"], res["record"]["l"], res["record"]["d"],
                     res["class_white"], res["class_black"])

        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        last_cand = cand_path.name

    log.info("arena stopped (STOP). %d promotions, best_elo=%.1f", promotions, best_elo)
    return 0


def _seed_tag(fen: str) -> str:
    """Short label for a seed FEN (reuse the trainer's compact tag)."""
    from chessckers_engine.selfplay_az_loop import _seed_tag as _t
    return _t(fen)


def _net_ts(path: Path) -> int:
    """Unix ts embedded in an archived champion filename `net-<ts>.pt` (sort key)."""
    try:
        return int(path.stem.split("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def _gc_nets(nets_dir: Path, keep: int) -> int:
    """Delete all but the newest `keep` archived champions (`net-*.pt`); returns the
    count removed. anchor.pt is a different filename and is never matched."""
    champs = sorted(nets_dir.glob("net-*.pt"), key=_net_ts)
    removed = 0
    for p in (champs[:-keep] if keep else []):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


if __name__ == "__main__":
    raise SystemExit(main())
