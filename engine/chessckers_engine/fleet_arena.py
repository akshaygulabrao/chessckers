"""Fleet arena — lc0/AGZ-style keep-best gating for the continuous trainer.

Runs on the trainer box, sharing the trainer's run-dir. `train_continuous` keeps
publishing candidate checkpoints (`iter-async-*.pt`); this arena decides which
candidate becomes the gated champion `best.pt` that self-play actually uses (the
server versions on `best.pt`, so workers reload once per PROMOTION, not per
iteration). It also archives every champion by timestamp, so we can read empirically
when the network stops improving.

The promotion gate is played by the FLEET, lc0-style: the arena opens a gate
(writes match.json), the server round-robins the candidate-vs-panel games to the
self-play clients, which POST each result back; the arena only TALLIES them. The
trainer/server host never plays a gate game itself (lc0's server/client
responsibility split) — it holds nets as files (served content-addressed by sha) and
decides promotion from the tally alone. With no client connected a gate simply never
completes — intentional; the local box always runs a loopback self-play client. (The
game-playing primitives below — `_play_from`, the pickers, `_make_net` — are still
defined here only because the fleet's client-side runner `fleet_match.MatchRunner`
imports them; the arena itself calls none of them.)

Promotion rule (single balanced seed -> defer to lc0, plus one deviation)
-------------------------------------------------------------------------
With a single balanced curriculum seed the per-side labelling/class-balancing older
revisions did is meaningless, so the gate is lc0's plain color-swapped win-rate:

  1. COLOR-SWAPPED SEED PAIRS (this IS lc0's both-sides logic). Each seed is played
     as pairs (candidate-as-White/best-as-Black, then swapped), so any structural
     side-advantage is handed to BOTH nets equally and cancels — the score is
     bias-free skill, not "who drew the winning side".
  2. PROMOTE iff  score-vs-best >= --threshold.

REGRESSION LADDER (the one deliberate deviation from lc0, BLOCKING — prevents
rock-paper-scissors). The SAME gate also plays the candidate against several PAST
champions — the rungs, at --ladder-rungs offsets back (1/4/16 by default) — and
REJECTS a candidate that beats the immediate best but regresses (score < --no-regress)
against any older champion. lc0's single-best gate can't see strength cycling
backwards; keeping the multi-champion comparison in the gate stops it promoting a net
that loses to one from N promotions ago. The whole panel (best + rungs) is fleet-played
in one gate.

On promotion: best.pt <- candidate, archive nets/net-<unix_ts>.pt, bump the cumulative
Elo by the gate margin. Archived champions are capped at the newest --keep-nets (floored
above the deepest ladder rung, so rungs always survive GC). Everything is appended to
gate_log.jsonl.

Run on the trainer box, sharing its run-dir:

    python -m chessckers_engine.fleet_arena --run-dir weights/run \\
      --seed-mix-file ../scripts/seed_mix.txt --d-hidden 256 --c-filters 96 --n-blocks 4
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.evaluate import _state_to_outcome
from chessckers_engine.mcts_puct import run_mcts
from chessckers_engine.model import ChesskersScorer, build_model
from chessckers_engine.runtime import setup_logging

log = logging.getLogger("chessckers_engine.fleet_arena")

# Native C++ search primitives, imported by the fleet's client-side match runner
# (fleet_match.MatchRunner) — NOT by the arena, which plays no game. Optional: the
# pickers fall back to the Python MCTS if the extension isn't built on this box.
try:
    import chessckers_cpp as _cpp
    from chessckers_engine.native_net import export_state_dict as _export_state_dict
    NATIVE_OK = True
except Exception:  # noqa: BLE001
    _cpp = None
    NATIVE_OK = False


# --- game-playing primitives (shared) ------------------------------------------
# These run a gate game with the search the fleet uses. The ARENA no longer plays any
# game (it dispatches + tallies); they live here because fleet_match.MatchRunner imports
# them verbatim, so a client gate game is the same computation end to end.

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


# --- the gate ------------------------------------------------------------------

def _score(outcome: str, cand_is_white: bool) -> float:
    """Candidate's points for a single game (1 win / 0.5 draw / 0 loss)."""
    if outcome == "draw":
        return 0.5
    cand_won = (outcome == "white") if cand_is_white else (outcome == "black")
    return 1.0 if cand_won else 0.0


def _score_opp(collected: dict, seeds: list[str], pairs: int) -> dict:
    """Plain color-swapped win-rate of ONE opponent's seed-paired games (lc0's both-sides
    score; one balanced seed, so no per-side class-balancing). `collected` maps
    (seed, cand_white) -> list of outcomes ('white'|'black'|'draw'); the first `pairs` per
    (seed, side) are scored (exactly 2*pairs games/seed). The candidate's mean points over
    those games is its score vs this opponent — the single definition of the gate math,
    shared by every panel opponent (best + the regression-ladder rungs)."""
    per_seed: dict[str, float] = {}
    rec = [0, 0, 0]  # candidate's aggregate [W, L, D] vs opponent across all games
    for seed in seeds:
        pts = 0.0
        for cand_white in (True, False):
            for outcome in collected.get((seed, cand_white), [])[:pairs]:
                s = _score(outcome, cand_white)
                pts += s
                rec[0 if s == 1.0 else 1 if s == 0.0 else 2] += 1  # candidate's W/L/D
        per_seed[seed] = pts / (2 * pairs)
    score = sum(per_seed.values()) / len(per_seed) if per_seed else 0.0
    return {
        "per_seed": {s: round(v, 3) for s, v in per_seed.items()},
        "score": round(score, 3),
        "record": {"w": rec[0], "l": rec[1], "d": rec[2]},  # candidate vs opponent, whole gate
    }


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

def _load_model(path: Path, arch: dict, device):
    m = build_model(**arch).to(device)
    load_checkpoint(m, str(path))
    m.eval()
    return m


def _export_bin(path: Path, arch: dict, bin_path: Path) -> None:
    """Export a checkpoint's weights to a flat native .bin (what the C++ client loads
    via /get_network). Atomic; arch-driven so V1 and the V2 transformer/gather net
    both serialize."""
    m = build_model(**arch)
    load_checkpoint(m, str(path))
    m.eval()
    tmp = str(bin_path) + ".tmp"
    _export_state_dict(m.state_dict(), tmp)
    os.replace(tmp, bin_path)


def _make_net(path: Path, arch: dict, bin_path: Path, device):
    """Load a checkpoint into a gate-playable net. Native: export the state_dict
    to a flat .bin and return a cc::ChesskersNet (CPU C++ search; supports both V1
    and the V2 transformer/gather net). Fallback: the torch model for the Python
    search when the native extension isn't built."""
    if not NATIVE_OK:
        return _load_model(path, arch, device)
    _export_bin(path, arch, bin_path)
    return _cpp.ChesskersNet(str(bin_path))


def _publish_gate_bin(pt_path: Path, arch: dict) -> None:
    """Phase 4 (lc0-split): write the C++-client .bin twin beside a SERVED gate net
    (.pt), so the native client can fetch the candidate / opponent nets by sha (GET
    /get_network) and play the gate game. Additive + best-effort: a failure never
    blocks the gate — the .pt path (Python match runner) is unaffected."""
    try:
        _export_bin(pt_path, arch, pt_path.with_suffix(".bin"))
    except Exception as e:  # noqa: BLE001
        log.warning("gate .bin export failed for %s (%s) — C++ client can't fetch it",
                    pt_path.name, e)


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    # Carry the arch sidecar (build_model recipe) so every copied checkpoint —
    # best.pt and the archived nets/ champions — stays self-describing for offline
    # loaders (checkpoints.load_scorer / the eval gauntlet). No-op if src has none.
    src_arch = Path(str(src) + ".arch.json")
    if src_arch.exists():
        dst_arch = Path(str(dst) + ".arch.json")
        atmp = Path(str(dst_arch) + ".tmp")
        shutil.copy2(src_arch, atmp)
        os.replace(atmp, dst_arch)


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
    p.add_argument("--arch-version", choices=["v1", "v2"], default="v1",
                   help="Net arch the gate nets use: v1 (pooled) or v2 (gather head + optional transformer). "
                        "MUST match the trainer; rides into match.json so gate clients rebuild the exact net.")
    p.add_argument("--tf-blocks", type=int, default=0,
                   help="V2 only: Transformer blocks interleaved into the trunk (0 = pure ResNet).")
    p.add_argument("--tf-heads", type=int, default=4, help="V2 transformer attention heads.")
    p.add_argument("--tf-ff-mult", type=int, default=4, help="V2 transformer feed-forward expansion.")
    p.add_argument("--sims", type=int, default=160, help="MCTS sims per move the fleet uses for gate games")
    p.add_argument("--pairs", type=int, default=4, help="color-swapped pairs per seed per opponent (2x games/seed)")
    p.add_argument("--threshold", type=float, default=0.55, help="win-rate vs current best to promote (lc0 gate)")
    p.add_argument("--ladder-rungs", default="1,4,16",
                   help="regression ladder: PAST champions to ALSO play the candidate against in the gate, "
                        "each a BLOCKING no-regress rung (see --no-regress). 'all' = EVERY previous champion "
                        "(unbounded — gate cost grows with history and nets/ is never GC'd); a comma list = "
                        "those offsets back (e.g. 1,4,16); empty = lc0 single-best gate.")
    p.add_argument("--no-regress", type=float, default=0.50,
                   help="min win-rate vs each ladder rung (older champion) to promote — the blocking "
                        "anti-rock-paper-scissors guard the single-best gate is blind to.")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--dirichlet-eps", type=float, default=0.15, help="root noise for gate-game diversity (light)")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--gate-seconds", type=float, default=60.0, help="poll cadence between gate cycles")
    p.add_argument("--keep-nets", type=int, default=32,
                   help="retain only the newest N archived champions in nets/ (0 = keep all); "
                        "floored above the deepest ladder rung so rungs always survive GC")
    args = p.parse_args()

    run_dir: Path = args.run_dir.resolve()
    weights_path = run_dir / "weights.pt"
    best_path = run_dir / "best.pt"
    nets_dir = run_dir / "nets"
    log_path = run_dir / "gate_log.jsonl"
    stop_path = run_dir / "STOP"
    match_path = run_dir / "match.json"           # open-gate manifest the server hands to clients
    results_dir = run_dir / "match_results"       # client gate outcomes (server writes them here)
    cand_served = run_dir / "cand.pt"             # candidate net (server serves it by sha via /get_network)
    served_dir = run_dir / "match_nets"           # per-opponent gate nets (served by sha via /get_network)
    nets_dir.mkdir(parents=True, exist_ok=True)
    if match_path.exists():
        match_path.unlink()                       # no gate open at startup; drop any stale manifest

    arch = {"version": args.arch_version, "d_hidden": args.d_hidden,
            "c_filters": args.c_filters, "n_blocks": args.n_blocks}
    if args.arch_version == "v2":
        arch.update(n_tf_blocks=args.tf_blocks, n_heads=args.tf_heads, tf_ff_mult=args.tf_ff_mult)
    ladder_all = args.ladder_rungs.strip().lower() == "all"
    ladder_offsets = [] if ladder_all else [int(x) for x in args.ladder_rungs.split(",") if x.strip()]
    # Keep enough champions that every ladder rung still exists after GC. Under `all` the
    # ladder IS the full history, so never GC (keep_floor=0 -> _gc_nets keeps everything).
    if ladder_all:
        keep_floor = 0
    else:
        keep_floor = max(args.keep_nets, max(ladder_offsets) + 1) if (args.keep_nets and ladder_offsets) else args.keep_nets

    seeds = [ln.strip() for ln in args.seed_mix_file.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    if not seeds:
        log.error("no seeds in %s", args.seed_mix_file)
        return 2
    log.info("arena up (dispatch+tally, plays no game): seeds=%d sims=%d pairs=%d threshold=%.2f no_regress=%.2f max_plies=%d",
             len(seeds), args.sims, args.pairs, args.threshold, args.no_regress, args.max_plies)
    log.info("regression ladder offsets=%s (BLOCKING) | keep-nets=%s",
             "all (every previous champion)" if ladder_all else (ladder_offsets or "off"),
             keep_floor or "all")

    # Establish best v0 (the gated champion). Adopt the trainer's current weights as the
    # first champion so self-play has something to pull immediately, before any gate runs.
    while not best_path.exists():
        if stop_path.exists():
            return 0
        if weights_path.exists():
            ts0 = int(time.time())
            _atomic_copy(weights_path, best_path)
            _atomic_copy(weights_path, nets_dir / f"net-{ts0}.pt")
            log.info("best v0 seeded from %s @ %s -> %s", weights_path.name, _clock(ts0), best_path.name)
            break
        log.info("waiting for trainer to publish weights.pt ...")
        time.sleep(5.0)

    # Wall-clock of the current champion: best.pt's mtime (the v0 seed just now, or a
    # pre-existing best on resume). Drives the "since last best" readouts. The arena holds
    # nets only as FILES (served content-addressed); it never loads one into memory.
    last_best_time = best_path.stat().st_mtime
    best_elo = _last_elo(log_path)
    last_cand: str | None = None
    promotions = 0
    idle_logged = False

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

        log.info("new candidate %s — opening gate over %d seed(s)", cand_path.name, len(seeds))
        # Opponent panel: the current best (PRIMARY — the candidate must beat it, lc0's gate) +
        # the regression-ladder rungs: past champions selected by --ladder-rungs (offsets back like
        # 1/4/16, or 'all' = every previous champion), each a BLOCKING no-regress guard so strength
        # can't cycle backwards (the one chessckers deviation from lc0's single-best gate). The
        # panel grows by one rung per promotion under 'all'. The arena plays NONE of it — it serves the
        # candidate + every opponent net (clients fetch both by sha via /get_network) and the
        # FLEET plays every game, so the panel is just (id, file) to serve; no net is loaded here.
        champ_paths = sorted(nets_dir.glob("net-*.pt"), key=_net_ts, reverse=True)  # newest first; [0]==best
        if ladder_all:
            rungs = [(p.stem, p) for p in champ_paths[1:]]   # EVERY previous champion (unbounded ladder)
        else:
            rungs = [(champ_paths[k].stem, champ_paths[k]) for k in ladder_offsets if 0 < k < len(champ_paths)]
        panel = [("best", best_path)] + rungs
        panel_oppids = [oppid for oppid, _src in panel]
        per_opp = len(seeds) * args.pairs * 2
        need = len(panel) * per_opp
        rung_names = ", ".join(n for n, _ in rungs[:6])
        if len(rungs) > 6:
            rung_names += f", ... (+{len(rungs) - 6} more)"
        log.info("GATE START %s | primary=best + %d ladder rung(s) [%s] | %d opponents x %d games = %d total (the fleet plays them; the arena tallies)",
                 cand_path.name, len(rungs), rung_names or "none yet",
                 len(panel), per_opp, need)

        # Open ONE gate covering the whole panel: clear stale results, serve the candidate and
        # every opponent net, write the manifest (opponent ids; index 0 is the primary vs-best).
        # The server then round-robins (opponent x seed x side) as `match` jobs to every client;
        # the arena waits below until the fleet has played the whole panel.
        match_id = int(time.time())
        for d in (results_dir, served_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        _atomic_copy(cand_path, cand_served)
        _publish_gate_bin(cand_served, arch)  # C++-client candidate net (fetch by sha)
        for oppid, src in panel:
            _atomic_copy(src, served_dir / f"{oppid}.pt")
            _publish_gate_bin(served_dir / f"{oppid}.pt", arch)  # C++-client opponent net
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
            r = _score_opp(collector.collected_for(oppid, seeds), seeds, args.pairs)
            opp_results.append((oppid, r))
            log.info("  panel %d/%d vs %-14s -> score=%.3f (cand W%d-L%d-D%d of %d)",
                     oi + 1, len(panel), oppid, r["score"],
                     r["record"]["w"], r["record"]["l"], r["record"]["d"], per_opp)
        log.info("  gate complete: %d fleet games tallied across %d opponent(s)", need, len(panel))

        res = opp_results[0][1]                         # vs immediate best — drives elo + the promote bar
        primary_ok = res["score"] >= args.threshold
        regress = [(n, rr["score"]) for n, rr in opp_results[1:] if rr["score"] < args.no_regress]
        promoted = primary_ok and not regress           # a rung regression BLOCKS promotion

        ladder_rec = [{"net": n, "score": rr["score"], "elo_gap": round(_elo_delta(rr["score"]), 1)}
                      for n, rr in opp_results[1:]]
        rec = {
            "ts": int(time.time()),
            "candidate": cand_path.name,
            "panel": [{"net": n, "score": rr["score"], "record": rr["record"]} for n, rr in opp_results],
            "ladder": ladder_rec,
            "promoted": promoted,
            **res,
        }

        if promoted:
            promotions += 1
            ts = int(time.time())
            _atomic_copy(cand_path, best_path)             # server versions on best.pt -> clients pull
            _publish_gate_bin(best_path, arch)             # C++-client champion net (fetch by sha)
            _atomic_copy(cand_path, nets_dir / f"net-{ts}.pt")
            best_elo += _elo_delta(res["score"])
            rec["best_elo"] = round(best_elo, 1)
            ladder_summ = "  ".join(
                f"{d['net']}={d['score']:.3f}({d['elo_gap']:+.0f}elo"
                + (" REGRESSED" if d['score'] < 0.5 else "") + ")"
                for d in ladder_rec) or "none yet"
            log.info("PROMOTED %s -> best #%d @ %s | %s since previous best | vs-best W%d-L%d-D%d (score=%.3f) | elo=%.1f | ladder: %s",
                     cand_path.name, promotions, _clock(ts), _fmt_dur(ts - last_best_time),
                     res["record"]["w"], res["record"]["l"], res["record"]["d"], res["score"], best_elo, ladder_summ)
            last_best_time = ts
            # Retention: cap nets/ at the newest keep_floor champions (>= the deepest ladder rung,
            # so rungs survive — they're gate opponents and MUST still exist at the next gate).
            if keep_floor:
                gone = _gc_nets(nets_dir, keep_floor)
                if gone:
                    log.info("  pruned %d old champion net(s) (kept newest %d)", gone, keep_floor)
        else:
            why = []
            if not primary_ok:
                why.append(f"vs-best score={res['score']:.3f} (need {args.threshold:.2f})")
            if regress:
                why.append("regressed vs " + ", ".join(f"{n}={s:.3f}<{args.no_regress:.2f}" for n, s in regress))
            log.info("rejected %s | %s | vs-best W%d-L%d-D%d",
                     cand_path.name, " ; ".join(why),
                     res["record"]["w"], res["record"]["l"], res["record"]["d"])

        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        last_cand = cand_path.name

    log.info("arena stopped (STOP). %d promotions, best_elo=%.1f", promotions, best_elo)
    return 0


def _net_ts(path: Path) -> int:
    """Unix ts embedded in an archived champion filename `net-<ts>.pt` (sort key)."""
    try:
        return int(path.stem.split("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def _gc_nets(nets_dir: Path, keep: int) -> int:
    """Delete all but the newest `keep` archived champions (`net-*.pt`); returns the
    count removed."""
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
