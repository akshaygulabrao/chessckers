"""Single-process, multi-threaded batched-inference self-play (lc0-style leaf batching).

Runs `--games` concurrent self-play games as THREADS that all share ONE model on the
GPU and ONE `InferenceServer`. Each game thread runs ordinary sequential PUCT MCTS, so
it has exactly ONE outstanding leaf at a time; with N games in flight the server packs
up to N leaves into a single batched forward — amortizing the per-call GPU dispatch cost
that dominates for the big V3 transformer net (≈4.5× a ResNet at batch-1, ≈8× cheaper
per position once batched on MPS).

Why threads / one process (NOT N worker processes):
  * ONE model copy on the GPU — N process copies OOM MPS unified memory;
  * the batcher pools leaves across games via an in-process queue + Futures (no IPC);
  * the heavy ops RELEASE the GIL — the batched GPU forward (torch) and any thread
    parked on its leaf Future — so game threads' CPU work overlaps the GPU batch.

GIL caveat (honest): the Rust movegen + check-detection + the per-leaf encoders all run
HOLDING the GIL (the crate has no `py.allow_threads`), so the *CPU* MCTS work across
games SERIALIZES. The win here is overlapping that serialized CPU with the batched GPU
forward — which is exactly why this pays off for the expensive V3 net and barely moves
the tiny V1. Releasing the GIL in the movegen crate is the follow-on lever for true CPU
parallelism across game threads.

Thread roles (all daemon, one process):
  * M game threads      — each: own PyVariantClient + RNG, loop play_az_game→queue→respawn
  * 1 inference thread  — InferenceServer: pools M leaves → one batch_eval (GIL released)
  * 1 writer thread     — drains finished games → ReplayBuffer (disk I/O off the hot path)
  * 1 monitor thread    — verbose throughput + batch-fill stats every --log-seconds

Tune `--games` (= the inference batch width) and `--batch-timeout-ms` to the exact GPU:
watch the logged `avg_bs/MAX (NN% full)` — if it sits well below 100%, the timeout is too
short to gather the pool (raise it) or you have too few games for the GPU (raise --games).

Example (local, V3 net on MPS, 16 concurrent games, run 2 minutes):
    .venv/bin/python -m chessckers_engine.selfplay_batched \
        --net weights/run/best.pt --games 16 --sims 200 --device mps --duration 120
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
from pathlib import Path

import torch

log = logging.getLogger("chessckers_engine.selfplay_batched")

# The d6/e6/f6-vs-8-pawns curriculum position V1/V2/V3 all trained from (see
# scripts/v3_run.sh / ab_v1_v2.sh). Self-play seeds here unless --seed-fen overrides.
DEFAULT_SEED_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"


class _Stats:
    """Thread-safe self-play counters, updated by the writer thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.games = 0
        self.examples = 0
        self.plies = 0
        self.outcomes = {"white": 0, "black": 0, "draw": 0}

    def record_game(self, outcome: str, n_plies: int, n_examples: int) -> None:
        with self._lock:
            self.games += 1
            self.plies += n_plies
            self.examples += n_examples
            self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1

    def snapshot(self) -> tuple[int, int, int, dict[str, int]]:
        with self._lock:
            return self.games, self.examples, self.plies, dict(self.outcomes)

    def count(self) -> int:
        with self._lock:
            return self.games


def _game_worker(
    tid: int,
    stop_event: threading.Event,
    server,
    results_q: "queue.Queue",
    params: dict,
    base_seed: int,
) -> None:
    """One game thread: play full self-play games back-to-back against the shared
    batched server, pushing each finished game to the writer. Respawns immediately
    so the inference pool always has ~M leaves to batch."""
    from chessckers_engine.selfplay_az import play_az_game
    from chessckers_engine.variant_py import PyVariantClient

    client = PyVariantClient()  # per-thread client: no shared mutable state / locks
    rng = torch.Generator().manual_seed(base_seed + tid)
    while not stop_event.is_set():
        try:
            game = play_az_game(
                server, client,
                n_sims=params["sims"],
                c_puct=params["c_puct"],
                temperature=params["temperature"],
                max_plies=params["max_plies"],
                rng=rng,
                dirichlet_alpha=params["dirichlet_alpha"],
                dirichlet_eps=params["dirichlet_eps"],
            )
        except Exception as e:  # noqa: BLE001 — one bad game must not kill the thread
            log.warning("game thread %d: play_az_game failed: %r", tid, e)
            continue
        # Block on a full queue (backpressure) but stay responsive to shutdown.
        while not stop_event.is_set():
            try:
                results_q.put(game, timeout=0.2)
                break
            except queue.Full:
                continue


def _writer_worker(
    stop_event: threading.Event,
    results_q: "queue.Queue",
    buffer,
    stats: _Stats,
    num_games: int,
) -> None:
    """Drain finished games → training examples → ReplayBuffer. Runs on its own
    thread so disk I/O (GIL-released) never stalls the game threads. Sets
    stop_event once --num-games is reached."""
    from chessckers_engine.selfplay_az import az_game_to_examples

    game_id = 0
    while True:
        try:
            game = results_q.get(timeout=0.2)
        except queue.Empty:
            if stop_event.is_set():
                break  # stopping and nothing left to drain
            continue
        game_id += 1
        examples = az_game_to_examples(game)
        try:
            gp = buffer.append_game(worker_id=0, game_id=game_id, examples=examples)
            try:
                Path(str(gp) + ".meta").write_text(
                    '{"worker_id": 0, "machine": "batched", "outcome": "%s", '
                    '"plies": %d, "seed_fen": %s}'
                    % (game.outcome, len(game.records),
                       _json_str(game.records[0].fen if game.records else None))
                )
            except OSError:
                pass
        except Exception as e:  # noqa: BLE001
            log.warning("writer: append_game failed: %r", e)
        stats.record_game(game.outcome, len(game.records), len(examples))
        if num_games and stats.count() >= num_games:
            stop_event.set()


def _json_str(s: str | None) -> str:
    if s is None:
        return "null"
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _monitor_worker(
    stop_event: threading.Event,
    server,
    stats: _Stats,
    log_seconds: float,
    max_batch: int,
    n_games: int,
) -> None:
    """Verbose, trackable progress: throughput + inference batch-fill every tick."""
    t0 = time.time()
    last_t, last_games = t0, 0
    while not stop_event.wait(log_seconds):
        now = time.time()
        games, examples, plies, oc = stats.snapshot()
        el = max(now - t0, 1e-6)
        dt = max(now - last_t, 1e-6)
        rate_inst = (games - last_games) / dt * 60.0
        rate_avg = games / el * 60.0
        s = server.stats()
        avg_bs = s["avg_batch_size"]
        fill = (avg_bs / max_batch * 100.0) if max_batch else 0.0
        log.info(
            "[t=%4.0fs] games=%-4d %5.1f/min (avg %5.1f) | examples=%-6d %4.0f pos/s | "
            "W/B/D=%d/%d/%d plies/g=%4.1f | inflight=%d | "
            "infer: batches=%-5d avg_bs=%4.1f/%d (%3.0f%% full) max=%d gpu=%4.1fs "
            "wait_first=%4.1fs wait_drain=%4.1fs",
            el, games, rate_inst, rate_avg, examples, examples / el,
            oc.get("white", 0), oc.get("black", 0), oc.get("draw", 0),
            plies / max(games, 1), n_games,
            s["n_batches"], avg_bs, max_batch, fill, s["max_batch_size_seen"],
            s["inference_secs"], s["wait_first_secs"], s["wait_drain_secs"],
        )
        last_t, last_games = now, games


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="chessckers_engine.selfplay_batched",
        description="Single-process, multi-threaded batched-inference self-play.",
    )
    ap.add_argument("--net", default="weights/run/best.pt",
                    help="checkpoint (.pt); arch read from its .arch.json sidecar")
    ap.add_argument("--fallback-version", default="v2",
                    help="arch version if the checkpoint has no sidecar")
    ap.add_argument("--games", type=int, default=16,
                    help="concurrent self-play games = inference batch width; "
                         "TUNE THIS to the GPU (watch avg_bs %% full in the logs)")
    ap.add_argument("--max-batch", type=int, default=0,
                    help="inference max batch size (0 = --games)")
    ap.add_argument("--batch-timeout-ms", type=float, default=12.0,
                    help="how long the batcher waits to gather the pool after the "
                         "first leaf lands; raise it if avg_bs sits below --games")
    ap.add_argument("--sims", type=int, default=200, help="MCTS sims per move")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--dirichlet-eps", type=float, default=0.25)
    ap.add_argument("--device", default="mps", help="mps | cpu | cuda")
    ap.add_argument("--seed-fen", default=DEFAULT_SEED_FEN,
                    help="self-play start FEN (sets CHESSCKERS_START_FEN)")
    ap.add_argument("--buffer-root", default="weights/batched_selfplay/buffer",
                    help="ReplayBuffer dir for the training chunks produced")
    ap.add_argument("--num-games", type=int, default=0,
                    help="stop after N games (0 = until --duration / Ctrl-C)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after S seconds (0 = until --num-games / Ctrl-C)")
    ap.add_argument("--log-seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="torch intra-op CPU threads (1 avoids fighting the game "
                         "threads for cores; the real compute is the batched GPU forward)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
    )

    os.environ["CHESSCKERS_START_FEN"] = args.seed_fen
    os.environ.setdefault("CHESSCKERS_MAX_PLIES", str(args.max_plies))
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    max_batch = args.max_batch or args.games

    from chessckers_engine.checkpoints import load_scorer
    from chessckers_engine.inference_server import InferenceServer
    from chessckers_engine.replay_buffer import ReplayBuffer

    device = torch.device(args.device)
    model = load_scorer(args.net, fallback_version=args.fallback_version).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "loaded %s | VERSION=%s params=%d | device=%s | games=%d max_batch=%d "
        "timeout=%.0fms sims=%d temp=%.2f max_plies=%d",
        args.net, getattr(model, "VERSION", "?"), n_params, device,
        args.games, max_batch, args.batch_timeout_ms, args.sims,
        args.temperature, args.max_plies,
    )
    log.info("seed FEN: %s", args.seed_fen)

    buffer = ReplayBuffer(args.buffer_root)
    log.info("training chunks -> %s", Path(args.buffer_root).resolve())

    server = InferenceServer(
        model, max_batch_size=max_batch, timeout_ms=args.batch_timeout_ms, log_every=0,
    )
    stop_event = threading.Event()
    results_q: "queue.Queue" = queue.Queue(maxsize=max(args.games * 2, 4))
    stats = _Stats()
    params = {
        "sims": args.sims, "c_puct": args.c_puct, "temperature": args.temperature,
        "max_plies": args.max_plies, "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_eps": args.dirichlet_eps,
    }

    writer = threading.Thread(
        target=_writer_worker, name="writer", daemon=True,
        args=(stop_event, results_q, buffer, stats, args.num_games),
    )
    monitor = threading.Thread(
        target=_monitor_worker, name="monitor", daemon=True,
        args=(stop_event, server, stats, args.log_seconds, max_batch, args.games),
    )
    games = [
        threading.Thread(
            target=_game_worker, name=f"game-{i}", daemon=True,
            args=(i, stop_event, server, results_q, params, args.seed),
        )
        for i in range(args.games)
    ]
    writer.start()
    monitor.start()
    for t in games:
        t.start()
    log.info("started %d game threads + writer + monitor", args.games)

    t_start = time.time()
    try:
        if args.duration > 0:
            stop_event.wait(args.duration)
            stop_event.set()
        else:
            # Run until --num-games (writer sets stop_event) or Ctrl-C.
            while not stop_event.wait(0.5):
                pass
    except KeyboardInterrupt:
        log.info("interrupted — stopping ...")
        stop_event.set()

    # Snappy stop: in-flight games are mid-search (no value target yet → not
    # persistable), so abandon them. Do NOT join the game threads — they're
    # daemon + mid-game (each would block the full timeout), so 16 sequential
    # joins = dead minutes. Just drain already-finished games from the queue via
    # the writer, then let the daemon game threads die at process exit.
    log.info("stopping: draining finished games (in-flight games abandoned) ...")
    writer.join(timeout=10)
    server.shutdown()  # logs the inference batching summary

    games_done, examples, plies, oc = stats.snapshot()
    el = time.time() - t_start
    log.info(
        "DONE in %.0fs | games=%d (%.1f/min) | examples=%d (%.0f pos/s) | "
        "W/B/D=%d/%d/%d | avg_plies=%.1f",
        el, games_done, games_done / max(el, 1e-6) * 60.0, examples,
        examples / max(el, 1e-6), oc.get("white", 0), oc.get("black", 0),
        oc.get("draw", 0), plies / max(games_done, 1),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
