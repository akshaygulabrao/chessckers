"""AlphaZero-style iteration loop for Chessckers.

Each iteration:
  1. Play `games_per_iter` self-play games using PUCT MCTS at every move.
  2. Convert each game's records into AZExamples (policy + value targets).
  3. Train the model on the combined examples (dual loss).
  4. Save a numbered checkpoint to `weights/iter-az-{N:03d}.pt`.
  5. Quick eval vs random both ways for a per-iteration win-rate datapoint.

CLI:
    uv run python -m chessckers_engine.selfplay_az_loop \
        --iterations 5 --games-per-iter 8 --sims 25 --epochs 3 \
        [--base path/to/start.pt]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import torch

from chessckers_engine.checkpoints import DEFAULT_WEIGHTS_DIR, load_checkpoint
from chessckers_engine.device import pick_device
from chessckers_engine.evaluate import evaluate as run_eval
from chessckers_engine.inference_server import InferenceServer
from chessckers_engine.mcts_puct import pick_puct
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.random_player import pick_random
from chessckers_engine.selfplay_az import JsonlWatchSink, az_game_to_examples, play_az_game
from chessckers_engine.train_az import save_checkpoint, train_az
from chessckers_engine.variant_py import PyVariantClient

log = logging.getLogger("chessckers_engine.selfplay_az_loop")


def _temperature_for(iter_idx: int, n_iters: int, t0: float, t_final: float) -> float:
    if n_iters <= 1:
        return t0
    return t0 + (t_final - t0) * (iter_idx / (n_iters - 1))


def _make_puct_picker(evaluator, client: PyVariantClient, n_sims: int):
    """Build a state→move picker driven by PUCT MCTS. `evaluator` is either a
    `ChesskersScorer` model (in-thread forward) or an `InferenceServer`
    (cross-game batched inference)."""
    def picker(state):
        return pick_puct(state, client, evaluator, n_sims=n_sims)

    return picker


def _make_random_picker():
    def picker(state):
        return pick_random(state.get("legalMoves") or [])

    return picker


def _gate_against_best(
    current: ChesskersScorer,
    best: ChesskersScorer,
    client,
    n_games: int,
    n_sims: int,
    threshold: float,
    parallel_workers: int = 0,
    model_arch: dict | None = None,
    weights_dir: Path | None = None,
) -> tuple[bool, float, dict]:
    """Head-to-head: play `n_games` between current and best, splitting sides
    evenly. Returns (accept, current_score, breakdown). `accept` is True if
    current's overall score (wins + 0.5×draws) / total_games ≥ threshold.

    parallel_workers > 0 enables the multiprocess path: each game runs in its
    own subprocess loading model state from disk. Requires model_arch +
    weights_dir for staging the per-side checkpoints."""
    half = max(1, n_games // 2)
    if parallel_workers > 0:
        if model_arch is None or weights_dir is None:
            raise ValueError("parallel gating requires model_arch + weights_dir")
        # Stage both models to disk once; workers reload per game (cheap vs game time).
        cur_path = weights_dir / "_gate_current.pt"
        best_path = weights_dir / "_gate_best.pt"
        torch.save(current.state_dict(), cur_path)
        torch.save(best.state_dict(), best_path)
        device = str(next(current.parameters()).device)
        # Half: current=White, best=Black.
        a = _run_eval_parallel(str(cur_path), str(best_path), half, n_sims,
                               model_arch, device, parallel_workers)
        b = _run_eval_parallel(str(best_path), str(cur_path), half, n_sims,
                               model_arch, device, parallel_workers)
    else:
        cur_picker = _make_puct_picker(current, client, n_sims)
        best_picker = _make_puct_picker(best, client, n_sims)
        # Half: current is White, best is Black. Outcome "white" = current wins.
        a = run_eval(cur_picker, best_picker, client, half)
        # Half: best is White, current is Black. Outcome "black" = current wins.
        b = run_eval(best_picker, cur_picker, client, half)
    cur_wins = a["white"] + b["black"]
    cur_losses = a["black"] + b["white"]
    draws = a["draw"] + b["draw"]
    total = cur_wins + cur_losses + draws
    score = (cur_wins + 0.5 * draws) / max(total, 1)
    accept = score >= threshold
    breakdown = {
        "as_white": {"wins": a["white"], "losses": a["black"], "draws": a["draw"]},
        "as_black": {"wins": b["black"], "losses": b["white"], "draws": b["draw"]},
        "score": round(score, 3),
    }
    return accept, score, breakdown


def _play_one_game(
    evaluator,
    n_sims: int, c_puct: float, temperature: float, seed: int,
    dirichlet_alpha: float | None, dirichlet_eps: float,
    sink, sink_context: dict | None,
    client_class=None,
    vloss_batch: int = 1,
    temp_cutoff_plies: int = 30,
):
    """Worker: build a fresh client and rng, play one self-play game.
    Each thread keeps its own connection pool (or in-process state for
    PyVariantClient) and torch.Generator so concurrent games don't share
    mutable state on the explicit-rng path. `evaluator` is shared across
    workers — either a model (each worker calls it sequentially) or an
    InferenceServer (workers' calls get batched on the inference thread)."""
    if client_class is None:
        from chessckers_engine.variant_py import PyVariantClient as _PVC
        client_class = _PVC
    client = client_class()
    rng = torch.Generator().manual_seed(seed)
    try:
        return play_az_game(
            evaluator, client,
            n_sims=n_sims, c_puct=c_puct,
            temperature=temperature, temp_cutoff_plies=temp_cutoff_plies, rng=rng,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            sink=sink, sink_context=sink_context,
            vloss_batch=vloss_batch,
        )
    finally:
        client.close()


def _eval_game_subprocess(payload: dict) -> str:
    """Play one eval/gating game in a worker subprocess. Returns outcome string.

    Each call builds its own model(s) + client. `white_model_path` and
    `black_model_path` may each be a checkpoint path (PUCT) or None (random
    opponent). For gating, both sides are model paths (current vs best); for
    vs-random eval, one side is a path and the other is None.

    Top-level so it can be pickled for spawn() workers."""
    import os as _os

    import torch as _torch

    _torch.set_num_threads(int(_os.environ.get("CHESSCKERS_TORCH_THREADS", "1")))

    from chessckers_engine.checkpoints import load_checkpoint
    from chessckers_engine.evaluate import play_game as _play_game
    from chessckers_engine.mcts_puct import pick_puct
    from chessckers_engine.model import ChesskersScorer as _Scorer
    from chessckers_engine.random_player import pick_random
    from chessckers_engine.variant_py import PyVariantClient as _PVC

    device = _torch.device(payload["device"])
    arch = payload["model_arch"]
    n_sims = payload["n_sims"]
    client = _PVC()

    def _build_picker(model_path: str | None):
        if model_path is None:
            return lambda state: pick_random(state.get("legalMoves") or [])
        m = _Scorer(**arch).to(device)
        load_checkpoint(m, model_path)
        m.eval()
        return lambda state: pick_puct(state, client, m, n_sims=n_sims)

    white_picker = _build_picker(payload["white_model_path"])
    black_picker = _build_picker(payload["black_model_path"])
    try:
        outcome = _play_game(white_picker, black_picker, client)
    finally:
        client.close()
    return outcome


def _play_game_subprocess(payload: dict):
    """Run one self-play game in a worker subprocess.

    Each call spins up its own model copy + InferenceServer + client. This is
    the multiprocess path that bypasses the GIL — workers actually run MCTS in
    parallel instead of contending for one Python interpreter. CUDA workloads
    REQUIRE this path because thread-based workers leave a 4090 at 1% util.

    Top-level so it can be pickled for spawn() workers (closures can't)."""
    # Imports here so the worker doesn't pay them at fork time, and so any
    # import errors surface inside the worker rather than during pool setup.
    import os as _os

    import torch as _torch

    # One torch thread per worker process. With N process workers, the default
    # (≈num_cores intra-op threads × N processes) oversubscribes the CPU — which
    # both slows self-play AND destabilizes the spawn pool (the broken-pipe
    # crash). Measured: threads=1 gives clean ~linear scaling to the core count.
    _torch.set_num_threads(int(_os.environ.get("CHESSCKERS_TORCH_THREADS", "1")))

    from chessckers_engine.checkpoints import load_checkpoint
    from chessckers_engine.inference_server import InferenceServer as _IS
    from chessckers_engine.model import ChesskersScorer as _Scorer
    from chessckers_engine.selfplay_az import play_az_game as _play
    from chessckers_engine.variant_py import PyVariantClient as _PVC

    device = _torch.device(payload["device"])
    model = _Scorer(**payload["model_arch"]).to(device)
    load_checkpoint(model, payload["state_path"])
    model.eval()

    use_server = payload["mcts_batch_size"] > 1
    server = _IS(model, max_batch_size=payload["mcts_batch_size"]) if use_server else None
    evaluator = server if server is not None else model

    client = _PVC()
    rng = _torch.Generator().manual_seed(payload["seed"])
    try:
        game = _play(
            evaluator, client,
            n_sims=payload["n_sims"], c_puct=payload["c_puct"],
            temperature=payload["temperature"], rng=rng,
            temp_cutoff_plies=payload.get("temp_cutoff_plies", 30),
            dirichlet_alpha=payload["dirichlet_alpha"],
            dirichlet_eps=payload["dirichlet_eps"],
            vloss_batch=payload["vloss_batch"],
        )
    finally:
        client.close()
        if server is not None:
            server.shutdown()
    return game


def _run_eval_parallel(
    white_model_path: str | None,
    black_model_path: str | None,
    n_games: int,
    n_sims: int,
    model_arch: dict,
    device: str,
    workers: int,
    stop_path: Path | None = None,
) -> dict[str, int]:
    """Multiprocess version of evaluate.evaluate(). Each game runs in its own
    worker subprocess so we get true parallelism on the GPU during eval/gating
    instead of single-game-at-a-time. Same return shape: {'white','black','draw','games'}.

    For gating: both args are checkpoint paths (current vs best).
    For vs-random eval: one arg is a path, the other is None.

    If `stop_path` is given and the file appears mid-cycle, the consumer loop
    terminates the pool early. Returned 'games' reflects how many actually
    completed, so the caller can spot a truncated cycle."""
    import multiprocessing as _mp

    counts = {"white": 0, "black": 0, "draw": 0}
    payloads = [
        {
            "device": device,
            "model_arch": model_arch,
            "n_sims": n_sims,
            "white_model_path": white_model_path,
            "black_model_path": black_model_path,
        }
        for _ in range(n_games)
    ]
    ctx = _mp.get_context("spawn")
    completed = 0
    with ctx.Pool(processes=min(workers, n_games)) as pool:
        it = pool.imap_unordered(_eval_game_subprocess, payloads)
        for i, outcome in enumerate(it):
            counts[outcome] += 1
            completed = i + 1
            log.info("game %d/%d -> %s  (running: W=%d B=%d D=%d)",
                     completed, n_games, outcome, counts["white"], counts["black"], counts["draw"])
            if stop_path is not None and stop_path.exists():
                log.warning("STOP file detected mid-eval — terminating pool after %d/%d games",
                            completed, n_games)
                pool.terminate()
                break
    return {**counts, "games": completed if stop_path is not None and completed < n_games else n_games}


def _atomic_write(path: Path, write_fn) -> None:
    """Write via temp file + os.replace so a kill mid-write never leaves
    a half-written file at `path`. `write_fn(tmp_path)` does the actual write."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def _save_resume_state(
    weights_dir: Path,
    completed_iter: int,
    total_iters: int,
    seed: int,
    replay_buffer: deque,
) -> None:
    """Persist per-iter state so a crash mid-run can be resumed.

    Order matters: replay buffer first, state.json last. state.json is the
    'commit marker' — if it advances to N, we promise iter N's checkpoint and
    replay buffer are both fully written. A kill between iter completion and
    here just costs that iter's compute on resume."""
    _atomic_write(
        weights_dir / "replay_buffer.pt",
        lambda p: torch.save(list(replay_buffer), p),
    )
    state = {
        "completed_iters": completed_iter,
        "total_iters": total_iters,
        "seed": seed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(
        weights_dir / "state.json",
        lambda p: p.write_text(json.dumps(state, indent=2)),
    )


def _load_resume_state(
    weights_dir: Path,
    model: ChesskersScorer,
    best_model: ChesskersScorer | None,
    buffer_iters: int,
) -> tuple[int, deque]:
    """Load model + best + replay buffer from a previous run.

    Returns (start_iter, replay_buffer). start_iter is the 0-indexed iter to
    begin at — the loop should iterate over `range(start_iter, n_iters)`.
    Raises FileNotFoundError if state.json is missing (caller decides whether
    to fall through to fresh-start)."""
    state_path = weights_dir / "state.json"
    state = json.loads(state_path.read_text())
    start_iter = int(state["completed_iters"])  # next iter to RUN (0-indexed)
    if start_iter < 1:
        log.warning("resume: state.json says no iters complete; starting fresh")
        return 0, deque(maxlen=max(1, buffer_iters))
    ckpt = weights_dir / f"iter-az-{start_iter:03d}.pt"
    log.info("resume: loading model from %s", ckpt)
    load_checkpoint(model, ckpt)
    if best_model is not None:
        best_path = weights_dir / "best.pt"
        if best_path.exists():
            log.info("resume: loading best from %s", best_path)
            load_checkpoint(best_model, best_path)
        else:
            log.warning("resume: best.pt missing, seeding best from current model")
            best_model.load_state_dict(model.state_dict())
    rb_path = weights_dir / "replay_buffer.pt"
    if rb_path.exists():
        # weights_only=False: replay buffer holds AZExample dataclasses, not just tensors.
        rb_list = torch.load(rb_path, map_location="cpu", weights_only=False)
        rb = deque(rb_list, maxlen=max(1, buffer_iters))
        n_examples = sum(len(b) for b in rb)
        log.info("resume: loaded replay buffer with %d iter(s), %d total examples",
                 len(rb), n_examples)
    else:
        log.warning("resume: replay_buffer.pt missing; starting with empty buffer "
                    "(first resumed iter will train on its own self-play only)")
        rb = deque(maxlen=max(1, buffer_iters))
    log.info("resume: continuing from iter %d/%d", start_iter + 1, state["total_iters"])
    return start_iter, rb


def _seed_tag(fen: str) -> str:
    """Compact label for a start FEN in per-seed logs: the overlay square keys
    (e.g. '[d3:kk,e3:kk]' -> 'd3+e3'), else a short prefix of the board field."""
    lb, rb = fen.find("["), fen.find("]")
    if 0 <= lb < rb:
        squares = [e.split(":")[0] for e in fen[lb + 1:rb].split(",") if ":" in e]
        if squares:
            return "+".join(squares)
    return fen.split(" ")[0][:16]


def _tally_seed(seed_outcomes: dict[str, dict[str, int]], game) -> None:
    """Bucket a finished game's outcome by its start FEN (curriculum seed), so
    a mixed-seed run can report per-seed W/B/D. The user grades/advances off
    this — it does NOT gate the curriculum."""
    sf = game.records[0].fen if game.records else "?"
    seed_outcomes.setdefault(sf, {"white": 0, "black": 0, "draw": 0})[game.outcome] += 1


def run_az_iterations(
    model: ChesskersScorer,
    client: PyVariantClient,
    n_iters: int,
    games_per_iter: int,
    n_sims: int,
    c_puct: float,
    epochs: int,
    lr: float,
    t_initial: float,
    t_final: float,
    eval_games: int,
    weights_dir: Path,
    seed: int = 0,
    dirichlet_alpha: float | None = 0.3,
    dirichlet_eps: float = 0.25,
    grad_clip: float | None = 1.0,
    watch_dir: Path | None = None,
    workers: int = 1,
    client_class=None,
    buffer_iters: int = 1,
    mcts_batch_size: int = 1,
    train_batch_size: int = 1,
    vloss_batch: int = 1,
    temp_cutoff_plies: int = 30,
    keep_best: bool = False,
    keep_best_threshold: float = 0.55,
    keep_best_games: int = 10,
    eval_sims: int | None = None,
    resume: bool = False,
    model_arch: dict | None = None,
    worker_mode: str = "auto",
) -> list[dict]:
    # Eval-vs-random uses MCTS on the trained model with no exploration noise;
    # too few sims and random can stumble into wins our model misjudges. Default
    # to the same as self-play sims for backwards compat, override via --eval-sims.
    eval_sims_effective = eval_sims if eval_sims is not None else n_sims
    weights_dir.mkdir(parents=True, exist_ok=True)
    sink = JsonlWatchSink(watch_dir) if watch_dir is not None else None
    summaries: list[dict] = []
    parallel = max(1, workers) > 1
    if parallel and sink is not None:
        log.warning("workers=%d: per-move sink snapshots disabled (would flap "
                    "between concurrent games); games.jsonl still written on completion.",
                    workers)
    use_server = mcts_batch_size > 1
    if use_server:
        log.info("MCTS inference batching enabled (max_batch=%d, workers=%d)",
                 mcts_batch_size, workers)

    # Worker-mode resolution. CUDA workloads require processes — thread-based
    # workers leave a 4090 at 1% util because the GIL serializes Python MCTS
    # work. MPS/CPU stay on threads (less startup overhead, no CUDA-context
    # cost). 'auto' applies this rule; explicit values override.
    model_device = next(model.parameters()).device
    if worker_mode == "auto":
        effective_mode = "processes" if model_device.type == "cuda" else "threads"
    else:
        effective_mode = worker_mode
    if parallel:
        log.info("worker mode: %s (device=%s)", effective_mode, model_device.type)

    # Best-vs-current gating. `best_model` shadows `model` and is used for
    # self-play data generation; `model` keeps training each iter and only
    # promotes to `best` when it beats it ≥ threshold in head-to-head.
    if keep_best:
        best_model = copy.deepcopy(model).eval()
        save_checkpoint(best_model, weights_dir / "best.pt")
        log.info("keep-best enabled: threshold=%.2f, gating-games=%d",
                 keep_best_threshold, keep_best_games)
    else:
        best_model = None

    # Replay buffer: deque of per-iter example lists. buffer_iters=1 reproduces
    # the original "train on current iter only" behavior. Larger values smooth
    # the value-target distribution across iters and dampen oscillation when
    # one iter's self-play is lopsided (e.g., 9/10 Black wins).
    start_iter = 0
    replay_buffer: deque[list] = deque(maxlen=max(1, buffer_iters))
    if resume and (weights_dir / "state.json").exists():
        start_iter, replay_buffer = _load_resume_state(
            weights_dir, model, best_model, buffer_iters
        )

    for it in range(start_iter, n_iters):
        temp = _temperature_for(it, n_iters, t_initial, t_final)
        log.info("iter %d/%d: playing %d games (sims=%d, τ=%.3f, workers=%d)",
                 it + 1, n_iters, games_per_iter, n_sims, temp, max(1, workers))
        outcomes = {"white": 0, "black": 0, "draw": 0}
        seed_outcomes: dict[str, dict[str, int]] = {}
        examples = []
        ply_counts: list[int] = []

        # Each game gets a unique seed so per-thread rngs don't correlate.
        game_seeds = [seed + it * games_per_iter + gi for gi in range(games_per_iter)]
        sink_ctx_factory = lambda gi: ({
            "iter": it + 1, "total_iters": n_iters,
            "game_idx": gi + 1, "total_games": games_per_iter,
        } if sink is not None else None)

        # Build the leaf evaluator: an InferenceServer when batching is on,
        # else the raw model. Server is per-iter so model updates from the
        # previous train step are picked up on the next iter. We shut it down
        # before training so the inference thread doesn't race the optimizer.
        # When keep-best is on, self-play data comes from the best model so
        # bad iters don't feed degraded training signal back in.
        play_model = best_model if (keep_best and best_model is not None) else model
        if use_server:
            evaluator = InferenceServer(play_model, max_batch_size=mcts_batch_size)
        else:
            evaluator = play_model

        if not parallel:
            for gi in range(games_per_iter):
                game = _play_one_game(
                    evaluator, n_sims=n_sims, c_puct=c_puct, temperature=temp,
                    temp_cutoff_plies=temp_cutoff_plies,
                    seed=game_seeds[gi],
                    dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
                    sink=sink, sink_context=sink_ctx_factory(gi),
                    client_class=client_class,
                    vloss_batch=vloss_batch,
                )
                outcomes[game.outcome] += 1
                _tally_seed(seed_outcomes, game)
                examples.extend(az_game_to_examples(game))
                ply_counts.append(len(game.records))
                log.info("  game %d/%d: %s in %d plies", gi + 1, games_per_iter, game.outcome, len(game.records))
        elif effective_mode == "processes":
            # Save model state to disk so worker subprocesses can load it.
            # Each worker rebuilds the model in its own CUDA context.
            if model_arch is None:
                raise ValueError(
                    "worker_mode='processes' requires model_arch to be passed "
                    "(d_hidden, c_filters, n_blocks). Pass it from main() or use "
                    "worker_mode='threads'."
                )
            worker_state = weights_dir / "_worker_state.pt"
            torch.save(play_model.state_dict(), worker_state)
            payloads = [
                {
                    "state_path": str(worker_state),
                    "model_arch": model_arch,
                    "device": str(model_device),
                    "mcts_batch_size": mcts_batch_size,
                    "n_sims": n_sims,
                    "c_puct": c_puct,
                    "temperature": temp,
                    "temp_cutoff_plies": temp_cutoff_plies,
                    "seed": game_seeds[gi],
                    "dirichlet_alpha": dirichlet_alpha,
                    "dirichlet_eps": dirichlet_eps,
                    "vloss_batch": vloss_batch,
                }
                for gi in range(games_per_iter)
            ]
            import multiprocessing as _mp
            ctx = _mp.get_context("spawn")  # required for CUDA
            with ctx.Pool(processes=workers) as pool:
                # imap_unordered streams results as workers finish — we get
                # interleaved log lines instead of waiting for all to complete.
                for gi, game in enumerate(pool.imap_unordered(_play_game_subprocess, payloads)):
                    outcomes[game.outcome] += 1
                    _tally_seed(seed_outcomes, game)
                    examples.extend(az_game_to_examples(game))
                    ply_counts.append(len(game.records))
                    log.info("  game %d/%d: %s in %d plies", gi + 1, games_per_iter,
                             game.outcome, len(game.records))
            # Skip the as_completed-based result loop below (only used by threads).
            pass  # noqa: PIE790
        else:  # effective_mode == "threads"
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _play_one_game, evaluator,
                        n_sims=n_sims, c_puct=c_puct, temperature=temp,
                        temp_cutoff_plies=temp_cutoff_plies,
                        seed=game_seeds[gi],
                        dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
                        sink=None,  # spectator sink fully disabled in parallel mode
                        sink_context=None,
                        client_class=client_class,
                        vloss_batch=vloss_batch,
                    ): gi for gi in range(games_per_iter)
                }
                for fut in as_completed(futures):
                    gi = futures[fut]
                    game = fut.result()
                    outcomes[game.outcome] += 1
                    _tally_seed(seed_outcomes, game)
                    examples.extend(az_game_to_examples(game))
                    ply_counts.append(len(game.records))
                    log.info("  game %d/%d: %s in %d plies", gi + 1, games_per_iter, game.outcome, len(game.records))

        # Self-play done — stop the inference thread before training mutates
        # the model. (Eval after training uses the model directly; eval is
        # sequential so batching wouldn't help anyway.)
        if use_server:
            evaluator.shutdown()

        mean_plies = sum(ply_counts) / max(len(ply_counts), 1)
        log.info(
            "iter %d/%d self-play done: %dW/%dB/%dD, mean %.0f plies/game, %d examples",
            it + 1, n_iters,
            outcomes["white"], outcomes["black"], outcomes["draw"],
            mean_plies, len(examples),
        )
        if len(seed_outcomes) > 1:           # per-seed dashboard for mixed curricula
            for sf in sorted(seed_outcomes):
                so = seed_outcomes[sf]
                n = so["white"] + so["black"] + so["draw"]
                log.info("    seed %-9s: %dW/%dB/%dD (n=%d)",
                         _seed_tag(sf), so["white"], so["black"], so["draw"], n)
        replay_buffer.append(examples)
        train_examples = [ex for batch in replay_buffer for ex in batch]
        if buffer_iters > 1:
            log.info(
                "  training on replay buffer: %d examples from last %d iter(s)",
                len(train_examples), len(replay_buffer),
            )
        result = train_az(
            model, train_examples,
            epochs=epochs, lr=lr, seed=seed + it, log_every=0,
            grad_clip=grad_clip,
            batch_size=train_batch_size,
        )
        ckpt = weights_dir / f"iter-az-{it + 1:03d}.pt"
        save_checkpoint(model, ckpt)

        # Best-vs-current gating: only promote `model` to `best` if it scores
        # ≥ threshold against the current best in head-to-head play. Self-play
        # next iter will use whichever wins this gate.
        gate_info = None
        if keep_best and best_model is not None:
            log.info("  gating: head-to-head current vs best (%d games)",
                     keep_best_games)
            accept, score, breakdown = _gate_against_best(
                model, best_model, client,
                n_games=keep_best_games, n_sims=n_sims,
                threshold=keep_best_threshold,
                parallel_workers=workers if effective_mode == "processes" else 0,
                model_arch=model_arch,
                weights_dir=weights_dir,
            )
            log.info("  gating result: current score=%.3f vs best (W:%d/%d/%d B:%d/%d/%d) → %s",
                     score,
                     breakdown["as_white"]["wins"], breakdown["as_white"]["losses"],
                     breakdown["as_white"]["draws"],
                     breakdown["as_black"]["wins"], breakdown["as_black"]["losses"],
                     breakdown["as_black"]["draws"],
                     "ACCEPT" if accept else "REJECT")
            if accept:
                best_model.load_state_dict(model.state_dict())
                save_checkpoint(best_model, weights_dir / "best.pt")
            gate_info = {**breakdown, "accepted": accept}

        # eval vs random both ways (uses the latest trained model). Sims here
        # are decoupled from self-play sims — eval is the regression check, so
        # we want enough search depth to make the result trustworthy.
        if eval_games <= 0:
            # Eval disabled (--eval-games 0): skip the per-iteration vs-random
            # regression check entirely so long self-play runs aren't held up
            # by it. The self-play W/B/D column still gives a per-iter signal.
            as_white = as_black = {"white": 0, "black": 0, "draw": 0}
        elif effective_mode == "processes" and model_arch is not None:
            # Parallel path: stage current model to disk, run both eval blocks in
            # subprocess pools. None for the random side.
            eval_state = weights_dir / "_eval_state.pt"
            torch.save(model.state_dict(), eval_state)
            device_str = str(model_device)
            as_white = _run_eval_parallel(str(eval_state), None, eval_games,
                                          eval_sims_effective, model_arch, device_str, workers)
            as_black = _run_eval_parallel(None, str(eval_state), eval_games,
                                          eval_sims_effective, model_arch, device_str, workers)
        else:
            puct_pick = _make_puct_picker(model, client, eval_sims_effective)
            rnd = _make_random_picker()
            as_white = run_eval(puct_pick, rnd, client, eval_games)
            as_black = run_eval(rnd, puct_pick, client, eval_games)

        summary = {
            "iter": it + 1,
            "temperature": round(temp, 3),
            "examples": len(examples),
            "mean_plies": round(mean_plies, 1),
            "policy_loss": round(result.epoch_losses[-1]["policy"], 4),
            "value_loss": round(result.epoch_losses[-1]["value"], 4),
            "self_play": outcomes,
            "puct_as_white_vs_random": as_white,
            "puct_as_black_vs_random": as_black,
            "checkpoint": str(ckpt),
            "gating": gate_info,
        }
        summaries.append(summary)
        # Commit marker — last write of the iter. If we crash before this lands,
        # state.json still says (it) iters done, and resume re-runs iter (it+1).
        _save_resume_state(weights_dir, it + 1, n_iters, seed, replay_buffer)
        log.info(
            "iter %d/%d done | self-play %dW/%dB/%dD ply̅=%.0f | "
            "vs.rand W:%d/%d/%d B:%d/%d/%d | policy=%.3f value=%.3f",
            it + 1, n_iters,
            outcomes["white"], outcomes["black"], outcomes["draw"], mean_plies,
            as_white["white"], as_white["black"], as_white["draw"],
            as_black["black"], as_black["white"], as_black["draw"],
            result.epoch_losses[-1]["policy"], result.epoch_losses[-1]["value"],
        )

    return summaries


def main() -> int:
    from chessckers_engine.runtime import setup_logging
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--games-per-iter", type=int, default=8)
    p.add_argument("--sims", type=int, default=25, help="PUCT simulations per move")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-final", type=float, default=0.3)
    p.add_argument("--temp-cutoff-plies", type=int, default=30,
                   help="AlphaZero per-ply temperature: sample at τ for the first "
                        "N plies, then play greedily (argmax). 30 = AZ-chess default; "
                        "use a small value (~4) for short endgames so wins convert.")
    p.add_argument("--eval-games", type=int, default=5)
    p.add_argument("--eval-sims", type=int, default=None,
                   help="MCTS sims for eval-vs-random. Defaults to --sims; bump higher "
                        "(e.g. 400) to reduce noise from undersampled positions.")
    p.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="Dirichlet noise concentration for root priors during self-play (set <=0 to disable)",
    )
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm; <=0 to disable")
    p.add_argument("--base", default=None)
    p.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    p.add_argument("--watch-dir", default=None,
                   help="If set, write current.json (live snapshot) and append finished games "
                        "to games.jsonl in this directory for the spectator UI.")
    p.add_argument("--workers", type=int, default=1,
                   help="Run this many self-play games concurrently per iter (threading; "
                        "spectator sink disabled when >1).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-pyvariant", action="store_true",
                   help="Deprecated no-op: PyVariant is always used (the scalachess "
                        "HTTP server was removed). Accepted for backward compat.")
    p.add_argument("--buffer-iters", type=int, default=1,
                   help="Replay buffer size in iterations. Each training step uses the "
                        "concatenated examples from the last N iters' self-play. "
                        "Default 1 = no buffer (train on current iter only). Recommended: 8-16.")
    p.add_argument("--device", default="auto",
                   help="Compute device: auto|cpu|cuda|mps. Default auto picks the best available.")
    p.add_argument("--mcts-batch-size", type=int, default=1,
                   help="Vectorized MCTS: batch up to N concurrent leaf-eval requests "
                        "into one model.batch_eval call. Default 1 = no batching (model "
                        "called directly). Set to ~workers count for cross-game batching, "
                        "higher on GPU. Real benefit requires GPU + multiple workers.")
    p.add_argument("--train-batch-size", type=int, default=1,
                   help="Mini-batch size for training. Default 1 = per-example (legacy). "
                        "64-256 recommended on GPU for big speedup; on CPU, set to 1.")
    p.add_argument("--vloss-batch", type=int, default=1,
                   help="Within-game virtual-loss batched MCTS. Default 1 = sequential. "
                        "When >1, B sims are selected with virtual loss + evaluated as one "
                        "batch (requires --mcts-batch-size>=B for the server to actually "
                        "batch). Multiplies effective inference batch size B× per game.")
    # Model sizing — defaults match the original ~2.4M-param laptop config.
    # AZ-chess scale would be roughly --model-blocks 20 --model-filters 256
    # (~25M params); AZ-go scale ~--model-blocks 40 --model-filters 256.
    p.add_argument("--model-blocks", type=int, default=4,
                   help="Residual blocks in the position trunk. Default 4 (~2.4M params). "
                        "AZ-chess used 20.")
    p.add_argument("--model-filters", type=int, default=96,
                   help="Conv channels per residual block. Default 96. AZ-chess used 256.")
    p.add_argument("--model-hidden", type=int, default=256,
                   help="Hidden dim for bottleneck + heads. Default 256.")
    # Keep-best gating: only adopt a new checkpoint if it beats the current
    # best in head-to-head play. Self-play always uses the best, so bad iters
    # never feed degraded data back into training.
    p.add_argument("--keep-best", action="store_true",
                   help="Enable best-vs-current gating. After each iter, head-to-head "
                        "current vs best; only update best if current wins ≥ threshold.")
    p.add_argument("--keep-best-threshold", type=float, default=0.55,
                   help="Win-rate threshold for adopting current as new best. AZ default 0.55.")
    p.add_argument("--keep-best-games", type=int, default=10,
                   help="Number of head-to-head games per gating round (split half each side).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest iter checkpoint in --weights-dir. "
                        "Reads state.json + replay_buffer.pt + iter-az-NNN.pt + best.pt. "
                        "If state.json is absent, starts fresh.")
    p.add_argument("--worker-mode", choices=["auto", "processes", "threads"], default="auto",
                   help="auto: processes for CUDA, threads for CPU/MPS. processes: always "
                        "spawn subprocesses (true GIL-free parallelism, required for CUDA "
                        "to drive the GPU). threads: legacy ThreadPoolExecutor.")
    args = p.parse_args()

    device = pick_device(args.device)

    # PyVariant is the only client now — the scalachess HTTP server was removed.
    # --use-pyvariant is accepted for backward compat (some launch scripts still
    # pass it) but is a no-op.
    client_class = PyVariantClient
    client = PyVariantClient()

    torch.manual_seed(args.seed)
    model = ChesskersScorer(
        d_hidden=args.model_hidden,
        c_filters=args.model_filters,
        n_blocks=args.model_blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "model: %d blocks × %d filters, hidden=%d → %d params (%.2fM)",
        args.model_blocks, args.model_filters, args.model_hidden,
        n_params, n_params / 1e6,
    )
    if args.base:
        log.info("warm-starting from %s", args.base)
        load_checkpoint(model, args.base)

    alpha = args.dirichlet_alpha if args.dirichlet_alpha > 0 else None
    grad_clip = args.grad_clip if args.grad_clip > 0 else None
    summaries = run_az_iterations(
        model=model,
        client=client,
        n_iters=args.iterations,
        games_per_iter=args.games_per_iter,
        n_sims=args.sims,
        c_puct=args.c_puct,
        epochs=args.epochs,
        lr=args.lr,
        t_initial=args.temperature,
        t_final=args.temperature_final,
        temp_cutoff_plies=args.temp_cutoff_plies,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        weights_dir=Path(args.weights_dir),
        seed=args.seed,
        dirichlet_alpha=alpha,
        dirichlet_eps=args.dirichlet_eps,
        grad_clip=grad_clip,
        watch_dir=Path(args.watch_dir) if args.watch_dir else None,
        workers=args.workers,
        client_class=client_class,
        buffer_iters=args.buffer_iters,
        mcts_batch_size=args.mcts_batch_size,
        train_batch_size=args.train_batch_size,
        vloss_batch=args.vloss_batch,
        keep_best=args.keep_best,
        keep_best_threshold=args.keep_best_threshold,
        keep_best_games=args.keep_best_games,
        resume=args.resume,
        worker_mode=args.worker_mode,
        model_arch={
            "d_hidden": args.model_hidden,
            "c_filters": args.model_filters,
            "n_blocks": args.model_blocks,
        },
    )

    print("\n  iter | τ     | ply̅  | policy | value  | self-play W/B/D | vs.rand W:W/L/D | vs.rand B:W/L/D")
    print("  -----|-------|------|--------|--------|-----------------|-----------------|-----------------")
    for s in summaries:
        sp = s["self_play"]
        w = s["puct_as_white_vs_random"]
        b = s["puct_as_black_vs_random"]
        print(
            f"  {s['iter']:4d} | {s['temperature']:.3f} | {s['mean_plies']:4.0f} | "
            f"{s['policy_loss']:6.3f} | {s['value_loss']:6.3f} | "
            f"{sp['white']:3d}/{sp['black']:3d}/{sp['draw']:3d}        | "
            f"{w['white']:2d}/{w['black']:2d}/{w['draw']:2d}           | "
            f"{b['black']:2d}/{b['white']:2d}/{b['draw']:2d}"
        )
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
