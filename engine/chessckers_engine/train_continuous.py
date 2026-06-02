"""Async / continuous AlphaZero trainer — decoupled from self-play.

This is the real-AZ architecture (vs the synchronous collect-200-then-train loop
in `selfplay_az_loop`). Self-play runs SEPARATELY as `selfplay_workers_only`
processes (local + leena), continuously writing game pkls into `<run-dir>/buffer`.
This trainer never pauses:

  - continuously INGESTS new game pkls into a rolling replay buffer (cap N positions),
  - does SGD steps NON-STOP on sampled minibatches,
  - PUBLISHES `weights.pt` on a timer (`--publish-seconds`, NOT every step) — the
    self-play workers + the leena sidecar hot-reload it on their mtime poll, so the
    ~1.3s/10MB leena copy cost is amortized (publish every ~45s -> a few % overhead),
  - throttles training to `--replay-factor` x (positions ingested) so it can't
    overfit a small buffer when self-play is slower than SGD,
  - checkpoints + logs every `--ckpt-seconds`.

Self-play and training OVERLAP (self-play on CPU, training on MPS), reclaiming the
~40% of wall the synchronous loop spends in its serialized train phase.

Run (after self-play workers are already producing into <run-dir>/buffer):

  python -m chessckers_engine.train_continuous \\
    --run-dir weights/run_async --base weights/base_curriculum_v3.pt \\
    --buffer-cap 50000 --batch-size 256 --publish-seconds 45 --ckpt-seconds 300

Stop: `touch <run-dir>/STOP`  (or SIGTERM).
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import signal
import time
from pathlib import Path

import torch
from torch import nn

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.device import pick_device
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.runtime import setup_logging
from chessckers_engine.selfplay_az_loop import _next_game_num, _seed_tag
from chessckers_engine.train_az import _batch_loss, save_checkpoint

log = logging.getLogger("chessckers_engine.train_continuous")


def _drain(buffer_dir: Path) -> tuple[list, list]:
    """Load + consume every COMPLETE game pkl in buffer_dir. Each pkl is a
    list[AZExample] (what selfplay_worker_async writes), with a sibling .meta
    JSON (machine/outcome/plies/seed_fen). Returns (examples, metas) — one meta
    dict per ingested game; skips partial/mid-rsync pkls (retried next tick),
    unlinks the consumed pkl + its .meta sidecar."""
    import json
    examples: list = []
    metas: list = []
    if not buffer_dir.exists():
        return examples, metas
    for pkl in sorted(buffer_dir.glob("*.pkl")):
        try:
            with open(pkl, "rb") as f:
                exs = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError, ValueError):
            continue  # mid-write / partial rsync — retry next tick
        examples.extend(exs)
        meta_path = Path(str(pkl) + ".meta")
        meta: dict = {}
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            pass
        metas.append(meta)
        for fp in (pkl, meta_path):
            try:
                fp.unlink()
            except OSError:
                pass
    return examples, metas


def _publish(model: ChesskersScorer, weights_path: Path) -> None:
    """Atomically publish weights.pt (tmp + os.replace) so the sidecar / workers
    never read a half-written file."""
    tmp = weights_path.with_suffix(".pt.tmp")
    save_checkpoint(model, tmp)
    os.replace(tmp, weights_path)  # atomic on POSIX


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Async/continuous AlphaZero trainer (decoupled self-play).")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--base", default="", help="warm-start checkpoint (else random init)")
    p.add_argument("--buffer-cap", type=int, default=50000, help="rolling replay buffer capacity (positions)")
    p.add_argument("--min-buffer", type=int, default=2000, help="start training once the buffer reaches this")
    p.add_argument("--replay-factor", type=float, default=8.0,
                   help="cap total samples at this x positions-ingested (prevents overfitting a small buffer "
                        "when self-play is slower than SGD); 0 = unthrottled")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--value-loss-weight", type=float, default=1.0)
    p.add_argument("--publish-seconds", type=float, default=45.0, help="how often to publish weights.pt to self-play")
    p.add_argument("--ckpt-seconds", type=float, default=300.0, help="how often to save an iter ckpt + log stats")
    p.add_argument("--device", default="auto", help="train device (auto -> mps if available)")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=96)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N SGD steps (0 = until STOP)")
    p.add_argument("--max-games", type=int, default=0,
                   help="stop the WHOLE run after N games ingested (local+leena); 0 = until STOP")
    args = p.parse_args()

    run_dir: Path = args.run_dir.resolve()
    buffer_dir = run_dir / "buffer"
    weights_path = run_dir / "weights.pt"
    stop_path = run_dir / "STOP"
    run_dir.mkdir(parents=True, exist_ok=True)
    buffer_dir.mkdir(parents=True, exist_ok=True)
    if stop_path.exists():
        stop_path.unlink()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model = ChesskersScorer(
        d_hidden=args.d_hidden, c_filters=args.c_filters, n_blocks=args.n_blocks,
    ).to(device)
    if args.base:
        load_checkpoint(model, args.base)
        log.info("warm-started from %s", args.base)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam, persistent across all steps (real continuous training)
    mse = nn.MSELoss()
    rng = random.Random(args.seed)

    # Publish an initial snapshot immediately so self-play has weights to load.
    _publish(model, weights_path)
    log.info("published initial weights -> %s", weights_path)

    def _on_term(*_a):
        try:
            stop_path.touch()
        except OSError:
            pass
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    buf: list = []  # rolling replay buffer (list, trimmed from the front; sample is O(bs))
    steps = 0
    games_seen = 0
    positions_ingested = 0
    last_publish = last_ckpt = time.time()
    ckpt_n = 0
    win_p = win_v = 0.0
    win_steps = 0
    log.info("continuous trainer up: device=%s buffer_cap=%d min=%d batch=%d "
             "replay_factor=%.1f publish=%.0fs ckpt=%.0fs",
             device, args.buffer_cap, args.min_buffer, args.batch_size,
             args.replay_factor, args.publish_seconds, args.ckpt_seconds)

    while not stop_path.exists():
        new_ex, metas = _drain(buffer_dir)
        if metas:
            buf.extend(new_ex)
            games_seen += len(metas)
            positions_ingested += len(new_ex)
            for m in metas:  # log EVERY ingested game (local + leena) — single unified per-game logger
                gn = _next_game_num()
                log.info("  game #%d [%s]: %s in %s plies (seed %s)",
                         gn, m.get("machine", "?"), m.get("outcome", "?"),
                         m.get("plies", "?"), _seed_tag(m.get("seed_fen") or ""))
            if len(buf) > args.buffer_cap:
                del buf[: len(buf) - args.buffer_cap]  # drop oldest in place
            if args.max_games and games_seen >= args.max_games:
                log.info("reached --max-games %d (games_seen=%d) — stopping run", args.max_games, games_seen)
                break

        if len(buf) < args.min_buffer:
            time.sleep(1.0)
            continue
        # Throttle: don't reuse data faster than replay_factor x ingest.
        if args.replay_factor and steps * args.batch_size >= args.replay_factor * positions_ingested:
            time.sleep(0.2)
            continue

        bs = min(args.batch_size, len(buf))
        batch = rng.sample(buf, bs)
        opt.zero_grad()
        p_loss, v_loss = _batch_loss(model, batch, mse)
        (p_loss + args.value_loss_weight * v_loss).backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()
        steps += 1
        win_p += float(p_loss.item()); win_v += float(v_loss.item()); win_steps += 1

        now = time.time()
        if now - last_publish >= args.publish_seconds:
            _publish(model, weights_path)
            last_publish = now
        if now - last_ckpt >= args.ckpt_seconds:
            ckpt_n += 1
            ckpt = run_dir / f"iter-async-{ckpt_n:04d}.pt"
            save_checkpoint(model, ckpt)
            log.info("checkpoint saved -> %s | step %d games_seen=%d/%s buf=%d | policy=%.4f value=%.4f | %.1f steps/s",
                     ckpt.name, steps, games_seen, (args.max_games or "inf"), len(buf),
                     win_p / max(win_steps, 1), win_v / max(win_steps, 1),
                     win_steps / max(now - last_ckpt, 1e-9))
            win_p = win_v = 0.0; win_steps = 0; last_ckpt = now
        if args.max_steps and steps >= args.max_steps:
            break

    stop_path.touch()  # signal local self-play workers (shared run-dir STOP) + the sidecar to tear down
    _publish(model, weights_path)
    save_checkpoint(model, run_dir / "iter-async-final.pt")
    log.info("stopped: %d steps, %d games seen, %d positions ingested (STOP signaled)",
             steps, games_seen, positions_ingested)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
