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
import json
import logging
import os
import pickle
import random
import signal
import time
from pathlib import Path

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.device import pick_device
from chessckers_engine.model import ChesskersScorer
from chessckers_engine.runtime import setup_logging
from chessckers_engine.selfplay_az_loop import _next_game_num, _seed_tag
from chessckers_engine.train_az import _batch_loss, save_checkpoint

log = logging.getLogger("chessckers_engine.train_continuous")


class _GameArchive:
    """Append-only durable archive of self-play games on a (slow, secondary)
    disk — the cold tier for re-training / re-labeling / data-window experiments.

    Each ingested game would otherwise be DISCARDED once the live buffer slides
    past it; the archive keeps the full AZExamples (targets included) so games
    can be re-trained on later without regenerating self-play.

    Best-effort by design: any write failure (e.g. a USB volume that dropped off
    the bus mid-run) is logged once and disables the archive for the rest of the
    run — it NEVER interrupts training. Games are bundled into rotating shards
    (not one tiny file per game) to stay friendly to flash random-IO + directory
    limits; a JSONL manifest lets you scan the archive (counts, seed mix, window
    sizing) without unpickling the shards."""

    SHARD_CAP_BYTES = 256 * 1024 * 1024  # rotate to a new shard at ~256 MB

    def __init__(self, root: Path, cap_bytes: int = 0):
        self.root = root
        self.cap_bytes = cap_bytes  # 0 = unbounded; else FIFO-evict oldest shards over cap
        self.enabled = True
        self.games_written = 0
        self._shard = None
        self._manifest = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            existing = sorted(self.root.glob("games-*.pkl"))
            # Resume the latest shard after a restart; else start at 1.
            self._shard_idx = int(existing[-1].stem.split("-")[1]) if existing else 1
            self._shard = open(self.root / f"games-{self._shard_idx:05d}.pkl", "ab")
            self._manifest = open(self.root / "manifest.jsonl", "a")
        except OSError as e:
            log.warning("game archive disabled (init failed for %s): %s", self.root, e)
            self.enabled = False

    def append(self, examples: list, meta: dict) -> None:
        if not self.enabled:
            return
        try:
            pickle.dump({"examples": examples, "meta": meta}, self._shard,
                        protocol=pickle.HIGHEST_PROTOCOL)
            self._shard.flush()
            rec = {k: meta.get(k) for k in ("machine", "outcome", "plies", "seed_fen")}
            rec["n_examples"] = len(examples)
            rec["shard"] = self._shard_idx
            self._manifest.write(json.dumps(rec) + "\n")
            self._manifest.flush()
            self.games_written += 1
            if self._shard.tell() >= self.SHARD_CAP_BYTES:  # rotate
                os.fsync(self._shard.fileno())
                self._shard.close()
                self._shard_idx += 1
                self._shard = open(self.root / f"games-{self._shard_idx:05d}.pkl", "ab")
                self._evict_over_cap()  # FIFO-drop oldest shards (only at rotation — rare)
        except OSError as e:
            log.warning("game archive write failed — disabling for this run: %s", e)
            self.enabled = False

    @staticmethod
    def _read_records(shard: Path):
        """Yield each {examples, meta} record from a shard, tolerating a
        truncated tail (a shard being appended-to, or a torn write)."""
        with open(shard, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    return
                except (pickle.UnpicklingError, ValueError):
                    return  # torn tail — stop at the last good record

    def load_recent(self, max_positions: int) -> list:
        """Most-recent ~max_positions AZExamples across the archive, for priming
        the in-RAM replay window on (re)start. Reads newest shards first and
        stops once it has enough — never the whole archive — so a 400 GB archive
        still primes from only the last shard or two."""
        if not self.enabled or max_positions <= 0:
            return []
        shards = sorted(self.root.glob("games-*.pkl"))  # ascending index = chronological
        newest_first: list[list] = []
        total = 0
        for shard in reversed(shards):
            exs: list = []
            try:
                for rec in self._read_records(shard):
                    exs.extend(rec["examples"])
            except OSError:
                continue
            newest_first.append(exs)
            total += len(exs)
            if total >= max_positions:
                break
        chronological = [e for exs in reversed(newest_first) for e in exs]
        return chronological[-max_positions:]

    def _evict_over_cap(self) -> None:
        """Delete oldest shards (and prune their manifest lines) until the
        archive fits cap_bytes. Keeps the active shard. No-op if cap_bytes=0."""
        if not self.cap_bytes:
            return
        try:
            shards = sorted(self.root.glob("games-*.pkl"))
            sizes = {s: s.stat().st_size for s in shards}
            total = sum(sizes.values())
            evicted: list[int] = []
            for shard in shards[:-1]:  # never evict the currently-open (last) shard
                if total <= self.cap_bytes:
                    break
                total -= sizes[shard]
                evicted.append(int(shard.stem.split("-")[1]))
                shard.unlink()
            if evicted:
                self._prune_manifest(set(evicted))
                log.info("archive over cap (%d MB): evicted %d oldest shard(s) %s",
                         self.cap_bytes // (1 << 20), len(evicted), evicted)
        except OSError as e:
            log.warning("archive eviction failed: %s", e)

    def _prune_manifest(self, evicted_idx: set[int]) -> None:
        """Rewrite manifest.jsonl dropping entries for evicted shards. Rare
        (only fires on eviction), so a full rewrite is fine."""
        path = self.root / "manifest.jsonl"
        self._manifest.close()
        kept = []
        for line in path.read_text().splitlines():
            try:
                if json.loads(line).get("shard") not in evicted_idx:
                    kept.append(line)
            except ValueError:
                continue
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(("\n".join(kept) + "\n") if kept else "")
        os.replace(tmp, path)
        self._manifest = open(path, "a")


def _drain(buffer_dir: Path, archive: "_GameArchive | None" = None) -> tuple[list, list]:
    """Load + consume every COMPLETE game pkl in buffer_dir. Each pkl is a
    list[AZExample] (what selfplay_worker_async writes), with a sibling .meta
    JSON (machine/outcome/plies/seed_fen). Returns (examples, metas) — one meta
    dict per ingested game; skips partial/mid-rsync pkls (retried next tick),
    archives the game (if `archive` set) before unlinking the consumed pkl + its
    .meta sidecar."""
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
        if archive is not None:
            archive.append(exs, meta)  # durable cold tier before the pkl is gone
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
    p.add_argument("--archive-dir", default="",
                   help="durable append-only game archive (e.g. /Volumes/hd0/chessckers_archive/<run>); "
                        "keeps every ingested game for re-training/re-labeling. Empty = off.")
    p.add_argument("--archive-cap-gb", type=float, default=0.0,
                   help="bound the archive to this many GB, FIFO-evicting oldest shards (0 = unbounded). "
                        "Set below the disk's free space, e.g. 380 on a 400 GB volume.")
    p.add_argument("--buffer-cap", type=int, default=50000, help="rolling replay buffer capacity (positions)")
    p.add_argument("--no-prime", action="store_true",
                   help="do NOT warm-start the in-RAM window from the archive. Keeps archiving, but the "
                        "trainer starts cold (waits for --min-buffer) and stays generation-bound — avoids "
                        "the startup burst (priming injects buffer-cap positions at once, handing the "
                        "replay-factor throttle a huge budget that hogs CPU and starves self-play workers).")
    p.add_argument("--min-buffer", type=int, default=2000, help="start training once the buffer reaches this")
    p.add_argument("--replay-factor", type=float, default=8.0,
                   help="cap total samples at this x positions-ingested (prevents overfitting a small buffer "
                        "when self-play is slower than SGD); 0 = unthrottled")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--value-loss-weight", type=float, default=1.0)
    p.add_argument("--mlh-loss-weight", type=float, default=0.3)
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

    archive = None
    if args.archive_dir:
        archive = _GameArchive(Path(args.archive_dir).resolve(),
                               cap_bytes=int(args.archive_cap_gb * (1 << 30)))
        if archive.enabled:
            cap = f"{args.archive_cap_gb:g} GB FIFO" if args.archive_cap_gb else "unbounded"
            log.info("[train] archiving every ingested game -> %s (%s)", archive.root, cap)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model = ChesskersScorer(
        d_hidden=args.d_hidden, c_filters=args.c_filters, n_blocks=args.n_blocks,
    ).to(device)
    if args.base:
        load_checkpoint(model, args.base)
        log.info("[train] warm-started from %s", args.base)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam, persistent across all steps (real continuous training)
    rng = random.Random(args.seed)

    # Publish an initial snapshot immediately so self-play has weights to load.
    _publish(model, weights_path)
    log.info("[train] published initial weights -> %s", weights_path)

    def _on_term(*_a):
        try:
            stop_path.touch()
        except OSError:
            pass
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    buf: list = []  # rolling replay buffer (list, trimmed from the front; sample is O(bs))
    # Prime the in-RAM window from the archive's most-recent games so a restart
    # resumes WARM (no cold min_buffer wait) and the window can be larger than
    # one process's freshly-ingested games. Stays RAM-resident — the slow flash
    # is read once here, never per training batch.
    if archive and archive.enabled and not args.no_prime:
        buf = archive.load_recent(args.buffer_cap)
        if buf:
            log.info("[train] primed replay window from archive: %d positions (cap %d)", len(buf), args.buffer_cap)
    elif args.no_prime:
        log.info("[train] cold start (--no-prime): not priming from archive; trainer waits for min_buffer")
    steps = 0
    games_seen = 0
    positions_ingested = len(buf)  # primed data counts as ingested (restores the replay-factor throttle state)
    last_publish = last_ckpt = time.time()
    ckpt_n = 0
    win_p = win_v = win_m = 0.0
    win_steps = 0
    log.info("[train] continuous trainer up: device=%s buffer_cap=%d min=%d batch=%d "
             "replay_factor=%.1f publish=%.0fs ckpt=%.0fs",
             device, args.buffer_cap, args.min_buffer, args.batch_size,
             args.replay_factor, args.publish_seconds, args.ckpt_seconds)

    while not stop_path.exists():
        new_ex, metas = _drain(buffer_dir, archive)
        if metas:
            buf.extend(new_ex)
            games_seen += len(metas)
            positions_ingested += len(new_ex)
            for m in metas:  # log EVERY ingested game (local + leena) — single unified per-game logger
                gn = _next_game_num()
                log.info("[selfplay] game #%d [%s]: %s in %s plies (seed %s)",
                         gn, m.get("machine", "?"), m.get("outcome", "?"),
                         m.get("plies", "?"), _seed_tag(m.get("seed_fen") or ""))
            if len(buf) > args.buffer_cap:
                del buf[: len(buf) - args.buffer_cap]  # drop oldest in place
            if args.max_games and games_seen >= args.max_games:
                log.info("[train] reached --max-games %d (games_seen=%d) — stopping run", args.max_games, games_seen)
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
        p_loss, v_loss, ml_loss = _batch_loss(model, batch)
        (p_loss + args.value_loss_weight * v_loss + args.mlh_loss_weight * ml_loss).backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()
        steps += 1
        win_p += float(p_loss.item()); win_v += float(v_loss.item()); win_m += float(ml_loss.item()); win_steps += 1

        now = time.time()
        if now - last_publish >= args.publish_seconds:
            _publish(model, weights_path)
            last_publish = now
        if now - last_ckpt >= args.ckpt_seconds:
            ckpt_n += 1
            ckpt = run_dir / f"iter-async-{ckpt_n:04d}.pt"
            save_checkpoint(model, ckpt)
            arch_stat = (f" archived={archive.games_written}" if archive and archive.enabled
                         else " archived=off" if not archive
                         else " archived=DISABLED")  # best-effort archive dropped out mid-run
            log.info("[train] step %d | policy=%.4f value=%.4f mlh=%.4f | %.1f steps/s | buf=%d games_seen=%d/%s",
                     steps,
                     win_p / max(win_steps, 1), win_v / max(win_steps, 1), win_m / max(win_steps, 1),
                     win_steps / max(now - last_ckpt, 1e-9),
                     len(buf), games_seen, (args.max_games or "inf"))
            log.info("[ckpt] saved %s | step %d |%s", ckpt.name, steps, arch_stat)
            win_p = win_v = win_m = 0.0; win_steps = 0; last_ckpt = now
        if args.max_steps and steps >= args.max_steps:
            break

    stop_path.touch()  # signal local self-play workers (shared run-dir STOP) + the sidecar to tear down
    _publish(model, weights_path)
    save_checkpoint(model, run_dir / "iter-async-final.pt")
    log.info("[train] stopped: %d steps, %d games seen, %d positions ingested, %d games archived (STOP signaled)",
             steps, games_seen, positions_ingested,
             archive.games_written if archive else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
