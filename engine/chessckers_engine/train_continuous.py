"""Async / continuous AlphaZero trainer — decoupled from self-play.

This is the real-AZ architecture: self-play runs SEPARATELY — the lc0 fleet
(`lczero-client` running the `akshay-chessckers-0` engine) plays the games and
the server feeds their chunks into `<run-dir>/buffer`. This trainer never pauses:

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
from collections import deque
from pathlib import Path

import torch

from chessckers_engine.checkpoints import load_checkpoint
from chessckers_engine.device import pick_device
from chessckers_engine.model import ChesskersScorer, build_model
from chessckers_engine.runtime import setup_logging
from chessckers_engine.train_az import _batch_loss, save_checkpoint
from chessckers_engine.training_chunk import ChunkDecodeError, decode_chunk

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


def _downsample_game(exs: list, keep_frac: float, rng: "random.Random | None") -> list:
    """Per-game downsampling (lc0-style SKIP): keep each position of a game
    independently with probability `keep_frac`. Consecutive plies of one game are
    near-duplicates, so taking ~1 in 1/keep_frac of them decorrelates the replay
    window without biasing it. keep_frac>=1 (or no rng) keeps everything; a
    non-empty game never downsamples to nothing (keeps >=1 position)."""
    if rng is None or keep_frac >= 1.0 or not exs:
        return exs
    kept = [e for e in exs if rng.random() < keep_frac]
    return kept or [rng.choice(exs)]


def _drain(buffer_dir: Path, archive: "_GameArchive | None" = None,
           keep_frac: float = 1.0, rng: "random.Random | None" = None) -> tuple[list, list]:
    """Load + consume every COMPLETE game pkl in buffer_dir. Each pkl is one game's
    AZExamples as a gzipped-JSON `ccz` chunk (training_chunk; data-only, never
    pickle.load on uploaded bytes), with a sibling .meta JSON
    (worker_id/machine/outcome/plies/seed_fen). Returns (examples, metas) —
    one meta dict per ingested game (stamped with meta['kept'] = positions taken
    into the live window); skips partial/mid-rsync pkls (retried next tick). The
    game is archived IN FULL (the cold tier keeps every position) BEFORE the live
    window is downsampled by `keep_frac`, then the pkl + .meta are unlinked."""
    examples: list = []
    metas: list = []
    if not buffer_dir.exists():
        return examples, metas
    for pkl in sorted(buffer_dir.glob("*.pkl")):
        try:
            exs = decode_chunk(pkl.read_bytes())
        except (OSError, ChunkDecodeError):
            continue  # mid-write / partial upload / foreign bytes — retry next tick
        meta_path = Path(str(pkl) + ".meta")
        meta: dict = {}
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            pass
        if archive is not None:
            archive.append(exs, meta)  # durable cold tier gets the WHOLE game (pre-downsample)
        kept = _downsample_game(exs, keep_frac, rng)
        meta["kept"] = len(kept)
        examples.extend(kept)
        metas.append(meta)
        for fp in (pkl, meta_path):
            try:
                fp.unlink()
            except OSError:
                pass
    return examples, metas


def _window_cap(games_seen: int, min_w: int, max_w: int, alpha: float) -> int:
    """lc0/KataGo-style growing replay window, measured in GAMES.

    Returns the current sliding-window game cap. With ``min_w`` <= 0 (or
    ``min_w`` >= ``max_w``) the window is FIXED at ``max_w`` — the legacy
    ``--window-games`` behavior. Otherwise it grows sublinearly from ``min_w``
    up to ``max_w`` as self-play games accumulate, using KataGo's growth law
    (with beta=0.4) anchored so the cap equals ``min_w`` at ``games_seen ==
    min_w``::

        cap(N) = min_w * (1 + (BETA/alpha) * ((N/min_w)**alpha - 1))

    clamped to ``[min_w, max_w]``. Narrow early so the near-random opening
    generations evict fast (the policy escapes random play instead of training
    on a window diluted by junk); wide late for variance reduction once the
    policy has stabilized. ``alpha`` < 1 makes the ramp progressively slower
    (more sublinear, wider spread); ``alpha`` ~ 1 is roughly linear in N."""
    if min_w <= 0 or min_w >= max_w:
        return max_w
    BETA = 0.4
    n = max(games_seen, min_w)
    cap = min_w * (1.0 + (BETA / alpha) * ((n / min_w) ** alpha - 1.0))
    return int(min(max_w, max(min_w, cap)))


# Replay-buffer snapshot: the in-RAM window persisted to disk so a restart (to
# tweak a hyperparameter) resumes the EXACT training state — buffer positions,
# the per-game sizes that drive window eviction, the ingest counters that drive
# the replay-factor throttle + window ramp, AND the optimizer state (SGD momentum
# + the step counter that drives the LR schedule) — with no cold rebuild. This is
# the lc0 "data lives on disk" property for our RAM-windowed trainer. Distinct
# from --archive-dir (a cold tier of WHOLE games for re-labeling): the snapshot
# is exactly the live window and round-trips game_sizes, which the archive can't.
_SNAPSHOT_VERSION = 1


def _to_cpu(o):
    """Recursively move tensors in an optimizer state_dict to CPU so the pickled
    snapshot stays device-agnostic (restartable on mps/cuda/cpu alike) and matches
    the all-CPU buffer payload rather than embedding live device tensors."""
    if torch.is_tensor(o):
        return o.detach().cpu()
    if isinstance(o, dict):
        return {k: _to_cpu(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_cpu(v) for v in o]
    return o


def _save_buffer_snapshot(path: Path, buf: list, game_sizes: "deque[int]",
                          games_seen: int, positions_ingested: int,
                          steps: int, opt_state: "dict | None") -> None:
    """Atomically persist the live replay window AND the optimization process (SGD
    momentum + the step counter that drives the LR schedule), so a restart resumes
    where it left off instead of re-warming the LR and zeroing momentum. Best-effort:
    a write failure is logged and never interrupts shutdown. tmp + os.replace so a
    torn write can never replace a good snapshot."""
    try:
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"version": _SNAPSHOT_VERSION, "buf": buf,
                         "game_sizes": list(game_sizes), "games_seen": games_seen,
                         "positions_ingested": positions_ingested,
                         "steps": steps, "opt_state": opt_state},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        log.info("[train] saved replay buffer snapshot: %d positions, %d games, step %d -> %s",
                 len(buf), len(game_sizes), steps, path.name)
    except (OSError, pickle.PicklingError) as e:
        log.warning("[train] replay buffer snapshot write failed (%s): %s", path.name, e)


def _load_buffer_snapshot(path: Path) -> "dict | None":
    """Restore a snapshot written by _save_buffer_snapshot, or None if it is
    absent / corrupt / a stale format (any of which falls back to a cold start).
    Drops game_sizes if it disagrees with the buffer length (e.g. an archive-
    primed snapshot) so the window simply rebuilds rather than mis-evicting."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            snap = pickle.load(f)
        if not isinstance(snap, dict) or snap.get("version") != _SNAPSHOT_VERSION:
            log.warning("[train] ignoring replay buffer snapshot %s: incompatible format", path.name)
            return None
        buf = snap["buf"]
        gs = snap.get("game_sizes") or []
        if sum(gs) != len(buf):  # game_sizes must partition buf for exact eviction
            gs = []
        snap["buf"], snap["game_sizes"] = buf, gs
        return snap
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, ValueError) as e:
        log.warning("[train] ignoring replay buffer snapshot %s: %s", path.name, e)
        return None


_STEP_STATE_VERSION = 1


def _save_step_state(path: Path, steps: int, opt_state: "dict | None") -> None:
    """Persist JUST the training clock (LR-schedule step counter) + SGD momentum,
    decoupled from the large replay-buffer snapshot so it can be written on the
    FREQUENT publish cadence (crash-safe) instead of only at clean shutdown. This is
    what keeps `steps` from resetting to 0 after a SIGKILL / crash / reboot (the
    buffer snapshot, written shutdown-only, would be stale or absent). Small (~MBs:
    momentum buffers only). tmp + os.replace so a torn write can't clobber a good one."""
    try:
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"version": _STEP_STATE_VERSION, "steps": steps,
                         "opt_state": opt_state}, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except (OSError, pickle.PicklingError) as e:
        log.warning("[train] step-state write failed (%s): %s", path.name, e)


def _load_step_state(path: Path) -> "dict | None":
    """Restore a sidecar written by _save_step_state, or None if absent/corrupt/stale."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            st = pickle.load(f)
        if not isinstance(st, dict) or st.get("version") != _STEP_STATE_VERSION:
            return None
        return st
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as e:
        log.warning("[train] ignoring step-state %s: %s", path.name, e)
        return None


def _publish(model: ChesskersScorer, weights_path: Path) -> None:
    """Atomically publish weights.pt (tmp + os.replace) so the sidecar / workers
    never read a half-written file."""
    tmp = weights_path.with_suffix(".pt.tmp")
    save_checkpoint(model, tmp)
    os.replace(tmp, weights_path)  # atomic on POSIX
    # save_checkpoint wrote the arch sidecar against the tmp name; move it to the
    # final name so offline loaders (checkpoints.load_scorer / eval) can rebuild
    # the exact net. (The fleet itself reads arch from fleet.env/CLI, not this.)
    tmp_arch = Path(str(tmp) + ".arch.json")
    if tmp_arch.exists():
        os.replace(tmp_arch, Path(str(weights_path) + ".arch.json"))
    # Phase 3B (lc0-split): also publish a C++-loadable native .bin so the no-Python
    # self-play client can fetch the net by sha (GET /get_network) and load it
    # directly — additive, the .pt path (Python clients) is untouched.
    from chessckers_engine.native_net import export_state_dict
    bin_path = weights_path.with_suffix(".bin")
    bin_tmp = bin_path.with_name(bin_path.name + ".tmp")
    export_state_dict(model.state_dict(), bin_tmp)
    os.replace(bin_tmp, bin_path)  # atomic on POSIX


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
                   help="do NOT warm-start the in-RAM window — neither from the replay-buffer snapshot "
                        "(replay_buffer.pkl) NOR the archive. Forces a cold start (waits for --min-buffer) "
                        "and stays generation-bound — avoids the startup burst (priming injects buffer-cap "
                        "positions at once, handing the replay-factor throttle a huge budget that hogs CPU "
                        "and starves self-play workers).")
    p.add_argument("--buffer-snapshot-seconds", type=float, default=0.0,
                   help="ALSO snapshot the replay buffer to <run-dir>/replay_buffer.pkl every N seconds "
                        "(crash safety), on top of the always-on snapshot taken at clean shutdown. The "
                        "snapshot lets a restart resume the exact window/throttle state with no cold "
                        "rebuild — so hyperparameters can be tweaked without losing progress. 0 = "
                        "shutdown-only (the buffer can be large; periodic writes cost I/O).")
    p.add_argument("--min-buffer", type=int, default=2000, help="start training once the buffer reaches this")
    p.add_argument("--replay-factor", type=float, default=8.0,
                   help="cap total samples at this x positions-ingested (prevents overfitting a small buffer "
                        "when self-play is slower than SGD); 0 = unthrottled")
    p.add_argument("--per-game-keep", type=float, default=1.0,
                   help="per-game downsampling (lc0-style SKIP): keep each position of a game with this "
                        "probability before it enters the live replay window — consecutive plies are "
                        "near-duplicates, so this decorrelates the buffer. 1.0 = keep all (default); the "
                        "durable archive is unaffected. E.g. 0.25 keeps ~1/4, widening the window's "
                        "game-diversity ~4x (positions_ingested grows ~4x slower, so the replay-factor "
                        "throttle permits proportionally fewer steps; raise --replay-factor to offset).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-2,
                   help="base LR (SGD-scale, ~20x the old Adam 1e-3); warmed up via --lr-warmup-steps")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD Nesterov momentum (lc0/AZ: 0.9)")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay (lc0/AZ: 1e-4)")
    p.add_argument("--grad-clip", type=float, default=1000.0)
    p.add_argument("--value-loss-weight", type=float, default=1.0)
    p.add_argument("--value-discount", type=float,
                   default=float(os.environ.get("CHESSCKERS_VALUE_DISCOUNT", "1.0")),
                   help="per-ply WDL value discount gamma (<1 pulls the outcome target toward draw by "
                        "gamma**(plies_to_end-1), so faster wins score higher — an incentive to convert "
                        "quickly). 1.0 = off (flat one-hot). E.g. 0.99 makes a 50-ply win-start worth "
                        "~0.6 win-mass vs ~0.17 for a 180-ply one. Applied at TRAIN time from the stored "
                        "moves_left, so it also reshapes the lc0-fork data without regenerating it.")
    p.add_argument("--value-q-ratio", type=float,
                   default=float(os.environ.get("CHESSCKERS_VALUE_Q_RATIO", "0.5")),
                   help="fraction of the value target taken from the SEARCH's root value q instead of "
                        "the game outcome z: target = (1-r)*z + r*q (both STM-relative WDL). 0.0 = off "
                        "(pure outcome, the old behavior). r>0 bootstraps value from search, removing the "
                        "conservatism that temperature / Dirichlet noise bake into the realized outcome "
                        "(Lever 3). Needs chunks carrying search_wdl (lc0-fork data); examples without it "
                        "fall back to pure z. Composes with --value-discount, which shapes the z term only.")
    p.add_argument("--mlh-loss-weight", type=float, default=0.3)
    p.add_argument("--publish-seconds", type=float, default=45.0, help="how often to publish weights.pt to self-play")
    p.add_argument("--ckpt-seconds", type=float, default=300.0, help="how often to save an iter ckpt + log stats")
    p.add_argument("--log-seconds", type=float, default=30.0,
                   help="how often to log a [loss] line (policy/value/mlh + grad-norm, clip-fraction, "
                        "update/weight ratio) — decoupled from --ckpt-seconds so you see the loss curve "
                        "without waiting a whole checkpoint interval")
    p.add_argument("--device", default="auto", help="train device (auto -> mps if available)")
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--c-filters", type=int, default=96)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--arch-version", choices=["v1", "v2", "v4"], default="v1",
                   help="Net arch: v1 (pooled), v2 (gather head + optional transformer), "
                        "v4 (gather head + Squeeze-Excitation ResNet blocks).")
    p.add_argument("--tf-blocks", type=int, default=0,
                   help="V2/V4: Transformer blocks interleaved into the trunk (0 = pure ResNet).")
    p.add_argument("--tf-heads", type=int, default=4, help="V2 transformer attention heads.")
    p.add_argument("--tf-ff-mult", type=int, default=4, help="V2 transformer feed-forward expansion.")
    p.add_argument("--se-ratio", type=int, default=8,
                   help="V4: Squeeze-Excitation reduction ratio (channels c -> c/r in the SE bottleneck).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N SGD steps (0 = until STOP)")
    p.add_argument("--max-games", type=int, default=0,
                   help="stop the WHOLE run after N games ingested (local+leena); 0 = until STOP")
    # --- Phase 1: publish gating on TRAINING PROGRESS, not wall-clock. Any set
    #     trigger fires a publish; --publish-seconds stays as an optional floor. ---
    p.add_argument("--publish-steps", type=int, default=0,
                   help="publish after this many SGD steps since the last publish (0 = off)")
    p.add_argument("--publish-games", type=int, default=0,
                   help="publish after this many games ingested since the last publish (0 = off)")
    # --- Phase 2: lc0-style generational training (windowed data, LR schedule, SWA/EMA). ---
    p.add_argument("--window-games", type=int, default=0,
                   help="cap the replay window to the last N GAMES (lc0 sliding window); "
                        "0 = bound by --buffer-cap positions only. --buffer-cap still applies as a ceiling. "
                        "With --window-games-min set, this is the RAMP CEILING (max window).")
    p.add_argument("--window-games-min", type=int, default=0,
                   help="lc0/KataGo growing window: start the replay window at this many GAMES and ramp "
                        "sublinearly up to --window-games as self-play accumulates. 0 = fixed window "
                        "(no ramp). Narrow early so random-net games evict fast; widens for stability "
                        "once the policy settles. Requires --window-games (the ceiling) > this.")
    p.add_argument("--window-ramp-alpha", type=float, default=0.75,
                   help="growth exponent for the --window-games-min ramp (KataGo default 0.75). "
                        "Lower = slower/wider ramp; ~1.0 = roughly linear in games generated.")
    p.add_argument("--lr-warmup-steps", type=int, default=0,
                   help="linearly warm LR from 0 -> --lr over this many steps (0 = no warmup)")
    p.add_argument("--lr-decay-steps", type=int, default=0,
                   help="multiply LR by --lr-gamma every this many steps (0 = constant LR)")
    p.add_argument("--lr-gamma", type=float, default=0.5,
                   help="LR decay factor applied every --lr-decay-steps")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="publish an EMA of the weights with this decay (continuous-training analog of "
                        "lc0 SWA; GroupNorm net has no BN stats to recalibrate). 0 = publish raw weights.")
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
    # One arch dict drives the build AND the published `.arch.json` sidecar (so
    # clients + arena rebuild the exact net). Transformer knobs are V2-only.
    arch_kwargs = {
        "d_hidden": args.d_hidden, "c_filters": args.c_filters, "n_blocks": args.n_blocks,
    }
    if args.arch_version in ("v2", "v4"):
        arch_kwargs.update(
            n_tf_blocks=args.tf_blocks, n_heads=args.tf_heads, tf_ff_mult=args.tf_ff_mult,
        )
    if args.arch_version == "v4":
        arch_kwargs.update(se_ratio=args.se_ratio)  # SE-ResNet blocks
    model = build_model(version=args.arch_version, **arch_kwargs).to(device)
    if args.base:
        load_checkpoint(model, args.base)
        log.info("[train] warm-started from %s", args.base)
    model.train()
    # SGD + Nesterov momentum + L2 weight decay — the AlphaZero/lc0 optimizer.
    # Persistent across all steps (real continuous training); the LR schedule
    # below (warmup -> step decay) drives g["lr"] per step.
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          nesterov=True, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)

    # --- Phase 2: lc0-style LR schedule (warmup -> step decay). Constant LR when
    #     neither warmup nor decay is configured (back-compat). Applied per step. ---
    def _lr_at(step: int) -> float:
        lr = args.lr
        if args.lr_warmup_steps and step < args.lr_warmup_steps:
            lr *= (step + 1) / args.lr_warmup_steps
        if args.lr_decay_steps:
            lr *= args.lr_gamma ** (step // args.lr_decay_steps)
        return lr

    # --- Phase 2: EMA of weights (continuous analog of lc0 SWA). The PUBLISHED net
    #     is the EMA; SGD keeps running on `model`. GroupNorm => no BN stats to
    #     recalibrate, so the average is directly loadable. ---
    ema_model = None
    if args.ema_decay:
        ema_model = build_model(version=args.arch_version, **arch_kwargs).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        log.info("[train] publishing EMA of weights (decay=%.4f)", args.ema_decay)

    def _update_ema() -> None:
        if ema_model is None:
            return
        d = args.ema_decay
        with torch.no_grad():
            msd, esd = model.state_dict(), ema_model.state_dict()
            for k, v in msd.items():
                if v.dtype.is_floating_point:
                    esd[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
                else:
                    esd[k].copy_(v)  # ints/longs (e.g. counters): track exactly

    def _pub_source():
        return ema_model if ema_model is not None else model

    def _write_stats(steps_per_s: float, games_per_s: float) -> None:
        """Phase 0: a small JSON heartbeat fleet_status.py reads — the two rates
        you can't tune windows/cadence without."""
        try:
            tmp = run_dir / "train_stats.json.tmp"
            tmp.write_text(json.dumps({
                "steps": steps, "games_seen": games_seen,
                "positions_ingested": positions_ingested, "buf": len(buf),
                "steps_per_s": round(steps_per_s, 2), "games_per_s": round(games_per_s, 4),
                "lr": round(_lr_at(steps), 8), "updated": time.time(),
            }))
            os.replace(tmp, run_dir / "train_stats.json")
        except OSError:
            pass

    # Publish an initial snapshot immediately so self-play has weights to load.
    _publish(_pub_source(), weights_path)
    log.info("[train] published initial weights -> %s", weights_path)

    def _on_term(*_a):
        try:
            stop_path.touch()
        except OSError:
            pass
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    buf: list = []  # rolling replay buffer (list, trimmed from the front; sample is O(bs))
    snapshot_path = run_dir / "replay_buffer.pkl"
    # Warm resume, in priority order: (1) the replay-buffer snapshot — the EXACT
    # live window from the last run (positions + game_sizes + ingest counters),
    # so a restart to tweak a hyperparameter loses no progress and the window
    # ramp / throttle pick up where they left off; (2) failing that, the archive's
    # most-recent games. Either stays RAM-resident (read once here, not per batch).
    # --no-prime forces a cold start (waits for --min-buffer).
    step_state_path = run_dir / "train_state.pkl"
    snap = None if args.no_prime else _load_buffer_snapshot(snapshot_path)
    # The training CLOCK (step counter + SGD momentum) persists via TWO sources: the
    # buffer snapshot (clean-shutdown only) and the train_state sidecar (written every
    # publish, so it survives a crash/SIGKILL). Resume from whichever is MORE ADVANCED
    # — after a clean stop the snapshot wins (it's >= the last publish); after a crash
    # the sidecar wins (the snapshot is stale/absent). This is what stops `steps` from
    # resetting to 0 on a non-clean restart; it persists until reset_fleet wipes the run-dir.
    step_state = None if args.no_prime else _load_step_state(step_state_path)
    snap_steps = snap.get("steps", -1) if snap else -1
    side_steps = step_state.get("steps", -1) if step_state else -1
    if side_steps > snap_steps:
        resume_steps, resume_opt = step_state["steps"], step_state.get("opt_state")
    else:
        resume_steps = snap.get("steps", 0) if snap else 0
        resume_opt = snap.get("opt_state") if snap else None
    if snap is not None:
        buf = snap["buf"]
        log.info("[train] restored replay buffer snapshot: %d positions, %d games "
                 "(games_seen=%d) — warm resume", len(buf), len(snap["game_sizes"]), snap["games_seen"])
    elif archive and archive.enabled and not args.no_prime:
        buf = archive.load_recent(args.buffer_cap)
        if buf:
            log.info("[train] primed replay window from archive: %d positions (cap %d)", len(buf), args.buffer_cap)
    elif args.no_prime:
        log.info("[train] cold start (--no-prime): not priming from snapshot/archive; trainer waits for min_buffer")
    # Resume the OPTIMIZATION process: the SGD momentum buffers. load_state_dict
    # overwrites the param-group knobs with the saved ones, so re-apply the CURRENT
    # CLI knobs afterward — a restart to TWEAK --lr/--momentum/--weight-decay must
    # still take effect; only the momentum BUFFERS carry over. Best-effort: an arch
    # change (param-shape mismatch) falls back to fresh momentum, and the LR schedule
    # still resumes from the saved step.
    if resume_opt is not None:
        _HYPER = ("lr", "momentum", "weight_decay", "nesterov", "dampening")
        fresh_hyper = [{k: g[k] for k in _HYPER if k in g} for g in opt.param_groups]
        try:
            opt.load_state_dict(resume_opt)
            for g, hp in zip(opt.param_groups, fresh_hyper):
                g.update(hp)
            log.info("[train] resumed optimizer state (SGD momentum + LR schedule at step %d)", resume_steps)
        except (ValueError, KeyError, RuntimeError) as e:
            log.warning("[train] could not resume optimizer state (%s) — fresh momentum; "
                        "LR schedule still resumes at step %d", e, resume_steps)
    steps = resume_steps   # resume the LR-schedule clock (warmup/decay)
    games_seen = snap["games_seen"] if snap else 0
    positions_ingested = snap["positions_ingested"] if snap else len(buf)  # restores the replay-factor throttle state
    last_publish = last_ckpt = time.time()
    last_pub_steps = last_pub_games = 0   # Phase 1: progress-based publish triggers
    games_at_ckpt = 0                     # Phase 0: games/s baseline for the ckpt window
    steps_at_splog = 0                    # Phase 0: steps/s baseline for the 60s stats heartbeat
    game_sizes: "deque[int]" = deque(snap["game_sizes"] if snap else ())  # Phase 2: per-game position counts for the game-count window
    last_buf_snap = time.time()           # periodic replay-buffer snapshot timer (--buffer-snapshot-seconds)
    win_cap = _window_cap(games_seen, args.window_games_min, args.window_games, args.window_ramp_alpha)  # live window cap (ramps)
    ckpt_n = 0
    win_p = win_v = win_m = 0.0
    win_steps = 0
    # Frequent [loss] window (reset every --log-seconds, independent of the ckpt window).
    lwin_p = lwin_v = lwin_m = 0.0
    lwin_gnorm = 0.0            # sum of pre-clip grad L2 norms
    lwin_steps = 0
    lwin_upd_ratio = 0.0       # last measured ||Δw|| / ||w|| (update-to-weight ratio)
    last_loss_log = time.time()
    sp_window: dict[str, int] = {}      # self-play games per machine since the last [selfplay] rollup
    last_sp_log = time.time()
    win_desc = ("off" if not args.window_games
                else f"{args.window_games_min}->{args.window_games}g ramp@a{args.window_ramp_alpha:g}"
                if 0 < args.window_games_min < args.window_games
                else f"{args.window_games}g")
    log.info("[train] continuous trainer up: device=%s buffer_cap=%d window=%s min=%d batch=%d "
             "replay_factor=%.1f per_game_keep=%.2f publish=[%ss/%dst/%dg] ckpt=%.0fs ema=%.4f",
             device, args.buffer_cap, win_desc, args.min_buffer, args.batch_size,
             args.replay_factor, args.per_game_keep,
             (f"{args.publish_seconds:.0f}" if args.publish_seconds else "off"),
             args.publish_steps, args.publish_games, args.ckpt_seconds, args.ema_decay)

    while not stop_path.exists():
        new_ex, metas = _drain(buffer_dir, archive, args.per_game_keep, rng)
        if metas:
            buf.extend(new_ex)
            games_seen += len(metas)
            positions_ingested += len(new_ex)
            for m in metas:  # tally per-machine self-play for the ~60s [selfplay] rollup
                mach = m.get("machine", "?")
                sp_window[mach] = sp_window.get(mach, 0) + 1
                game_sizes.append(int(m.get("kept", 0)))  # per-game size for the window
            if args.window_games:
                # Phase 2 — lc0 sliding window: evict whole oldest GAMES past the
                # game cap, with --buffer-cap kept as a hard position ceiling.
                # The cap RAMPS (lc0/KataGo growing window) when --window-games-min
                # is set: narrow early (random games evict fast), widening toward
                # --window-games as self-play accumulates. Fixed when min is unset.
                # (Exact under cold start / --no-prime, where buf is wholly game-sourced.)
                win_cap = _window_cap(games_seen, args.window_games_min,
                                      args.window_games, args.window_ramp_alpha)
                while len(game_sizes) > win_cap or (
                        args.buffer_cap and len(buf) > args.buffer_cap and len(game_sizes) > 1):
                    del buf[: game_sizes.popleft()]
            elif len(buf) > args.buffer_cap:
                del buf[: len(buf) - args.buffer_cap]  # drop oldest in place
            if args.max_games and games_seen >= args.max_games:
                log.info("[train] reached --max-games %d (games_seen=%d) — stopping run", args.max_games, games_seen)
                break

        # Per-machine self-play throughput rollup (~60s): one glance shows every box
        # (local + leena) producing games concurrently — the "harmonious fleet" signal,
        # alongside the arena's [arena] gate lines in the same /tmp/cc_train.log stream.
        sp_now = time.time()
        if sp_now - last_sp_log >= 60.0:
            elapsed = sp_now - last_sp_log
            _write_stats((steps - steps_at_splog) / elapsed, sum(sp_window.values()) / elapsed)
            if sp_window:
                backlog = sum(1 for _ in buffer_dir.glob("*.pkl"))
                split = " ".join(f"{k}:{v}" for k, v in sorted(sp_window.items()))
                win_now = (_window_cap(games_seen, args.window_games_min, args.window_games,
                                       args.window_ramp_alpha) if args.window_games else 0)
                log.info("[selfplay] +%d games/%ds across workers | %s | buffer backlog=%d%s | %.1f steps/s",
                         sum(sp_window.values()), int(elapsed), split, backlog,
                         (f" | window={win_now}/{args.window_games}g" if win_now else ""),
                         (steps - steps_at_splog) / elapsed)
            sp_window = {}
            last_sp_log = sp_now
            steps_at_splog = steps

        if len(buf) < args.min_buffer:
            time.sleep(1.0)
            continue
        # Throttle: don't reuse data faster than replay_factor x ingest.
        if args.replay_factor and steps * args.batch_size >= args.replay_factor * positions_ingested:
            time.sleep(0.2)
            continue

        for g in opt.param_groups:           # Phase 2: LR schedule (no-op if constant)
            g["lr"] = _lr_at(steps)
        bs = min(args.batch_size, len(buf))
        batch = rng.sample(buf, bs)
        opt.zero_grad()
        p_loss, v_loss, ml_loss = _batch_loss(
            model, batch, gamma=args.value_discount, q_ratio=args.value_q_ratio)
        (p_loss + args.value_loss_weight * v_loss + args.mlh_loss_weight * ml_loss).backward()
        # Always measure the pre-clip grad norm (max_norm=inf computes it WITHOUT scaling
        # when clipping is disabled), so "is the LR too high?" is a logged number.
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=(args.grad_clip if args.grad_clip and args.grad_clip > 0 else float("inf")),
        ))
        # On a log-due step, snapshot weights before/after to measure ||Δw||/||w|| (the
        # scale-free "update-to-weight ratio"; healthy SGD ~1e-3, >1e-2 => steps too big).
        measure_upd = (time.time() - last_loss_log) >= args.log_seconds
        if measure_upd:
            from torch.nn.utils import parameters_to_vector
            w_before = parameters_to_vector(model.parameters()).detach().clone()
        opt.step()
        if measure_upd:
            w_after = parameters_to_vector(model.parameters()).detach()
            lwin_upd_ratio = float((w_after - w_before).norm() / (w_before.norm() + 1e-12))
        steps += 1
        _update_ema()                        # Phase 2: track the published EMA (no-op if off)
        win_p += float(p_loss.item()); win_v += float(v_loss.item()); win_m += float(ml_loss.item()); win_steps += 1
        lwin_p += float(p_loss.item()); lwin_v += float(v_loss.item()); lwin_m += float(ml_loss.item()); lwin_steps += 1
        lwin_gnorm += gnorm

        # Phase 1: publish on TRAINING PROGRESS (steps and/or games), with
        # --publish-seconds as an optional time floor. Any configured trigger fires.
        now = time.time()
        pub = ((args.publish_steps and steps - last_pub_steps >= args.publish_steps)
               or (args.publish_games and games_seen - last_pub_games >= args.publish_games)
               or (args.publish_seconds and now - last_publish >= args.publish_seconds))
        if pub:
            _publish(_pub_source(), weights_path)
            # Crash-safe clock: persist steps + momentum on the publish cadence so a
            # non-clean restart keeps the LR-schedule step counter (no reset to 0).
            _save_step_state(step_state_path, steps, _to_cpu(opt.state_dict()))
            last_publish, last_pub_steps, last_pub_games = now, steps, games_seen
        # Frequent loss/health log (every --log-seconds) — the loss curve + LR-too-high
        # instruments (grad norm, update/weight ratio), without waiting a
        # whole --ckpt-seconds interval. Saves nothing to disk; pure stdout.
        if now - last_loss_log >= args.log_seconds and lwin_steps:
            log.info("[loss] step %d | policy=%.4f value=%.4f mlh=%.4f | gnorm=%.2f "
                     "upd/w=%.1e | lr=%.2e",
                     steps,
                     lwin_p / lwin_steps, lwin_v / lwin_steps, lwin_m / lwin_steps,
                     lwin_gnorm / lwin_steps, lwin_upd_ratio, _lr_at(steps))
            lwin_p = lwin_v = lwin_m = 0.0
            lwin_gnorm = 0.0; lwin_steps = 0
            last_loss_log = now
        if now - last_ckpt >= args.ckpt_seconds:
            ckpt_n += 1
            ckpt = run_dir / f"iter-async-{ckpt_n:04d}.pt"
            save_checkpoint(model, ckpt)
            arch_stat = (f" archived={archive.games_written}" if archive and archive.enabled
                         else " archived=off" if not archive
                         else " archived=DISABLED")  # best-effort archive dropped out mid-run
            elapsed = max(now - last_ckpt, 1e-9)
            steps_per_s = win_steps / elapsed
            games_per_s = (games_seen - games_at_ckpt) / elapsed
            log.info("[train] step %d | policy=%.4f value=%.4f mlh=%.4f | %.1f steps/s %.3f games/s "
                     "| lr=%.2e buf=%d games_seen=%d/%s",
                     steps,
                     win_p / max(win_steps, 1), win_v / max(win_steps, 1), win_m / max(win_steps, 1),
                     steps_per_s, games_per_s, _lr_at(steps),
                     len(buf), games_seen, (args.max_games or "inf"))
            log.info("[ckpt] saved %s | step %d |%s", ckpt.name, steps, arch_stat)
            _write_stats(steps_per_s, games_per_s)
            win_p = win_v = win_m = 0.0; win_steps = 0; last_ckpt = now; games_at_ckpt = games_seen
        if args.buffer_snapshot_seconds and now - last_buf_snap >= args.buffer_snapshot_seconds:
            _save_buffer_snapshot(snapshot_path, buf, game_sizes, games_seen, positions_ingested,
                                  steps, _to_cpu(opt.state_dict()))
            last_buf_snap = now
        if args.max_steps and steps >= args.max_steps:
            break

    stop_path.touch()  # signal local self-play workers (shared run-dir STOP) + the sidecar to tear down
    _save_buffer_snapshot(snapshot_path, buf, game_sizes, games_seen, positions_ingested,
                          steps, _to_cpu(opt.state_dict()))  # persist window + optimizer for a warm restart
    _publish(_pub_source(), weights_path)
    save_checkpoint(model, run_dir / "iter-async-final.pt")
    log.info("[train] stopped: %d steps, %d games seen, %d positions ingested, %d games archived (STOP signaled)",
             steps, games_seen, positions_ingested,
             archive.games_written if archive else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
