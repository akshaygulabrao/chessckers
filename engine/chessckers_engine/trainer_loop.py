"""Continuous trainer loop for async AlphaZero training.

Runs forever (until `stop_event` is set), draining mini-batches from a
shared `ReplayBuffer` and updating the model. Periodically broadcasts
weights to a shared file that self-play workers mtime-poll, and dumps
durable checkpoints at a coarser interval.

Decoupled from self-play: this loop has no idea games are being produced
in parallel. It just keeps training as fast as `_batch_loss` + the buffer
let it.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.replay_buffer import ReplayBuffer
from chessckers_engine.train_az import _batch_loss

log = logging.getLogger("chessckers_engine.trainer_loop")


class TrainerLoop:
    def __init__(
        self,
        model: ChesskersScorer,
        buffer: ReplayBuffer,
        weights_path: str | os.PathLike,
        checkpoint_dir: str | os.PathLike,
        device: str = "cpu",
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_save_every: int = 200,
        checkpoint_every: int = 2000,
        min_buffer_games: int = 20,
        value_loss_weight: float = 1.0,
        grad_clip: float = 1.0,
        log_every: int = 50,
        wait_poll_seconds: float = 2.0,
        max_steps: Optional[int] = None,
        stop_event: Optional[mp.Event] = None,
        resume_from: Optional[str | os.PathLike] = None,
    ):
        self.model = model.to(device)
        self.buffer = buffer
        self.weights_path = Path(weights_path)
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.weight_save_every = weight_save_every
        self.checkpoint_every = checkpoint_every
        self.min_buffer_games = min_buffer_games
        self.value_loss_weight = value_loss_weight
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.wait_poll_seconds = wait_poll_seconds
        self.max_steps = max_steps
        self.stop_event = stop_event
        self.resume_from = Path(resume_from) if resume_from else None
        self.step = 0
        # Set lazily in run() so we can rehydrate optimizer state too.
        self._opt: Optional[torch.optim.Optimizer] = None

    def _stopped(self) -> bool:
        return bool(self.stop_event and self.stop_event.is_set())

    def save_weights_atomic(self) -> None:
        """Save current weights so workers polling `weights_path` can hot-reload.
        Atomic via .tmp + rename so a worker never reads a half-written file."""
        tmp = self.weights_path.with_suffix(self.weights_path.suffix + ".tmp")
        torch.save(self.model.state_dict(), tmp)
        os.replace(tmp, self.weights_path)

    def save_checkpoint(self, suffix: str = "") -> Path:
        """Durable checkpoint with model + optimizer + step. Resumable.

        Atomic write so a kill mid-save can't leave a half-written file at the
        target path. The bare-`state_dict()` form is preserved as the contents
        of `weights.pt`; this richer dict lives only in `checkpoint_dir`.
        """
        name = f"step_{self.step:08d}{('_' + suffix) if suffix else ''}.pt"
        path = self.checkpoint_dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "model": self.model.state_dict(),
            "step": self.step,
        }
        if self._opt is not None:
            payload["optimizer"] = self._opt.state_dict()
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def _restore_from(self, path: Path) -> None:
        """Restore model + optimizer + step from a richer checkpoint.

        Tolerates the legacy bare-state_dict format: that case loads model
        weights only and starts the optimizer + step fresh. Logs which path
        was taken so silent half-resumes don't go unnoticed."""
        target_device = next(self.model.parameters()).device
        obj = torch.load(path, map_location=target_device, weights_only=True)
        if isinstance(obj, dict) and "model" in obj:
            self.model.load_state_dict(obj["model"], strict=False)
            if "step" in obj:
                self.step = int(obj["step"])
            if "optimizer" in obj and self._opt is not None:
                self._opt.load_state_dict(obj["optimizer"])
                log.info("resumed trainer from %s: step=%d (with optimizer state)",
                         path, self.step)
            else:
                log.info("resumed trainer from %s: step=%d (model only — no optimizer)",
                         path, self.step)
        else:
            # Legacy: file is a raw state_dict. Model weights only.
            self.model.load_state_dict(obj, strict=False)
            log.info("resumed trainer from %s: model only (legacy bare-state_dict format)",
                     path)

    def _wait_for_buffer(self) -> bool:
        """Block until buffer has min_buffer_games. Returns False if stopped."""
        while self.buffer.count_games() < self.min_buffer_games:
            if self._stopped():
                return False
            time.sleep(self.wait_poll_seconds)
        return True

    def run(self) -> int:
        """Main loop. Returns the number of training steps completed."""
        if not self._wait_for_buffer():
            return self.step

        self._opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Restore optimizer + step from a previous checkpoint if requested.
        if self.resume_from is not None and self.resume_from.exists():
            self._restore_from(self.resume_from)
        value_loss_fn = nn.MSELoss()
        self.model.train()
        # Broadcast initial weights so workers can start with the trainer's
        # version of the network (rather than each worker's random init).
        self.save_weights_atomic()
        last_log = time.perf_counter()

        while not self._stopped():
            if self.max_steps is not None and self.step >= self.max_steps:
                break
            batch = self.buffer.sample(self.batch_size)
            if not batch:
                time.sleep(self.wait_poll_seconds)
                continue
            self._opt.zero_grad()
            p_loss, v_loss = _batch_loss(self.model, batch, value_loss_fn)
            total = p_loss + self.value_loss_weight * v_loss
            total.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self._opt.step()
            self.step += 1
            # Yield the GIL so a co-resident inference-server thread (in the
            # shared-inference architecture) gets fair scheduling. On CPU
            # with a tiny model, the trainer can sustain 100+ steps/s of
            # Python-side work between GIL releases, starving the server
            # and causing workers to block forever on response queues.
            time.sleep(0)

            if self.step % self.weight_save_every == 0:
                self.save_weights_atomic()
            if self.step % self.checkpoint_every == 0:
                self.save_checkpoint()
            if self.log_every and self.step % self.log_every == 0:
                now = time.perf_counter()
                steps_per_s = self.log_every / max(now - last_log, 1e-9)
                last_log = now
                log.info(
                    "step=%d policy=%.4f value=%.4f total=%.4f steps/s=%.2f games=%d",
                    self.step, p_loss.item(), v_loss.item(), total.item(),
                    steps_per_s, self.buffer.count_games(),
                )

        self.save_weights_atomic()
        self.save_checkpoint(suffix="final")
        return self.step
