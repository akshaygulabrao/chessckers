"""Cross-game inference batching for vectorized MCTS.

Each MCTS worker calls `server.submit(fen, legal_moves)` and blocks on the
returned Future. A background thread drains the request queue (up to
`max_batch_size` requests, or until `timeout_ms` elapses since the first
queued request) and runs `model.batch_eval` once per batch, fanning the
results back to each future.

Effective batch size scales with the number of concurrent game workers.
On GPU this amortizes per-call dispatch overhead — a 16-way batched
forward is roughly the cost of 1-2 single-position forward passes.

Usage:
    with InferenceServer(model, max_batch_size=16) as srv:
        # game worker thread:
        value, priors = srv.submit(fen, legal_moves).result()
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any

import torch

from chessckers_engine.encoding import encoders_for
from chessckers_engine.model import ChesskersScorer

LegalMove = dict[str, Any]
log = logging.getLogger("chessckers_engine.inference_server")


class InferenceServer:
    """Thread-safe batched inference for MCTS game workers."""

    def __init__(
        self,
        model: ChesskersScorer,
        max_batch_size: int = 16,
        timeout_ms: float = 5.0,
        log_every: int = 0,
    ) -> None:
        self.model = model
        # Pick encoders by the model's arch VERSION so the batched path matches
        # the net: V2/V3 need the 16ch/10x10 position + 114-dim gather move
        # encoding, NOT V1's 15ch/8x8 + 240-dim. Without this, batch_eval gets
        # the wrong tensor shapes for a transformer/gather net.
        self._enc_pos, _, self._enc_move = encoders_for(getattr(model, "VERSION", "v1"))
        self.max_batch_size = max_batch_size
        self.timeout = timeout_ms / 1000.0
        self.log_every = log_every
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        # Stats — updated under _stats_lock so stats()/log_summary() are safe.
        # Histogram buckets: [1, 2-3, 4-7, 8-15, 16-31, 32+].
        self._stats_lock = threading.Lock()
        self._n_batches = 0
        self._n_requests = 0
        self._max_bs_seen = 0
        self._hist = [0] * 6
        self._inference_secs = 0.0
        self._wait_first_secs = 0.0
        self._wait_drain_secs = 0.0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="InferenceServer"
        )
        self._thread.start()

    def submit(
        self, fen: str, legal_moves: list[LegalMove]
    ) -> "Future[tuple[float, list[float]]]":
        """Submit a (fen, moves) request. Returns a Future resolving to
        (value, priors) where value ∈ [-1, 1] and priors is a list of
        len(legal_moves) probabilities (or [] when legal_moves is empty).

        Encoding happens here, in the caller's thread, so concurrent worker
        threads encode in parallel rather than serializing through the
        single inference thread. The server only stacks pre-encoded tensors
        and runs the batched forward."""
        pos_tensor = self._enc_pos(fen)  # (15,8,8) v1 / (16,10,10) v2
        if legal_moves:
            moves_tensor = torch.stack([self._enc_move(m) for m in legal_moves])
        else:
            moves_tensor = None
        future: Future = Future()
        self._queue.put((pos_tensor, moves_tensor, future))
        return future

    def _run(self) -> None:
        while not self._stop.is_set():
            t_wait_first = time.perf_counter()
            batch, drain_secs = self._collect_batch()
            if not batch:
                continue
            wait_first_secs = time.perf_counter() - t_wait_first - drain_secs
            try:
                t0 = time.perf_counter()
                self._process_batch(batch)
                inf_secs = time.perf_counter() - t0
                self._record(len(batch), inf_secs, wait_first_secs, drain_secs)
            except Exception as e:  # noqa: BLE001
                # Resolve all futures with the exception so callers don't hang.
                log.exception("inference batch of %d failed: %s", len(batch), e)
                for _, _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)

    def _collect_batch(self) -> tuple[list[tuple[Any, Any, Future]], float]:
        """Block for one request (with periodic shutdown polling), then drain
        more (up to max_batch_size) within timeout. Returns (batch, drain_secs)
        where drain_secs is time spent collecting items 2..N (i.e. the window
        we waited for batch-mates after the first request landed). drain_secs
        is 0 when the batch is a singleton."""
        batch: list = []
        try:
            batch.append(self._queue.get(timeout=0.1))
        except queue.Empty:
            return batch, 0.0
        t_drain_start = time.perf_counter()
        deadline = t_drain_start + self.timeout
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
        return batch, time.perf_counter() - t_drain_start

    def _bucket(self, n: int) -> int:
        if n <= 1:
            return 0
        if n <= 3:
            return 1
        if n <= 7:
            return 2
        if n <= 15:
            return 3
        if n <= 31:
            return 4
        return 5

    def _record(
        self,
        batch_size: int,
        inference_secs: float,
        wait_first_secs: float,
        drain_secs: float,
    ) -> None:
        with self._stats_lock:
            self._n_batches += 1
            self._n_requests += batch_size
            if batch_size > self._max_bs_seen:
                self._max_bs_seen = batch_size
            self._hist[self._bucket(batch_size)] += 1
            self._inference_secs += inference_secs
            self._wait_first_secs += wait_first_secs
            self._wait_drain_secs += drain_secs
            n_batches = self._n_batches
            sum_reqs = self._n_requests
            max_bs = self._max_bs_seen
            hist = list(self._hist)
            inf = self._inference_secs
        if self.log_every and n_batches % self.log_every == 0:
            avg_bs = sum_reqs / n_batches
            log.info(
                "inference: batches=%d reqs=%d avg_bs=%.2f max_bs=%d "
                "hist[1,2-3,4-7,8-15,16-31,32+]=%s gpu_secs=%.2f",
                n_batches, sum_reqs, avg_bs, max_bs, hist, inf,
            )

    def stats(self) -> dict[str, Any]:
        """Snapshot of accumulated batching stats. Safe to call from any thread."""
        with self._stats_lock:
            n = max(self._n_batches, 1)
            return {
                "n_batches": self._n_batches,
                "n_requests": self._n_requests,
                "avg_batch_size": self._n_requests / n,
                "max_batch_size_seen": self._max_bs_seen,
                "hist_1_2_4_8_16_32+": list(self._hist),
                "inference_secs": self._inference_secs,
                "wait_first_secs": self._wait_first_secs,
                "wait_drain_secs": self._wait_drain_secs,
            }

    def log_summary(self) -> None:
        s = self.stats()
        if s["n_batches"] == 0:
            log.info("inference summary: no batches processed")
            return
        log.info(
            "inference summary: batches=%d reqs=%d avg_bs=%.2f max_bs=%d "
            "hist[1,2-3,4-7,8-15,16-31,32+]=%s gpu_secs=%.2f "
            "wait_first_secs=%.2f wait_drain_secs=%.2f",
            s["n_batches"], s["n_requests"], s["avg_batch_size"],
            s["max_batch_size_seen"], s["hist_1_2_4_8_16_32+"],
            s["inference_secs"], s["wait_first_secs"], s["wait_drain_secs"],
        )

    def _process_batch(
        self, batch: list[tuple[torch.Tensor, torch.Tensor | None, Future]]
    ) -> None:
        device = next(self.model.parameters()).device
        # Stack the pre-encoded position tensors (encoding already done in
        # caller threads). Single CPU→device transfer for the whole batch.
        positions = torch.stack([pos for pos, _, _ in batch]).to(device)
        moves_list: list[torch.Tensor | None] = []
        for _, moves, _ in batch:
            moves_list.append(moves.to(device) if moves is not None else None)
        with torch.no_grad():
            values, priors_list = self.model.batch_eval(positions, moves_list)
        values_cpu = values.tolist()
        for i, (_, _, fut) in enumerate(batch):
            priors = (
                priors_list[i].tolist() if priors_list[i].numel() > 0 else []
            )
            fut.set_result((float(values_cpu[i]), priors))

    def shutdown(self, wait_seconds: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=wait_seconds)
        self.log_summary()

    def __enter__(self) -> "InferenceServer":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
