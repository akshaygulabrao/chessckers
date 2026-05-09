"""Cross-process inference server: one model on GPU, many worker subprocesses.

The in-process `InferenceServer` (inference_server.py) batches leaves WITHIN
one worker. With async self-play running 8 workers, that's 8 small batched
forwards in parallel — most of the GPU time is per-call kernel-launch
overhead, not actual compute.

This module collapses all 8 workers' leaves into a single coordinator-side
inference thread. Workers encode positions locally, ship pre-encoded
tensors to the server via mp.Queue, and block on a per-worker response
queue. The server drains the request queue (up to `max_batch_size`,
within `timeout_ms`), runs ONE batched forward on the live model
(shared with the trainer thread), and dispatches results back.

Drop-in replacement: `CrossInferenceClient.submit(fen, legal_moves)`
returns a `Future[(value, priors)]` matching `InferenceServer.submit`.

Bonus: workers no longer need their own GPU model copy. Self-play
worker subprocesses become pure-CPU: PyVariantClient + MCTS +
encoding + IPC. Frees ~200 MB GPU per worker and skips ~4 s of
per-worker CUDA init.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from multiprocessing import queues as mp_queues
from typing import Any

import torch

from chessckers_engine.encoding import encode_move, encode_position
from chessckers_engine.model import ChesskersScorer

LegalMove = dict[str, Any]
log = logging.getLogger("chessckers_engine.cross_inference")

# (worker_id, request_id, position_tensor, moves_tensor_or_None)
Request = tuple[int, int, torch.Tensor, "torch.Tensor | None"]
# (request_id, value, priors_list)
Response = tuple[int, float, list[float]]


class CrossInferenceServer:
    """Coordinator-side server: one thread, batches across all workers.

    Owns the model (typically the same object the trainer holds). After every
    trainer SGD step, the next forward sees the updated weights for free —
    no weights file IPC, no hot-reload poll.
    """

    def __init__(
        self,
        model: ChesskersScorer,
        request_q: mp_queues.Queue,
        response_qs: list[mp_queues.Queue],
        max_batch_size: int = 64,
        timeout_ms: float = 5.0,
        log_every: int = 0,
    ) -> None:
        self.model = model
        self.request_q = request_q
        self.response_qs = response_qs
        self.max_batch_size = max_batch_size
        self.timeout = timeout_ms / 1000.0
        self.log_every = log_every
        self._stop = threading.Event()
        self._stats_lock = threading.Lock()
        self._n_batches = 0
        self._n_requests = 0
        self._max_bs_seen = 0
        self._inference_secs = 0.0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="CrossInferenceServer"
        )

    def start(self) -> None:
        self._thread.start()

    def _collect_batch(self) -> list[Request]:
        """Block briefly for the first request, then drain more within timeout."""
        batch: list[Request] = []
        try:
            batch.append(self.request_q.get(timeout=0.1))
        except queue.Empty:
            return batch
        deadline = time.perf_counter() + self.timeout
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(self.request_q.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        # Outer try ensures the thread NEVER dies silently. A dead server
        # thread leaves all workers blocked on response_q forever.
        while not self._stop.is_set():
            try:
                batch = self._collect_batch()
                if not batch:
                    continue
                try:
                    t0 = time.perf_counter()
                    self._process_and_dispatch(batch)
                    self._record(len(batch), time.perf_counter() - t0)
                except Exception as e:  # noqa: BLE001
                    log.exception(
                        "cross-inference batch of %d failed: %s", len(batch), e,
                    )
                    # Best-effort: send error sentinels so workers don't deadlock.
                    for worker_id, request_id, _, _ in batch:
                        try:
                            self.response_qs[worker_id].put((request_id, 0.0, []))
                        except Exception:
                            pass
            except Exception as outer:  # noqa: BLE001
                # collect_batch or anything else raised — log and keep looping.
                log.exception("cross-inference outer loop error: %s", outer)
                time.sleep(0.05)

    def _process_and_dispatch(self, batch: list[Request]) -> None:
        device = next(self.model.parameters()).device
        positions = torch.stack([req[2] for req in batch]).to(device)
        moves_list: list[torch.Tensor | None] = [
            (req[3].to(device) if req[3] is not None else None) for req in batch
        ]
        with torch.no_grad():
            values, priors_list = self.model.batch_eval(positions, moves_list)
        values_cpu = values.tolist()
        for i, (worker_id, request_id, _, _) in enumerate(batch):
            priors = priors_list[i].tolist() if priors_list[i].numel() > 0 else []
            self.response_qs[worker_id].put(
                (request_id, float(values_cpu[i]), priors)
            )

    def _record(self, batch_size: int, inference_secs: float) -> None:
        with self._stats_lock:
            self._n_batches += 1
            self._n_requests += batch_size
            if batch_size > self._max_bs_seen:
                self._max_bs_seen = batch_size
            self._inference_secs += inference_secs
            n = self._n_batches
        if self.log_every and n % self.log_every == 0:
            with self._stats_lock:
                avg = self._n_requests / max(self._n_batches, 1)
                gpu_secs = self._inference_secs
            log.info(
                "x-inference: batches=%d reqs=%d avg_bs=%.2f max_bs=%d gpu_secs=%.2f",
                n, self._n_requests, avg, self._max_bs_seen, gpu_secs,
            )

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            n = max(self._n_batches, 1)
            return {
                "n_batches": self._n_batches,
                "n_requests": self._n_requests,
                "avg_batch_size": self._n_requests / n,
                "max_batch_size_seen": self._max_bs_seen,
                "inference_secs": self._inference_secs,
            }

    def shutdown(self, wait_seconds: float = 3.0) -> None:
        self._stop.set()
        self._thread.join(timeout=wait_seconds)


class CrossInferenceClient:
    """Worker-side client: encodes locally, ships to server, blocks for response.

    Returns Futures for API parity with `InferenceServer.submit`. Within a
    worker, MCTS submits one leaf at a time and immediately calls .result(),
    so submit() blocks synchronously on the response queue. (Per-worker
    response queue is single-consumer, so order is preserved.)
    """

    def __init__(
        self,
        worker_id: int,
        request_q: mp_queues.Queue,
        response_q: mp_queues.Queue,
    ) -> None:
        self.worker_id = worker_id
        self.request_q = request_q
        self.response_q = response_q
        self._next_id = 0

    def submit(
        self, fen: str, legal_moves: list[LegalMove]
    ) -> "Future[tuple[float, list[float]]]":
        pos = encode_position(fen)
        moves = (
            torch.stack([encode_move(m) for m in legal_moves])
            if legal_moves else None
        )
        rid = self._next_id
        self._next_id += 1
        self.request_q.put((self.worker_id, rid, pos, moves))
        # Per-worker response queue is FIFO single-consumer → next message is ours.
        response_rid, value, priors = self.response_q.get()
        if response_rid != rid:
            raise RuntimeError(
                f"cross-inference response order violated: "
                f"expected rid={rid}, got rid={response_rid}"
            )
        future: Future = Future()
        future.set_result((value, priors))
        return future

    def shutdown(self) -> None:
        # Queues are owned by the coordinator; nothing for the worker to clean up.
        return None
