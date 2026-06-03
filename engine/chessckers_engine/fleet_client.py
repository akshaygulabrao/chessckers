"""Fleet client — lc0-style self-play client sync bridge.

Runs on every self-play box (the local machine over loopback AND leena / future
volunteers over the LAN). The actual game generation is the existing
`selfplay_workers_only` process writing pkls into `--run-dir/buffer` and
mtime-polling `--run-dir/weights.pt`; this bridge is the only thing that talks to
the server, so a volunteer needs no inbound SSH — just outbound HTTP. It replaces
the rsync `leena_sync.sh` sidecar.

Each tick (`--poll-seconds`):
  1. GET /control (carrying an `X-Client-Id` heartbeat so the server can list live
     boxes in /status) — if STOP, signal the local workers (touch run-dir/STOP) and
     exit. A tick that raises an unexpected error is logged and retried next poll —
     the loop never dies (an unattended volunteer box must outlive transient faults).
  2. GET /version — if changed, GET /weights and write run-dir/weights.pt atomically.
     The local workers hot-reload it on their own mtime poll. Version tracks the
     trainer's newest checkpoint, so this fires once per trainer ITERATION.
  3. Upload finished games: each `*.pkl` older than --min-age with its `.meta`
     present is POSTed (meta first, then pkl, so the server-side trainer sees a
     complete pair), then deleted locally — each game uploaded exactly once.
  4. If a keep-best gate is open, contribute up to --match-games-per-tick gate
     games (GET /next_game, play, POST /match_result) via `fleet_match`. This is
     the only heavy step (torch + move-gen ext), imported lazily and only while a
     match is open; a box without those deps logs once and stays self-play-only.

The sync path is stdlib only (urllib) — no requests/aiohttp dep, so it runs on a
bare volunteer venv; only the optional step 4 pulls in the engine.

Run (on a self-play box):

    python -m chessckers_engine.fleet_client \\
      --server http://192.168.1.50:8000 --run-dir ~/chessckers/run --poll-seconds 15
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from chessckers_engine.runtime import setup_logging

log = logging.getLogger("chessckers_engine.fleet_client")


def _get(url: str, timeout: float, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _post(url: str, data: bytes, timeout: float) -> None:
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()


def _pull_weights_if_new(server: str, weights: Path, last_version: str, timeout: float) -> str:
    """Pull weights.pt iff the server's version changed. Returns the (possibly
    unchanged) current version so the caller can track it."""
    try:
        version = _get(f"{server}/version", timeout).decode().strip()
    except (urllib.error.URLError, OSError) as e:
        log.debug("version poll failed (server down/restarting?): %s", e)
        return last_version
    if version == last_version or version == "none":
        return version
    try:
        data = _get(f"{server}/weights", timeout)
    except (urllib.error.URLError, OSError) as e:
        log.debug("weights fetch failed: %s", e)
        return last_version  # retry next tick
    tmp = weights.with_suffix(".pt.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, weights)  # atomic; bumps mtime -> local workers hot-reload
    except OSError as e:
        log.warning("weights write failed: %s", e)
        return last_version
    log.info("pulled net %s -> %s (%d KB)", version, weights, len(data) // 1024)
    return version


def _upload_games(server: str, buffer: Path, min_age: float, timeout: float) -> int:
    """POST each complete, settled game (pkl + .meta) to the server, then delete it
    locally. meta first so the server-side trainer never globs a pkl whose meta has
    not landed. Returns the number of games uploaded this tick."""
    if not buffer.exists():
        return 0
    now = time.time()
    uploaded = 0
    for pkl in sorted(buffer.glob("*.pkl")):
        try:
            if now - pkl.stat().st_mtime < min_age:
                continue  # still being written / its .meta not yet flushed
        except OSError:
            continue
        meta = Path(str(pkl) + ".meta")
        try:
            # meta is best-effort on the worker side; upload it first when present,
            # else upload the pkl alone (attribution lost, data kept).
            if meta.exists():
                _post(f"{server}/game/{meta.name}", meta.read_bytes(), timeout)
            _post(f"{server}/game/{pkl.name}", pkl.read_bytes(), timeout)
        except (urllib.error.URLError, OSError) as e:
            log.debug("upload %s failed (retry next tick): %s", pkl.name, e)
            break  # server down — stop this tick, keep games for retry
        for fp in (meta, pkl):
            try:
                fp.unlink()
            except OSError:
                pass
        uploaded += 1
    return uploaded


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Fleet client: pull net / push games over HTTP (lc0-style).")
    p.add_argument("--server", required=True, help="e.g. http://192.168.1.50:8000")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Self-play run-dir (shared with selfplay_workers_only): "
                        "weights.pt is written here, games are read from buffer/.")
    p.add_argument("--poll-seconds", type=float, default=15.0)
    p.add_argument("--min-age", type=float, default=2.0,
                   help="Only upload pkls older than this (s) so their .meta has flushed.")
    p.add_argument("--timeout", type=float, default=30.0, help="per-request HTTP timeout (s)")
    p.add_argument("--match-games-per-tick", type=int, default=1,
                   help="keep-best gate games to contribute per tick while a match is open "
                        "(0 = self-play only; needs torch + the move-gen ext, degrades gracefully)")
    p.add_argument("--client-id", default="",
                   help="fleet liveness id sent as X-Client-Id (default: this box's hostname)")
    args = p.parse_args()

    server = args.server.rstrip("/")
    client_id = args.client_id or socket.gethostname()
    hb = {"X-Client-Id": client_id}  # per-tick heartbeat header (server tracks last-seen)
    run_dir = args.run_dir.resolve()
    weights = run_dir / "weights.pt"
    buffer = run_dir / "buffer"
    stop_path = run_dir / "STOP"
    buffer.mkdir(parents=True, exist_ok=True)

    log.info("fleet client up: server=%s run-dir=%s poll=%.0fs id=%s",
             server, run_dir, args.poll_seconds, client_id)
    last_version = ""
    total_up = 0
    runner = None
    match_disabled = args.match_games_per_tick <= 0
    while True:
        try:
            # 1. control (also the per-tick liveness heartbeat via the X-Client-Id header)
            try:
                control = _get(f"{server}/control", args.timeout, hb).decode().strip()
            except (urllib.error.URLError, OSError):
                control = "RUN"  # server unreachable — keep self-playing on current weights
            if control == "STOP":
                log.info("server signaled STOP -> stopping local workers + exiting")
                try:
                    stop_path.touch()
                except OSError:
                    pass
                break
            # 2. net
            last_version = _pull_weights_if_new(server, weights, last_version, args.timeout)
            # 3. games
            n = _upload_games(server, buffer, args.min_age, args.timeout)
            if n:
                total_up += n
                log.info("uploaded %d game(s) (total %d)", n, total_up)
            # 4. contribute gate games while a keep-best match is open (lc0-style).
            if not match_disabled:
                if runner is None:
                    try:
                        from chessckers_engine.fleet_match import MatchRunner
                        runner = MatchRunner(run_dir)
                    except Exception as e:  # noqa: BLE001 — torch/native ext absent: self-play only
                        log.info("gate-game contribution unavailable (%s) — self-play only", e)
                        match_disabled = True
                if runner is not None:
                    played = 0
                    for _ in range(args.match_games_per_tick):
                        try:
                            if runner.step(server, args.timeout) == 0:
                                break  # no match open right now
                        except (urllib.error.URLError, OSError) as e:
                            log.debug("match step failed (retry next tick): %s", e)
                            break
                        except Exception as e:  # noqa: BLE001 — one bad game shouldn't kill the loop
                            log.warning("match step error (retry next tick): %s", e)
                            break
                        played += 1
                    if played:
                        log.info("contributed %d gate game(s)", played)
        except Exception as e:  # noqa: BLE001 — unattended box: never die on an unexpected tick error
            log.warning("tick error (continuing): %s", e)
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
