"""Fleet client — lc0-style self-play client sync bridge.

Runs on every self-play box (the local machine over loopback AND leena / future
volunteers over the LAN). The actual game generation is the existing
`selfplay_workers_only` process writing pkls into `--run-dir/buffer` and
mtime-polling `--run-dir/weights.pt`; this bridge is the only thing that talks to
the server, so a volunteer needs no inbound SSH — just outbound HTTP. It replaces
the rsync `leena_sync.sh` sidecar.

With `--spawn-workers` it also OWNS that worker process (lc0's client-owns-engine
model): it launches `selfplay_workers_only` once weights land, restarts it if it
dies, and heartbeats its liveness to the server — so a zombie box (client up,
workers dead, producing nothing) shows up in /status instead of silently lying.

Each tick (`--poll-seconds`):
  1. GET /control (carrying `X-Client-Id` + `X-Client-Version` heartbeat headers so
     the server can list live boxes AND their code version in /status — a box on a
     stale commit is then visible, not a silent runtime crash) — if STOP, signal the
     local workers (touch run-dir/STOP) and exit. A tick that raises an unexpected
     error is logged and retried next poll — the loop never dies (an unattended
     volunteer box must outlive transient faults).
  1b. GET /client-version — the trainer host's code sha. With --update-cmd set, a
     mismatch against the sha this client booted on triggers a pull + native rebuild
     and an os.execv re-exec onto the fresh code (retried with backoff so it self-
     corrects even if the code lands here just after the server advertises it). Without
     --update-cmd the drift is only logged, so a stale box stays visible in /status.
  2. GET /version — if changed, GET /weights and write run-dir/weights.pt atomically.
     The local workers hot-reload it on their own mtime poll. Version tracks the
     trainer's newest checkpoint, so this fires once per trainer ITERATION.
  2b. GET /selfplay — mirror the server's canonical self-play params into
     run-dir/selfplay.json (only on a content change). The workers re-read it each
     game, so the whole fleet self-plays with the SAME, operator-tunable params and
     can be annealed mid-run without a relaunch.
  3. Upload finished games: each `*.pkl` older than --min-age with its `.meta`
     present is POSTed (meta first, then pkl, so the server-side trainer sees a
     complete pair), then deleted locally — each game uploaded exactly once.
  4. If a keep-best gate is open, DRAIN it: contribute gate games (GET /next_game,
     play, POST /match_result) via `fleet_match` until the per-tick budget
     (--match-burst-seconds) elapses or the gate closes — the box helps carry the
     whole gate instead of trickling one game per poll, WITHOUT pausing self-play
     (the workers are a separate process; this is a side task on the poll thread).
     The only heavy step (torch + move-gen ext), imported lazily and only while a
     match is open; a box without those deps logs once and stays self-play-only.
  5. (--spawn-workers) Supervise the self-play worker subprocess: spawn it once
     weights.pt has landed, restart it on unexpected exit, and heartbeat its state
     (X-Client-Workers: up/down/off) on steps 1/2b so /status can flag a zombie box.

The sync path is stdlib only (urllib) — no requests/aiohttp dep, so it runs on a
bare volunteer venv; only the optional step 4 pulls in the engine. Step 5 shells
out to the worker (never imports it), so the client itself stays stdlib-only.

Run (on a self-play box):

    python -m chessckers_engine.fleet_client \\
      --server http://192.168.1.50:8000 --run-dir ~/chessckers/run --poll-seconds 15 \\
      --spawn-workers -- --workers 4 --native --device cpu \\
      --d-hidden 256 --c-filters 96 --n-blocks 4 --worker-id-base 300

Everything after `--` is the worker command (selfplay_workers_only's flags); the
client injects --run-dir/--weights, so the launcher only passes box-specific config.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from chessckers_engine.runtime import setup_logging

log = logging.getLogger("chessckers_engine.fleet_client")

UPDATE_RETRY_S = 300.0  # re-attempt a failed/too-early self-update to a given sha no more often than this


def _git_version() -> str:
    """Best-effort short git sha of the running client code (resolved once at
    startup), reported so a box on a STALE commit — the silent-runtime-crash class
    we hit with the mismatched native .so — shows up plainly in the server's
    /status. 'unknown' when not run from a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — version is diagnostics-only; never block the client on it
        return "unknown"


def _split_worker_argv(argv: list) -> tuple:
    """Split CLI args at the first literal `--`: everything before is the client's own
    flags, everything after is the worker command (for --spawn-workers). No `--` => no
    worker command."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _spawn_workers(worker_argv: list, run_dir: Path, weights: Path, log_path: Path):
    """Launch selfplay_workers_only as a supervised child (lc0's client-owns-engine
    model). The client injects --run-dir/--weights (it owns those paths); `worker_argv`
    carries the box-specific config the launcher passed after `--`. Worker stdout/stderr
    go to workers.log so the client's own log stays readable. Shelling out (not
    importing) keeps the client stdlib-only. Returns (Popen, open-file)."""
    cmd = [sys.executable, "-m", "chessckers_engine.selfplay_workers_only",
           "--run-dir", str(run_dir), "--weights", str(weights), *worker_argv]
    f = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    log.info("spawned workers (pid %d) -> %s | args: %s",
             proc.pid, log_path, " ".join(worker_argv) or "(none)")
    return proc, f


def _self_update(want: str, current: str, update_cmd: str, run_dir: Path,
                 stop_path: Path, worker_proc) -> None:
    """Pull + rebuild this box onto code sha `want` via `update_cmd`, then re-exec this
    client onto the fresh code (lc0's client-updates-itself, adapted). On success
    os.execv replaces this process and never returns. On failure (cmd errored, or the
    working tree didn't actually advance to `want`) it logs and returns, leaving the box
    visibly on `current` — the caller's backoff retries later (e.g. once the code lands).
    Owned workers are wound down before the rebuild so a stale .so isn't mid-search."""
    log.warning("code self-update: server on %s, running %s — running update-cmd", want, current)
    if worker_proc is not None:
        try:
            stop_path.touch()
            worker_proc.terminate()
        except OSError:
            pass
    try:
        r = subprocess.run(update_cmd, shell=True, cwd=str(run_dir.parent),
                           timeout=600, capture_output=True, text=True)
    except (subprocess.TimeoutExpired, OSError) as e:
        log.error("update-cmd failed to run (%s) — staying on %s", e, current)
        return
    try:
        (run_dir / "update.log").write_text(
            f"$ {update_cmd}\nrc={r.returncode}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    except OSError:
        pass
    now_sha = _git_version()
    if r.returncode == 0 and now_sha == want:
        log.warning("update OK; box now on %s — re-exec client onto fresh code", now_sha)
        try:
            stop_path.unlink()  # clear the wind-down sentinel for the fresh workers
        except OSError:
            pass
        os.execv(sys.executable,
                 [sys.executable, "-m", "chessckers_engine.fleet_client", *sys.argv[1:]])
    log.error("update-cmd rc=%d, tree now %s (wanted %s) — staying on %s; see run/update.log",
              r.returncode, now_sha, want, current)


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


def _pull_selfplay_if_new(server: str, params_path: Path, last: bytes, timeout: float,
                          headers: dict) -> bytes:
    """Mirror the server's canonical self-play params into the local run-dir. The
    workers re-read this file each game, so a server-side change anneals this box at
    the next game boundary — same mechanism as the weights pull, one level finer.
    Only rewrites on a CONTENT change so an unchanged file's mtime stays put and the
    workers don't needlessly re-parse. Returns the current bytes for tracking."""
    try:
        data = _get(f"{server}/selfplay", timeout, headers)
    except (urllib.error.URLError, OSError) as e:
        log.debug("selfplay poll failed (server down/restarting?): %s", e)
        return last
    if data == last:
        return last
    tmp = params_path.with_suffix(".json.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, params_path)
    except OSError as e:
        log.warning("selfplay params write failed: %s", e)
        return last
    log.info("self-play params updated from server -> %s",
             data.decode("utf-8", "replace").strip())
    return data


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
    # The worker command (for --spawn-workers) is everything after a literal `--`;
    # split it off before argparse so the client's own flags parse cleanly.
    argv, worker_argv = _split_worker_argv(sys.argv[1:])
    p = argparse.ArgumentParser(description="Fleet client: pull net / push games over HTTP (lc0-style).")
    p.add_argument("--server", required=True, help="e.g. http://192.168.1.50:8000")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Self-play run-dir (shared with selfplay_workers_only): "
                        "weights.pt is written here, games are read from buffer/.")
    p.add_argument("--poll-seconds", type=float, default=15.0)
    p.add_argument("--min-age", type=float, default=2.0,
                   help="Only upload pkls older than this (s) so their .meta has flushed.")
    p.add_argument("--timeout", type=float, default=30.0, help="per-request HTTP timeout (s)")
    p.add_argument("--match-burst-seconds", type=float, default=10.0,
                   help="per-tick wall-clock budget for contributing keep-best gate games while a "
                        "gate is open: the client DRAINS the open gate (lc0 lets a worker dedicate "
                        "itself to a match) WITHOUT pausing self-play — the self-play workers are a "
                        "separate process, this runs on the poll thread. 0 disables gate contribution. "
                        "Keep < --poll-seconds so heartbeats/uploads stay on cadence. (needs torch + "
                        "the move-gen ext; degrades to self-play-only if absent)")
    p.add_argument("--match-games-per-tick", type=int, default=0,
                   help="optional hard cap on gate games contributed per tick (0 = no cap; bounded "
                        "only by --match-burst-seconds). A ceiling for when individual games are fast.")
    p.add_argument("--client-id", default="",
                   help="fleet liveness id sent as X-Client-Id (default: this box's hostname)")
    p.add_argument("--spawn-workers", action="store_true",
                   help="Own the self-play worker subprocess (lc0-style): launch "
                        "selfplay_workers_only, restart it if it dies, report liveness. "
                        "Worker flags follow a `--` separator; --run-dir/--weights injected.")
    p.add_argument("--update-cmd", default="",
                   help="Shell command that pulls + rebuilds this box's code when the server "
                        "advertises a newer version (e.g. 'cd ~/chessckers && git pull --ff-only "
                        "&& cd engine && PATH=.venv/bin:$PATH cpp/build.sh'). On success the "
                        "client re-execs onto the fresh code. Empty = warn on drift only.")
    args = p.parse_args(argv)

    server = args.server.rstrip("/")
    client_id = args.client_id or socket.gethostname()
    client_version = _git_version()
    # per-tick heartbeat headers: id + code version (server tracks last-seen + version)
    hb = {"X-Client-Id": client_id, "X-Client-Version": client_version}
    run_dir = args.run_dir.resolve()
    weights = run_dir / "weights.pt"
    params_path = run_dir / "selfplay.json"
    buffer = run_dir / "buffer"
    stop_path = run_dir / "STOP"
    buffer.mkdir(parents=True, exist_ok=True)

    log.info("fleet client up: server=%s run-dir=%s poll=%.0fs id=%s version=%s",
             server, run_dir, args.poll_seconds, client_id, client_version)
    if args.spawn_workers:
        log.info("owning workers (lc0-style): selfplay_workers_only %s",
                 " ".join(worker_argv) or "(no extra args)")
    last_version = ""
    last_selfplay = b""
    total_up = 0
    runner = None
    worker_proc = None
    worker_log = None
    update_cmd = args.update_cmd
    update_backoff: dict = {}  # sha -> last attempt time, so a self-update isn't retried every tick
    match_disabled = args.match_burst_seconds <= 0
    while True:
        try:
            now = time.time()
            # 0. worker liveness for this tick's heartbeat (so the server can flag a
            #    zombie box: client heartbeating but its workers dead). off = not yet
            #    spawned (waiting on weights); up/down = supervised child alive/exited.
            if args.spawn_workers:
                ws = ("down" if (worker_proc is not None and worker_proc.poll() is not None)
                      else "up" if worker_proc is not None else "off")
                hb_tick = {**hb, "X-Client-Workers": ws}
            else:
                hb_tick = hb
            # 1. control (also the per-tick liveness heartbeat via the X-Client-Id header)
            try:
                control = _get(f"{server}/control", args.timeout, hb_tick).decode().strip()
            except (urllib.error.URLError, OSError):
                control = "RUN"  # server unreachable — keep self-playing on current weights
            if control == "STOP":
                log.info("server signaled STOP -> stopping local workers + exiting")
                try:
                    stop_path.touch()
                except OSError:
                    pass
                if worker_proc is not None:
                    try:
                        worker_proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        log.warning("workers slow to stop; terminating")
                        worker_proc.terminate()
                break
            # 1b. code self-update (lc0-style): re-exec onto the trainer host's code if
            #     we booted on an older sha. Backoff so it self-corrects if the code lands
            #     here just after the server starts advertising it.
            try:
                want = _get(f"{server}/client-version", args.timeout, hb_tick).decode().strip()
            except (urllib.error.URLError, OSError):
                want = ""
            if (want and want not in ("unknown", client_version)
                    and now - update_backoff.get(want, 0.0) > UPDATE_RETRY_S):
                update_backoff[want] = now
                if not update_cmd:
                    log.warning("code drift: server on %s, this box booted on %s "
                                "(no --update-cmd; update manually)", want, client_version)
                else:
                    _self_update(want, client_version, update_cmd, run_dir, stop_path, worker_proc)
            # 2. net
            last_version = _pull_weights_if_new(server, weights, last_version, args.timeout)
            # 2b. self-play params (server-published; workers live-apply per game)
            last_selfplay = _pull_selfplay_if_new(server, params_path, last_selfplay, args.timeout, hb_tick)
            # 3. games
            n = _upload_games(server, buffer, args.min_age, args.timeout)
            if n:
                total_up += n
                log.info("uploaded %d game(s) (total %d)", n, total_up)
            # 4. DRAIN an open keep-best gate (lc0-style): keep contributing gate games
            #    until the per-tick budget elapses or the gate closes, so the box helps
            #    carry the whole gate instead of trickling one game per poll. Self-play is
            #    a SEPARATE process and is never paused for this — it's a side task here.
            if not match_disabled:
                if runner is None:
                    try:
                        from chessckers_engine.fleet_match import MatchRunner
                        runner = MatchRunner(run_dir)
                    except Exception as e:  # noqa: BLE001 — torch/native ext absent: self-play only
                        log.info("gate-game contribution unavailable (%s) — self-play only", e)
                        match_disabled = True
                if runner is not None:
                    played, t0 = 0, time.time()
                    cap = args.match_games_per_tick
                    while time.time() - t0 < args.match_burst_seconds and (cap <= 0 or played < cap):
                        try:
                            if runner.step(server, args.timeout) == 0:
                                break  # no match open right now -> back to the normal loop
                        except (urllib.error.URLError, OSError) as e:
                            log.debug("match step failed (retry next tick): %s", e)
                            break
                        except Exception as e:  # noqa: BLE001 — one bad game shouldn't kill the loop
                            log.warning("match step error (retry next tick): %s", e)
                            break
                        played += 1
                    if played:
                        log.info("contributed %d gate game(s) in %.0fs", played, time.time() - t0)
            # 5. own the worker subprocess (lc0 client-owns-engine). Spawn once weights
            #    have landed; restart on unexpected exit (self-heal). STOP is handled
            #    above, so reaching here always means a (re)start is wanted.
            if args.spawn_workers:
                if worker_proc is None:
                    if weights.exists():
                        worker_proc, worker_log = _spawn_workers(
                            worker_argv, run_dir, weights, run_dir / "workers.log")
                elif worker_proc.poll() is not None:
                    log.warning("workers exited (rc=%s) — restarting", worker_proc.returncode)
                    if worker_log is not None:
                        worker_log.close()
                    worker_proc, worker_log = _spawn_workers(
                        worker_argv, run_dir, weights, run_dir / "workers.log")
        except Exception as e:  # noqa: BLE001 — unattended box: never die on an unexpected tick error
            log.warning("tick error (continuing): %s", e)
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
