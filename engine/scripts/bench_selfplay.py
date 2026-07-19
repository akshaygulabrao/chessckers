#!/usr/bin/env python3
"""League-selfplay throughput benchmark. Runs ON the box against a quiesced GPU
(operator stops client/trainer/audits first; restore with restart_fleet.sh).

Each config launches the fork engine with the production --training=true flag set
(captured from the live client 2026-07-18) plus one override, time-boxed. The
engine never reaches --games; we kill at the cap and measure steady-state
throughput as the slope of gameready completions in the tail window (ramp
excluded), so configs with different game latencies compare fairly.

Outputs under /workspace/bench-league/:
  results.jsonl   one line per config (games/min, plies, league share, GPU util)
  <name>.log      timestamped engine stdout
  PROGRESS / DONE
"""
import json
import os
import re
import signal
import subprocess
import threading
import time

BIN = "/workspace/chessckers/akshay-chessckers-0/build/release/akshay-chessckers-0"
ROOT = "/workspace/bench-league"
CACHE = "/root/.cache/chessckers/client-cache"
BEST = f"{CACHE}/c9550aa74b0da09a467a407950959707b64f11222de13e21dd19880dd5e0cfb1.bin"
POOL = [
    f"{CACHE}/fb176aee9572a605bbc27e5efb6cc2ad5b94ce188158e0a13e5e1487c31e2b14.bin",
    f"{CACHE}/a95eda74c916b33cf4d002f5d3a155cfe8443b746beea3c3e1c87d98ca0072b7.bin",
    f"{CACHE}/3f434da1f3c21370e8b2d6c522fead1a0120bd46f11610ea5a32ec5b8453b787.bin",
    f"{CACHE}/b1f29e3e5f2f5a619e990280547745cdf46a7deabc57437f86bb654caada7bb2.bin",
    f"{CACHE}/3394e319b18ef47bbac5f3909b8baee2b3670357347d31eb3a82461368c812f4.bin",
    f"{CACHE}/0c93e6d89f4af58e724c84ca2f437ba956809d9d7ed9010d820b1b65b1fd71a9.bin",
    f"{CACHE}/4cfe379052cfcfc1b9e03243514258d6e1b9abb9e619d4cb47ab0392d073483d.bin",
]
LEAGUE = [
    "--league-weights=" + ",".join(POOL),
    "--league-fraction=0.2",
    "--league-probs=0.159,0.119,0.143,0.185,0.112,0.122,0.159",
]


def flags(parallelism=32, visits=800, league=True, extra=()):
    f = [
        "--backend=chessckers",
        f"--parallelism={parallelism}",
        "--noise-epsilon=0.25",
        "--noise-alpha=0.3",
        "--temperature=1.0",
        "--tempdecay-moves=15",
        f"--visits={visits}",
        "--training=true",
        f"--weights={BEST}",
        "--games=2000",
    ]
    if league:
        f += LEAGUE
    f += list(extra)
    return f


# (name, flag overrides, wall cap seconds). Smoke config first: 128v completes
# games in ~1-2 min, validating gameready parsing before the expensive 800v runs.
CONFIGS = [
    ("v128_P32_smoke", dict(visits=128), 420),
    ("base_P32", dict(), 1080),
    ("P16", dict(parallelism=16), 720),
    ("P48", dict(parallelism=48), 1320),
    ("noleague_P32", dict(league=False), 840),
    ("noshare_P32", dict(extra=("--no-share-trees",)), 960),
    ("mb128_P32", dict(extra=("--minibatch-size=128",)), 840),
]

GAMEREADY = re.compile(r"^gameready ")


def gpu_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        util, mem = out.strip().split(",")
        load1 = open("/proc/loadavg").read().split()[0]
        return {"util": float(util), "mem": float(mem), "load1": float(load1)}
    except Exception:
        return None


def slope_games_per_min(tail):
    if len(tail) < 2:
        return None
    if len(tail) < 5:
        dt = tail[-1][0] - tail[0][0]
        return 60.0 * (tail[-1][1] - tail[0][1]) / dt if dt > 0 else None
    ts = [t for t, _ in tail]
    cs = [c for _, c in tail]
    tm = sum(ts) / len(ts)
    cm = sum(cs) / len(cs)
    denom = sum((t - tm) ** 2 for t in ts)
    if not denom:
        return None
    return 60.0 * sum((t - tm) * (c - cm) for t, c in tail) / denom


def run_config(name, kw, cap):
    data_dir = os.path.join(ROOT, "data_" + name)
    os.makedirs(data_dir, exist_ok=True)
    cmd = [BIN, "selfplay"] + flags(**kw)
    log = open(os.path.join(ROOT, name + ".log"), "w")
    log.write("# " + " ".join(cmd) + "\n")
    log.flush()
    events = []  # (t_rel, plies, opponent_idx)
    samples = []
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=data_dir, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)

    def reader():
        for line in proc.stdout:
            t = time.time() - t0
            line = line.rstrip("\n")
            log.write(f"{t:8.1f}  {line}\n")
            log.flush()
            if GAMEREADY.match(line):
                toks = line.split()
                plies = len(toks) - toks.index("moves") - 1 if "moves" in toks else 0
                opp = -1
                if "opponent" in toks:
                    try:
                        opp = int(toks[toks.index("opponent") + 1])
                    except (ValueError, IndexError):
                        pass
                events.append((t, plies, opp))

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()

    while time.time() - t0 < cap and proc.poll() is None:
        time.sleep(10)
        s = gpu_sample()
        if s:
            s["t"] = round(time.time() - t0, 1)
            samples.append(s)

    for sig, wait_s in ((signal.SIGINT, 8), (signal.SIGTERM, 5), (signal.SIGKILL, 3)):
        if proc.poll() is not None:
            break
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        for _ in range(wait_s * 2):
            if proc.poll() is not None:
                break
            time.sleep(0.5)
    rt.join(timeout=5)
    log.close()
    wall = time.time() - t0

    n = len(events)
    tail_from = max(180.0, 0.35 * cap)
    tail = [(t, i + 1) for i, (t, _, _) in enumerate(events) if t >= tail_from]
    plies = [p for _, p, _ in events]
    utils = [s["util"] for s in samples]
    res = {
        "name": name,
        "cap_s": cap,
        "wall_s": round(wall, 1),
        "games_done": n,
        "games_per_min_tail": None,
        "tail_events": len(tail),
        "tail_from_s": tail_from,
        "first_game_s": round(events[0][0], 1) if events else None,
        "mean_plies": round(sum(plies) / n, 1) if n else None,
        "league_games": sum(1 for _, _, o in events if o >= 0),
        "gpu_util_mean": round(sum(utils) / len(utils), 1) if utils else None,
        "gpu_util_max": max(utils) if utils else None,
        "gpu_mem_max_mb": max((s["mem"] for s in samples), default=None),
        "load1_max": max((s["load1"] for s in samples), default=None),
        "flags": " ".join(cmd[2:]),
    }
    g = slope_games_per_min(tail)
    if g is not None:
        res["games_per_min_tail"] = round(g, 3)
    with open(os.path.join(ROOT, "results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def main():
    os.makedirs(ROOT, exist_ok=True)
    prog = os.path.join(ROOT, "PROGRESS")
    for i, (name, kw, cap) in enumerate(CONFIGS):
        with open(prog, "a") as f:
            f.write(f"[{i + 1}/{len(CONFIGS)}] {time.strftime('%H:%M:%S')} "
                    f"running {name} (cap {cap}s)\n")
        try:
            r = run_config(name, kw, cap)
            line = (f"    -> {r['games_done']} games, tail {r['games_per_min_tail']}"
                    f" g/min ({r['tail_events']} tail ev), gpu {r['gpu_util_mean']}%")
        except Exception as e:  # keep sweeping; a broken config is one bad row
            line = f"    -> FAILED {e!r}"
            with open(os.path.join(ROOT, "results.jsonl"), "a") as f:
                f.write(json.dumps({"name": name, "error": repr(e)}) + "\n")
        with open(prog, "a") as f:
            f.write(line + "\n")
        time.sleep(10)
    with open(os.path.join(ROOT, "DONE"), "w") as f:
        f.write(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    main()
