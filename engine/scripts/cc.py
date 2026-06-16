#!/usr/bin/env python3
"""cc — Chessckers fleet command-center.

Auto-resolves the live vast.ai box (so nothing hardcodes an ssh endpoint, which
changes whenever the instance is recreated) and dispatches the diagnostic
scripts to run on it.

  cc box [--refresh]            # show the resolved box (ssh + server URL + paths)
  cc ssh [cmd...]               # ssh to the box (interactive, or run one command)
  cc run <script.py> [args]     # run engine/scripts/<script.py> ON the box
  cc doctor [args]              # one-shot run health/convergence report
  cc status [args]              # fleet dashboard (live arena + gate decisions; runs fleet_status.py)
  cc plot [args]                # plot the run metrics time-series
  cc ladder [args]              # round-robin champion nets -> terminal Elo + score matrix
  cc gauntlet [args]            # current net vs ALL previous snapshots -> strength + regression curve
  cc games [opts] [watch args]  # pull a RECORDED fleet self-play game + render it
  cc watch [watch args]         # pull the latest fleet net + watch it self-play live
  cc restart-trainer [LR]       # clean warm-restart the trainer (optionally change LR)
  cc play [play args]           # play a human-vs-net game against the latest fleet net
  cc launch                     # print the fresh-run runbook

cc games — render the network's actual self-play games (newest by default):
  cc games                      # newest recorded game, board move-by-move (no net needed)
  cc games --list [K]           # list the K newest chunks with ages (default 15)
  cc games --index N            # a specific training.N.gz
  cc games --eval               # also pull the fleet net + show per-ply WDL
  cc games --step               # any extra args pass through to watch_game.py

Env: CC_INSTANCE=<id> forces a box when several are running.
Run as `python scripts/cc.py <cmd>` or alias `cc` to it (see scripts/README.md).
"""
import json, os, shlex, subprocess, sys, time, urllib.request, urllib.error

CACHE = os.path.expanduser("~/.cache/cc_box.json")
CACHE_TTL = 600
ENGINE_DIR = "/workspace/chessckers/engine"          # repo path ON THE BOX
SERVER_DIR = "/workspace/chessckers/lczero-server"
LOCAL_ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_CACHE = os.path.expanduser("~/.cache/cc_games")  # pulled chunks + fleet net land here


def _instances():
    out = subprocess.run(["vastai", "show", "instances", "--raw"],
                         capture_output=True, text=True, timeout=40).stdout
    s = out.find("[")
    data = json.loads(out[s:]) if s >= 0 else []
    res = []
    for i in data:
        if i.get("actual_status") != "running":
            continue
        p = (i.get("ports") or {}).get("10100/tcp")
        d = (i.get("ports") or {}).get("22/tcp")   # DIRECT ssh; vast's proxy (sshN.vast.ai) is unreliable/dead
        res.append({"id": i["id"], "ssh_host": i.get("ssh_host"), "ssh_port": i.get("ssh_port"),
                    "ip": i.get("public_ipaddr"), "server_port": p[0]["HostPort"] if p else None,
                    "ssh_port_direct": d[0]["HostPort"] if d else i.get("ssh_port"),
                    "gpu": i.get("gpu_name")})
    return res


def _serves(ip, port):
    if not (ip and port):
        return False
    try:
        urllib.request.urlopen(f"http://{ip}:{port}/", timeout=3)
        return True
    except urllib.error.HTTPError:
        return True                       # responded at all => it's the server
    except Exception:
        return False


def resolve(refresh=False):
    if not refresh and os.path.exists(CACHE):
        c = json.load(open(CACHE))
        if time.time() - c.get("_ts", 0) < CACHE_TTL:
            return c
    inst = _instances()
    if not inst:
        sys.exit("cc: no running vast instances")
    force = os.environ.get("CC_INSTANCE")
    box = (next((x for x in inst if str(x["id"]) == str(force)), None) if force
           else inst[0] if len(inst) == 1
           else next((x for x in inst if _serves(x["ip"], x["server_port"])), None))
    if box is None:
        sys.exit(f"cc: {len(inst)} running boxes ({', '.join(str(x['id']) for x in inst)}); "
                 f"set CC_INSTANCE=<id>")
    box = {**box, "server_url": f"http://{box['ip']}:{box['server_port']}" if box["server_port"] else None,
           "_ts": time.time()}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(box, open(CACHE, "w"))
    return box


def _ssh(box):
    # DIRECT endpoint (public ip + container port-22 host mapping): vast's proxy
    # ssh (sshN.vast.ai:NNNNN) is unreliable/dead, so prefer the direct mapping.
    host = box.get("ip") or box["ssh_host"]
    port = box.get("ssh_port_direct") or box["ssh_port"]
    return ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
            "-p", str(port), f"root@{host}"]


def _q(s):
    return "'" + s.replace("'", "'\\''") + "'" if (not s or any(c in s for c in " \"'$&|;<>()")) else s


def _run_on_box(script, rest):
    box = resolve()
    remote = f"cd {ENGINE_DIR} && .venv/bin/python scripts/{script} " + " ".join(_q(a) for a in rest)
    return subprocess.call(_ssh(box) + [remote])


def _ssh_out(box, cmd):
    """Run one command on the box, return its stdout (banner/motd go to stderr -> dropped)."""
    return subprocess.run(_ssh(box) + [cmd], capture_output=True, text=True).stdout


def _fetch(box, remote, local):
    """Copy a remote file down via `cat` over the existing ssh — binary-safe and avoids
    scp/sftp-subsystem quirks on the box (ssh already works, so this always does too).
    Returns 0 on success; cleans up a partial/empty file on failure."""
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "wb") as f:
        rc = subprocess.call(_ssh(box) + [f"cat {shlex.quote(remote)}"], stdout=f,
                             stderr=subprocess.DEVNULL)  # drop the ssh motd banner
    if (rc != 0 or os.path.getsize(local) == 0) and os.path.exists(local):
        os.remove(local)
        return rc or 1
    return 0


def _games_dir(box):
    """Newest games/<run>/ dir on the box (so this never hardcodes run1)."""
    d = _ssh_out(box, "ls -dt /workspace/chessckers/lczero-server/games/*/ 2>/dev/null | head -1").strip()
    return d or f"{SERVER_DIR}/games/run1/"


def _watch_game(extra):
    """Invoke the local watch_game.py renderer with extra args appended."""
    py = os.path.join(LOCAL_ENGINE, ".venv/bin/python")
    wg = os.path.join(LOCAL_ENGINE, "scripts/watch_game.py")
    return subprocess.call([py, wg, *extra])


def _fetch_fleet_net(box):
    """Pull the live fleet net (+ its .arch.json sidecar, needed to rebuild the exact
    V1/V2/V4 arch) into the local cache. Returns the local path, or None on failure."""
    remote = f"{SERVER_DIR}/trainer/run1/weights.pt"
    local = os.path.join(GAMES_CACHE, "fleet_weights.pt")
    print(f"# fetching net {remote}", flush=True)
    if _fetch(box, remote, local) != 0:
        return None
    if _fetch(box, remote + ".arch.json", local + ".arch.json") != 0:
        print("# warning: no .arch.json sidecar — eval may load a wrong (fallback) arch")
    return local


def cmd_games(args):
    """Pull a RECORDED self-play game off the box and render its ACTUAL moves locally."""
    box = resolve()
    gdir = _games_dir(box)
    if "--list" in args:
        i = args.index("--list")
        k = int(args[i + 1]) if i + 1 < len(args) and args[i + 1].isdigit() else 15
        out = _ssh_out(box, f"find {shlex.quote(gdir)} -name 'training.*.gz' "
                            f"-printf '%T@ %f\\n' 2>/dev/null | sort -n | tail -{k}")
        now = time.time()
        print(f"# {gdir}  (newest last)")
        for ln in out.splitlines():
            parts = ln.split(None, 1)
            if len(parts) == 2:
                print(f"  {parts[1]:<22} {(now - float(parts[0]))/60:6.1f}m ago")
        return 0

    eval_on = "--eval" in args
    args = [a for a in args if a != "--eval"]
    idx = None
    if "--index" in args:
        i = args.index("--index")
        idx = args[i + 1]
        args = args[:i] + args[i + 2:]

    if idx is not None:
        remote = f"{gdir.rstrip('/')}/training.{idx}.gz"
    else:
        # find (not an ls glob): a run dir holds tens of thousands of chunks, so
        # `ls training.*.gz` overflows ARG_MAX and silently returns nothing.
        line = _ssh_out(box, f"find {shlex.quote(gdir)} -name 'training.*.gz' "
                             f"-printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1").strip()
        remote = line.split(None, 1)[1] if line else ""
        if not remote:
            sys.exit(f"cc games: no chunks under {gdir}")
    local = os.path.join(GAMES_CACHE, os.path.basename(remote))
    print(f"# fetching {remote}", flush=True)  # flush: precede the subprocess render
    if _fetch(box, remote, local) != 0:
        sys.exit("cc games: fetch failed (is the box up? try `cc box --refresh`)")

    extra = ["--chunk", local]
    net = _fetch_fleet_net(box) if eval_on else None
    extra += ["--weights", net] if net else ["--no-eval"]
    return _watch_game(extra + args)


def cmd_watch(args):
    """Pull the latest fleet net and watch it self-play a fresh game from the start FEN."""
    box = resolve()
    net = _fetch_fleet_net(box)
    if not net:
        sys.exit("cc watch: net fetch failed (is the box up? try `cc box --refresh`)")
    extra = ["--weights", net]
    if "--device" not in args:
        extra += ["--device", "mps"]
    return _watch_game(extra + args)


def cmd_play(args):
    """Pull the latest fleet net and play an interactive human-vs-net game against it."""
    box = resolve()
    net = _fetch_fleet_net(box)
    if not net:
        sys.exit("cc play: net fetch failed (is the box up? try `cc box --refresh`)")
    py = os.path.join(LOCAL_ENGINE, ".venv/bin/python")
    pn = os.path.join(LOCAL_ENGINE, "scripts/play_net.py")
    extra = ["--weights", net]
    if "--device" not in args:
        extra += ["--device", "mps"]
    return subprocess.call([py, pn, *extra, *args])


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 0
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "box":
        b = resolve("--refresh" in args)
        print(f"id={b['id']}  gpu={b['gpu']}")
        print(f"ssh:    ssh -p {b.get('ssh_port_direct') or b['ssh_port']} root@{b.get('ip') or b['ssh_host']}  (direct; proxy sshN.vast.ai is dead)")
        print(f"server: {b['server_url']}")
        print(f"engine (on box): {ENGINE_DIR}")
    elif cmd == "ssh":
        b = resolve()
        os.execvp("ssh", _ssh(b) + ["-t"] + ([" ".join(args)] if args else []))
    elif cmd == "run":
        if not args:
            sys.exit("usage: cc run <script.py> [args...]")
        return _run_on_box(args[0], args[1:])
    elif cmd == "doctor":
        return _run_on_box("run_doctor.py", args)
    elif cmd == "plot":
        return _run_on_box("plot_run.py", args)
    elif cmd == "ladder":
        return _run_on_box("ladder.py", args)
    elif cmd == "gauntlet":
        return _run_on_box("gauntlet.py", args)
    elif cmd == "status":
        # fleet_status.py lives in lczero-server (outside engine) — run it from there.
        box = resolve()
        remote = (f"cd {SERVER_DIR} && {ENGINE_DIR}/.venv/bin/python scripts/fleet_status.py "
                  + " ".join(_q(a) for a in args))
        return subprocess.call(_ssh(box) + [remote])
    elif cmd == "games":
        return cmd_games(args)
    elif cmd == "watch":
        return cmd_watch(args)
    elif cmd == "restart-trainer":
        # Hardened clean warm-restart (snapshot-guarded, env-captured, self-verifying).
        # The heavy lifting is a box-side script so nothing long is ever pasted.
        box = resolve()
        remote = f"bash {SERVER_DIR}/scripts/restart_trainer.sh " + " ".join(_q(a) for a in args)
        return subprocess.call(_ssh(box) + [remote])
    elif cmd == "play":
        return cmd_play(args)
    elif cmd == "launch":
        b = resolve()
        print(f"# Fresh run on box {b['id']} ({b['ssh_host']}:{b['ssh_port']}).")
        print(f"# See scripts/README.md 'Launching a run'. In short, on the box:")
        print(f"#   {SERVER_DIR}/scripts/reset_fleet.sh         # wipe prior run (DESTRUCTIVE)")
        print(f"#   {SERVER_DIR}/scripts/run_server_vast.sh     # server + trainer bridge")
        print(f"#   {ENGINE_DIR}/../lczero-client/scripts/launch_vast_direct.sh   # self-play")
        print(f"# Set the start FEN in akshay-chessckers-0/src/chess/board.cc (kStartposFen).")
    else:
        sys.exit(f"cc: unknown command {cmd!r}\n{__doc__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
