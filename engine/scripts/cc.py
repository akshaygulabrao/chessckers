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
  cc strength [args]           # strength TABLE from the in-fleet gate matches (fast; no games played)
  cc games [opts] [watch args]  # pull a RECORDED fleet self-play game + render it
  cc watch [watch args]         # pull the latest fleet net + watch it self-play live
  cc restart-trainer [LR]       # clean warm-restart the trainer (optionally change LR)
  cc restart                    # relaunch the whole fleet (warm-resume) if down — idempotent
  cc play [play args]           # play a human-vs-net game against the latest fleet net
  cc lengths [--window=50]      # average game length over training (survival→mate curve)
  cc fresh-run [--run-name=X] [--arch=v5] [--parallelism=32] [--base=<box-net.pt>]
                              # provision + launch a fresh training run from scratch

cc games — render the network's actual self-play games (newest by default):
  cc games                      # newest recorded game, board move-by-move (no net needed)
  cc games --list [K]           # list the K newest chunks with ages (default 15)
  cc games --index N            # a specific training.N.gz
  cc games --eval               # also pull the fleet net + show per-ply WDL
  cc games --step               # any extra args pass through to watch_game.py

Env: CC_INSTANCE=<id> forces a box when several are running.
Run as `python scripts/cc.py <cmd>` or alias `cc` to it (see scripts/README.md).
"""

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
import urllib.error

CACHE = os.path.expanduser("~/.cache/cc_box.json")
CACHE_TTL = 600
ENGINE_DIR = "/workspace/chessckers/engine"  # repo path ON THE BOX
SERVER_DIR = "/workspace/chessckers/lczero-server"
LOCAL_ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_CACHE = os.path.expanduser(
    "~/.cache/cc_games"
)  # pulled chunks + fleet net land here


def _instances():
    out = subprocess.run(
        ["vastai", "show", "instances", "--raw"],
        capture_output=True,
        text=True,
        timeout=40,
    ).stdout
    s = out.find("[")
    data = json.loads(out[s:]) if s >= 0 else []
    res = []
    for i in data:
        if i.get("actual_status") != "running":
            continue
        p = (i.get("ports") or {}).get("10100/tcp")
        d = (i.get("ports") or {}).get(
            "22/tcp"
        )  # DIRECT ssh; vast's proxy (sshN.vast.ai) is unreliable/dead
        res.append(
            {
                "id": i["id"],
                "ssh_host": i.get("ssh_host"),
                "ssh_port": i.get("ssh_port"),
                "ip": i.get("public_ipaddr"),
                "server_port": p[0]["HostPort"] if p else None,
                "ssh_port_direct": d[0]["HostPort"] if d else i.get("ssh_port"),
                "gpu": i.get("gpu_name"),
            }
        )
    return res


def _serves(ip, port):
    if not (ip and port):
        return False
    try:
        urllib.request.urlopen(f"http://{ip}:{port}/", timeout=3)
        return True
    except urllib.error.HTTPError:
        return True  # responded at all => it's the server
    except Exception:
        return False


def resolve(refresh=False):
    if not refresh and os.path.exists(CACHE):
        with open(CACHE) as f:
            c = json.load(f)
        if time.time() - c.get("_ts", 0) < CACHE_TTL:
            return c
    inst = _instances()
    if not inst:
        sys.exit("cc: no running vast instances")
    force = os.environ.get("CC_INSTANCE")
    if force:
        box = next((x for x in inst if str(x["id"]) == str(force)), None)
    elif len(inst) == 1:
        box = inst[0]
    else:
        box = next((x for x in inst if _serves(x["ip"], x["server_port"])), None)
    if box is None:
        sys.exit(
            f"cc: {len(inst)} running boxes ({', '.join(str(x['id']) for x in inst)}); "
            f"set CC_INSTANCE=<id>"
        )
    box = {
        **box,
        "server_url": f"http://{box['ip']}:{box['server_port']}"
        if box["server_port"]
        else None,
        "_ts": time.time(),
    }
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(box, f)
    return box


def _ssh(box):
    # DIRECT endpoint (public ip + container port-22 host mapping): vast's proxy
    # ssh (sshN.vast.ai:NNNNN) is unreliable/dead, so prefer the direct mapping.
    host = box.get("ip") or box["ssh_host"]
    port = box.get("ssh_port_direct") or box["ssh_port"]
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=20",
        "-p",
        str(port),
        f"root@{host}",
    ]


def _q(s):
    return (
        "'" + s.replace("'", "'\\''") + "'"
        if (not s or any(c in s for c in " \"'$&|;<>()"))
        else s
    )


def _run_on_box(script, rest):
    box = resolve()
    remote = f"cd {ENGINE_DIR} && .venv/bin/python scripts/{script} " + " ".join(
        _q(a) for a in rest
    )
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
        rc = subprocess.call(
            _ssh(box) + [f"cat {shlex.quote(remote)}"],
            stdout=f,
            stderr=subprocess.DEVNULL,
        )  # drop the ssh motd banner
    if (rc != 0 or os.path.getsize(local) == 0) and os.path.exists(local):
        os.remove(local)
        return rc or 1
    return 0


def _games_dir(box):
    """Newest games/<run>/ dir on the box (so this never hardcodes run1)."""
    d = _ssh_out(
        box, "ls -dt /workspace/chessckers/lczero-server/games/*/ 2>/dev/null | head -1"
    ).strip()
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
        print(
            "# warning: no .arch.json sidecar — eval may load a wrong (fallback) arch"
        )
    return local


def cmd_fresh_run(args):
    """Provision + launch a fresh training run on the resolved vast.ai box.

    One command that does everything between "empty box" and "fleet running":
      1. provision the box (toolchain, repos, build server + engine)
      2. rsync + build the akshay-chessckers-0 fork with the right flags
      3. rsync + build the lczero-client
      4. reset fleet state (wipe old DB/nets/games)
      5. launch server + trainer in tmux 'cc'
      6. launch self-play client in tmux 'cc-client'

    Flags override defaults:
      cc fresh-run --run-name V5_myexp --arch v5 --parallelism 32
    """
    box = resolve()
    host = box.get("ip") or box["ssh_host"]
    port = box.get("ssh_port_direct") or box["ssh_port"]
    ssh = _ssh(box)

    def sh(cmd):
        print(f"  $ {cmd}")
        return subprocess.call(ssh + [cmd])

    def sh_ok(cmd):
        r = sh(cmd)
        if r != 0:
            sys.exit(f"Command failed (exit {r}): {cmd}")
        return r

    run_name = "V5_e8d8"
    arch = "v5"
    parallelism = "32"
    base = ""  # warm-start: a net path ON THE BOX (must survive reset_fleet, e.g.
    #            /workspace/run8_seed/weights.pt). Empty = cold random init.
    for a in args:
        if a.startswith("--run-name="):
            run_name = a.split("=", 1)[1]
        elif a.startswith("--arch="):
            arch = a.split("=", 1)[1]
        elif a.startswith("--parallelism="):
            parallelism = a.split("=", 1)[1]
        elif a.startswith("--base="):
            base = a.split("=", 1)[1]

    print(
        f"=== fresh-run: box={host}:{port}  run={run_name}  arch={arch}  p={parallelism}"
        f"  init={'warm:' + base if base else 'cold'} ==="
    )

    # 1. Provision (server + engine, no state seed).
    print("\n--- 1/6: provisioning box (toolchain, server, engine) ---")
    prov = os.path.join(
        LOCAL_ENGINE, "..", "..", "lczero-server", "scripts", "provision_server_vast.sh"
    )
    subprocess.run(
        ["bash", prov],
        env={
            **os.environ,
            "VAST_HOST": host,
            "VAST_PORT": str(port),
            "SEED_STATE": "false",
        },
        check=True,
    )

    # 2. Rsync + build the fork.
    print("\n--- 2/6: building akshay-chessckers-0 ---")
    fork_local = os.path.join(LOCAL_ENGINE, "..", "..", "akshay-chessckers-0")
    sh_ok(
        "pip3 install meson ninja 2>/dev/null; apt-get install -y libopenblas-dev 2>/dev/null"
    )
    rsync_cmd = [
        "rsync",
        "-az",
        "-e",
        f"ssh -p {port} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20",
        f"{fork_local}/",
        f"root@{host}:/workspace/chessckers/akshay-chessckers-0/",
        "--exclude=build/",
        "--exclude=.git/",
    ]
    subprocess.run(rsync_cmd, check=True)
    sh_ok(
        "export PATH=$HOME/.local/bin:$PATH && cd /workspace/chessckers/akshay-chessckers-0 && "
        "rm -rf build/release && meson setup build/release --buildtype release "
        "-Dblas=false -Dplain_cuda=false -Donnx=false -Dbuild_backends=false && "
        "ninja -C build/release akshay-chessckers-0"
    )

    # 3. Rsync + build the client.
    print("\n--- 3/6: building lczero-client ---")
    client_local = os.path.join(LOCAL_ENGINE, "..", "..", "lczero-client")
    rsync_cmd2 = [
        "rsync",
        "-az",
        "-e",
        f"ssh -p {port} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20",
        f"{client_local}/",
        f"root@{host}:/workspace/chessckers/lczero-client/",
        "--exclude=.git/",
    ]
    subprocess.run(rsync_cmd2, check=True)
    sh_ok(
        "export PATH=/usr/local/go/bin:$PATH && cd /workspace/chessckers/lczero-client && go build -o lc0-client ."
    )
    sh_ok(
        "mkdir -p /workspace/chessckers/lczero-client/.enginebin && "
        "ln -sfT /workspace/chessckers/akshay-chessckers-0/build/release/akshay-chessckers-0 "
        "/workspace/chessckers/lczero-client/.enginebin/akshay-chessckers-0"
    )

    # 4. Reset fleet (DESTRUCTIVE).
    print("\n--- 4/6: resetting fleet (wipe old state) ---")
    sh_ok("cd /workspace/chessckers/lczero-server && bash scripts/reset_fleet.sh")

    # 5. Launch server + trainer in tmux 'cc'.
    print("\n--- 5/6: launching server + trainer ---")
    # Interpolate absolute box paths (SERVER_DIR/ENGINE_DIR) directly into the
    # send-keys command. Do NOT use shell $SRV/$ENG here: the pane runs its own
    # shell where those are undefined, and single-quoting them (the old bug) types
    # a literal `cd '$SRV'` that fails. $PATH stays single-quoted so the PANE
    # expands it at runtime.
    # Warm-start: pass BASE so launch_trainer.sh feeds the trainer a seed net
    # (--base) instead of cold random init. Empty = cold (the default).
    base_env = f"BASE={base} " if base else ""
    sh_ok(
        f"tmux kill-session -t cc 2>/dev/null; sleep 1; "
        f"cd {SERVER_DIR} && tmux new-session -d -s cc -n server -c {SERVER_DIR} && "
        f"tmux send-keys -t cc:server "
        f"'cd {SERVER_DIR} && PATH=/usr/local/go/bin:$PATH RUN_NAME={run_name} scripts/launch_server.sh 2>&1 | tee -a server.log' C-m && "
        f"tmux new-window -t cc -n trainer -c {SERVER_DIR} && sleep 0.5 && "
        f"tmux send-keys -t cc:trainer "
        f"'cd {SERVER_DIR} && sleep 6 && {base_env}ENGINE_DIR={ENGINE_DIR} SERVER=http://localhost:10100 ARCH_VERSION={arch} scripts/launch_trainer.sh 2>&1 | tee -a trainer.log' C-m"
    )

    # 6. Launch client in tmux 'cc-client'.
    print("\n--- 6/6: launching self-play client ---")
    cl = "/workspace/chessckers/lczero-client"
    sh_ok(
        f"tmux kill-session -t cc-client 2>/dev/null; sleep 1; "
        f"cd {cl} && tmux new-session -d -s cc-client -n selfplay -c {cl} && sleep 0.5 && "
        f"tmux send-keys -t cc-client "
        f"'export PATH={cl}/.enginebin:$PATH; cd {cl}; ./lc0-client -hostname http://localhost:10100 -user vast -password chessckers -run 1 -parallelism {parallelism} 2>&1 | tee -a client.log' C-m"
    )

    # Install/refresh the @reboot auto-restart cron so a vast.ai reboot (which
    # kills tmux but keeps disk state) self-heals instead of silently dying.
    print("\n--- installing @reboot auto-restart cron ---")
    cron_line = (
        f"@reboot RUN_NAME={run_name} ARCH_VERSION={arch} PARALLELISM={parallelism} "
        f"{SERVER_DIR}/scripts/restart_fleet.sh --boot >> /workspace/restart_fleet.log 2>&1")
    sh_ok(
        f"chmod +x {SERVER_DIR}/scripts/restart_fleet.sh; "
        f"( crontab -l 2>/dev/null | grep -v restart_fleet.sh; echo '{cron_line}' ) | crontab -; "
        f"echo '[cron] @reboot auto-restart installed'")

    print("\n=== fresh-run complete ===")
    print(f"  ssh -p {port} root@{host} -t tmux attach -t cc     # server + trainer")
    print(f"  ssh -p {port} root@{host} -t tmux attach -t cc-client  # self-play")
    print("  cc status                                           # fleet dashboard")
    return 0


def cmd_games(args):
    """Pull a RECORDED self-play game off the box and render its ACTUAL moves locally."""
    box = resolve()
    gdir = _games_dir(box)
    if "--list" in args:
        i = args.index("--list")
        k = int(args[i + 1]) if i + 1 < len(args) and args[i + 1].isdigit() else 15
        out = _ssh_out(
            box,
            f"find {shlex.quote(gdir)} -name 'training.*.gz' "
            f"-printf '%T@ %f\\n' 2>/dev/null | sort -n | tail -{k}",
        )
        now = time.time()
        print(f"# {gdir}  (newest last)")
        for ln in out.splitlines():
            parts = ln.split(None, 1)
            if len(parts) == 2:
                print(f"  {parts[1]:<22} {(now - float(parts[0])) / 60:6.1f}m ago")
        return 0

    eval_on = "--eval" in args
    args = [a for a in args if a != "--eval"]
    idx = None
    if "--index" in args:
        i = args.index("--index")
        idx = args[i + 1]
        args = args[:i] + args[i + 2 :]

    if idx is not None:
        remote = f"{gdir.rstrip('/')}/training.{idx}.gz"
    else:
        # find (not an ls glob): a run dir holds tens of thousands of chunks, so
        # `ls training.*.gz` overflows ARG_MAX and silently returns nothing.
        line = _ssh_out(
            box,
            f"find {shlex.quote(gdir)} -name 'training.*.gz' "
            f"-printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1",
        ).strip()
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


def cmd_restart_trainer(args):
    """Warm-restart the trainer bridge in the existing tmux cc:trainer window.

    Optional positional arg = the Adam LR. It is exported as the LR *env var*
    because launch_trainer.sh reads ${LR}; positional args to that script are
    ignored, so `cc restart-trainer 0.001` used to be a silent no-op."""
    box = resolve()
    _ssh_out(box, f"rm -f {SERVER_DIR}/trainer/run1/STOP")
    lr_env = f"LR={args[0]} " if args else ""
    remote = (
        f"tmux send-keys -t cc:trainer C-c && sleep 0.3 && "
        f"tmux send-keys -t cc:trainer "
        f"'{lr_env}ENGINE_DIR={ENGINE_DIR} SERVER=http://localhost:10100 ARCH_VERSION=v4 "
        f"bash {SERVER_DIR}/scripts/launch_trainer.sh 2>&1 | tee -a {SERVER_DIR}/trainer.log' C-m"
    )
    return subprocess.call(_ssh(box) + [remote])


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
        print(__doc__)
        return 0
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "box":
        b = resolve("--refresh" in args)
        print(f"id={b['id']}  gpu={b['gpu']}")
        print(
            f"ssh:    ssh -p {b.get('ssh_port_direct') or b['ssh_port']} root@{b.get('ip') or b['ssh_host']}  (direct; proxy sshN.vast.ai is dead)"
        )
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
    elif cmd == "strength":
        # Read the in-fleet GATE's match results (fast read-only DB query, NO games
        # played). The gate already plays each net vs best with the fast C++ engine.
        # For a slow deep OFFLINE check (fleet paused), use `cc gauntlet` instead.
        return _run_on_box("strength.py", args)
    elif cmd == "status":
        # fleet_status.py lives in lczero-server (outside engine) — run it from there.
        box = resolve()
        remote = (
            f"cd {SERVER_DIR} && {ENGINE_DIR}/.venv/bin/python scripts/fleet_status.py "
            + " ".join(_q(a) for a in args)
        )
        return subprocess.call(_ssh(box) + [remote])
    elif cmd == "games":
        return cmd_games(args)
    elif cmd == "watch":
        return cmd_watch(args)
    elif cmd == "restart-trainer":
        return cmd_restart_trainer(args)
    elif cmd == "restart":
        # Relaunch the whole fleet (server + trainer warm-resume + client) if it's
        # down — idempotent, the same script the @reboot cron runs. No rebuild/wipe.
        box = resolve()
        return subprocess.call(_ssh(box) + [f"bash {SERVER_DIR}/scripts/restart_fleet.sh"])
    elif cmd == "play":
        return cmd_play(args)
    elif cmd == "launch":
        b = resolve()
        print(f"# Fresh run on box {b['id']} ({b['ssh_host']}:{b['ssh_port']}).")
        print("# See scripts/README.md 'Launching a run'. In short, on the box:")
        print(
            f"#   {SERVER_DIR}/scripts/reset_fleet.sh         # wipe prior run (DESTRUCTIVE)"
        )
        print(
            f"#   {SERVER_DIR}/scripts/run_server_vast.sh     # server + trainer bridge"
        )
        print(
            "# Set the start FEN in akshay-chessckers-0/src/chess/board.cc (kStartposFen)."
        )
        print("# (Or just use `cc fresh-run`, which does all of the above in one command.)")
    elif cmd == "lengths":
        return _run_on_box("game_lengths.py", args)
    elif cmd == "fresh-run":
        return cmd_fresh_run(args)
    else:
        sys.exit(f"cc: unknown command {cmd!r}\n{__doc__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
