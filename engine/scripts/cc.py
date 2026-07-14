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
  cc ladder [args]              # round-robin champion nets (REAL lc0 fork; --mcts for Python) -> Elo + matrix
  cc champs [args]              # ladder the gate's REAL champions + rejected candidates (server .bin nets)
  cc gauntlet [args]            # current net vs ALL previous snapshots -> strength + regression curve
  cc anchor [args]              # current net vs FIXED anchors (random/search bot/seed13) -> absolute strength trajectory
  cc strength [args]           # strength TABLE from the in-fleet gate matches (fast; no games played)
  cc games [opts] [watch args]  # pull a RECORDED fleet self-play game + render it
  cc verify-chunks [opts]       # scan recent chunks for oracle-illegal moves (fork-vs-PyVariant parity)
  cc watch [watch args]         # pull the latest fleet net + watch it self-play live
  cc restart-trainer [LR]       # clean warm-restart the trainer (optionally change LR)
  cc restart                    # relaunch the whole fleet (warm-resume) if down — idempotent
  cc play [play args]           # play a human-vs-net game against the latest fleet net
  cc lengths [--window=50]      # average game length over training (survival→mate curve)
  cc backup                     # pull irreplaceable telemetry off the box to telemetry/<run>/
  cc compare [file...]          # compare anchor_gauntlet.jsonl runs — sparklines + alignment table
  cc fresh-run [--run-name=X] [--arch=v5] [--parallelism=32] [--base=<box-net.pt>]
               [--c-filters=N] [--n-blocks=N] [--se-ratio=N]
               [--policy-target=visits|improved] [--value-q-ratio=R]
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
import math
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
FORK_BINARY = "/workspace/chessckers/akshay-chessckers-0/build/release/akshay-chessckers-0"  # lc0 fork on the box
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
    # Arch dims: empty = defer to launch_trainer.sh's own defaults (c48/b5). Set these to
    # match the --base net when warm-starting a scaled arch (e.g. c64/b6), else the trainer
    # builds the default-size model and the base load fails on shape mismatch.
    c_filters = ""
    n_blocks = ""
    se_ratio = ""
    # Training-target knobs: empty = defer to launch_trainer.sh defaults (visits / 0.5).
    policy_target = ""
    value_q_ratio = ""
    for a in args:
        if a.startswith("--run-name="):
            run_name = a.split("=", 1)[1]
        elif a.startswith("--arch="):
            arch = a.split("=", 1)[1]
        elif a.startswith("--parallelism="):
            parallelism = a.split("=", 1)[1]
        elif a.startswith("--base="):
            base = a.split("=", 1)[1]
        elif a.startswith("--c-filters="):
            c_filters = a.split("=", 1)[1]
        elif a.startswith("--n-blocks="):
            n_blocks = a.split("=", 1)[1]
        elif a.startswith("--se-ratio="):
            se_ratio = a.split("=", 1)[1]
        elif a.startswith("--policy-target="):
            policy_target = a.split("=", 1)[1]
        elif a.startswith("--value-q-ratio="):
            value_q_ratio = a.split("=", 1)[1]

    dims = f" c{c_filters}/b{n_blocks}" if (c_filters or n_blocks) else ""
    tgt = f"  target={policy_target}" if policy_target else ""
    qr = f"  qratio={value_q_ratio}" if value_q_ratio else ""
    print(
        f"=== fresh-run: box={host}:{port}  run={run_name}  arch={arch}{dims}{tgt}{qr}  p={parallelism}"
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
    # Trainer knob env (only for the ones set): arch dims so a scaled net (c64/b6) is
    # built to match --base, plus the policy-target / value-q-ratio training knobs.
    knob_env = "".join(
        f"{k}={v} " for k, v in (
            ("C_FILTERS", c_filters), ("N_BLOCKS", n_blocks), ("SE_RATIO", se_ratio),
            ("POLICY_TARGET", policy_target), ("VALUE_Q_RATIO", value_q_ratio),
        ) if v
    )
    sh_ok(
        f"tmux kill-session -t cc 2>/dev/null; sleep 1; "
        f"cd {SERVER_DIR} && tmux new-session -d -s cc -n server -c {SERVER_DIR} && "
        f"tmux send-keys -t cc:server "
        f"'cd {SERVER_DIR} && PATH=/usr/local/go/bin:$PATH RUN_NAME={run_name} scripts/launch_server.sh 2>&1 | tee -a server.log' C-m && "
        f"tmux new-window -t cc -n trainer -c {SERVER_DIR} && sleep 0.5 && "
        f"tmux send-keys -t cc:trainer "
        f"'cd {SERVER_DIR} && sleep 6 && {base_env}{knob_env}ENGINE_DIR={ENGINE_DIR} SERVER=http://localhost:10100 ARCH_VERSION={arch} scripts/launch_trainer.sh 2>&1 | tee -a trainer.log' C-m"
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
        f"@reboot RUN_NAME={run_name} ARCH_VERSION={arch} {knob_env}PARALLELISM={parallelism} "
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
    ignored, so `cc restart-trainer 0.001` used to be a silent no-op.

    The run's knob env (ARCH_VERSION, C_FILTERS/N_BLOCKS/SE_RATIO, POLICY_TARGET,
    VALUE_Q_RATIO) is derived from the @reboot cron line fresh-run installed — the
    persisted source of truth for the live run — so a bare restart can't revert the
    trainer to launch_trainer.sh defaults (a mismatched net shape SIGTRAPs the
    warm-resume; a default policy target silently flips the Gumbel A/B arm)."""
    box = resolve()
    _ssh_out(box, f"rm -f {SERVER_DIR}/trainer/run1/STOP")
    cron = _ssh_out(box, "crontab -l 2>/dev/null | grep restart_fleet.sh | tail -1")
    keep = ("ARCH_VERSION", "C_FILTERS", "N_BLOCKS", "SE_RATIO",
            "POLICY_TARGET", "VALUE_Q_RATIO")
    knob_env = " ".join(
        t for t in cron.split() if "=" in t and t.split("=", 1)[0] in keep)
    if "ARCH_VERSION=" not in knob_env:
        knob_env = f"ARCH_VERSION=v4 {knob_env}".strip()  # pre-cron boxes: old behavior
    lr_env = f"LR={args[0]} " if args else ""
    remote = (
        f"tmux send-keys -t cc:trainer C-c && sleep 0.3 && "
        f"tmux send-keys -t cc:trainer "
        f"'{lr_env}{knob_env} ENGINE_DIR={ENGINE_DIR} SERVER=http://localhost:10100 "
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


def cmd_verify_chunks(args):
    """Scan recent self-play chunks for oracle-illegal moves (fork vs PyVariant parity).

    Pulls the newest N chunks (or a specific --index) off the box and replays each
    through PyVariant: any ply whose recorded transition no PyVariant-legal move can
    reproduce is a fork rules divergence (the class of bug behind training.218.gz —
    a quiet diagonal sliding through a White piece). Exit 1 if any are found.

      cc verify-chunks                 # newest 200 chunks
      cc verify-chunks --count 1000    # newest 1000
      cc verify-chunks --index 218     # one specific chunk
    """
    box = resolve()
    gdir = _games_dir(box)
    if "--index" in args:
        i = args.index("--index")
        remotes = [f"{gdir.rstrip('/')}/training.{args[i + 1]}.gz"]
    else:
        count = 200
        if "--count" in args:
            count = int(args[args.index("--count") + 1])
        out = _ssh_out(
            box,
            f"find {shlex.quote(gdir)} -name 'training.*.gz' "
            f"-printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -{count}",
        )
        remotes = [ln.split(None, 1)[1] for ln in out.splitlines() if len(ln.split(None, 1)) == 2]
        if not remotes:
            sys.exit(f"cc verify-chunks: no chunks under {gdir}")

    locals_ = []
    for remote in remotes:
        # Always re-fetch (overwrite): chunk numbers reset per run, so a same-named
        # file cached from a PRIOR run is stale and would scan the wrong game.
        local = os.path.join(GAMES_CACHE, os.path.basename(remote))
        if _fetch(box, remote, local) != 0:
            print(f"# skip (fetch failed): {remote}")
            continue
        locals_.append(local)
    print(f"# scanning {len(locals_)} chunk(s) for oracle-illegal moves ...", flush=True)
    checker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_chunk_parity.py")
    return subprocess.call([sys.executable, checker, *locals_])


_TELEMETRY_ROOT = os.path.join(os.path.dirname(LOCAL_ENGINE), "telemetry")
_BACKUP_MARKER = os.path.join(_TELEMETRY_ROOT, ".last-backup")
_BACKUP_STALE_SECS = 6 * 3600


def _backup_stale():
    """True if no backup has run in the last 6 hours (or marker absent)."""
    try:
        return time.time() - os.path.getmtime(_BACKUP_MARKER) > _BACKUP_STALE_SECS
    except OSError:
        return True


def _spawn_background_backup():
    """Fire-and-forget backup subprocess — never fatal."""
    try:
        subprocess.Popen(
            [sys.executable, __file__, "backup"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("(telemetry backup started in background)")
    except Exception:
        pass


def _run_label(box):
    """Best-effort run label for the telemetry dir name (e.g. 'run19' or box id)."""
    try:
        cron = _ssh_out(box, "crontab -l 2>/dev/null | grep restart_fleet.sh | tail -1")
        for tok in cron.split():
            if tok.startswith("RUN_NAME="):
                name = tok.split("=", 1)[1]
                # strip leading runNN_ prefix if present, else use as-is
                return name
    except Exception:
        pass
    return f"box{box.get('id', 'unknown')}"


def cmd_backup(args):
    """Pull irreplaceable telemetry off the box to telemetry/<run>/ under the repo root.

    Files pulled (overwritten — the jsonls/db are append-only so latest wins):
      anchor_gauntlet.jsonl, champs_audit.jsonl (skipped if absent), chessckers.db,
      ALERTS.log, server.log (last 2000 lines), trainer.log (last 2000 lines).
    """
    box = resolve()
    label = _run_label(box)
    dest = os.path.join(_TELEMETRY_ROOT, label)
    os.makedirs(dest, exist_ok=True)

    run1 = f"{SERVER_DIR}/trainer/run1"

    def _pull_file(remote, local_name):
        local = os.path.join(dest, local_name)
        rc = _fetch(box, remote, local)
        if rc != 0 and not os.path.exists(local):
            print(f"  {local_name:<32} absent")
        else:
            size = os.path.getsize(local)
            print(f"  {local_name:<32} {size:>10,} bytes")

    def _pull_tail(remote, local_name, lines=2000):
        """Pull the last N lines of a remote log via ssh_out, write locally."""
        local = os.path.join(dest, local_name)
        text = _ssh_out(box, f"tail -n {lines} {shlex.quote(remote)} 2>/dev/null")
        if text:
            with open(local, "w") as f:
                f.write(text)
            print(f"  {local_name:<32} {os.path.getsize(local):>10,} bytes  (last {lines} lines)")
        else:
            print(f"  {local_name:<32} absent")

    print(f"backup → {dest}/")
    _pull_file(f"{run1}/anchor_gauntlet.jsonl", "anchor_gauntlet.jsonl")
    # champs_audit.jsonl may not exist yet — _fetch silently skips on failure
    champs_remote = f"{run1}/champs_audit.jsonl"
    champs_local = os.path.join(dest, "champs_audit.jsonl")
    rc = _fetch(box, champs_remote, champs_local)
    if rc != 0 and not os.path.exists(champs_local):
        print(f"  {'champs_audit.jsonl':<32} absent (not yet written — ok)")
    else:
        print(f"  {'champs_audit.jsonl':<32} {os.path.getsize(champs_local):>10,} bytes")
    _pull_file(f"{SERVER_DIR}/chessckers.db", "chessckers.db")
    _pull_file("/workspace/chessckers/ALERTS.log", "ALERTS.log")
    _pull_tail(f"{SERVER_DIR}/server.log", "server.log")
    _pull_tail(f"{SERVER_DIR}/trainer.log", "trainer.log")

    # update marker
    os.makedirs(_TELEMETRY_ROOT, exist_ok=True)
    with open(_BACKUP_MARKER, "w") as f:
        f.write(str(time.time()))
    return 0


def _parse_jsonl(path):
    """Parse a JSONL file, skipping blank/malformed lines. Returns list of dicts."""
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return rows


def _sparkline(values):
    """Unicode block sparkline for a sequence of floats (▁–▇)."""
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo
    return "".join(
        blocks[min(7, int((v - lo) / (span + 1e-9) * 7))]
        for v in values
    )


def _elo_slope(points, interval_h=8):
    """Elo/24h slope over the last 3 rows. points = [(ts_or_None, elo)]; uses real
    timestamps when present (the cron cadence varies), else assumes interval_h."""
    tail = points[-3:]  # up to 3 points
    if len(tail) < 2:
        return None
    rise = tail[-1][1] - tail[0][1]
    ts0, ts1 = tail[0][0], tail[-1][0]
    if ts0 and ts1 and ts1 > ts0:
        hours = (ts1 - ts0) / 3600.0
    else:
        hours = (len(tail) - 1) * interval_h
    return rise / hours * 24  # Elo per 24h


def cmd_compare(args):
    """Compare anchor_gauntlet.jsonl runs — sparklines + seed13 alignment table.

    Usage:  cc compare [file.jsonl ...]
      Default: every telemetry/*/anchor_gauntlet.jsonl (refresh with `cc backup`).
      Each file is one run.  For each anchor, prints:
        • a sparkline of Elo over rows, first/last Elo, slope Elo/24h (last 3 rows)
      Then: a per-row alignment table for anchor "seed13" so two runs can be
      compared at matched progress (rows share an index, not a timestamp).
    """
    import glob as _glob

    # Collect files
    if args:
        files = args
    else:
        files = sorted(_glob.glob(os.path.join(_TELEMETRY_ROOT, "*/anchor_gauntlet.jsonl")))
    if not files:
        print("cc compare: no files found. Run `cc backup` first, or pass paths explicitly.")
        return 1

    # Load and parse each run
    runs = []  # list of (label, rows)
    for path in files:
        rows = _parse_jsonl(path)
        # derive a short label from the path (parent dir name)
        parts = path.replace("\\", "/").split("/")
        try:
            label = parts[parts.index("telemetry") + 1]
        except (ValueError, IndexError):
            label = os.path.basename(os.path.dirname(path))
        runs.append((label, rows))

    if not runs:
        print("cc compare: no data loaded.")
        return 1

    # ------------------------------------------------------------------ per-run blocks
    for label, rows in runs:
        print(f"\n=== {label} ({len(rows)} rows) ===")
        if not rows:
            print("  (no rows)")
            continue

        # Collect per-anchor series (jsonl rows carry the per-anchor list under "anchors")
        anchors = {}
        for row in rows:
            for result in row.get("anchors", []):
                anchor = result.get("anchor", "")
                elo = result.get("elo")
                if anchor and elo is not None:
                    anchors.setdefault(anchor, []).append(
                        (row.get("best_net"), elo, row.get("ts")))

        for anchor, entries in sorted(anchors.items()):
            elos = [e for _, e, _ in entries]
            nets = [n for n, _, _ in entries]
            spark = _sparkline(elos)
            first_e, last_e = elos[0], elos[-1]
            slope = _elo_slope([(t, e) for _, e, t in entries])
            slope_str = f"{slope:+.0f} Elo/24h" if slope is not None else "n/a"
            net_range = ""
            non_null = [n for n in nets if n is not None]
            if non_null:
                net_range = f"  nets {non_null[0]}..{non_null[-1]}"
            print(f"  {anchor:<12} {spark}  [{first_e:+.0f} → {last_e:+.0f}]  slope {slope_str}{net_range}")

    # ------------------------------------------------------------------ seed13 alignment table
    print("\n--- seed13 alignment table (row index vs run) ---")
    # Gather per-run seed13 series
    seed13_series = {}
    for label, rows in runs:
        elos = []
        for row in rows:
            for result in row.get("anchors", []):
                if result.get("anchor") == "seed13":
                    elo = result.get("elo")
                    if elo is not None:
                        elos.append((row.get("best_net"), elo))
        seed13_series[label] = elos

    max_rows = max((len(v) for v in seed13_series.values()), default=0)
    if max_rows == 0:
        print("  (no seed13 data)")
        return 0

    run_labels = [label for label, _ in runs]
    col_w = max(12, max(len(l) for l in run_labels))
    header = f"  {'row':>4}  " + "  ".join(f"{l:>{col_w}}" for l in run_labels)
    print(header)
    for i in range(max_rows):
        cells = []
        for label in run_labels:
            series = seed13_series[label]
            if i < len(series):
                net, elo = series[i]
                net_str = f"/{net}" if net else ""
                cells.append(f"{elo:>+.0f}{net_str}"[:col_w].rjust(col_w))
            else:
                cells.append("-".rjust(col_w))
        print(f"  {i:>4}  " + "  ".join(cells))

    return 0


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
        rc = _run_on_box("run_doctor.py", args)
        if _backup_stale():
            _spawn_background_backup()
        return rc
    elif cmd == "plot":
        return _run_on_box("plot_run.py", args)
    elif cmd == "ladder":
        # Default to the REAL lc0 fork (always built on the box); `cc ladder --mcts`
        # forces the Python PUCT reference. Inject an explicit binary path (not a
        # bare --engine) so ladder.py's nargs='?' can't swallow a following net path.
        if "--mcts" in args:
            args = [a for a in args if a != "--mcts"]
        elif "--engine" not in args:
            args = ["--engine", FORK_BINARY, *args]
        return _run_on_box("ladder.py", args)
    elif cmd == "champs":
        # champ_ladder always runs the fork (server nets are .bin-only) and
        # defaults the binary itself — no --engine injection needed here.
        return _run_on_box("champ_ladder.py", args)
    elif cmd == "gauntlet":
        return _run_on_box("gauntlet.py", args)
    elif cmd == "anchor":
        return _run_on_box("anchor_gauntlet.py", args)
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
        rc = subprocess.call(_ssh(box) + [remote])
        if _backup_stale():
            _spawn_background_backup()
        return rc
    elif cmd == "games":
        return cmd_games(args)
    elif cmd == "verify-chunks":
        return cmd_verify_chunks(args)
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
    elif cmd == "backup":
        return cmd_backup(args)
    elif cmd == "compare":
        return cmd_compare(args)
    elif cmd == "fresh-run":
        return cmd_fresh_run(args)
    else:
        sys.exit(f"cc: unknown command {cmd!r}\n{__doc__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
