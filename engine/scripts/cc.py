#!/usr/bin/env python3
"""cc — Chessckers fleet command-center.

Auto-resolves the live vast.ai box (so nothing hardcodes an ssh endpoint, which
changes whenever the instance is recreated) and dispatches the diagnostic
scripts to run on it.

  cc box [--refresh]            # show the resolved box (ssh + server URL + paths)
  cc ssh [cmd...]               # ssh to the box (interactive, or run one command)
  cc run <script.py> [args]     # run engine/scripts/<script.py> ON the box
  cc doctor [args]              # one-shot run health/convergence report
  cc plot [args]                # plot the run metrics time-series
  cc validate "<FEN>" [args]    # (LOCAL) is this start winnable? mate distance?
  cc launch                     # print the fresh-run runbook

Env: CC_INSTANCE=<id> forces a box when several are running.
Run as `python scripts/cc.py <cmd>` or alias `cc` to it (see scripts/README.md).
"""
import json, os, subprocess, sys, time, urllib.request, urllib.error

CACHE = os.path.expanduser("~/.cache/cc_box.json")
CACHE_TTL = 600
ENGINE_DIR = "/workspace/chessckers/engine"          # repo path ON THE BOX
SERVER_DIR = "/workspace/chessckers/lczero-server"
LOCAL_ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    elif cmd == "validate":
        if not args:
            sys.exit('usage: cc validate "<FEN>" [extra args]')
        py = os.path.join(LOCAL_ENGINE, ".venv/bin/python")
        return subprocess.call([py, os.path.join(LOCAL_ENGINE, "scripts/solve_endgame.py"),
                                "--validate", "--fen", args[0], *args[1:]])
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
