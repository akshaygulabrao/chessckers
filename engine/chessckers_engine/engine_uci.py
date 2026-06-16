"""Drive the `akshay-chessckers-0` lc0 fork as a UCI opponent (one process per game).

The fork is the production Chessckers player; here we use its **UCI mode**: feed a
position, ask it to search, read back its move. Two hard-won gotchas are baked in:

  * **Always position via an explicit FEN, never `position startpos`.** The fork's
    `startpos` is its *training* start (a different position than the standard
    game start), so `startpos` and PyVariant disagree. `position fen <FEN>` round-
    trips exactly — the fork's emitted UCI tokens match PyVariant's move-gen.
  * **No `temperature`/`noise`** exists in the fork: at a fixed node count the
    search is deterministic. Strength is set purely by `go nodes N`.

We keep the process alive across the game and read stdout until `bestmove`, which
also sidesteps the one-shot "quit-race" (a piped `quit` can abort the search
before it prints `bestmove`).
"""
from __future__ import annotations

import os
import select
import subprocess
from typing import Any

_PKG = os.path.dirname(os.path.abspath(__file__))   # .../engine/chessckers_engine
_ENG = os.path.dirname(_PKG)                          # .../engine
_REPO = os.path.dirname(_ENG)                         # .../chessckers
DEFAULT_BIN = os.path.join(
    _REPO, "akshay-chessckers-0", "build", "release", "akshay-chessckers-0"
)


def _parse_search(lines: list[str]) -> dict[str, Any]:
    """Pull the move + last analysis line out of a `go` response.

    Returns `{uci, score_cp?, mate?, nodes?, depth?, pv?}`; `score_cp`/`pv` are
    from the engine's POV (the side it just searched for). `uci` is None if the
    search produced no `bestmove`."""
    best: str | None = None
    info: dict[str, Any] = {}
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0] == "info" and "pv" in t:
            d: dict[str, Any] = {}
            for key, name in (("cp", "score_cp"), ("mate", "mate"),
                              ("nodes", "nodes"), ("depth", "depth")):
                if key in t:
                    try:
                        d[name] = int(t[t.index(key) + 1])
                    except (ValueError, IndexError):
                        pass
            d["pv"] = t[t.index("pv") + 1:]
            info = d  # keep the last (deepest) info line
        elif t[0] == "bestmove":
            best = t[1] if len(t) > 1 else None
    return {"uci": best, **info}


class UciEngine:
    """A live UCI process speaking Chessckers. Construct once, call `bestmove`
    per move, `close()` when done (also usable as a context manager)."""

    def __init__(
        self,
        net: str,
        *,
        binary: str | None = None,
        backend: str | None = None,
        threads: int | None = None,
        handshake_timeout: float = 30.0,
    ) -> None:
        self.binary = binary or os.environ.get("CHESSCKERS_ENGINE_BIN", DEFAULT_BIN)
        if not os.path.exists(self.binary):
            raise FileNotFoundError(
                f"engine binary not found: {self.binary}\n"
                "build the fork (meson+ninja) or pass --engine-bin / set CHESSCKERS_ENGINE_BIN."
            )
        if not os.path.exists(net):
            raise FileNotFoundError(f"network file not found: {net}")
        self.net = net
        args = [self.binary, "-w", net]
        if backend:
            args.append(f"--backend={backend}")
        if threads:
            args.append(f"--threads={threads}")
        # stderr→stdout so any load error is visible in the line stream; UCI
        # replies are matched by their leading token, so log lines are ignored.
        self.p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._send("uci")
        if not self._read_until("uciok", handshake_timeout):
            raise RuntimeError(
                f"engine did not reply 'uciok' within {handshake_timeout:.0f}s "
                f"(binary {self.binary!r}); run it by hand to see the error."
            )
        self._send("isready")
        self._read_until("readyok", handshake_timeout)
        self._send("ucinewgame")

    def _send(self, cmd: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _read_until(self, token: str, timeout: float) -> list[str]:
        """Read lines until one begins with `token` (or the process dies). Returns
        every line read (including the matching one); empty list on timeout/death."""
        import time
        assert self.p.stdout is not None
        end = time.time() + timeout
        out: list[str] = []
        while time.time() < end:
            r, _, _ = select.select([self.p.stdout], [], [], max(0.0, end - time.time()))
            if not r:
                continue
            ln = self.p.stdout.readline()
            if not ln:  # process exited
                break
            ln = ln.rstrip("\n")
            out.append(ln)
            if ln.split()[:1] == [token]:
                return out
        return out

    def bestmove(self, fen: str, nodes: int, *, timeout: float = 300.0) -> dict[str, Any]:
        """Search `fen` for `nodes` playouts; return `_parse_search(...)`."""
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        return _parse_search(self._read_until("bestmove", timeout))

    def close(self) -> None:
        if self.p.poll() is not None:
            return
        try:
            self._send("quit")
            self.p.wait(timeout=5)
        except Exception:  # noqa: BLE001 — best-effort teardown
            self.p.kill()

    def __enter__(self) -> "UciEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
