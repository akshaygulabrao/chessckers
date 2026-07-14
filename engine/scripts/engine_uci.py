#!/usr/bin/env python3
"""Minimal UCI driver for the akshay-chessckers-0 lc0 fork — the REAL production engine.

Spawns the engine binary pinned to one net (``--weights``) and drives it move by
move over UCI stdin/stdout: ``position fen <fen>`` + ``go nodes N`` -> ``bestmove <uci>``.
Lets match tools (ladder.py) play the actual production search instead of the Python
MCTS reference — same net loader (``cc::ChesskersNet`` reads the flat ``.bin`` that
``chessckers_engine.native_net.export_state_dict`` writes; see ``src/chessckers/nn.hpp``
``load_weights`` and ``src/neural/backends/chessckers_backend.cc``), same search, same
UCI move notation (verified byte-identical to PyVariant's — chains ``c<cad>:<from>~..-><to>``,
deploys ``<from><to>[n]``, charges, quiets — so ``bestmove`` feeds straight into
``PyVariantClient.make_move`` with no translation).

The client always launches the fork with ``--backend=chessckers`` (lc0_main.go), so we
do too. NEVER send ``position startpos`` — the Chessckers start FEN is non-standard;
always pass an explicit FEN.
"""
from __future__ import annotations

import os
import subprocess

# Engine binary on the box (built + symlinked by cc.py's provision step). Callers
# on the Mac / elsewhere pass an explicit path.
DEFAULT_BINARY = "/workspace/chessckers/akshay-chessckers-0/build/release/akshay-chessckers-0"


class UciEngine:
    """One persistent engine process pinned to a single net's weights. Reuse it
    across many games — call ``new_game()`` between games to clear the search tree.

    Node count per move is fixed at construction (``visits``, default 128 to match
    the fleet gate's ``matchParams --visits=128``); ``go nodes N`` visit-limits the
    search the same way (VisitsStopper), so ladder numbers track gate conditions.
    ``extra_args`` carries the opening-diversity flags (temperature etc.)."""

    def __init__(self, weights: str, binary: str = DEFAULT_BINARY, *,
                 visits: int = 128, extra_args: list[str] | None = None) -> None:
        self.weights = weights
        self.visits = visits
        self._argv = [binary, "--backend=chessckers", f"--weights={weights}",
                      *(extra_args or [])]
        # Engine stderr goes to a file, not DEVNULL — when the fork dies mid-game
        # (CUDA abort, assert, segv) the reason is otherwise lost. Appended across
        # respawns; keyed by net + visits so asymmetric-visits engines don't mix.
        self._err_path = os.path.join(
            os.environ.get("UCI_ERR_DIR", "/tmp"),
            f"uci-{os.path.basename(weights)}.{visits}v.err")
        self._errf = None
        self._spawn()

    def _spawn(self) -> None:
        self._errf = open(self._err_path, "ab")
        self.proc = subprocess.Popen(
            self._argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._errf, text=True, bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")      # engine ready to accept options/positions
        self._send("isready")
        self._read_until("readyok")    # net loaded

    def restart(self) -> None:
        """Respawn the process after a crash — the fork segfaults intermittently, so
        callers restart rather than lose a long run."""
        self.close()
        self._spawn()

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _readline(self) -> str:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if line == "":  # EOF: the process closed stdout / died
            raise RuntimeError(
                f"engine closed stdout (died?) — weights={self.weights}. Check the "
                "binary path, that this build supports --backend=chessckers, and that "
                f"the .bin loads (cc::ChesskersNet::load_weights). stderr: {self._err_path}")
        return line.rstrip("\n")

    def _read_until(self, prefix: str) -> str:
        """Read lines (skipping `info ...` etc.) until one starts with `prefix`."""
        while True:
            line = self._readline()
            if line.startswith(prefix):
                return line

    def new_game(self) -> None:
        """Clear the search tree — call once per game before the first position."""
        self._send("ucinewgame")

    def bestmove(self, fen: str) -> str | None:
        """Search `fen` for `visits` nodes; return the fork's chosen UCI move, or
        None if the engine reports no move (`bestmove (none)`).

        Clears the tree (`ucinewgame`) before EVERY search: the fork has an
        intermittent SIGSEGV in its tree-REUSE path (Edge::GetMove race — the class
        the selfplay `--no-share-trees` note avoids), and carrying the prior move's
        subtree in UCI mode trips it. Fresh-tree-per-move also gives every position a
        full N-node search, which is what a fixed-budget ladder wants anyway."""
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go nodes {self.visits}")
        parts = self._read_until("bestmove").split()
        mv = parts[1] if len(parts) > 1 else "(none)"
        return None if mv == "(none)" else mv

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send("quit")
                self.proc.wait(timeout=3)
            except Exception:  # noqa: BLE001 — best-effort shutdown
                self.proc.kill()
        if self._errf is not None:
            self._errf.close()
            self._errf = None

    def __enter__(self) -> "UciEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
