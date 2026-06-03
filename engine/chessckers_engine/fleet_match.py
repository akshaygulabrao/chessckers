"""Fleet match player — a self-play client contributes keep-best GATE games.

When `fleet_arena` opens a gate it publishes a match (candidate vs current best)
through the server: `GET /next_game` hands out (seed, side) units, `GET /net/best`
and `GET /net/cand` serve the two nets, `POST /match_result` collects outcomes.
This module is the client side — pull a unit, play one gate game with the SAME
search the arena uses (native C++ if built, else Python MCTS), POST the outcome.
The arena tallies these and plays locally only the units no client supplied, so
idle boxes share the gating cost instead of the trainer host carrying all of it.

Heavy (torch + model + the native ext), so `fleet_client` imports this lazily and
only when a match is actually open; a box without the deps stays self-play-only.
It reuses the arena's game runner / net loader / pickers verbatim, so a client
gate game is the same computation as a local one.
"""
from __future__ import annotations

import itertools
import json
import logging
import urllib.request
from pathlib import Path

from chessckers_engine import fleet_arena as arena
from chessckers_engine.device import pick_device
from chessckers_engine.variant_py import PyVariantClient

log = logging.getLogger("chessckers_engine.fleet_match")


def _get(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _post(url: str, data: bytes, timeout: float) -> None:
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()


class MatchRunner:
    """Holds the current match's two pickers; rebuilds them when the server rotates
    to a new candidate (match_id change). One instance per client process."""

    def __init__(self, run_dir: Path, device: str = "cpu") -> None:
        self.cache = run_dir / "_match"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.device = pick_device(device)
        self.client = PyVariantClient()
        self.counter = itertools.count(1)   # per-move Dirichlet seed (native diversity)
        self.mid: int | None = None
        self.cand_pick = None
        self.best_pick = None
        self.max_plies = 200

    def _picker(self, net, sims, c_puct, dir_alpha, dir_eps):
        if arena.NATIVE_OK:
            return arena._native_picker(net, sims, c_puct, dir_alpha, dir_eps, self.counter)
        return arena._model_picker(net, self.client, sims, c_puct, dir_alpha, dir_eps)

    def _ensure(self, server: str, mid: int, arch: dict, params: dict, timeout: float) -> None:
        """Download + build both gate nets for `mid` if we don't already hold them."""
        if mid == self.mid:
            return
        best_pt = self.cache / f"best-{mid}.pt"
        cand_pt = self.cache / f"cand-{mid}.pt"
        best_pt.write_bytes(_get(f"{server}/net/best", timeout))
        cand_pt.write_bytes(_get(f"{server}/net/cand", timeout))
        best_net = arena._make_net(best_pt, arch, self.cache / f"best-{mid}.bin", self.device)
        cand_net = arena._make_net(cand_pt, arch, self.cache / f"cand-{mid}.bin", self.device)
        self.best_pick = self._picker(best_net, params["sims"], params["c_puct"],
                                      params["dir_alpha"], params["dir_eps"])
        self.cand_pick = self._picker(cand_net, params["sims"], params["c_puct"],
                                      params["dir_alpha"], params["dir_eps"])
        self.max_plies = int(params["max_plies"])
        self.mid = mid
        for p in self.cache.iterdir():            # prune the previous match's files
            if str(mid) not in p.name:
                try:
                    p.unlink()
                except OSError:
                    pass
        log.info("loaded gate nets for match %d (backend=%s)", mid,
                 "native" if arena.NATIVE_OK else "python")

    def step(self, server: str, timeout: float) -> int:
        """One assignment: GET /next_game; if a match is open, play its unit and POST
        the outcome. Returns 1 if a gate game was played, 0 for self-play (no match)."""
        a = json.loads(_get(f"{server}/next_game", timeout))
        if a.get("mode") != "match":
            return 0
        self._ensure(server, a["match_id"], a["arch"], a["params"], timeout)
        cand_white = bool(a["cand_white"])
        white_pick, black_pick = ((self.cand_pick, self.best_pick) if cand_white
                                  else (self.best_pick, self.cand_pick))
        outcome = arena._play_from(white_pick, black_pick, self.client, a["seed"], self.max_plies)
        _post(f"{server}/match_result", json.dumps({
            "match_id": a["match_id"], "seed": a["seed"],
            "cand_white": cand_white, "outcome": outcome,
        }).encode(), timeout)
        return 1
