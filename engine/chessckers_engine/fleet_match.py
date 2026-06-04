"""Fleet match player — a self-play client contributes keep-best GATE games.

When `fleet_arena` opens a gate it publishes a match (candidate vs a panel of the
last N champions) through the server: `GET /next_game` hands out (opponent, seed,
side) units, `GET /net/cand` serves the candidate and `GET /net/opp/<id>` serves
each opponent net, `POST /match_result` collects outcomes tagged with the opponent.
This module is the client side — pull a unit, play one gate game with the SAME
search the arena uses (native C++ if built, else Python MCTS), POST the outcome.
The arena tallies these and plays locally only the units no client supplied, so
idle boxes share the gating cost instead of the trainer host carrying all of it.
Serving every champion net (not just best) is what lets the older-champion ladder
games offload too — lc0 ships both shas to its clients; we ship cand + each champ.

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
    """Holds the current gate's candidate picker plus a per-opponent picker cache, all
    rebuilt when the server rotates to a new gate (match_id change). Opponents are fetched
    lazily as `/next_game` hands them out, so a client only downloads the champions it's
    actually asked to play. One instance per client process."""

    def __init__(self, run_dir: Path, device: str = "cpu") -> None:
        self.cache = run_dir / "_match"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.device = pick_device(device)
        self.client = PyVariantClient()
        self.counter = itertools.count(1)   # per-move Dirichlet seed (native diversity)
        self.mid: int | None = None
        self.arch: dict | None = None
        self.params: dict | None = None
        self.max_plies = 200
        self.cand_pick = None
        self.opp_picks: dict = {}           # oppid -> picker (one per opponent we've been handed this gate)

    def _picker(self, net, sims, c_puct, dir_alpha, dir_eps):
        if arena.NATIVE_OK:
            return arena._native_picker(net, sims, c_puct, dir_alpha, dir_eps, self.counter)
        return arena._model_picker(net, self.client, sims, c_puct, dir_alpha, dir_eps)

    def _build_pick(self, url_path: str, fname: str, server: str, timeout: float):
        """Download one gate net to `fname` and build its search picker (params/arch are
        the open gate's, set by `_ensure_match`)."""
        pt = self.cache / fname
        pt.write_bytes(_get(f"{server}{url_path}", timeout))
        net = arena._make_net(pt, self.arch, self.cache / (fname[:-3] + ".bin"), self.device)
        return self._picker(net, self.params["sims"], self.params["c_puct"],
                            self.params["dir_alpha"], self.params["dir_eps"])

    def _ensure_match(self, server: str, mid: int, arch: dict, params: dict, timeout: float) -> None:
        """On a new gate (match_id change): reset, download the candidate, drop stale files."""
        if mid == self.mid:
            return
        self.arch = arch
        self.params = params
        self.max_plies = int(params["max_plies"])
        self.cand_pick = None
        self.opp_picks = {}
        self.mid = mid
        for p in self.cache.iterdir():            # prune the previous gate's files
            if str(mid) not in p.name:
                try:
                    p.unlink()
                except OSError:
                    pass
        self.cand_pick = self._build_pick("/net/cand", f"cand-{mid}.pt", server, timeout)
        log.info("loaded candidate for match %d (backend=%s)", mid,
                 "native" if arena.NATIVE_OK else "python")

    def _ensure_opp(self, server: str, oppid: str, timeout: float):
        """Lazily download + build a gate opponent's picker (cached for this gate). Clients
        only ever held best.pt before; serving every champion net is what lets an
        older-champion ladder game be offloaded exactly like the vs-best gate."""
        pick = self.opp_picks.get(oppid)
        if pick is None:
            pick = self._build_pick(f"/net/opp/{oppid}", f"opp-{self.mid}-{oppid}.pt", server, timeout)
            self.opp_picks[oppid] = pick
            log.info("loaded opponent %s for match %d", oppid, self.mid)
        return pick

    def step(self, server: str, timeout: float) -> int:
        """One assignment: GET /next_game; if a match is open, play its (opponent, seed,
        side) unit and POST the outcome tagged with the opponent. Returns 1 if a gate game
        was played, 0 for self-play (no match)."""
        a = json.loads(_get(f"{server}/next_game", timeout))
        if a.get("mode") != "match":
            return 0
        self._ensure_match(server, a["match_id"], a["arch"], a["params"], timeout)
        oppid = a.get("opp") or "best"
        opp_pick = self._ensure_opp(server, oppid, timeout)
        cand_white = bool(a["cand_white"])
        white_pick, black_pick = ((self.cand_pick, opp_pick) if cand_white
                                  else (opp_pick, self.cand_pick))
        outcome = arena._play_from(white_pick, black_pick, self.client, a["seed"], self.max_plies)
        _post(f"{server}/match_result", json.dumps({
            "match_id": a["match_id"], "seed": a["seed"], "opp": oppid,
            "cand_white": cand_white, "outcome": outcome,
        }).encode(), timeout)
        return 1
