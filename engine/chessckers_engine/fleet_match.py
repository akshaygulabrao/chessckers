"""Fleet match player — plays one keep-best GATE game for the client.

When `fleet_arena` opens a gate it publishes a panel (the candidate vs the last N
champions). The server hands each (opponent, seed, side) unit to a self-play box as
a `match` job over `POST /next_game` (lc0-shaped; nets identified by sha256). The box
fetches the two nets by content address (`GET /get_network?sha=`), plays the unit with
the SAME search the arena uses (native C++ if built, else Python MCTS), and POSTs the
outcome to `/match_result`. The arena tallies these and plays locally only the units no
client supplied, so idle boxes share the gating cost instead of the trainer host
carrying all of it.

This module is the in-process PLAYER only. `fleet_client` owns all the HTTP (it has the
interface-bound opener + the heartbeat headers), so it fetches the nets and hands this
runner the job dict plus the two already-fetched net files; the runner loads them and
plays — it never talks to the network. It is heavy (torch + model + the native ext), so
`fleet_client` imports it LAZILY and only when a gate is actually open; a box without the
deps stays self-play-only. It reuses the arena's game runner / net loader / pickers
verbatim, so a client gate game is the same computation as a local one.
"""
from __future__ import annotations

import itertools
import logging
from pathlib import Path

from chessckers_engine import fleet_arena as arena
from chessckers_engine.device import pick_device
from chessckers_engine.variant_py import PyVariantClient

log = logging.getLogger("chessckers_engine.fleet_match")


class MatchRunner:
    """Plays gate games for one client process. Search pickers are cached by net content
    sha (building a native net is expensive) and dropped when the server rotates to a new
    gate (match_id change), so at most one gate's panel is ever resident. The client fetches
    nets content-addressed and passes their file paths to `play`, so this runner is pure
    compute — no HTTP, no knowledge of the wire. One instance per client process."""

    def __init__(self, cache_dir: Path, device: str = "cpu") -> None:
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.device = pick_device(device)
        self.client = PyVariantClient()
        self.counter = itertools.count(1)   # per-move Dirichlet seed (native diversity)
        self.mid: int | None = None
        self.arch: dict | None = None
        self.params: dict | None = None
        self.max_plies = 200
        self.picks: dict[str, object] = {}  # net sha -> search picker (rebuilt each gate)

    def _picker(self, net):
        if arena.NATIVE_OK:
            return arena._native_picker(net, self.params["sims"], self.params["c_puct"],
                                        self.params["dir_alpha"], self.params["dir_eps"],
                                        self.counter)
        return arena._model_picker(net, self.client, self.params["sims"], self.params["c_puct"],
                                   self.params["dir_alpha"], self.params["dir_eps"])

    def _pick(self, sha: str, path: Path):
        """Build (and cache, by content sha) the search picker for one net file. The native
        net is held in memory after construction, so the picker stays valid even if the
        backing file is later removed."""
        pick = self.picks.get(sha)
        if pick is None:
            net = arena._make_net(Path(path), self.arch, self.cache / f"{sha}.bin", self.device)
            pick = self._picker(net)
            self.picks[sha] = pick
        return pick

    def play(self, job: dict, cand_path: Path, opp_path: Path) -> str:
        """Play one gate unit described by a `POST /next_game` match job, given the two
        already-fetched (content-addressed) net files. Returns 'white' | 'black' | 'draw'
        from the winner's perspective. On a new gate (match_id change) the per-net picker
        cache is dropped first, so only the current gate's nets stay resident."""
        if job["match_id"] != self.mid:
            self.mid = job["match_id"]
            self.picks = {}
        self.cache.mkdir(parents=True, exist_ok=True)
        self.arch = job["arch"]
        self.params = job["params"]
        self.max_plies = int(job["params"]["max_plies"])
        cand_pick = self._pick(job["candidate_sha"], cand_path)
        opp_pick = self._pick(job["opponent_sha"], opp_path)
        white_pick, black_pick = ((cand_pick, opp_pick) if job["cand_white"]
                                  else (opp_pick, cand_pick))
        return arena._play_from(white_pick, black_pick, self.client, job["seed"], self.max_plies)
