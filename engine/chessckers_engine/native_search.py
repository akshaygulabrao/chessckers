"""Native (C++) PUCT search as a drop-in for mcts_puct.run_mcts in self-play.

`make_native_search_fn(net_box)` returns a function with run_mcts's call shape
that play_az_game can use via its `search_fn` hook. It runs the fully-native
cpp.run_mcts_native (native move-gen + apply + encode + NN forward) and returns
a result carrying only `visit_distribution` (play_az_game samples the move
itself) and a no-reuse root (the native search builds a fresh tree per ply; its
per-search speed more than compensates). `net_box[0]` is the current
cc::ChesskersNet, hot-swapped by the worker on weight reload.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import chessckers_cpp as cpp


class _NoReuseRoot:
    __slots__ = ("children",)

    def __init__(self) -> None:
        # .children.get(uci) -> None, so play_az_game's reuse_root becomes None.
        self.children: dict = {}


@dataclass
class _NativeResult:
    chosen: Any
    visit_distribution: dict
    value: float | None = None  # root->q(): STM-relative expected outcome, for resignation
    root: _NoReuseRoot = field(default_factory=_NoReuseRoot)


def make_native_search_fn(net_box: list):
    seed = itertools.count(1)

    def search_fn(state, client, model, *, n_sims, c_puct, dirichlet_alpha, dirichlet_eps,
                  vloss_batch, reuse_root):
        _chosen, visit_dist, root_value = cpp.run_mcts_native(
            cpp.parse_fen(state["fen"]), net_box[0], int(n_sims), float(c_puct),
            float(dirichlet_alpha or 0.0), float(dirichlet_eps), next(seed),
        )
        return _NativeResult(chosen=None, visit_distribution=dict(visit_dist), value=float(root_value))

    return search_fn
