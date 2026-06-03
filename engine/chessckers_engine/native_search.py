"""Native (C++) PUCT search as a drop-in for mcts_puct.run_mcts in self-play.

`make_native_search_fn(net_box)` returns a function with run_mcts's call shape
that play_az_game can use via its `search_fn` hook. It runs the fully-native
cpp.run_mcts_native (native move-gen + apply + encode + NN forward) and returns
a result carrying `visit_distribution` (play_az_game samples the move itself),
`value` (root Q, for resignation), and a `root` shim wrapping the native tree
handle so the played move's subtree carries into the next ply (Lc0 tree reuse —
carried visits count toward n_sims). `net_box[0]` is the current cc::ChesskersNet,
hot-swapped by the worker on weight reload.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import chessckers_cpp as cpp


class _RootShim:
    """Bridges play_az_game's `result.root.children.get(uci)` reuse hook to the
    native tree handle: `.children.get(uci)` detaches and returns the subtree
    NativeTree under the played move (or None), which play_az_game passes back as
    `reuse_root` next ply. None -> run_mcts_native rebuilds fresh (the same as a
    fen mismatch)."""
    __slots__ = ("_tree",)

    def __init__(self, tree) -> None:
        self._tree = tree

    @property
    def children(self):
        return self

    def get(self, uci):
        return self._tree.child(uci) if self._tree is not None else None


@dataclass
class _NativeResult:
    chosen: Any
    visit_distribution: dict
    value: float | None = None  # root->q(): STM-relative expected outcome, for resignation
    root: Any = None            # _RootShim over the native tree handle (tree reuse)


def make_native_search_fn(net_box: list):
    seed = itertools.count(1)

    def search_fn(state, client, model, *, n_sims, c_puct, dirichlet_alpha, dirichlet_eps,
                  vloss_batch, reuse_root):
        # reuse_root is the NativeTree subtree from the previous ply (or None on the
        # first ply / a fen mismatch); run_mcts_native re-roots it if its position
        # matches, else searches fresh.
        _chosen, visit_dist, root_value, tree = cpp.run_mcts_native(
            cpp.parse_fen(state["fen"]), net_box[0], int(n_sims), float(c_puct),
            float(dirichlet_alpha or 0.0), float(dirichlet_eps), next(seed),
            reuse_root,
        )
        return _NativeResult(chosen=None, visit_distribution=dict(visit_dist),
                             value=float(root_value), root=_RootShim(tree))

    return search_fn
