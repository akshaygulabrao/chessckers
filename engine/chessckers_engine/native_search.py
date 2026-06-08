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


def play_game_native(net, *, start_fen: str, n_sims: int = 100, c_puct: float = 1.5,
                     temperature: float = 1.0, temp_cutoff_plies: int = 30,
                     max_plies: int = 400, dirichlet_alpha: float | None = 0.3,
                     dirichlet_eps: float = 0.25, seed: int = 0,
                     resign_threshold: float = 0.0, resign_no_resign_frac: float = 0.1,
                     resign_consecutive: int = 2, resign_min_ply: int = 8):
    """Play one FULLY-NATIVE self-play game (cpp.play_game_native — search+sample+
    apply+record per ply, zero Python in the hot loop) and adapt it to an AZGame so
    the existing `az_game_to_examples` → chunk path is unchanged. `net` is a
    cc::ChesskersNet. Phase 1 of the lc0-split migration (see the migration plan)."""
    from chessckers_engine.selfplay_az import AZGame, AZRecord

    records_raw, outcome, final_status = cpp.play_game_native(
        cpp.parse_fen(start_fen), net, int(n_sims), float(c_puct), float(temperature),
        int(temp_cutoff_plies), int(max_plies), float(dirichlet_alpha or 0.0),
        float(dirichlet_eps), int(seed), float(resign_threshold),
        float(resign_no_resign_frac), int(resign_consecutive), int(resign_min_ply),
    )
    records = [
        AZRecord(fen=fen, legal_moves=list(legal), visit_counts=list(vc), side_to_move=side)
        for (fen, legal, vc, side) in records_raw
    ]
    return AZGame(records=records, final_status=final_status, outcome=outcome)
