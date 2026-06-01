"""The AlphaZero per-ply temperature cutoff: after `temp_cutoff_plies`, moves
are played greedily (argmax of visit counts), regardless of `temperature`.

With Dirichlet noise disabled to isolate temperature, a game played with
`temp_cutoff_plies=0` (every ply past the cutoff = argmax) must be fully
deterministic — independent of the RNG seed AND of the temperature value, since
argmax never consults either. That's the guarantee the cutoff provides: sharp,
reproducible conversion after the opening.
"""
from __future__ import annotations

import torch

from chessckers_engine.model import ChesskersScorer
from chessckers_engine.selfplay_az import play_az_game
from chessckers_engine.variant_py import PyVariantClient

# d4/e4 endgame seed (Black to move) — a few plies of real play, not a 1-ply end.
SEED_FEN = "8/8/8/8/3kk3/8/8/4K3[d4:kk,e4:kk] b - - 0 1"


def _fen_sequence(model, client, *, seed: int, temperature: float, cutoff: int) -> list[str]:
    rng = torch.Generator().manual_seed(seed)
    game = play_az_game(
        model, client,
        n_sims=16, temperature=temperature, temp_cutoff_plies=cutoff,
        max_plies=6, rng=rng,
        dirichlet_alpha=None,   # isolate the temperature effect from root noise
    )
    return [r.fen for r in game.records]


def test_cutoff_zero_is_argmax_and_deterministic(monkeypatch):
    monkeypatch.setenv("CHESSCKERS_START_FEN", SEED_FEN)
    model = ChesskersScorer().eval()
    client = PyVariantClient()

    # cutoff=0 → every ply is argmax → the move sequence cannot depend on the
    # RNG seed or on the (nominally high) temperature.
    a = _fen_sequence(model, client, seed=1, temperature=2.0, cutoff=0)
    b = _fen_sequence(model, client, seed=999, temperature=2.0, cutoff=0)

    assert len(a) > 1, "expected a multi-ply game to make the test meaningful"
    assert a == b, "argmax play (cutoff=0) must be independent of the RNG seed"

    # Same argmax sequence regardless of the temperature value, too.
    c = _fen_sequence(model, client, seed=1, temperature=0.1, cutoff=0)
    assert a == c, "past the cutoff, temperature must not affect the move choice"
