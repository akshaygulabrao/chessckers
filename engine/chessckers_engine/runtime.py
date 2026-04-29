"""Construct the dict of named pickers used by both the HTTP server and the
evaluation harness.

`build_pickers` returns `{"random": ..., "material": ..., "nn": ...}`. The NN
picker is wrapped in a try/except so a missing torch install or unloadable
checkpoint doesn't take down the random and material pickers.
"""

from __future__ import annotations

import logging

from chessckers_engine.http_server import GameState, LegalMove, Picker


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging and silence httpx/httpcore.

    Each MCTS sim makes ~30 API calls, so httpx INFO-level "POST /api/game/..."
    lines drown out training progress unless suppressed. Anything that
    actually went wrong still logs at WARNING from those libs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
from chessckers_engine.material_player import pick_material
from chessckers_engine.mcts import pick_mcts
from chessckers_engine.random_player import pick_random
from chessckers_engine.server_client import ServerClient


def build_pickers(
    client: ServerClient,
    model_path: str | None,
    log: logging.Logger,
    mcts_sims: int = 100,
    puct_sims: int = 50,
) -> dict[str, Picker]:
    pickers: dict[str, Picker] = {}

    def random_picker(state: GameState) -> LegalMove | None:
        return pick_random(state.get("legalMoves") or [])

    pickers["random"] = random_picker

    def material_picker(state: GameState) -> LegalMove | None:
        return pick_material(state, client)

    pickers["material"] = material_picker

    def mcts_picker(state: GameState) -> LegalMove | None:
        return pick_mcts(state, client, n_sims=mcts_sims)

    pickers["mcts"] = mcts_picker

    try:
        from chessckers_engine.checkpoints import load_checkpoint
        from chessckers_engine.model import ChesskersScorer
        from chessckers_engine.nn_player import pick_nn

        model = ChesskersScorer()
        if model_path:
            log.info("loading NN weights from %s", model_path)
            load_checkpoint(model, model_path)
        else:
            log.info("no model path provided; NN picker uses random-init weights")
        model.eval()

        def nn_picker(state: GameState) -> LegalMove | None:
            return pick_nn(state, model)

        pickers["nn"] = nn_picker

        from chessckers_engine.mcts_puct import pick_puct

        def puct_picker(state: GameState) -> LegalMove | None:
            return pick_puct(state, client, model, n_sims=puct_sims)

        pickers["puct"] = puct_picker
    except Exception as e:  # noqa: BLE001
        log.warning("NN-based pickers unavailable (%s: %s); 'random'/'material' still work",
                    type(e).__name__, e)

    return pickers
