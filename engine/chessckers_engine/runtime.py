"""Construct the dict of named pickers used by both the HTTP server and the
evaluation harness.

`build_pickers` returns `{"random": ..., "material": ..., "nn": ...}`. The NN
picker is wrapped in a try/except so a missing torch install or unloadable
checkpoint doesn't take down the random and material pickers.
"""

from __future__ import annotations

import logging

from chessckers_engine.http_server import GameState, LegalMove, Picker
from chessckers_engine.material_player import pick_material
from chessckers_engine.random_player import pick_random
from chessckers_engine.server_client import ServerClient


def build_pickers(
    client: ServerClient, model_path: str | None, log: logging.Logger
) -> dict[str, Picker]:
    pickers: dict[str, Picker] = {}

    def random_picker(state: GameState) -> LegalMove | None:
        return pick_random(state.get("legalMoves") or [])

    pickers["random"] = random_picker

    def material_picker(state: GameState) -> LegalMove | None:
        return pick_material(state, client)

    pickers["material"] = material_picker

    try:
        import torch

        from chessckers_engine.model import ChesskersScorer
        from chessckers_engine.nn_player import pick_nn

        model = ChesskersScorer()
        if model_path:
            log.info("loading NN weights from %s", model_path)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        else:
            log.info("no model path provided; NN picker uses random-init weights")
        model.eval()

        def nn_picker(state: GameState) -> LegalMove | None:
            return pick_nn(state, model)

        pickers["nn"] = nn_picker
    except Exception as e:  # noqa: BLE001
        log.warning("NN picker unavailable (%s: %s); 'random' and 'material' still work",
                    type(e).__name__, e)

    return pickers
