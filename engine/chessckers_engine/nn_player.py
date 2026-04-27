"""Neural-net move picker for the Chessckers engine.

Counterpart to random_player.pick_random: given the current GameState
(as returned by the API) and a ChesskersScorer model, encodes the
position and each legal move, runs a single forward pass, and returns
the highest-scoring LegalMove dict.

With random-init weights the chosen move is effectively random (just
through a NN instead of random.choice). The point of this module is
having the inference path wired up so a future milestone can swap in
trained weights without changing anything downstream.
"""

from __future__ import annotations

from typing import Any

import torch

from chessckers_engine.encoding import encode_move, encode_position
from chessckers_engine.model import ChesskersScorer

GameState = dict[str, Any]
LegalMove = dict[str, Any]


def pick_nn(state: GameState, model: ChesskersScorer) -> LegalMove | None:
    legal_moves = state.get("legalMoves") or []
    if not legal_moves:
        return None
    pos = encode_position(state["fen"]).unsqueeze(0)
    moves = torch.stack([encode_move(m) for m in legal_moves])
    model.eval()
    with torch.no_grad():
        logits = model(pos, moves)
    return legal_moves[int(logits.argmax())]
