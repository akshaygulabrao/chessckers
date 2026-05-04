"""Self-play game generation and outcome-based training-target labeling.

Plays games where both sides sample moves from `softmax(NN_logits / τ)`
(stochastic policy — necessary so two copies of the same model don't
play identical games). Records every (fen, move, side_to_move) decision
along the way, then labels each decision with the outcome from that
side's perspective: +1 win, -1 loss, 0 draw.

The labeled examples drop straight into the existing `train.py` pipeline:
same `(fen, move, target)` shape, same MSE loss. Only the target *meaning*
changes — material delta becomes win-rate.

Why stochastic sampling instead of argmax: identical-policy collapse. With
deterministic argmax and identical weights on both sides, every self-play
game is bitwise-identical and the gradient is zero. Temperature τ=1.0 is
a safe default; anneal toward τ→0 in later iterations to play closer to
the policy as it sharpens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from chessckers_engine.encoding import encode_move, encode_position
from chessckers_engine.model import ChesskersScorer

log = logging.getLogger("chessckers_engine.selfplay")

GameState = dict[str, Any]
LegalMove = dict[str, Any]


class _Mover(Protocol):
    def new_game(self, fen: str | None = None) -> GameState: ...
    def make_move(self, fen: str, uci: str) -> GameState: ...


@dataclass
class Decision:
    fen: str
    move: LegalMove
    side_to_move: str  # "white" or "black"


@dataclass
class SelfPlayGame:
    decisions: list[Decision]
    final_status: str | None  # "mate" | "variantEnd" | "stalemate" | None (max_plies)
    outcome: str  # "white" | "black" | "draw"


def _outcome_from_status(status: str | None) -> str:
    if status == "mate":
        return "black"
    if status == "variantEnd":
        return "white"
    return "draw"


def sample_move(
    model: ChesskersScorer,
    state: GameState,
    rng: torch.Generator | None,
    temperature: float,
) -> LegalMove | None:
    """Sample a move from softmax(logits / temperature) over the legal moves.

    `temperature → 0` reduces to argmax; `temperature = 1.0` matches the policy.
    `temperature → ∞` reduces to uniform random.
    """
    legal = state.get("legalMoves") or []
    if not legal:
        return None
    device = next(model.parameters()).device
    pos = encode_position(state["fen"]).unsqueeze(0).to(device)
    moves = torch.stack([encode_move(m) for m in legal]).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(pos, moves)
    if temperature <= 0:
        idx = int(logits.argmax())
    else:
        probs = torch.softmax(logits / temperature, dim=0)
        idx = int(torch.multinomial(probs, num_samples=1, generator=rng).item())
    return legal[idx]


def play_self_game(
    model: ChesskersScorer,
    client: _Mover,
    temperature: float = 1.0,
    max_plies: int = 400,
    rng: torch.Generator | None = None,
) -> SelfPlayGame:
    """Play one self-play game with a single shared model on both sides."""
    state = client.new_game()
    decisions: list[Decision] = []
    ply = 0
    while not state.get("status") and ply < max_plies:
        chosen = sample_move(model, state, rng, temperature)
        if chosen is None:
            break
        decisions.append(Decision(fen=state["fen"], move=chosen, side_to_move=state["turn"]))
        try:
            state = client.make_move(state["fen"], chosen["uci"])
        except Exception as e:  # noqa: BLE001
            # Same scalachess server bug we tolerate elsewhere.
            log.debug("make_move failed at ply %d uci=%s: %s; ending game as draw", ply, chosen["uci"], e)
            return SelfPlayGame(decisions=decisions, final_status=None, outcome="draw")
        ply += 1
    status = state.get("status")
    return SelfPlayGame(decisions=decisions, final_status=status, outcome=_outcome_from_status(status))


def decisions_to_examples(game: SelfPlayGame) -> list[dict[str, Any]]:
    """Convert a finished self-play game into (fen, move, target) examples.

    target = +1 if this side won, -1 if this side lost, 0 for draws.
    """
    if game.outcome == "draw":
        target_white, target_black = 0.0, 0.0
    elif game.outcome == "white":
        target_white, target_black = 1.0, -1.0
    else:  # black
        target_white, target_black = -1.0, 1.0

    examples: list[dict[str, Any]] = []
    for d in game.decisions:
        target = target_white if d.side_to_move == "white" else target_black
        examples.append({"fen": d.fen, "move": d.move, "target": target})
    return examples
