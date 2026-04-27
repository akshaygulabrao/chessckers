"""AlphaZero-style self-play data generation for Chessckers.

At each move during a game we run PUCT MCTS, record the resulting visit
distribution, play the most-visited move (or sample from `visits**1/τ` if
exploration is desired), and continue. After the game ends, every recorded
position gets a value target equal to the eventual outcome from that
position's side-to-move perspective: +1 win, -1 loss, 0 draw.

Each AZExample produces, for one position visited during play:
- `fen`               — the position
- `legal_moves`       — the candidates considered at that position
- `visit_distribution`— normalized visit counts (a probability over
                        `legal_moves`) that becomes the policy target
- `value_target`      — outcome from STM's perspective (training target
                        for the value head)

These examples drop into the dual-loss training step in `train.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from chessckers_engine.mcts_puct import run_mcts
from chessckers_engine.model import ChesskersScorer

log = logging.getLogger("chessckers_engine.selfplay_az")

GameState = dict[str, Any]
LegalMove = dict[str, Any]


class _Mover(Protocol):
    def new_game(self, fen: str | None = None) -> GameState: ...
    def make_move(self, fen: str, uci: str) -> GameState: ...


@dataclass
class AZRecord:
    fen: str
    legal_moves: list[LegalMove]
    visit_counts: list[int]   # aligned with legal_moves
    side_to_move: str          # "white" or "black"


@dataclass
class AZGame:
    records: list[AZRecord]
    final_status: str | None
    outcome: str  # "white" | "black" | "draw"


@dataclass
class AZExample:
    fen: str
    legal_moves: list[LegalMove]
    visit_distribution: list[float]  # probabilities, sum to ~1
    value_target: float              # in {-1.0, 0.0, 1.0}


def _outcome_from_status(status: str | None) -> str:
    if status == "mate":
        return "black"
    if status == "variantEnd":
        return "white"
    return "draw"


def _aligned_visits(visit_dist: dict[str, int], legal_moves: list[LegalMove]) -> list[int]:
    return [visit_dist.get(m["uci"], 0) for m in legal_moves]


def _sample_move_index_from_visits(
    visits: list[int],
    temperature: float,
    rng: torch.Generator | None,
) -> int:
    """Sample an index into `visits` with probabilities ∝ visits**(1/τ).

    τ → 0 reduces to argmax; τ = 1.0 samples in proportion to visit counts."""
    if not visits:
        return 0
    if temperature <= 0:
        return int(max(range(len(visits)), key=lambda i: visits[i]))
    counts = torch.tensor(visits, dtype=torch.float32)
    if counts.sum() == 0:
        return 0
    probs = counts.pow(1.0 / temperature)
    probs = probs / probs.sum()
    return int(torch.multinomial(probs, num_samples=1, generator=rng).item())


def play_az_game(
    model: ChesskersScorer,
    client: _Mover,
    n_sims: int = 100,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    max_plies: int = 400,
    rng: torch.Generator | None = None,
) -> AZGame:
    """Play one self-play game using PUCT MCTS at each move."""
    state = client.new_game()
    records: list[AZRecord] = []
    ply = 0
    while not state.get("status") and ply < max_plies:
        legal = state.get("legalMoves") or []
        if not legal:
            break
        result = run_mcts(state, client, model, n_sims=n_sims, c_puct=c_puct)
        visits = _aligned_visits(result.visit_distribution, legal)
        records.append(
            AZRecord(
                fen=state["fen"],
                legal_moves=legal,
                visit_counts=visits,
                side_to_move=state["turn"],
            )
        )
        idx = _sample_move_index_from_visits(visits, temperature, rng)
        chosen = legal[idx]
        try:
            state = client.make_move(state["fen"], chosen["uci"])
        except Exception as e:  # noqa: BLE001
            log.debug("make_move failed at ply %d uci=%s: %s; ending game as draw", ply, chosen["uci"], e)
            return AZGame(records=records, final_status=None, outcome="draw")
        ply += 1

    status = state.get("status")
    return AZGame(records=records, final_status=status, outcome=_outcome_from_status(status))


def az_game_to_examples(game: AZGame) -> list[AZExample]:
    """Convert an AZGame to dual-target training examples."""
    if game.outcome == "draw":
        v_white, v_black = 0.0, 0.0
    elif game.outcome == "white":
        v_white, v_black = 1.0, -1.0
    else:
        v_white, v_black = -1.0, 1.0

    out: list[AZExample] = []
    for rec in game.records:
        total = sum(rec.visit_counts) or 1
        dist = [v / total for v in rec.visit_counts]
        target_v = v_white if rec.side_to_move == "white" else v_black
        out.append(
            AZExample(
                fen=rec.fen,
                legal_moves=rec.legal_moves,
                visit_distribution=dist,
                value_target=target_v,
            )
        )
    return out
