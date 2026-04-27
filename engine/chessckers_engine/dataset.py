"""Training-data generator for the supervised material-target phase.

Walks random self-play games using a `ServerClient`-like Mover. At every ply,
asks the server to apply *each* legal move and records the resulting position's
material score (from the moving side's perspective, with king_value=0 — king
capture is game-ending and handled at the status level rather than as a
+/-1000 outlier).

Each example is `{"fen": str, "move": LegalMove, "target": float}`. Examples
are written one-per-line as JSONL so they can be regenerated cheaply and
streamed at training time without holding everything in memory.

Generation cost scales as O(n_games * avg_plies * avg_legal_moves) server
round-trips; for a local sbt API at ~2 ms per round-trip, ~9k examples per
minute is realistic.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from chessckers_engine.material import material_for_side_to_move

log = logging.getLogger("chessckers_engine.dataset")

GameState = dict[str, Any]
LegalMove = dict[str, Any]
Example = dict[str, Any]

MAX_PLIES_PER_GAME = 200


class _Mover(Protocol):
    def new_game(self, fen: str | None = None) -> GameState: ...
    def make_move(self, fen: str, uci: str) -> GameState: ...


def _examples_at(state: GameState, client: _Mover) -> Iterator[Example]:
    """Score every legal move at the given position. Yields one Example per move."""
    fen = state["fen"]
    for move in state.get("legalMoves") or []:
        post = client.make_move(fen, move["uci"])
        # material_for_side_to_move(post_fen) is from the *next* mover's
        # perspective. Negate to get the perspective of the player who just moved.
        target = -material_for_side_to_move(post["fen"], king_value=0)
        yield {"fen": fen, "move": move, "target": float(target)}


def generate_examples(
    n_games: int,
    client: _Mover,
    rng: random.Random | None = None,
    max_plies: int = MAX_PLIES_PER_GAME,
) -> list[Example]:
    """Play random-vs-random self-play games and return all per-move examples."""
    rng = rng or random.Random()
    out: list[Example] = []
    for game in range(n_games):
        state = client.new_game()
        ply = 0
        while not state.get("status") and ply < max_plies:
            legal = state.get("legalMoves") or []
            if not legal:
                break
            out.extend(_examples_at(state, client))
            chosen = rng.choice(legal)
            state = client.make_move(state["fen"], chosen["uci"])
            ply += 1
        log.info("game %d/%d: %d plies, %d examples so far", game + 1, n_games, ply, len(out))
    return out


def save_jsonl(examples: Iterable[Example], path: str | Path) -> int:
    """Write examples one-per-line as JSON. Returns count written."""
    n = 0
    with Path(path).open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex))
            f.write("\n")
            n += 1
    return n


def load_jsonl(path: str | Path) -> list[Example]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
