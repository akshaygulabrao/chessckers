"""Load human-vs-engine games saved by the UI and convert them into
demonstration-bootstrapped training examples.

Each saved game in `games/games.jsonl` looks like:
  {
    "history": [{"fen": "...", "uci": "..."}, ...],  # one entry per ply
    "final_fen": "...",
    "final_status": "mate" | "variantEnd" | "stalemate" | None,
    "outcome": "white" | "black" | "draw" | "incomplete",
    "controllers": {"white": "...", "black": "..."},
    "saved_at": "...",
  }

`extract_examples(games, color, client)` returns a list of training examples
for the specified color's moves only:
  {"fen": str, "move": LegalMove dict, "target": float}
where target ∈ {+1, -1, 0} from `color`'s perspective (win / loss / draw).

The LegalMove dict comes from re-querying the API at each saved FEN and
matching the saved UCI; this is needed because encode_move() requires the
full LegalMove fields (from, to, waypoints, deployCount, etc.), and the UI
saves only the UCI string.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("chessckers_engine.demos")

GameRecord = dict[str, Any]
Example = dict[str, Any]

_FEN_TURN = re.compile(r"^[^\s\[]+(?:\[[^\]]*\])?\s+([wb])\b")


class _Mover(Protocol):
    def new_game(self, fen: str | None = None) -> dict[str, Any]: ...


def load_games(path: str | Path) -> list[GameRecord]:
    """Read JSONL and return the parsed game records."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def filter_games_for_color(games: list[GameRecord], color: str) -> list[GameRecord]:
    """Keep only games where the specified color was human-played AND the game finished."""
    out = []
    for g in games:
        controllers = g.get("controllers") or {}
        if controllers.get(color) != "player":
            continue
        if g.get("outcome") in (None, "incomplete"):
            continue
        out.append(g)
    return out


def _turn_from_fen(fen: str) -> str:
    m = _FEN_TURN.match(fen)
    if not m:
        raise ValueError(f"unrecognized FEN: {fen!r}")
    return "white" if m.group(1) == "w" else "black"


def _target_for_color(outcome: str, color: str) -> float:
    if outcome == "draw":
        return 0.0
    return 1.0 if outcome == color else -1.0


def extract_examples(
    games: list[GameRecord],
    color: str,
    client: _Mover,
) -> list[Example]:
    """For each finished game where `color` was human-played, emit one example
    per move that color made, with target = +1/-1/0 from `color`'s perspective.
    Each example carries the full LegalMove dict so encode_move() can be applied."""
    examples: list[Example] = []
    for game in games:
        outcome = game["outcome"]
        target = _target_for_color(outcome, color)
        history = game.get("history") or []
        for entry in history:
            fen = entry["fen"]
            uci = entry["uci"]
            try:
                turn = _turn_from_fen(fen)
            except ValueError:
                continue
            if turn != color:
                continue
            # Re-query legal moves at this FEN to find the matching dict.
            try:
                state = client.new_game(fen)
            except Exception as e:  # noqa: BLE001
                log.debug("skipping fen=%s due to API error: %s", fen[:40], e)
                continue
            legal = state.get("legalMoves") or []
            match = next((m for m in legal if m.get("uci") == uci), None)
            if match is None:
                log.debug("skipping (fen, uci)=(%s, %s) — no matching legal move", fen[:40], uci)
                continue
            examples.append({"fen": fen, "move": match, "target": float(target)})
    return examples
