"""Evaluation harness for the Chessckers engine.

Plays full games between two named pickers via the live API and reports
W/D/L from White's perspective.

CLI:
    uv run python -m chessckers_engine.evaluate \
        --white nn --black material --games 50

Status mapping (per `Chessckers.scala` and chessckers.md §5):
- 'mate'        → Black wins (White king captured)
- 'variantEnd'  → White wins (Black eliminated or Black stalemated)
- 'stalemate'  → draw (only triggers when White has no legal moves)
- max_plies hit → draw (safety bound; not a real outcome)

The harness only orchestrates pickers + the API; it has no understanding
of variant rules itself, so it stays correct as long as the server's
status field is honest.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Callable

from chessckers_engine.checkpoints import latest_checkpoint
from chessckers_engine.runtime import build_pickers
from chessckers_engine.variant_py import PyVariantClient

log = logging.getLogger("chessckers_engine.evaluate")

GameState = dict[str, Any]
LegalMove = dict[str, Any]
Picker = Callable[[GameState], LegalMove | None]

Outcome = str  # "white" | "black" | "draw"


def play_game(
    white_picker: Picker,
    black_picker: Picker,
    client: Any,  # anything implementing new_game / make_move
    max_plies: int = 400,
) -> Outcome:
    state = client.new_game()
    ply = 0
    while not state.get("status") and ply < max_plies:
        picker = white_picker if state["turn"] == "white" else black_picker
        chosen = picker(state)
        if chosen is None:
            break
        try:
            state = client.make_move(state["fen"], chosen["uci"])
        except Exception as e:
            legal = state.get("legalMoves") or []
            legal_ucis = [m["uci"] for m in legal]
            in_list = chosen["uci"] in legal_ucis
            log.error(
                "make_move failed at ply %d (turn=%s, uci=%s, in_legalMoves=%s): %s\n"
                "  fen=%s\n  matching legalMoves entry: %s",
                ply, state["turn"], chosen["uci"], in_list, e, state["fen"],
                next((m for m in legal if m["uci"] == chosen["uci"]), "<not found>"),
            )
            return "draw"
        ply += 1
    return _state_to_outcome(state)


def _state_to_outcome(state: dict) -> Outcome:
    """`winner` is authoritative; status-only fallback is for test stubs.
    See `selfplay_az._outcome_from_state` for the rationale (Black king
    capture returns status='variantEnd', winner='black')."""
    winner = state.get("winner")
    if winner == "white":
        return "white"
    if winner == "black":
        return "black"
    status = state.get("status")
    if status == "mate":
        return "black"
    if status == "variantEnd":
        return "white"
    return "draw"


# Kept for any external callers; same fallback semantics as before.
def _status_to_outcome(status: str | None) -> Outcome:
    return _state_to_outcome({"status": status})


def evaluate(
    white_picker: Picker,
    black_picker: Picker,
    client: Any,
    n_games: int,
    max_plies: int = 400,
) -> dict[str, int]:
    counts = {"white": 0, "black": 0, "draw": 0}
    for i in range(n_games):
        outcome = play_game(white_picker, black_picker, client, max_plies=max_plies)
        counts[outcome] += 1
        log.info("game %d/%d -> %s  (running: W=%d B=%d D=%d)",
                 i + 1, n_games, outcome, counts["white"], counts["black"], counts["draw"])
    return {**counts, "games": n_games}


def format_results(white_name: str, black_name: str, counts: dict[str, int]) -> str:
    n = counts["games"]
    w, b, d = counts["white"], counts["black"], counts["draw"]
    score_white = (w + 0.5 * d) / n if n else 0.0
    return (
        f"\n  {white_name} (W) vs {black_name} (B), {n} games\n"
        f"    White wins: {w:3d}  ({100 * w / max(n, 1):5.1f}%)\n"
        f"    Black wins: {b:3d}  ({100 * b / max(n, 1):5.1f}%)\n"
        f"    Draws:      {d:3d}  ({100 * d / max(n, 1):5.1f}%)\n"
        f"    Score (W's perspective): {score_white:.3f}\n"
    )


def main() -> int:
    from chessckers_engine.runtime import setup_logging
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--white", required=True, choices=["random", "material", "mcts", "nn"])
    p.add_argument("--black", required=True, choices=["random", "material", "mcts", "nn"])
    p.add_argument("--mcts-sims", type=int, default=100, help="MCTS simulations per move")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--model", default=None, help="Path to .pt weights for nn (default: auto-discovery)")
    p.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8080"))
    p.add_argument("--use-server", action="store_true",
                   help="Deprecated no-op: PyVariant is always used (scalachess server removed).")
    args = p.parse_args()

    model_path = args.model
    if not model_path:
        latest = latest_checkpoint()
        if latest is not None:
            model_path = str(latest)
            log.info("auto-selected latest checkpoint: %s", model_path)

    # PyVariant is the only client now (the scalachess HTTP server was removed).
    # --use-server / --api-url are accepted but ignored.
    client = PyVariantClient()

    pickers = build_pickers(client, model_path, log, mcts_sims=args.mcts_sims)
    if args.white not in pickers or args.black not in pickers:
        log.error("requested picker not available; available=%s", sorted(pickers))
        client.close()
        return 2

    counts = evaluate(pickers[args.white], pickers[args.black], client, args.games, max_plies=args.max_plies)
    print(format_results(args.white, args.black, counts))
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
