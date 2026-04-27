import random
from typing import Any

LegalMove = dict[str, Any]


def pick_random(legal_moves: list[LegalMove], rng: random.Random | None = None) -> LegalMove | None:
    if not legal_moves:
        return None
    return (rng or random).choice(legal_moves)
