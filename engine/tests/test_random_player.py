import random

from chessckers_engine.random_player import pick_random


def test_empty_returns_none() -> None:
    assert pick_random([]) is None


def test_singleton_returns_only_element() -> None:
    only = {"uci": "e2e4"}
    assert pick_random([only]) is only


def test_uniform_covers_all_elements() -> None:
    moves = [{"uci": f"a{i}"} for i in range(5)]
    rng = random.Random(0)
    seen = {pick_random(moves, rng=rng)["uci"] for _ in range(500)}
    assert seen == {m["uci"] for m in moves}
