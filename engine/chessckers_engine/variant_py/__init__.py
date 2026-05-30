"""Pure-Python implementation of the Chessckers variant — the engine's
authoritative game logic: FEN/overlay parsing (`state.py`), White move
generation via python-chess (`moves_white.py`), Black swarm/chain move
generation (`moves_black.py`), make_move, and win detection.

The public surface is `PyVariantClient` (`client.py`), an in-process API.
A Rust extension (`chessckers_movegen`) accelerates the Black move-gen hot
path and is kept equivalent to the Python.
"""

from chessckers_engine.variant_py.client import PyVariantClient  # noqa: F401
