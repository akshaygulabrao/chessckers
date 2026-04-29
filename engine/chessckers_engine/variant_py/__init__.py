"""Pure-Python implementation of the Chessckers variant.

Replaces the scalachess HTTP server for the engine's hot path. The public
surface is `PyVariantClient`, which mirrors the
`chessckers_engine.server_client.ServerClient` API so it can be dropped in
without changes elsewhere.

Implementation status (this is being built incrementally):
- [ ] FEN parser (board + bracket overlay) — `state.py`
- [ ] White move generation (chess via `python-chess`) — `moves_white.py`
- [ ] Black move generation (Chessckers swarm/chain logic) — `moves_black.py`
- [ ] State transition / make_move
- [ ] Win condition detection (king capture, Black elimination, max plies)
- [ ] Hop-based chain step / chain end APIs

See `tests/test_pyvariant_diff.py` for the differential test harness that
compares this implementation against scalachess on identical FENs.
"""

from chessckers_engine.variant_py.client import PyVariantClient  # noqa: F401
