"""Regression: per spec §5, every Black move type EXCEPT charge promotes
all Stones in the moving stack when the destination is on rank 1.

Bug history (2026-05-10): the spectator viewer surfaced ~118 positions
where Black Stones (`s`/`S`) sat on rank 1 unpromoted. Root cause:
`_apply_quiet_or_sprint` and `_apply_deploy` in moves_black.py applied
the move via `_move_full_tower` without ever calling `_promote_all_stones`.
Capture hops/chains and (correctly) charges were unaffected.
"""

from chessckers_engine.variant_py import PyVariantClient


def test_quiet_diagonal_to_rank1_promotes():
    c = PyVariantClient()
    fen = "k7/8/8/8/8/8/2p5/4K3[c2:s] b - - 0 1"
    out = c.make_move(fen, "c2b1")
    assert "b1:k" in out["fen"], f"expected b1:k, got {out['fen']}"


def test_deploy_to_rank1_promotes():
    c = PyVariantClient()
    fen = "k7/8/8/8/8/8/2p5/4K3[c2:ss] b - - 0 1"
    out = c.make_move(fen, "c2b1[1]")
    assert "b1:k" in out["fen"], f"expected b1:k, got {out['fen']}"
    # remainder at c2 stays as a single stone (not on rank 1, no promotion)
    assert "c2:s" in out["fen"]


def test_quiet_diagonal_off_rank1_does_not_promote():
    c = PyVariantClient()
    fen = "k7/8/8/8/2p5/8/8/4K3[c4:s] b - - 0 1"
    out = c.make_move(fen, "c4b3")
    assert "b3:s" in out["fen"], f"expected b3:s, got {out['fen']}"


def test_charge_to_rank1_does_not_promote():
    """Per spec §3C: 'Landing on rank 1 does not promote any Stones to
    Kings; only diagonal moves promote.'"""
    c = PyVariantClient()
    fen = "k7/8/8/8/8/7p/8/4K3[h3:kk] b - - 0 1"
    out = c.make_move(fen, "h3h1")
    # Charge demotes both kings (n_kings == d == 2) → SS, not kk and not promoted.
    assert "h1:SS" in out["fen"], f"expected h1:SS, got {out['fen']}"


def test_merge_onto_rank1_promotes_existing_stack():
    """Per spec §5: 'every Stone in the tower is promoted to a King.' The
    'tower' after a merge includes the existing pieces at dest, so they
    must promote too."""
    c = PyVariantClient()
    fen = "k7/8/8/8/8/8/2p5/1p2K3[b1:s,c2:s] b - - 0 1"
    out = c.make_move(fen, "c2b1")
    assert "b1:kk" in out["fen"], f"expected b1:kk, got {out['fen']}"
