"""Regression test for the arch-sidecar atomic-write naming bug.

The fleet save sites write a checkpoint to `<name>.pt.tmp` (via train_az.save_checkpoint,
which also drops a `<path>.arch.json` sidecar) then `os.replace` the `.pt` into place.
The bug: the sidecar was written against the *temp* name (`weights.pt.tmp.arch.json`) and
the replace moved only the `.pt`, orphaning the sidecar — so `weights.pt.arch.json` never
existed and `best.pt` (copied via _atomic_copy) got no sidecar at all. The fleet itself is
unaffected (it reads arch from fleet.env/CLI), but checkpoints.load_scorer / the eval
gauntlet rely on the sidecar to rebuild a V2/V3 net off a bare `.pt`.

These tests pin: (1) _publish lands the sidecar at the FINAL name (not `.pt.tmp.arch.json`),
(2) _atomic_copy carries the sidecar to the destination, (3) load_scorer round-trips to v2.
"""

import json

import torch

from chessckers_engine import checkpoints
from chessckers_engine.fleet_arena import _atomic_copy
from chessckers_engine.model import build_model
from chessckers_engine.train_continuous import _publish


def _tiny_v3(seed: int = 0):
    torch.manual_seed(seed)
    # Small but V3-shaped: gather head + a transformer block (n_heads must divide c_filters).
    return build_model(
        version="v2", d_hidden=32, c_filters=16, n_blocks=1,
        n_tf_blocks=1, n_heads=4, tf_ff_mult=2,
    )


def test_publish_lands_sidecar_at_final_name(tmp_path):
    model = _tiny_v3()
    weights = tmp_path / "weights.pt"
    _publish(model, weights)

    assert weights.exists()
    good = tmp_path / "weights.pt.arch.json"
    orphan = tmp_path / "weights.pt.tmp.arch.json"
    assert good.exists(), "sidecar not written at the final name"
    assert not orphan.exists(), "orphaned temp-named sidecar left behind"
    assert not (tmp_path / "weights.pt.tmp").exists(), "temp .pt not cleaned up"

    arch = json.loads(good.read_text())
    assert arch["version"] == "v2"
    assert arch["n_tf_blocks"] == 1  # the V3 transformer recipe survived the round-trip


def test_atomic_copy_carries_sidecar(tmp_path):
    model = _tiny_v3()
    weights = tmp_path / "weights.pt"
    _publish(model, weights)

    best = tmp_path / "best.pt"
    _atomic_copy(weights, best)
    best_arch = tmp_path / "best.pt.arch.json"
    assert best_arch.exists(), "_atomic_copy dropped the arch sidecar"
    assert json.loads(best_arch.read_text()) == json.loads((tmp_path / "weights.pt.arch.json").read_text())
    assert not (tmp_path / "best.pt.arch.json.tmp").exists()


def test_load_scorer_round_trips_v3_via_sidecar(tmp_path):
    model = _tiny_v3()
    weights = tmp_path / "weights.pt"
    _publish(model, weights)
    best = tmp_path / "best.pt"
    _atomic_copy(weights, best)

    # Without the sidecar this would fall back to a v1 ResNet and shape-mismatch.
    loaded = checkpoints.load_scorer(best)
    assert loaded.VERSION == "v2"
    assert sum(p.numel() for p in loaded.parameters()) == sum(p.numel() for p in model.parameters())
