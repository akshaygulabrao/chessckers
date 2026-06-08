"""Phase 3B-3 (orchestrator+engine, true lc0): the local-job ENGINE mode — NO HTTP.

`cpp.run_jobs_local(run_dir, ...)` is the engine half: fleet_client.py (the orchestrator)
mints jobs into run_dir/jobs/ and syncs run_dir/weights.bin; this loop claims each job
(atomic rename, mirroring selfplay_worker_async._claim_job), plays it with the NATIVE
engine, and writes the output to run_dir/buffer/ (train chunk) or run_dir/match_out/
(gate outcome) — exactly the files the Python worker wrote, for the orchestrator to ship.

Gate (stronger than tensor equality): each train chunk in buffer/ is FIELD-identical to
`decode_chunk(play_game_chunk(seed=base_seed+k))` for the k-th train job, and each gate
outcome equals `play_match_game(...)` with the same per-game Dirichlet seed — i.e. the
local-job plumbing dispatches + seeds deterministically and reuses the Phase-2/3A/4b
primitives whose cross-language parity is already proven.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import torch

from chessckers_engine.model import ChesskersScorer, ChesskersScorerV2
from chessckers_engine.native_net import export_state_dict
from chessckers_engine.training_chunk import decode_chunk

cpp = pytest.importorskip("chessckers_cpp")

SEED_FEN = "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1"
CC_SELFPLAY = Path(__file__).resolve().parent.parent / "cpp" / "build" / "cc_selfplay"

# Train-job play params (job JSON uses "sims"; parse_job maps it to n_sims). The oracle
# play_game_chunk takes n_sims directly — keep the two aligned.
TRAIN_PARAMS = dict(sims=24, c_puct=1.5, temperature=1.0, temp_cutoff_plies=8, max_plies=40,
                    dirichlet_alpha=0.3, dirichlet_eps=0.25)
MATCH_PARAMS = dict(sims=16, c_puct=1.5, dir_alpha=0.3, dir_eps=0.25, max_plies=24)


def _export_bin(model, path: Path) -> Path:
    export_state_dict(model.state_dict(), str(path))
    return path


def _chunk_oracle_params() -> dict:
    """play_game_chunk kwargs equivalent to TRAIN_PARAMS (sims -> n_sims)."""
    p = dict(TRAIN_PARAMS)
    p["n_sims"] = p.pop("sims")
    return p


def test_run_jobs_local_train_and_match_parity(tmp_path):
    torch.manual_seed(0)
    run_dir = tmp_path / "run"
    jobs = run_dir / "jobs"
    jobs.mkdir(parents=True)

    # weights.bin = the train net the engine self-plays with (mtime hot-reload).
    train_model = ChesskersScorerV2(n_blocks=2, n_tf_blocks=1, n_heads=4, tf_ff_mult=2)
    _export_bin(train_model, run_dir / "weights.bin")
    train_net = cpp.ChesskersNet(str(run_dir / "weights.bin"))  # oracle net (same bytes)

    # Two gate nets the orchestrator would have pre-fetched to local .bin paths.
    cand_bin = _export_bin(ChesskersScorer(), tmp_path / "cand.bin")
    opp_bin = _export_bin(ChesskersScorer(), tmp_path / "opp.bin")
    cand_net = cpp.ChesskersNet(str(cand_bin))
    opp_net = cpp.ChesskersNet(str(opp_bin))

    # Mint jobs: 0,1,2 = train; 3,4 = match. Single digits so the lexicographic claim
    # order is the numeric order (the engine globs jobs/*.json sorted).
    for i in range(3):
        (jobs / f"{i}.json").write_text(json.dumps(
            {"type": "train", "bin_sha": "ignored-uses-weights.bin", "params": TRAIN_PARAMS}))
    match_specs = [
        {"match_id": 7, "opponent": "best", "cand_white": True},
        {"match_id": 8, "opponent": "net-3", "cand_white": False},
    ]
    for n, spec in enumerate(match_specs):
        (jobs / f"{n + 3}.json").write_text(json.dumps({
            "type": "match", "seed": SEED_FEN,
            "cand_bin": str(cand_bin), "opp_bin": str(opp_bin),
            "params": MATCH_PARAMS, **spec,
        }))

    base_seed = 5
    handled = cpp.run_jobs_local(str(run_dir), SEED_FEN, worker_id=400, machine="testbox",
                                 base_seed=base_seed, max_jobs=5, seq_start=1)
    assert handled == 5

    # --- train: 3 chunks, field-identical to the seeded native game ---
    pkls = sorted((run_dir / "buffer").glob("*.pkl"))
    assert len(pkls) == 3
    op = _chunk_oracle_params()
    for k, pkl in enumerate(pkls):  # buffer seq order == train-claim order == train_count k
        oracle = decode_chunk(cpp.play_game_chunk(
            cpp.parse_fen(SEED_FEN), train_net, seed=base_seed + k, **op))
        assert decode_chunk(pkl.read_bytes()) == oracle
        meta = json.loads(Path(str(pkl) + ".meta").read_text())
        assert meta["worker_id"] == 400 and meta["machine"] == "testbox"
        assert meta["outcome"] in ("white", "black", "draw")

    # --- match: 2 outcomes, identical to play_match_game with the per-game Dirichlet seed ---
    mp = MATCH_PARAMS
    for m, (jseq, spec) in enumerate(zip(("3", "4"), match_specs)):
        out = json.loads((run_dir / "match_out" / f"{jseq}.json").read_text())
        dseed = base_seed + m * (mp["max_plies"] + 1)
        white = cand_net if spec["cand_white"] else opp_net
        black = opp_net if spec["cand_white"] else cand_net
        exp_outcome, _ = cpp.play_match_game(white, black, SEED_FEN, mp["sims"], mp["c_puct"],
                                             mp["dir_alpha"], mp["dir_eps"], mp["max_plies"], dseed)
        assert out["outcome"] == exp_outcome
        assert out["match_id"] == spec["match_id"] and out["opp"] == spec["opponent"]
        assert out["cand_white"] is spec["cand_white"] and out["seed"] == SEED_FEN

    # --- the queue drained cleanly: no unclaimed jobs, no orphaned claims/tmp left ---
    assert not list(jobs.glob("*.json"))
    assert not list(jobs.glob("*.c400"))
    assert not list((run_dir / "buffer").glob("*.tmp"))


def test_run_jobs_local_stops_on_stop_file(tmp_path):
    """A STOP sentinel makes the (unbounded) engine exit immediately, handling nothing."""
    run_dir = tmp_path / "run"
    (run_dir / "jobs").mkdir(parents=True)
    (run_dir / "STOP").touch()
    # A queued job that must NOT be played because STOP is checked first.
    (run_dir / "jobs" / "0.json").write_text(json.dumps({"type": "train", "params": TRAIN_PARAMS}))

    handled = cpp.run_jobs_local(str(run_dir), SEED_FEN, base_seed=0, max_jobs=0)
    assert handled == 0
    assert (run_dir / "jobs" / "0.json").exists()  # untouched (never claimed)


@pytest.mark.slow
def test_cc_selfplay_jobs_local_executable(tmp_path):
    """The standalone binary (NO Python) the orchestrator shells out to: run it in
    --jobs-local mode against a file-only run-dir, feed it train jobs, and confirm it
    plays them into buffer/ and exits on STOP."""
    if not CC_SELFPLAY.exists():
        pytest.skip("cc_selfplay not built (run cpp/build.sh)")
    torch.manual_seed(0)
    run_dir = tmp_path / "run"
    jobs = run_dir / "jobs"
    jobs.mkdir(parents=True)
    _export_bin(ChesskersScorer(), run_dir / "weights.bin")
    for i in range(2):
        (jobs / f"{i}.json").write_text(json.dumps({"type": "train", "params": TRAIN_PARAMS}))

    proc = subprocess.Popen(
        [str(CC_SELFPLAY), "--jobs-local", "--run-dir", str(run_dir), "--worker-id", "402",
         "--machine", "exebox", "--seed", "1", "--start-fen", SEED_FEN],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 60
        while time.time() < deadline and len(list((run_dir / "buffer").glob("*.pkl"))) < 2:
            time.sleep(0.5)
        (run_dir / "STOP").touch()  # unbounded loop -> exit
        out, err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    pkls = sorted((run_dir / "buffer").glob("*.pkl"))
    assert len(pkls) == 2
    for pkl in pkls:
        assert decode_chunk(pkl.read_bytes())
        assert json.loads(Path(str(pkl) + ".meta").read_text())["machine"] == "exebox"
