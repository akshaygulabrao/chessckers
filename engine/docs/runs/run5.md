# Run 5 — V5 e8/d8 KK-vs-K endgame

> Backfilled from the former `engine/docs/v5-training-run.md` plus the live-run
> observations previously kept only in Claude memory. The V5 design delta below is
> verbatim from that doc; the Log/Result are reconstructed and dated approximately.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_e8d8` (the e8/d8 campaign also ran under `V4_e8d8`) |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (two kk towers d8/e8, K on e1) |
| Arch | SE-ResNet gather head, c_filters=48, n_blocks=5, d_hidden=256, se_ratio=8, ~364K params, tag `v5` |
| Optimizer | Adam, lr=**2e-2** (set by accident — intended a lower LR; corrected to 1e-3 in [run 6](run6.md)) |
| Key commit / branch | `a872b5f` (optimizer revert to Adam), `c7a44e6` (V5 design doc) |
| Fleet box | vast (various across the campaign) |
| Started | ~2026-06 |
| Status | superseded → [run 6](run6.md) |

## Hypothesis

From `3kk3/.../4K3[d8:kk,e8:kk]`, Black can force mate (or at minimum a not-loss via
perpetual back-rank check). Search can't solve it — the win is ~mate-in-11+ with very
high branching — so the net must **learn** the conversion. Success = self-play converges
to the Black win and the net plays the mate at temperature 0.

## Design delta

What changed from V4 (encoding tables now live in
[`../encoding-reference.md`](../encoding-reference.md)):

1. **Input encoding → per-depth tower channels.** V4's 5 aggregate Black-tower channels
   (8–12: height/counts/top-markers) discarded piece *order* — `"kSs"` and `"skS"` looked
   identical. V5 replaces them with 5 per-depth channels (`8+d` = piece at stack depth `d`,
   bottom-to-top; `s=0.33`, `S=0.67`, `k=1.0`, `none=0`). No info lost; order now visible.
   Channel count unchanged.
2. **Tower height cap at 5.** Friendly merges that would exceed 5 are illegal (PyVariant +
   fork `movegen.hpp` + both FEN parsers). All `/24.0` denominators → `/MAX_TOWER_HEIGHT`.
3. **Optimizer → Adam** replacing V4's SGD+Nesterov (lr=0.02, momentum=0.9). Revert in
   `a872b5f`. This run ran at Adam **lr=2e-2** (set accidentally); [run 6](run6.md) corrects
   it to 1e-3 to test learning efficiency.
4. **Network tag** `.arch.json "version":"v5"`; trunk unchanged (`ChesskersScorerV2`).

Run parameters: replay window 400→4000 (α=0.75), min buffer 200, buffer cap 500k positions,
batch 1024, publish every 100 games, EMA 0.999, value discount 1.0 (WDL from moves-left head),
Q-ratio 0.5, replay factor 8×, self-play visits 800, temp 1.0 decay/15, Dirichlet ε=0.25 α=0.3.

Files changed for V5: spec `chessckers.md` (max-height rule); Python `encoding.py`,
`state.py`, `moves_black.py`, `model.py`, `train_continuous.py`; fork C++ `board.hpp`,
`encode.hpp`, `movegen.hpp`, `board.cc` (start FEN); bridge `trainer_bridge.py`.

## Log

- `~06-10` Tiny baseline (c16/b1 V4, ~104k params) on this start judged "extremely good
  training" — learned the conversion. Motivated the c48/b5 v5 scale-up.
- `06-13` Long stretch read as a stall (~97% White; Black never sustained the back-rank
  check / perpetual). Built diagnostics (`eval_history.py`, `gen_probe_suite.py`,
  `solve_endgame.py` box-shrink reference policy). Diagnosis split by depth: shallow skills
  saturated, the deep mate looked stuck.
- `06-13` **Resolved: undertrained, not stuck.** Self-play flipped at ~#10k games, then
  climbed to and **stabilized ~99.5% Black through ~38k games, where the run was stopped**.
  Net plays the mate at temp 0 (~mate-in-15). It just took ~10k games to break the
  asymmetric cold-start trap.
- `06-15`→`06-16` Campaign churned start positions: detour to d6/e6/f6 scale-up, reverted to
  e8/d8 + Adam 1e-3 cold wipe, then reverted again to d6/e6/f6 (White-to-move) — see run 6.

## Result

The e8/d8 KK-vs-K curriculum **converged**: Black learned the forced mate. Key lesson —
on a fixed asymmetric-win start, a flat early win-rate is **undertraining, not a stall**;
it needed only ~10k games (at Adam lr=2e-2) to escape the cold-start equilibrium — the run
was then left to run to ~38k games before being stopped. This run is the
baseline the d6/e6/f6 scale-up (run 6) builds on. Best-net/checkpoints lived in
`../lczero-server/trainer/run1/` (`weights.pt` + `iter-async-*.pt`).

> ⚠️ Oracle-dependent metrics that made this run measurable (box-shrink curriculum, probe
> "correct moves", self-play W/B balance as a strength signal) do **not** transfer to a run
> whose correct outcome is unknown. See run 6.
