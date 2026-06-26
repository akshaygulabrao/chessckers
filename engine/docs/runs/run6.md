# Run 6 — e8/d8 KK-vs-K, Adam LR ablation (2e-2 → 1e-3)

> **Active run.** Identical to [run 5](run5.md) except the Adam learning rate. This is a
> controlled A/B: keep everything else fixed so the LR effect is isolated. Fill in commit/box
> at launch and keep the Log current.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_e8d8` — consider a distinct tag (e.g. `V5_e8d8_lr1e3`) to keep run-5/run-6 game dirs separate |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (unchanged from run 5) |
| Arch | SE-ResNet gather head, c_filters=48, n_blocks=5, ~364K params, tag `v5` (unchanged) |
| Optimizer | Adam, lr=**1e-3** (run 5 ran 2e-2) |
| Key commit / branch | TBD |
| Fleet box | vast id TBD |
| Started | 2026-06-25 |
| Status | active |

## Hypothesis

Run 5 converged on this exact position (Black learns the forced mate) but ran Adam at **2e-2**
and took ~38k self-play games to break the cold-start trap, with an oscillatory result curve.
Dropping the LR to **1e-3** (everything else identical) should make learning **drastically more
efficient** — convergence to the Black forced-mate in materially fewer games and/or a smoother,
less oscillatory result trajectory.

**Success criterion:** at temperature 0 the net plays the e8/d8 Black mate, reached in
**notably fewer games than run 5's ~38k** (or with clearly more stable training dynamics).

## Design delta

vs [run 5](run5.md) — **one change only**:

- **Adam LR 2e-2 → 1e-3.** Set via the trainer LR env. **Plumbing verified end-to-end
  (2026-06-25):** `launch_trainer.sh:63` `LR="${LR:-1e-3}"` → `:101` `--lr "$LR"` →
  `trainer_bridge.py:240` forwards `--lr` verbatim → `train_continuous.py:431` (`--lr`
  default `1e-3`) → `:543` `Adam(model.parameters(), lr=args.lr)`. No layer drops/overrides
  it; the `_lr_at` schedule (`:548`) is a no-op unless `LR_WARMUP_STEPS`/`LR_DECAY_STEPS` > 0.
  So `LR=0.001` ⇒ Adam trains at exactly 0.001.
  - **Config-passing simplified (2026-06-25)** to remove the drift that likely caused run 5's
    wrong LR: `trainer_bridge.py` is now a transparent pass-through (forwarded hyperparams
    `default=None`, forwarded only when set), so **`train_continuous.py` is the single source
    of truth** for every default (it now also defaults `--arch-version` to `v2`). `cc
    restart-trainer 0.001` now actually sets the LR (passes it as the `LR` env var, which
    `launch_trainer.sh` reads — previously a positional no-op).
  - ⚠️ **Still must verify on the box:** the GPU box runs **rsync copies** of these scripts, so
    these fixes only take effect once synced there. run 5's `2e-2` most plausibly came from the
    box's `launch_trainer.sh` carrying a stale SGD-era `LR=0.02` default (committed default is
    `1e-3`). **At launch, confirm the effective LR from the trainer log** (`launch_trainer.sh`
    prints `lr=…`; `train_continuous` logs it too) rather than trusting the repo.
- Everything else held fixed: start FEN, c48/b5 `v5` arch, encoding
  ([`../encoding-reference.md`](../encoding-reference.md)), replay window 400→4000, batch 1024,
  self-play visits 800, temp/noise. No code change beyond the LR.
- **Start cold (random init)** so the LR comparison isn't contaminated by run-5 weights — the
  trainer warm-resumes from `weights.pt` by default, so force a fresh init (set `BASE=""`) and
  wipe prior state (`reset_fleet.sh`). Cold init is deterministic (seed 0) → the first
  `upload_network` may 400 "Network already exists" (benign SHA dedup).

## Log

- `06-25` Run defined: same e8/d8 start + arch as run 5, Adam **1e-3** (run 5 was 2e-2 by
  accident). To launch cold from random init for a clean LR A/B.

## Result

<leave empty while active — record games-to-convergence vs run 5's ~38k, and whether the
result curve is smoother>
