# Run 26 — mate_bench: Gumbel S2 (Sequential Halving @ 64v) vs run-25 control

> Tests Gumbel AlphaZero STAGE 2 — Gumbel-top-m root sampling + Sequential Halving
> visit allocation (Danihelka et al. 2022) — at a 12.5× smaller visit budget
> (`--visits=64` vs 800). Design mirrors run 25's gates-off 2+2 exactly: the two
> **control trials are run 25's arm A** (A1/A2, seeds 0–1: V@90% = **147.2M / 187.2M**,
> gates-off, no PCR, `3688a2a`-era engine) and this run adds only the S2 arm
> (2 trials, seeds 0–1). Metric: search visits to MATE-crossing (`bench_visits.py`,
> window 1000, threshold 0.9, self-play-only).

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `run26_e8d8_gumbelS2_bench` |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (compiled `board.cc`, unchanged) |
| Arch / trainer | v5 c64/b6, Adam 1e-3, `improved` policy target, pure z, cold, EMA 0.99, publish 400, seeds 0–1 — identical to run-25 arms |
| Self-play params | `--visits=64 --gumbel-sh=true --gumbel-m=16` and NOTHING else (no Dirichlet flags, no temperature flags — Gumbel root perturbation is the exploration mechanism and the SH winner is the played move) |
| Gates | **disabled** (`matches.disabled=true`, carried over from run 25) → league/panel inert, every upload auto-promotes |
| Engine | fork `chessckers-port` + Gumbel S2 commit (on top of `4545d43` slim-edge+memo) |
| Fleet box | vast `44287736` (RTX 3060) |
| Control | run 25 gates-off arm A stamps + tars (`engine/weights/run25-bench-artifacts/`) — NOT re-run |

## Hypothesis / decision rules (pre-committed)

- **H1 (the S2 bet):** SH at n=64 preserves enough policy-improvement signal that
  MATE-crossing arrives at **fewer total search visits** than the 800v control
  (arm A median 167.2M, spread 147–187M). The mechanism: 64v games are ~12.5×
  cheaper per move, so the trainer sees ~an order of magnitude more games/positions
  per visit; Gumbel's improved-policy target is explicitly designed to stay a
  policy improvement at tiny n.
- **Decisive win:** both S2 trials cross below 147M (the control's best seed).
  **Decisive loss:** both above 187M (worst control seed) or a 10h DNF.
  In between: seed-paired reads + games/plies texture, verdict "suggestive, n=2".
- DNF (10h) counts against the arm. Babysitter restarts logged, trial stands.
- Visit accounting: no PCR → every ply is a full search → V = records × 64
  (`bench_visits.py` reads `--visits` from the DB trainParams; verify
  `full_visits: 64, pcr_full_prob: 1.0` in its output before trusting V).
- Caveats accepted upfront: control ran on `3688a2a`, S2 arm runs on the newer
  slim-edge+memo+S2 engine (run 25 verified slim-edge semantically neutral); the
  temperature/noise exploration mechanism differs BY DESIGN (that IS Gumbel S2);
  wall-clock remains tenant-noise-dominated — visits is the metric.

## Design delta (vs run 25 arm A)

Gumbel S2 implemented in the fork (`src/search/classic/`): when `--gumbel-sh` is on,
root child selection ignores PUCT/Dirichlet/temperature. At search start the root
samples g(a)+logP(a) per edge (Gumbel-top-m, m=16 default), then a Sequential
Halving schedule allocates the remaining visit budget (VisitsStopper bound −
initial tree visits, so tree reuse Just Works) in phases — 16×1 → 8×2 → 4×4 → 2×8
at n=64 — re-ranking survivors between phases by g + logP + σ(q̂) with the same
σ convention as the S1 improved-policy target (c_visit=50, c_scale=0.1, min-max q).
The played move is the final SH winner (root exploration = the Gumbel draw itself).
Non-root selection stays PUCT (documented deviation from the paper; subtrees at
n=64 are tiny). Phase advancement waits for in-flight visits to back up (root
collision + batch flush). S1 improved-policy chunk target unchanged. UCI and
default-flag selfplay byte-path unchanged (flag defaults off).

- `params.{h,cc}`: `--gumbel-sh` (bool, default false), `--gumbel-m` (int, default 16)
- `search.{h,cc}`: `SetGumbelVisitBudget`, `GumbelMaybeInit`/`GumbelPickRootChild`/
  `GumbelBestCandidate`, root-branch in `PickNodesToExtendTask`, SH-winner override
  in `EnsureBestMoveKnown`, Dirichlet skipped under gumbel
- `selfplay/game.cc`: budget handoff after Search construction
- `lczero-server` bootstrap: env-driven `VISITS` / `GUMBEL_SH` / `GUMBEL_M` →
  trainParams (S2 arm emits ONLY the three S2 flags)

## Log

- `07-24` Implemented + verified on Mac (Metal): 4-game smoke at 64v — npm 64.4
  (budget consumed exactly), varied openings across games (Gumbel stochasticity
  live), all three outcomes observed; SH winner's root visit share ≈ 0.27–0.28
  (= 15–17/63, textbook schedule). Training chunks decode with **PyVariant oracle
  parity on every record** (192 records / 2 games), improved_policy present +
  normalized. Rules battery green (8000-position parity corpus, rules scenarios,
  FEN invariants — 0 mismatches). UCI default path sane (`bestmove` at 128 nodes,
  gumbel off). Observation for the file: `--temperature=1.0 --tempdecay-moves=15`
  yields 0 games on Mac/Metal at 64v on the PRISTINE `4545d43` binary too —
  pre-existing local quirk (suspect: NDEBUG `assert(sum)` no-op path in
  `GetBestRootChildWithTemperature`), not an S2 regression, not a box blocker
  (run 25 ran those flags for 8.7k games); S2 bypasses temperature entirely.

## Result

<pending>
