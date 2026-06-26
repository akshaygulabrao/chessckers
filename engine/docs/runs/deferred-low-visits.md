# Deferred idea — low self-play visits (800 → 100)

> **DEFERRED 2026-06-26** (was the planned run 7). After run 6 converged, the user judged that
> self-play visits "won't change very much," so this was dropped from the run sequence and the
> charge rule change (ex-run-8) took the run-7 slot. Kept here so the idea/knob isn't lost — pick
> it up later if a run wants a throughput lever. Knob: `bootstrap/main.go:43` `--visits` (or
> `CC_VISITS` on the client).

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V5_e8d8_v100` (proposed) |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (assume same as run 6 — confirm at launch) |
| Arch | inherit run 6 (SE-ResNet v5 c48/b5) — TBD |
| Optimizer | inherit run 6's winner (run 6 tests Adam 1e-3 vs run 5's 2e-2) — TBD |
| **Self-play visits** | **100** (run 6 / run 5 ran **800**) ← the change |
| Key commit / branch | TBD |
| Fleet box | TBD |
| Started | TBD (after run 6) |
| Status | planned |

## Hypothesis

Drop self-play search from **800 → 100 visits/move**. Each game becomes ~8× cheaper, so the
fleet generates far more games per unit **wall-clock**. For this near-solved e8/d8 mate, the
bet is that the network learns the mate faster from **sheer data volume** — many cheap mate
examples per hour — rather than needing deep per-move search to find it. Framed by the user:
the net should *remember* which moves lead to mate (lots of fast examples) instead of having
to *search-discover* it expensively each game; the relevant (mate) data accumulates far quicker.

**Success criterion:** reach Black-mate convergence in **less wall-clock time** than the
800-visit baseline (run 5 ≈ 38k games / ~8 days; run 6 TBD). Measure both **games-to-converge**
and **wall-clock-to-converge** — the bet only pays if higher throughput (games/hr) more than
offsets any increase in games-needed.

## Design delta

vs run 6 — **one change**:

- **Self-play `--visits` 800 → 100.** Seeded in the DB training-run params at
  `lczero-server/cmd/bootstrap/main.go:43` (`trainParams` JSON, the `--visits=800` entry).
  Edit there → takes effect on the next **cold bootstrap** (`reset_fleet.sh` / `cc fresh-run`).
  For a quick no-rebuild test, set `CC_VISITS=100` in the client env instead
  (`lczero-client/lc0_main.go:478` reads it; default 100 only applies when the server sends no
  `--visits`, which it does — so the env/bootstrap value is what actually binds).
- Everything else inherits run 6's winning config (start FEN, arch, LR, replay window, batch,
  temp/noise). No other change.

## Watch-for (the real tension)

The throughput win is not free — lower visits degrades search quality:

- **Noisier policy targets.** The AZ policy target *is* the MCTS visit distribution; at 100
  visits it's spread thin → less peaked / noisier than at 800. `CLAUDE.md` warns `n_sims ≥ 50`
  to avoid degenerate visit distributions — 100 clears that floor but sits closer to it.
- **Slower cold-start mate discovery.** Run 5 needed ~38k games at 800 partly because Black's
  mate line is deep; 100-visit search may discover it even more slowly (the
  `chessckers-monitoring-findings` notes already flagged 800 as *shallow* for Black's line). So
  it's plausible run 7 needs **more games** — the experiment is whether games/hr wins anyway.
- Anchor the comparison on **wall-clock**, and track games-to-first-Black-win + games-to-converge
  vs run 6/run 5, not raw win-rate alone (which isn't a strength signal without an anchor).

## Log

- `06-26` Idea parked: self-play visits 800→100 for throughput-driven faster learning. Knob =
  `bootstrap/main.go:43`. Awaiting run 6's outcome to set the inherited config (arch/LR).

## Result

<leave empty — fill once run 7 runs; compare wall-clock-to-converge vs run 6 (and run 5's ~38k games / ~8 days at 800 visits)>
