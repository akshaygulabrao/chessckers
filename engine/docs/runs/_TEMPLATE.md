# Run N — <short name, e.g. e8d8 KK-vs-K>

> Copy this file to `runN.md`, fill the Identity table from `cc fresh-run`'s stdout,
> write a one-line Hypothesis, then append to the Log as the run progresses.
> Keep it short — this is a ledger entry, not a design doc.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V?_<tag>` |
| Start FEN | `... w - - 0 1` (compiled into fork `src/chess/board.cc` `kStartposFen`) |
| Arch | SE-ResNet gather head, c_filters=?, n_blocks=?, ~?K params, tag `v?` |
| Optimizer | Adam/SGD, lr=? |
| Key commit / branch | `<sha>` |
| Fleet box | vast id `<id>` |
| Started | YYYY-MM-DD |
| Status | active / abandoned / superseded → runM |

## Hypothesis

<1–2 lines: what this run tests and why this start position. What would count as success?>

## Design delta

What changed vs the previous run, with commit refs. If the **encoding/arch** changed,
link to [`../encoding-reference.md`](../encoding-reference.md) and note the delta here
rather than re-pasting channel tables.

- ...

## Log

Append-only, dated. One bullet per notable event (launch, revert, measurement, pivot).

- `MM-DD` ...

## Decision rules (pre-committed)

Pre-decide triggers so mid-run judgment calls don't drift. Fill before launch; amend only with a dated Log entry.

- **LR drop** — trigger: discriminative anchor (`seed13`) gains < +40 Elo over 3 consecutive anchor rows (~24h) with ingest healthy (trainer step-rate normal, buffer not starved). Confirm headroom via a deep-vs-shallow visits match (800v vs 128v, ≥ 40g; real search-scaling ⇒ headroom exists). If confirmed: `cc restart-trainer <LR × 0.3>`.
- **Plateau definition** — `seed13` flat ± noise (95% CI overlaps zero gain) for ≥ 3 rows AND `cc champs` round-robin spans < 100 Elo across the same stretch. Both instruments must agree; either alone is insufficient.
- **RPS check cadence** — daily `cc champs` audit (via `install_monitor_crons.sh`). A>B>C>A cycles beyond 1σ multiple-comparison noise over ≥ 3 consecutive audits = RPS signature → investigate league fraction or pool spacing.
- **Anchor rotation** — pin a new rung (e.g. the current best at saturation) when `seed13` saturates and loses discriminative power. Note the new anchor in the Log with the row index where it was added.
- **Abandon / pivot** — anchor trajectory clearly slower than the paired control run for ≥ 48h with plateau confirmed → write a Result entry and open a new run doc.

## Result

<Final outcome once the run ends: did it converge? best net path? why pivoted/reverted?
link to the successor run. Leave empty while active.>
