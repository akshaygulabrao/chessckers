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

## Result

<Final outcome once the run ends: did it converge? best net path? why pivoted/reverted?
link to the successor run. Leave empty while active.>
