# Run 8 — forced bottom-N charge demotion (rule change)

> **PLANNED / PARKED** (drafted 2026-06-26 to capture the idea while [run 6](run6.md) trains).
> Unlike run 7 (a config knob), this is a **variant rule change** — it touches the spec, both
> rules implementations (PyVariant oracle + the fork's C++ copy), and the move encoding. Treat
> it as a new variant generation. Independent of run 7's visits change — could be combined into
> one run or sequenced.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V6_e8d8_botcharge` (proposed; arch/rule bump → likely a new tag) |
| Start FEN | e8/d8 (inherit run 6 — confirm at launch) |
| Arch | inherit run 6 — TBD |
| Optimizer / visits | inherit run 6 (and optionally run 7's visits=100) — TBD |
| **Rule change** | **Charge of *d* squares demotes the bottom *d* Kings (forced); no player choice** |
| Status | planned (rule change — needs spec + dual-impl edits + parity re-validation) |

## Hypothesis

Today a charge of distance *d* from a tower with `n_kings > d` enumerates **`C(n_kings, d)`
separate legal moves** — one per choice of which Kings to demote (`moves_black.py:1088`). Forcing
the demotion to the **bottom *d* Kings** collapses each (from→to) charge to a **single move**,
removing that combinatorial fan-out from Black's legal-move set and MCTS branching.

The bet: this **trains the net faster** (smaller move set / less branching / sharper visit
distributions per node) while **barely losing real-world strength**, because demoting from the
bottom **preserves the top Kings** — i.e. keeps the tower King-top and fully mobile, which is
usually the strategically preferred choice anyway. The discarded choices (demoting a higher King)
are rare and marginal.

**Success criterion:** measurable drop in mean Black legal-move count / branching on charge-heavy
positions, faster convergence (games and wall-clock) vs the choice-version baseline, with strength
(net-vs-net, or vs run 6's converged net on a fixed eval) **not materially worse**.

## Rule change (precise)

Spec §3C.3 today: *"If the tower has more Kings than the cost requires, the player picks which
ones to demote (1-indexed from bottom)."* → Change to: *"The bottom *d* Kings are demoted
(positions 1..*d* from the bottom); there is no choice."* A demoted King still becomes a
`Stone(hasMoved=true)`; cost, path-capture, ram, rim-overshoot, and mandate rules are unchanged.
The `{a,b,…}` demoted-King notation suffix becomes obsolete (the choice is gone).

## Landing sites (implementation checklist)

A move-gen rule change must stay in sync across both oracles + spec + encoding (see CLAUDE.md
"Rule-change landing sites"):

1. **Spec** — `chessckers.md` §3C.3 (rewrite "Choice of demoted Kings"), and the §3C **Notation**
   `{a,b,…}` suffix (drop it). Bump version note.
2. **PyVariant oracle** — `engine/chessckers_engine/variant_py/moves_black.py`:
   `black_charge_moves` (stop enumerating demotion combinations; emit one move demoting bottom *d*)
   and `_apply_charge` (~L924-927; `chosen = bottom-d` instead of `move["demotedKings"]`). The
   `demotedKings`/`demotionsRequired` move fields become derived, not chosen.
3. **Fork C++ rules copy** — `../akshay-chessckers-0/src/chessckers/` (charge gen in `movegen.hpp`)
   — mirror exactly; this is the production player.
4. **Encoding / policy** — `encoding.py` move features (`is_ortho`, `demotions_required`) + however
   charges index into the policy move space. **Verify first** whether the policy currently
   distinguishes demotion choices at all — the 240-dim move vector has `demotions_required` but no
   "which Kings" field, so the choices may already collide in feature space; if so, the net can't
   even express the choice today, which *strengthens* the "barely loses strength" argument and
   means the main win is MCTS branching, not policy-head size.
5. **Notation / display** — `watch_game.py` / `render_board.py` charge rendering (drop the
   demoted-King suffix).
6. **Parity corpus** — `corpus/rules_scenarios.jsonl` charge scenarios; re-validate PyVariant↔fork
   parity after the change (the dual-impl must agree).

## Watch-for

- **Parity is non-negotiable** — PyVariant and the fork must produce identical charge moves after
  the change, or self-play data and the oracle diverge. Re-run the parity corpus.
- **Strength check** — confirm the bet: play the bottom-N net vs run 6's choice-version net (fixed
  eval / gauntlet). "Barely loses" must be *measured*, not assumed.
- **Encoding collision** (point 4) — settle whether choices were ever distinguished; it changes
  where the speedup actually comes from.

## Log

- `06-26` Idea parked: force charge demotion to bottom *d* Kings (no choice) to kill the
  `C(n_kings,d)` charge fan-out. Rule change — spec + PyVariant + fork + encoding + parity.
  Independent of run 7 (visits); combinable.

## Result

<leave empty — fill once implemented + run; report branching reduction + strength delta vs the choice-version>
