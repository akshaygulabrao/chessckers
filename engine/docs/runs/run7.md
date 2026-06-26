# Run 7 — forced bottom-N charge demotion (rule change)

> **IN IMPLEMENTATION** (2026-06-26). This is the renumbered ex-run-8: the low-self-play-visits
> idea (old run 7) was **deferred** — see [deferred-low-visits.md](deferred-low-visits.md) — on
> the judgment that visits won't move the needle much, so the charge rule change took the run-7
> slot. Unlike a config knob, this is a **variant rule change** — it touches the spec, both rules
> implementations (PyVariant oracle + the fork's C++ copy), and the move encoding. Treat it as a
> new variant generation.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `V6_e8d8_botcharge` (proposed; rule bump → new tag) |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (e8/d8, from run 6) |
| Arch | SE-ResNet v5 c48/b5 ~364K (from run 6) |
| Optimizer | Adam **lr=1e-3** (run 6's winner) |
| Self-play visits | 800 (visits ablation deferred — see deferred-low-visits.md) |
| **Rule change** | **Charge of *d* squares demotes the bottom *d* Kings (forced); no player choice** |
| Status | in implementation (spec + dual-impl edits + parity re-validation) |

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

- `06-26` Idea parked (as run 8): force charge demotion to bottom *d* Kings (no choice) to kill
  the `C(n_kings,d)` charge fan-out. Rule change — spec + PyVariant + fork + encoding + parity.
- `06-26` **Promoted to run 7 + implementation started.** Run 6 converged (~8k games, lr=1e-3);
  the visits ablation was deferred, so this took the run-7 slot. Inherits run 6's e8/d8 / v5
  c48b5 / Adam 1e-3. Beginning with the PyVariant oracle (`black_charge_moves` / `_apply_charge`).
- `06-26` **PyVariant oracle (#2) DONE.** `black_charge_moves` collapses to one move/charge
  demoting `king_positions[:d]` (bottom *d*); `_apply_charge` unchanged (uses the explicit
  `demotedKings`). Dropped the `combinations` import. Full non-slow suite green (164 passed); the
  one recorded-game replay using the old `f6e6{2}` top-king-demote was updated to `f6e6`.
- `06-26` **Spec (#1) DONE.** §3C.3 rewritten to forced bottom-first (v6 note + `binom{n}{d}`
  rationale); notation `{a,b,…}` suffix removed.
- `06-26` **Encoding (#4): no `encoding.py` change needed, but the earlier "net can't express the
  choice" claim was WRONG (corrected).** Two distinct encodings: the 240-dim **move** vector has
  `demotions_required` but no "which-Kings" field, so demotion choices of the same `from→to` share
  the same **policy prior** (collide at the move-prior level). BUT the **per-depth tower channels
  8–12** (the v5 position encoding) *do* encode the resulting tower, so the net distinguishes the
  choices via the **resulting-position value/policy** through MCTS — a colliding prior is not
  inexpressible. So forcing bottom-*d* removes a **real** degree of freedom, and "barely loses
  strength" rests **only** on the strategic argument (bottom-*d* keeps the top Kings ≈ optimal) and
  **must be measured** vs run 6's net — it is NOT an encoding free pass. The speedup is still the
  legal-move/branching reduction; `encoding.py` itself needs no edit (move vector unchanged).
- **Remaining:** fork C++ rules copy (#3, `../akshay-chessckers-0/src/chessckers/movegen.hpp` +
  rebuild) — the production player; display/parse spot-check (#5); parity re-validation (#6).

## Result

<leave empty — fill once implemented + run; report branching reduction + strength delta vs the choice-version>
