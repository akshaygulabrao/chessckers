# Run 27 — the run-26 ABLATION: PUCT @ 64 visits (budget vs algorithm)

> Run 26 showed Gumbel S2 @64v crossing on 5.3× fewer visits than the 800v
> control — but it changed TWO things at once: the visit budget (800→64) AND the
> root algorithm (PUCT+Dirichlet+temperature → Gumbel-top-m + Sequential
> Halving). This run holds the budget at 64 and reverts the algorithm to the
> control's PUCT, completing the 2×2:
>
> | | 800 visits | 64 visits |
> |---|---|---|
> | **PUCT + Dirichlet + temp** | run 25 arm A: **147.2M / 187.2M** | **run 27 (this run)** |
> | **Gumbel + Seq. Halving** | (not run) | run 26: **31.9M / 31.8M** |
>
> `800v PUCT → 64v PUCT` isolates the **budget** effect.
> `64v PUCT → 64v SH` isolates the **algorithm** effect — the open question.

## Identity

| Field | Value |
|---|---|
| `RUN_NAME` | `run27_e8d8_puct64_bench` |
| Start FEN | `3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1` (unchanged) |
| Arch / trainer | v5 c64/b6, Adam 1e-3, `improved` policy target, pure z, cold, EMA 0.99, publish 400, seeds 0–1 — identical to runs 25/26 |
| Self-play params | `--noise-epsilon=0.25 --noise-alpha=0.3 --temperature=1.0 --tempdecay-moves=15 --visits=64` (= run-25 arm A's flag set with 800→64; `--gumbel-sh` deliberately ABSENT) |
| Gates | disabled (`matches.disabled=true`), as runs 25/26 |
| Engine | same build as run 26 (S2 present but unused — flag off) |
| Comparators | run 25 arm A (800v PUCT) + run 26 (64v SH), both already banked |

## Hypothesis / decision rules (pre-committed)

- **H_algo (literature prior):** PUCT@64v lands materially WORSE than SH@64v
  (31.8M median) — because PUCT's visit-count-derived signal and root Dirichlet
  noise degrade badly at small n, which is precisely the problem Gumbel was
  designed to fix. Under this hypothesis the run-26 win belongs to Sequential
  Halving.
- **H_budget (the deflationary alternative):** PUCT@64v lands near SH@64v
  (within the ~27% control seed spread) — the run-26 win was mostly "800 visits
  is simply overkill on this task", and SH contributed little. This would be a
  genuinely useful negative result: it would say drop the visit budget on this
  task and skip the search-algorithm complexity.
- **Read:** compare V@90% medians three ways. Decisive for H_algo if PUCT@64v
  ≥ 2× SH@64v (i.e. ≥ ~64M) with both trials above SH's range. Decisive for
  H_budget if PUCT@64v ≤ 1.3× SH@64v (≤ ~41M). Between = unresolved at n=2.
- A PUCT@64v **DNF** (10h) is a legitimate result for H_algo, not a bug —
  but see the throughput guard below before scoring it.
- **Throughput guard (pre-registered, from the pre-launch smoke):** standalone
  `selfplay --games=N` runs with `--temperature`+`--tempdecay-moves` complete
  only 3/10 (800v) and 1/10 (64v) of requested games, vs 9/10 for gumbel — an
  artifact of `--games=N` budget accounting in standalone mode, NOT a fleet
  path (run 25 ran the identical 800v flag set for 8,766 games, and its fleet
  throughput of 2,020 games/h matches pure-compute scaling: observed 13.9×
  vs run 26 where visits/game predicts 16.8×; a 70% game-loss rate would have
  shown ~46×). **Guard:** if run-27's early games/h falls far below what
  compute scaling predicts (~28k games/h, i.e. near run-26's rate since the
  budget is identical), STOP and investigate discards before trusting the
  arm — a silently game-shedding PUCT arm would be unfair to PUCT.

## Log

- `07-24` Pre-launch smoke on the box surfaced the standalone temperature-path
  flakiness above; ruled out as a fleet issue by the throughput argument
  (recorded so it isn't re-litigated). `bootstrap` gained a `VISITS` override on
  the baseline branch so this exact flag set can run at 64 visits (unset VISITS
  ⇒ byte-identical to the previous literal).

- `07-24 18:55 UTC` **Launched, trainParams verified exact:**
  `["--noise-epsilon=0.25","--noise-alpha=0.3","--temperature=1.0","--tempdecay-moves=15","--visits=64"]`
  — the run-25 arm-A flag set with 800→64, `--gumbel-sh` absent. Seed 0.
- `07-24 19:05 UTC` **THROUGHPUT GUARD TRIPPED → INVESTIGATED → EXPERIMENT VALID
  (no game loss).** Run 27 produces ~5.7k games/h vs run 26's 28.2k at the SAME
  visit budget. Checked the two possible causes:
  - *Longer games?* NO — 65.4 plies/game vs run-26's 59.0. Not the explanation.
  - *Lost games (the unfair case)?* **NO.** Client reports 708 games completed
    in 7m50s (5,415/h); server stores 111 games/60s (6,660/h) — same order, so
    games the engine finishes DO reach the server. No upload loss, no restart
    churn in the client pane, GPU at 81% (not starved), engine process carrying
    the correct flags.
  - *Actual cause:* run 27 genuinely does less search per hour — npm 75.9 (vs
    S2's 64.4, +18%) plus, hypothesised, poorer GPU batch formation: PUCT at
    n=64 is sequentially dependent so each gather pass collects few nodes,
    whereas SH's phase structure makes many distinct root children eligible at
    once. **Mechanism is a hypothesis; the throughput numbers are measured.**
  - **Verdict: this is a WALL-CLOCK property, not a visits property.** The
    metric is visits-to-crossing precisely to be immune to throughput; V is
    computed as plies × 64 from the landed chunks, which is exactly the search
    embodied in the training data. With no game loss there is no survivorship
    bias, so the run-26 vs run-27 comparison stands.
  - **Bonus finding for S2 (secondary, mechanism unconfirmed):** at equal visit
    budget Gumbel+SH delivers **~4.4× more visits/hour on the same GPU**
    (106M/h vs 24M/h) — i.e. S2's advantage in run 26 was sample efficiency
    *and* it is separately cheaper in wall-clock per visit.
  - **Consequence for this run:** at ~24M visits/h the 10h cap corresponds to
    ~240M visits. So H_budget (~32M ⇒ ~1.3h) and a moderate H_algo result
    (~150M ⇒ ~6h) both fit; only a PUCT cost >240M would DNF — and that DNF
    would itself be decisive for H_algo (>7× worse than SH), not a defect.

## Result

<pending>
