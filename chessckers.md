# Chessckers: Formal Specification (v3)

**Players.** White (Chess) and Black (Swarm). White plays standard FIDE chess; Black follows the rules in §3.

**Board.** The standard 8×8 grid, $\{1..8\} \times \{1..8\}$.

**Rim.** The 10×10 perimeter just outside the board. Black towers may step onto rim squares mid-chain during a diagonal capture (§3B), but a turn can never end on the rim. White pieces never enter the rim at all.

**Distance.** $dist(p_1, p_2) = \max(|x_1 - x_2|, |y_1 - y_2|)$ (Chebyshev).

---

## Initial Position

**White.** Standard FIDE setup on ranks 1 and 2.

**Black.** Single-piece towers on every square of ranks 6–8: Stones on ranks 6 and 8, Kings on rank 7 (24 towers total).

---

## 1. Piece & Tower Definitions

*   **White piece.** Any standard FIDE piece — Pawn, Knight, Bishop, Rook, Queen, or King.
*   **Black piece.** Either a **Stone** or a **King**. Each Stone carries a `hasMoved` flag used by Back Rank Sprint (§3A). The flag is per-stone, not per-tower: it persists when the stone's tower merges into another, and once true it never resets.
*   **Tower.** An ordered sequence of Black pieces sharing a square, written bottom-to-top. Its **height** is the number of pieces; call it $n$. The **top piece** governs the tower's capabilities, and a tower whose top piece is a King is a **King Tower**. The height $n$ doubles as the maximum distance the tower can move on a non-capturing turn and the maximum distance it can scan along a diagonal when capturing.

<!-- §1 code anchors:
  per-stone hasMoved persistence — Chessckers.scala:1022 (Stone.hasMoved field); preserved through Vector concatenation on merge (lines 297, 313, 331).
  height = max non-capture distance + max diagonal-capture scan — Chessckers.scala:209-229 (walkRay caps at n), 392 (jump scan up to n+1).
-->


---

## 2. White: The Chess Legion

*   **Movement and capture.** Standard FIDE chess.
*   **Scope.** White stays on the board and never enters or interacts with the rim.
*   **Capture effect.** When a White piece captures a Black tower, the **entire tower** is removed regardless of its height.

<!-- §2 code anchor:
  entire-tower removal on White capture — Chessckers.scala:163-169 (updateChessckersData clears data.stacks at move.dest whenever move.capture is non-empty).
-->


---

## 3. Black: The Swarm Mechanics

### A. Non-Capturing Actions (Ends Turn)

These are Black's three ways to move without capturing. The destination is always either an empty square or a friendly tower; a non-capturing move never lands on White. Landing on a friendly tower **merges** the moving piece(s) onto its top.

1.  **Diagonal move.** A whole tower slides up to $n$ squares along one diagonal, where $n$ is its height. If the top piece is a Stone, only forward diagonals are allowed (toward rank 1); if the top piece is a King, any diagonal is fine.
2.  **Deploy.** Take the top $s$ pieces off a tower ($1 \le s < n$) and move them as a smaller sub-tower up to $s$ squares along a diagonal. The sub-tower's top piece is the same as the original tower's top, so direction follows the same Stone-vs-King rule. The remaining $n - s$ pieces stay put.
3.  **Back Rank Sprint.** A height-1 Stone-top tower at rank 8 with `hasMoved = false` may move 2 squares forward-diagonal. The path must be clear. After the move, the Stone's `hasMoved` flips to true regardless of where it landed.

<!-- §3A code anchors:
  diagonal move (full tower) — Chessckers.scala:295 (walkRay), 297 (merge concat).
  deploy (sub-tower) — Chessckers.scala:308 (sub-tower top governs direction), 311 (walkRay over s), 313 (deploy-merge concat).
  sprint — Chessckers.scala:320-337 (height-1 Stone(false) at rank 8, distance=2, forward only), 327 (Stone(true) post-sprint), 331 (sprint-merge concat).
-->
### B. Diagonal Captures (Hops & Chains)

A **hop** walks along one diagonal, captures White pieces along the way, and lands somewhere on the same diagonal. A turn can be a single hop or a **chain** — several hops, each in a different direction, sharing one **cadence**.

**Single hop.** Each hop has five steps:

1.  **Find the first enemy.** From the tower's starting square $p_1$, walk up to $n$ steps along one diagonal. A Stone-top tower walks only forward (toward rank 1); a King-top tower walks along any diagonal. Friendly towers block the walk; the path may step onto rim squares, but Whites are never on the rim. The first White piece encountered, at distance $d \in [1, n]$, is the **first enemy**. If no White is found within $n$ steps, no hop is available in that direction. A White reached only at step $n+1$ is *not* a first enemy: the $n+1$ slot is reserved for landings.
2.  **Pick a landing distance, $k$.** Choose any $k \in [d+1, n+1]$. Each value of $k$ is a separate candidate hop; the player picks which one to play. A ram (landing on White) is only legal when $k > d$ — the moving tower must overshoot the first enemy and land on a *different* White, capturing the first enemy in transit (step 4). Landing on the first enemy itself ($k = d$) is **not** a legal hop: there is nothing to hop over.
3.  **Trace the path.** Walk $k$ steps from $p_1$ in the chosen direction. The path is a single straight diagonal: it **never reflects or bounces**. It may pass through or land on rim squares (the one-square ring just outside the board), and the step-$k$ landing may even fall *off* the 10×10 grid (more than one square outside the board); how each kind of landing resolves is covered in step 5. (Because a straight diagonal that reaches the rim is already heading outward, a hop touches the rim at most once before leaving the grid; no path clips the rim and returns to the board.)
4.  **Capture path Whites (overrun).** Every White piece on a board square at steps $1..k{-}1$ is captured. Rim steps capture nothing. Because $k > d$ (step 2), the first enemy at $d$ is always among the captured path Whites.
5.  **Land.** What happens at step $k$ depends on what's there:
    *   **Empty board square:** the hop lands normally; the tower survives.
    *   **Friendly tower:** that $k$ is illegal. Other values of $k$ in the same direction may still be legal.
    *   **White piece (ram):** the tower is destroyed at the landing square. The landing White is **not** captured (only the path Whites from step 4). A ram ends the chain. A ram requires $k > d$ per step 2, so its reachability is already proven by the trace reaching step $k$.
    *   **Rim square:** the hop genuinely lands on the rim (step $k$ itself is a rim square). The tower is now on the rim. The chain may continue with another hop from the rim square (see *Chain*); but if no continuation is available — or the player stops — the tower is left on the rim and *End-of-turn fallback* decides where it actually comes to rest. The rim landing is real: path captures (step 4) and any rank-1 promotion still apply; only the final resting square is adjusted by the fallback.
    *   **Off-grid overshoot:** step $k$ lies beyond the rim, so the hop overshoots the grid — there is no square to land on, and **the tower never rests off the grid**. The hop is legal **only if it captured at least one White on the way** (step 4) — i.e. it had a genuine first enemy; an overshoot with nothing captured is a non-move. When it is legal, the tower **settles on the last on-board square its path crossed**, and the **turn ends** — there is no rim square from which to continue a chain. As with the rim fallback, all path captures and any rank-1 promotion are kept; only the resting square changes.

**Chain.** After a non-ram hop that landed on a board or rim square, the tower *may* continue with another hop in any direction except the 180° reverse of the one just played. A chain has a **cadence** — the value of $k$ used by its first hop — and every continuation hop walks exactly *cadence* steps. A direction is **available** only if it has a first enemy at $d \in [1, \text{cadence}]$; with no enemy on that diagonal it is a non-move. A continuation hop still *runs* when its cadence landing falls on the rim or overshoots the grid — it captures its path Whites either way — but the two differ in what follows: a rim landing may continue the chain, whereas an off-grid overshoot forces the chain to end (step 5, *Off-grid overshoot*). The player can stop the chain after any capture; continuing is always optional. A ram, an off-grid overshoot, or running out of available directions ends the chain.

**End-of-turn fallback.** Black must end its turn on the board, never on the rim or off the grid. Two situations leave the tower off the board at the end of a hop, and both resolve the same way — **fall back to the last on-board square the path occupied**, walking the path backwards to the most recent board square and resting there:

*   The final hop **lands on a rim square** and has no continuation (none available, or the player stops). The sequence is explicit: land on the rim, look for a continuation hop, find none, fall back.
*   A hop **overshoots off the grid** (step 5, *Off-grid overshoot*). The tower could never stand there, so it settles back and the turn ends immediately.

In both cases all captures made along the way, and any promotion from crossing rank 1, are kept — only the resting square moves. A hop into the corner rim with no follow-up rests on the last board square it came from; it does not die at trace time.

**Notation.** A diagonal-capture move is written **`c<N>:<from>~<hop₁>~…~<hopₖ>→<rest>`**:
*   **`c<N>`** — the **cadence**, leading. It is the uniform hop length locked by the first hop and shared by every hop in the chain, and it is what tells two otherwise-similar moves apart.
*   **`<from>`** — the starting square.
*   **`~<hopᵢ>`** — the successive hops' grid keys: each hop's landing square, or, for a final hop that overshoots, the **last on-grid square** that hop reached. Always a key on the 10×10 grid (board or rim) — an overshoot is never written as an off-grid coordinate.
*   **`→<rest>`** — the square the tower actually **comes to rest on**, always shown and always on the board. It equals the last landing for an on-board hop, or the fall-back square for a rim dead-end or off-grid overshoot.

All keys are on the 10×10 grid (files `z, a–h, i` with rim `z`/`i`; ranks `0–9` with rim `0`/`9`) — the notation never names a square off the grid. **Cadence is the discriminator:** a rim landing and an off-grid overshoot can share the same on-grid keys and the same `→<rest>` (e.g. `c2:g3~i1→h2` lands on i1; `c3:g3~i1→h2` overshoots past i1), and the leading `c<N>` is what tells them apart.

*Worked example* (FEN `8/8/8/8/8/6k1/7R/K7[g3:sk] b - - 0 1`). A King-top `sk` tower on g3 ($n = 2$, so $k \in \{2, 3\}$), a White Rook on h2, the White King tucked on a1 (off g3's diagonals, so the Rook is the only target), Black to move. The Rook sits on g3's immediate diagonal, so the capture mandate fires. Hopping down-right, step 1 captures the Rook on h2. The tower offers two **distinct candidate hops**: $k = 2$, landing on the rim square **i1**; and $k = 3$, landing off the grid one step past i1. Both are legal *here* — each captures the Rook — but they are **separate moves, not one**: $k = 2$ has cadence 2 and lands on the rim, while $k = 3$ has cadence 3 and overshoots the grid, ending the turn (it never rests off the grid — it settles back). Neither continues in this position (from i1, no direction has an enemy within cadence 2), so each falls back to **h2**, the last on-board square its path occupied (now empty, the Rook captured). They reach the same square here, but they remain different candidates — and they are not always both legal: in another position one may be available and the other not, or the rim landing ($k = 2$) could continue a chain where the off-grid overshoot ($k = 3$) cannot. In the move notation the two are `c2:g3~i1→h2` and `c3:g3~i1→h2` — identical on-grid keys and rest, told apart by the leading cadence alone.

*Worked example — cadence lock and the off-grid overshoot* (FEN `8/8/8/5k2/8/7R/8/4K2N[f5:sk] b - - 0 1`). The `sk` tower now on f5, the Rook on h3 (distance 2), a White Knight on h1, the White King on e1, Black to move. The Rook is at $d = 2$ with $n = 2$, so the only capturing hop has $k = 3$ — **cadence 3** — landing on the rim square **i2** (capturing the Rook in transit). From i2 the Knight on h1 is one step away, but a continuation must walk *exactly* cadence 3: i2 → h1 (capture Knight) → g0 (rim) → overshoots off the grid. The hop captured the Knight, so it is legal, and since it never rests off the grid it **settles on h1** — the last on-board square it crossed — and the turn ends. Net result: f5 captures Rook + Knight and rests on **h1** — written **`c3:f5~i2~g0→h1`** (g0 is the last on-grid square of the overshooting hop). Remove the Knight and that diagonal has no enemy, so the continuation is a non-move; the chain dead-ends on i2 and falls back to **h3**: **`c3:f5~i2→h3`** (Rook only). Contrast the cadence-2 version (tower on g4, Rook adjacent at $d=1$): there the same Knight is captured by a hop that *lands* on the rim at g0 (a rim dead-end), giving **`c2:g4~i2~g0→h1`**; here cadence 3 overshoots past g0. Both finish on h1, and the leading cadence reads them apart — `c2:…g0→h1` (rim landing) versus `c3:…g0→h1` (overshoot).

**Promotion.** If any hop's path touches rank 1 — by landing there or by stepping through it on the way to a rim square — every Stone in the tower is promoted to a King *before* the next hop is considered. See §5 for the full statement.

<!-- §3B code anchors:
  scan, first enemy, n+1 gating — Chessckers.scala:399-553 (findSlideCaptureOptionsFrom); 490 (suicide n+1 guard: step <= n || capturesSoFar.nonEmpty).
  ram requires k > d (path capture must already exist) — Chessckers.scala (findSlideCaptureOptionsFrom White-branch: `if capturesSoFar.nonEmpty then options += CaptureHop(...isSuicide = true...)`).
  off-grid overshoot (NEW rule; no bounce) — a hop whose cadence landing overshoots the grid still captures its path Whites and SETTLES on the last on-board square (never off the grid), ending the turn; legal only if ≥1 capture. NOTE: Chessckers.scala:547-551 currently *terminates* on off-grid (old no-settle behavior) and must be updated to match; neither version reflects/bounces.
  overrun (path-intermediate captures) — Chessckers.scala:514-516 (capturesSoFar accumulation as scan walks board squares).
  landing cases (empty/friendly/ram/rim) — Chessckers.scala:451-535 (CaptureHop emitted with isSuicide=true when landing on White; friendly landing skipped, no emit).
  chain direction filter (no 180° reverse) — Chessckers.scala:629 (dirs.filter((f,r) => !(f == -ldf && r == -ldr))).
  cadence (chain k uniformity) — Chessckers.scala:637-639 (suicideFiltered.filter(_.waypoints.size == c)).
  end-of-turn fallback — Chessckers.scala:809-818 (buildFinalMove: walk allWaypoints backwards via Square.fromKey for last on-board square).
  promotion via rank-1 crossing — Chessckers.scala:217-218 (hopPromotes: hop.crossesRank1 || landing on rank 1); crossedRank1 flag set at 442.
-->

### C. Charge Movement & Capture

A King-top tower may move along a rank or file in a single straight line. Path Whites are captured for free; the cost is paid in King demotions, one per square moved.

1.  **Requirement.** The top piece must be a King. The path (the squares strictly between $p_1$ and $p_2$) must contain no friendly towers. The path may step onto rim squares — rim has no pieces, so no path captures happen there. If any step would go off the 10×10 grid (file or rank reaches ±2 from the board), the charge is invalid at that cost.
2.  **Cost.** One King is demoted per square moved, and the tower must start with at least that many Kings. Capturing is free — only the squares-moved count is paid.
3.  **Choice of demoted Kings.** If the tower has more Kings than the cost requires, the player picks which ones to demote. Kings are 1-indexed from the bottom (`1` = bottom-most, `n` = top). A demoted King becomes a Stone with `hasMoved = true`. The choice is strategic: the post-demotion top piece governs the tower's mobility on its next turn.
4.  **Path captures.** Every White on the path's intermediate squares is captured automatically.
5.  **Landing.** What happens at $p_2$:
    *   **Empty:** the demoted tower lands at $p_2$.
    *   **Friendly:** the demoted tower is placed on top of the destination tower (same merge convention as §3A).
    *   **White (ram):** the tower is destroyed at $p_2$. The landing White is **not** captured (path captures still apply). A ram is legal only when step 4 captured at least one White — the charge must overshoot at least one enemy before crashing. A charge of distance 1 (no intermediate squares to capture) therefore cannot ram.
    *   **Rim:** the path stepped onto rim at $p_2$. The tower falls back to the last on-board square in the path (parallel to §3B end-of-turn fallback). That square is always empty after path captures apply, so the fallback landing is always an empty board square. Rams are not possible on a fallback landing (the rim itself has no piece, and the prior on-board square was either empty or captured during the path).

**Restrictions.** A charge ends Black's turn — no chaining. Landing on rank 1 does *not* promote any Stones to Kings; only diagonal moves promote (see §5). For how charges interact with the mandate, see §4: capturing charges (including rams) satisfy it, and non-capturing charges are suppressed while it is active.

<!-- §3C code anchors:
  King-top requirement, cost, path & landing — Chessckers.scala:872-955 (genBlackOrtho).
  cost = squares moved, captures free — Chessckers.scala:928 (totalDemotions = dist).
  demotion choice (1-indexed from bottom) — Chessckers.scala:859-863 (demotionChoices).
  demoted King → Stone(true) — Chessckers.scala:853.
  landing on empty / friendly merge — Chessckers.scala:909-924 (merge), 940-951 (empty).
  ram (orthoSuicide) — Chessckers.scala:985-1008.
  mandate fulfillment by charge captures/rams — Chessckers.scala:99-100 (validMoves: orthoCaptures and orthoSuicide returned in mandatory set; non-capturing charge suppressed).
-->

---

## 4. Mandatory Logic

1.  **Trigger.** At the start of Black's turn, every Black tower is scanned for adjacent Whites along the diagonals it can move (forward only for Stone-top towers, any diagonal for King-top towers). If at least one such adjacent White admits at least one **normal** capture (a diagonal hop that lands on an empty board square), the mandate fires. The trigger is recomputed every turn — fulfilling the mandate this turn does not silence threats that may exist at the start of the next.

2.  **Fulfillment.** While the mandate is active, any capturing move from any Black tower satisfies it: a diagonal hop (normal landing or ram), a charge that captures at least one White on its path, or a charging ram. Non-capturing moves (quiet diagonals, deploys, sprints, and non-capturing charges) are suppressed for that turn — they cannot be played at all.

3.  **Ram is never required.** The trigger looks only at normal-landing diagonal hops, so a position whose adjacent-White diagonals lead only to rams or rim-only landings does not fire the mandate. Once the mandate is active, the player can choose a ram as the fulfillment — rams count as valid fulfillments — but a ram will never be the only legal option.

4.  **Chain continuation is never required.** The mandate applies only to the *first* capture of the turn; after that, the player can stop the chain whenever they want. See §3B's "Chain" and "End-of-turn fallback" for the chain mechanics.

<!-- §4 code anchors:
  trigger condition (adjacent White + normal landing exists) — Chessckers.scala:141-161 (hasMandatoryCapture; line 158: filterNot(_.isSuicide).exists(_.landingSquare.isDefined)).
  fulfillment set when mandatory — Chessckers.scala:97-101 (validMoves: jumps ++ diagSuicide ++ orthoCaptures ++ orthoSuicide when mandatory; non-capturing charge and quiet diag/deploy/sprint suppressed).
-->

---

<!-- §5 anchors: promotion — Chessckers.scala:179-205 (promoteStack/applyPromotion/hopPromotes); White-win/Black-stalemate-loss — Chessckers.scala:54-67 (specialEnd) -->
## 5. Promotion & Win Conditions

**Promotion.** Whenever the path traced by any Black move other than a charge (quiet diagonal, deploy, sprint, capture hop, or capture chain) touches rank 1 — either by landing there or by stepping through it on the way to a rim square — every Stone in the tower is promoted to a King. The promotion takes effect immediately, so any later hops in the same chain use the now-promoted (King-top) tower for direction and capability. Charges never promote.

**White wins** if Black has no pieces left on the board, or if Black has no legal moves on their turn. (Chessckers does not treat Black being unable to move as a draw — being stuck loses the game.)

**Black wins** by checkmating the White king under standard FIDE rules.
