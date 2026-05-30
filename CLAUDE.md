# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Chessckers is a chess-vs-checkers hybrid game built as three cooperating subprojects in a monorepo:

- **scalachess** (`scalachess/`) — Fork of lichess.org's Scala 3 chess engine with a new `Chessckers` variant. This is the game logic layer: move generation, validation, FEN/UCI parsing, bitboard representation.
- **server** (`server/`) — Scala 3 HTTP API (http4s + Circe) that wraps scalachess and exposes game state, legal moves, and move execution over REST.
- **chessground** (`chessground/`) — Fork of lichess.org's TypeScript board UI. Renders the board, handles drag/click interaction. The Chessckers frontend lives in `chessground/chessckers.html`.

## Build & Run

### scalachess (Scala 3 / sbt)

Requires Java 21 (Temurin). Start an sbt shell from `scalachess/`:

```
cd scalachess && sbt
```

Inside sbt:
- `compile` — compile all modules
- `testKit/test` — run full test suite
- `testKit/test -- *ChessckersTest*` — run only Chessckers tests
- `prepare` — run scalafmt + scalafix (run before committing)
- `check` — verify formatting without modifying files

### server (Scala 3 / sbt)

The server depends on scalachess via `ProjectRef(file("../scalachess"), "scalachess")`. From `server/`:

```
cd server && sbt run
```

Starts on `http://localhost:8080`. API endpoints:
- `POST /api/game/new` — new game (optional `{"fen": "..."}` body)
- `POST /api/game/move` — apply a move (`{"fen": "...", "uci": "..."}`)
- `POST /api/game/moves-at` — legal moves from a square (`{"fen": "...", "square": "e2"}`)

### chessground (TypeScript / pnpm)

```
cd chessground && pnpm install
```

- `pnpm run compile` — TypeScript compile (or `--watch`)
- `pnpm run dist` — compile + esbuild bundle
- `pnpm test` — vitest (jsdom)
- `pnpm run lint` — oxlint
- `pnpm run format` — oxfmt

To play: start the server (`cd server && sbt run`), then open `chessground/chessckers.html` in a browser. The HTML file loads chessground from `dist/` and talks to `localhost:8080`.

## Architecture

### Chessckers Variant Design (scalachess)

The variant follows the **Crazyhouse pattern**: `Board` stays as plain bitboards (no subclassing), and variant-specific state rides in `History.chessckersData: Option[Chessckers.Data]`, exactly mirroring `History.crazyData`. This is a deliberate architectural choice — `Board` is a case class whose mutating helpers (`taking`, `move`, `put`) construct new `Board` instances and would silently drop subclass fields.

Key invariant: for every Black square on the board, `chessckersData.stacks(sq).head` matches the bitboard top piece (King = `byRole(King)`, Stone = `byRole(Pawn)`). Bitboards are truth for top piece; the overlay is truth for everything below.

**Black pieces on bitboards:** Stones are encoded as `Black-Pawn`, Kings as `Black-King`. This reuse means standard chess move generation correctly treats Black squares as blockers/captures for White. Black moves go exclusively through the Chessckers generators (`genBlackQuiet`, `genBlackJumps`, `genBlackOrtho`), never through `Standard.validMoves`.

**One Move per chain:** Full diagonal capture chains are computed inside the generator and emitted as a single `Move(orig=start, dest=end)` with the complete bitboard + overlay delta already applied. Chain disambiguation uses optional `waypoints` field on `Move`.

**Transient boundary (T):** The 10x10 perimeter outside the 8x8 board is computed in raw `(Int, Int)` coordinates during chain enumeration — `Square` (0..63) is never extended. Edge reflection reverses the out-of-range component; corner retroreflects both.

### FEN Extension

Chessckers FEN appends a bracketed stack overlay after the standard board field: `[a6:s,a7:k,a8:Sks,...]`. Each entry is `square:pieces` where pieces are bottom-to-top: `s`=Stone(unmoved), `S`=Stone(moved), `k`=King.

### Server ↔ UI Data Flow

The server is stateless — each request carries the full FEN. `GameApi.toGameState` returns the full position: FEN, turn, check, status, legal moves (with UCI + from/to/piece/waypoints), and a `stacks` map (square → piece list). The UI renders stack height badges via SVG autoShapes and uses the legal moves map for chessground's `movable.dests`.

### chessground Chessckers Customizations

- `assets/chessground.chessckers.css` — renders Black pawns as dark checker disks and Black kings as crowned white disks (inline SVG data URIs)
- `chessckers.html` — standalone game UI with move log, undo, and chain path visualization

## Game Rules Reference

The formal spec is in `chessckers.md` (monorepo root). Key points:
- White plays standard FIDE chess (ranks 1-2)
- Black has 24 checker pieces on ranks 6-8 (Stones on 6+8, Kings on 7), organized as stacks
- Black moves: diagonal movement (range = stack height), stacking/unstacking, back rank sprint, diagonal capture chains (with T-boundary bounces), orthogonal capture (King-top stacks, costs King demotions)
- Mandatory diagonal capture rule: if any Black stack has a White piece at immediate diagonal, Black must capture
- Suicide captures: landing on enemy destroys the stack but captures all enemies on path; never mandatory
- Win: White wins by eliminating all Black stacks; Black wins by checkmating White's king

## Code Style

- **Scala**: max 110 chars, Scala 3 syntax (`given`/`using`, opaque types, enums). Run `sbt prepare` before committing.
- **TypeScript**: oxlint + oxfmt. Strict TypeScript with `noUnusedLocals`/`noUnusedParameters`. pnpm as package manager.
