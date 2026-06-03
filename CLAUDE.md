# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Chessckers is a chess-vs-checkers hybrid game. The repository is now a single authoritative engine plus its training/operations tooling:

- **engine** (`engine/`) — Python (managed by `uv`) + a Rust extension (`chessckers_movegen`, built with `maturin`). This is the whole stack: the game logic (move generation, validation, FEN/UCI parsing), an AlphaZero-style neural engine (PUCT MCTS, self-play, training), and a terminal board renderer for debugging.

The game logic lives in **PyVariant** (`engine/chessckers_engine/variant_py/`), a pure-Python reimplementation of the Chessckers rules. The Rust extension accelerates the hot path (Black move generation + check detection) and is kept byte-for-byte equivalent to PyVariant.

> History: the game logic used to live in three forks — `scalachess` (Scala 3 engine), `server` (HTTP API), and `chessground` (browser board UI). All three were removed once PyVariant + Rust became the authority and terminal rendering replaced the browser UI. A 2GB backup of the forks exists at `~/chessckers-backups/forks-20260528.tar.gz`. The `git mv`'d formal spec (`chessckers.md`) and the engine are all that remain.

## Build & Run

Everything runs from `engine/`. Dependencies are managed with `uv`; the Rust extension with `maturin`.

```
cd engine
uv sync                  # install Python deps into .venv
```

### Rust extension

The move-gen accelerator is a PyO3 crate at `engine/rust/chessckers_movegen/`. After editing `src/lib.rs`, rebuild and reinstall into the venv:

```
cd engine/rust/chessckers_movegen
VIRTUAL_ENV=../../.venv ../../.venv/bin/maturin develop --release
```

Use `--release` for anything performance-sensitive (self-play, the full test suite); the debug build is several× slower. Set `CHESSCKERS_NO_RUST=1` to bypass the extension and run the pure-Python move-gen (used by the spec tests, which also monkeypatch `_rs_movegen = None`).

### C++ engine (lc0-style port, in progress)

A native C++ self-play/search/inference engine is being ported under `engine/cpp/` (pybind11 module `chessckers_cpp`), following lc0 as the reference architecture; **training stays in Python**. The C++ rules are a third implementation (alongside PyVariant + the Rust crate), translated from the validated Rust and held byte-equivalent via the existing parity tests (PyVariant `CHESSCKERS_NO_RUST` is the move-gen oracle; the Rust crate is the cross-check). After editing anything under `cpp/src/`, rebuild + reinstall into the venv:

```
cd engine
cpp/build.sh            # cmake + clang++ -> installs chessckers_cpp.*.so into .venv
```

The slice roadmap (0 = board+FEN round-trip, done) lives in the `project-cpp-port` memory. The C++ module mirrors the Rust crate's bb-decomposed call surface so the same parity tests serve as its oracle.

### Tests

```
cd engine
.venv/bin/python -m pytest -q          # full suite
.venv/bin/python -m pytest -q -m "not slow"   # skip subprocess-spawning integration tests
```

The `slow` marker tags end-to-end tests that spawn subprocess workers (self-play, inference server). They are not excluded by default but are the slowest part of the suite — the heaviest plays two full 400-ply games and takes minutes. There is no `pytest-timeout` config by default; add `--timeout=N` only for diagnosis.

### Run / train

- `python -m chessckers_engine.selfplay_az_loop [...]` — AlphaZero self-play + training loop.
- Operational scripts live in `scripts/` (`launch_workers.sh`, `train_cloud*.sh`, `watchdog.sh`, `status.sh`, the launchd plist, remote-fleet sync). These drive multi-worker / cloud self-play.

The legacy HTTP layer is gone: there is no Scala server, no `ServerClient`, and no `python -m chessckers_engine` HTTP server. All game logic runs in-process via `PyVariantClient`. Some launch scripts still pass `--use-pyvariant`/`--use-server`; those flags are accepted as no-ops.

## Architecture

### Chessckers Variant Design (PyVariant)

A position is a `State` (`variant_py/state.py`): a python-chess `Board` (bitboards) plus a `stacks` overlay `dict[square -> str]`. This follows the lichess **Crazyhouse pattern** — keep the board as plain bitboards and carry variant-specific state in a side overlay, rather than subclassing `Board`.

**Black pieces on bitboards:** Stones are encoded as `Black-Pawn`, Kings as `Black-King`. This reuse means standard chess move generation correctly treats Black squares as blockers/captures for White. Black moves go exclusively through the Chessckers generators in `variant_py/moves_black.py` (quiet diagonals, deploys, charges, diagonal-capture hops/chains), never through python-chess. White moves go through `variant_py/moves_white.py`, which filters python-chess pseudo-legal moves with a **Chessckers-correct check predicate** (`_is_white_in_chessckers_check`) — python-chess's own `is_check` is wrong here because it treats the Black-King encoding as a standard 8-direction king.

Key invariant: for every Black square, `stacks[sq]`'s top piece matches the bitboard top piece (King = `Black-King`, Stone = `Black-Pawn`). Bitboards are truth for the top piece; the overlay is truth for everything below.

**Rust acceleration:** `engine/rust/chessckers_movegen/src/lib.rs` mirrors the Black generators and the check predicate (`black_can_capture_white_king`, a bool early-exit) for speed. It MUST stay equivalent to the Python — when you change a Black move-gen rule, change both, rebuild with `maturin develop`, and verify equivalence (run the suite with Rust on, and the spec tests with it bypassed).

**One Move per chain:** a full diagonal capture chain is computed inside the generator and emitted as a single move with the complete bitboard + overlay delta applied; `waypoints`/`chainHops` carry the path for disambiguation/display.

### In-process API (PyVariantClient)

`variant_py/client.py` exposes `PyVariantClient` — an in-process API whose methods (`new_game`, `make_move`, `moves_at`, plus the MCTS fast-path `parse`/`apply_known`/`status_and_legal`) return the same JSON-shaped dicts the old Scala server returned: `fen`, `turn`, `check`, `status`, `winner`, `legalMoves` (with UCI + from/to/piece/waypoints), and a `stacks` map. This replaced the HTTP `ServerClient`; the engine is stateless (each call carries the full FEN).

### Terminal rendering

`chessckers_engine/render_board.py` renders the 10×10 board (the 8×8 grid plus the rim ring) into text, with stacks shown as `s`/`k` strings (rightmost = top, White uppercase) and an optional numbered path overlay. This is the debugging surface that replaced the browser UI — it is LLM-readable, so move-gen bugs can be reproduced and inspected from a FEN without a human in the loop.

### AlphaZero engine

`model.py` (`ChesskersScorer`: policy + value heads), `encoding.py` (board/move tensors), `mcts_puct.py` (PUCT MCTS), `selfplay_az.py` / `selfplay_az_async.py` / `selfplay_az_loop.py` (self-play + training loop), `selfplay_worker_async.py` / `selfplay_workers_only.py` (subprocess workers), `inference_server.py` + `cross_inference.py` (batched GPU eval shared across workers), `replay_buffer.py`, `train_az.py`, `trainer_loop.py`. Self-play correctness depends on the move-gen + check detection above; `n_sims` should be ≥ 50 (lower yields degenerate visit distributions).

### FEN Extension

Chessckers FEN appends a bracketed stack overlay after the standard board field: `[a6:s,a7:k,a8:Sks,...]`. Each entry is `square:pieces` where pieces are bottom-to-top: `s`=Stone(unmoved), `S`=Stone(moved), `k`=King.

## Game Rules Reference

The formal spec is in `chessckers.md` (monorepo root). Key points (v3 terms):
- White plays standard FIDE chess (ranks 1-2).
- Black has 24 checker pieces (Stones + Kings) on ranks 6-8, organized as **Towers** (stacks).
- Black moves: diagonal movement (range = stack height), deploys (stacking/unstacking), back-rank sprint, diagonal **capture hops/chains**, and orthogonal **Charges** (King-top towers).
- §3B capture rules: a hop walks one diagonal and captures Whites in transit; a chain is several hops sharing a **cadence** (the first hop's length). Paths **never bounce off the rim** — a hop whose cadence landing overshoots the 10×10 grid settles on the last on-board square and ends the turn (legal only if it captured ≥1 White). Intermediate chain stops are optional. Notation: `c<N>:<from>~<hops>-><rest>` (cadence leading; `<rest>` always on-board).
- **Mandate** (mandatory capture): if any Black tower has a normal-landing capture available, Black must capture this turn.
- **Ram** (suicide capture): landing on an enemy destroys the tower but captures all enemies on the path; never mandatory; does not capture the landing piece (so a ram onto the King does NOT capture it — only path-captures do).
- Win: White wins by eliminating all Black towers; Black wins by checkmating (capturing) White's king.

## Code Style

- **Python**: type hints throughout, dict shapes preserved at the `PyVariantClient` boundary (these mirror what the old Scala server returned). Keep changes surgical.
- **Rust**: `rustfmt`; the crate mirrors PyVariant and must stay equivalent — any rule change goes in both, with the equivalence verified before committing.
