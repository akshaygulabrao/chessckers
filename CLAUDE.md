# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Chessckers is a chess-vs-checkers hybrid game. The repository centers on one `engine/` tree (Python + C++) — now serving as the project's trainer, rules authority, and analysis tooling (production play + the fleet have moved to the lc0-ecosystem forks; see *Current role* below):

- **engine** (`engine/`) — Python (managed by `uv`) + a native C++ extension (`chessckers_cpp`, built with `cmake`). It holds the game logic (move generation, validation, FEN/UCI parsing), an AlphaZero-style neural stack (model, encoders, PUCT MCTS, training), and a terminal board renderer.

**Current role (post lc0-split):** this repo is now the **trainer + rules authority + analysis tooling**, not the production player. The live, load-bearing piece is the continuous AlphaZero trainer `chessckers_engine.train_continuous`, which the lc0 fleet drives to produce nets (see *Run / train*). Production **self-play and fleet coordination moved out** to the lc0-ecosystem forks: `akshay-chessckers-0` (the lc0 engine fork) plays the games, `lczero-client` runs them, `lczero-server` coordinates. The in-repo C++ self-play client (`cc_selfplay`) and the old HTTP fleet (`scripts/launch_*.sh`, `fleet_client`/`fleet_server`) are **superseded** by that ecosystem and kept only as a parity reference / legacy. `scripts/watch_game.py` (analysis) and `chessckers.md` (rules spec) remain first-class here. See the `chessckers-lczero-fleet` and `chessckers-lc0-fork` memories.

The game logic lives in **PyVariant** (`engine/chessckers_engine/variant_py/`), a pure-Python reimplementation of the Chessckers rules — the authoritative reference and the parity oracle for the C++ port. Its main live consumer today is `scripts/watch_game.py`; it remains the rules authority of record, but the lc0 fork carries its own rules port and does not call back into this repo at runtime.

> History: the game logic used to live in three Scala/JS forks (`scalachess`, `server`, `chessground`) — all removed once PyVariant became the authority. A Rust move-gen accelerator (`chessckers_movegen`) then served as PyVariant's hot path and one of the C++ port's parity oracles; it was **retired** once the C++ engine became the production engine (lc0-split migration), leaving PyVariant as the single oracle; production play has since moved once more, out to the `akshay-chessckers-0` lc0 fork. A 2GB backup of the forks exists at `~/chessckers-backups/forks-20260528.tar.gz`. The `git mv`'d formal spec (`chessckers.md`) and the engine are all that remain.

## Build & Run

Everything runs from `engine/`. Dependencies are managed with `uv`; the C++ extension with `cmake`.

```
cd engine
uv sync                  # install Python deps into .venv
```

### C++ engine (lc0-style)

`engine/cpp/` is a native C++ self-play/search/inference build (pybind11 module `chessckers_cpp` + standalone `cc_selfplay` client), following lc0 as the reference architecture; **training stays in Python**. **It is no longer the production player** — the `akshay-chessckers-0` lc0 fork plays the fleet's games (it carries its own, byte-identical copy of `nn.hpp`/`nn_metal`). `engine/cpp/` is retained as the in-process engine for tooling and as the parity reference: its C++ rules are held byte-equivalent to PyVariant (the oracle) via the parity tests in `tests/test_cpp_*.py`. After editing anything under `cpp/src/`, rebuild + reinstall into the venv:

```
cd engine
cpp/build.sh            # cmake + clang++ -> installs chessckers_cpp.*.so into .venv
```

The slice roadmap lives in the `project-cpp-port` memory. The C++ module's bb-decomposed call surface is what the PyVariant parity tests exercise as its oracle.

### Tests

```
cd engine
.venv/bin/python -m pytest -q          # full suite
.venv/bin/python -m pytest -q -m "not slow"   # skip subprocess-spawning integration tests
```

The `slow` marker tags end-to-end tests that spawn subprocess workers (self-play, inference server). They are not excluded by default but are the slowest part of the suite — the heaviest plays two full 400-ply games and takes minutes. There is no `pytest-timeout` config by default; add `--timeout=N` only for diagnosis.

### Run / train

**Training is this repo's live job.** `chessckers_engine.train_continuous` is the continuous AlphaZero trainer: it ingests ccz1 game chunks into a rolling replay buffer, does non-stop SGD, and publishes `weights.pt` + a C++-loadable `weights.bin` on a timer. In the current fleet it is **driven by the lc0 ecosystem**, not launched here directly — `lczero-server/scripts/launch_trainer.sh` runs a bridge (`lczero-server/trainer/trainer_bridge.py`) that spawns `train_continuous`, feeds it the games the server collected, and uploads each fresh `weights.bin` back for promotion.

**The current fleet lives in the lc0-ecosystem forks** (`~/AAworkspace/{lczero-server,lczero-client,akshay-chessckers-0}`), run as foreground tabs:
- `lczero-server/scripts/launch_server.sh` — the lc0 server: collects ccz1 games, distributes nets, runs promotion matches (port 9830).
- `lczero-server/scripts/launch_trainer.sh` — the trainer bridge → `chessckers_engine.train_continuous` (this repo).
- `lczero-client/scripts/launch_client.sh` (+ `launch_leena.sh`) — self-play clients running the `akshay-chessckers-0` engine, uploading games to the server.

See the `chessckers-lczero-fleet` and `chessckers-lc0-fork` memories.

**Analysis:** `scripts/watch_game.py "<FEN>" [--device mps --sims N --explore 0]` — watch the trained net play both sides from any FEN, with the 10×10 board + WDL eval + top lines rendered each ply. It loads the net via its `.arch.json` sidecar, so V1/V2/V3 (incl. transformer trunks) all load correctly.

**Superseded / legacy in this repo** (present but off the live path): `selfplay_az_loop` (synchronous self-play+train loop), the in-repo HTTP fleet (`scripts/launch_{server,local,leena}.sh`, `fleet_client`/`fleet_server`), and `cc_selfplay` as a player. The pre-lc0 Scala layer is fully gone — no Scala server, no `ServerClient`, no `python -m chessckers_engine` HTTP server; leftover `--use-pyvariant`/`--use-server` flags are accepted as no-ops.

## Architecture

### Chessckers Variant Design (PyVariant)

A position is a `State` (`variant_py/state.py`): a python-chess `Board` (bitboards) plus a `stacks` overlay `dict[square -> str]`. This follows the lichess **Crazyhouse pattern** — keep the board as plain bitboards and carry variant-specific state in a side overlay, rather than subclassing `Board`.

**Black pieces on bitboards:** Stones are encoded as `Black-Pawn`, Kings as `Black-King`. This reuse means standard chess move generation correctly treats Black squares as blockers/captures for White. Black moves go exclusively through the Chessckers generators in `variant_py/moves_black.py` (quiet diagonals, deploys, charges, diagonal-capture hops/chains), never through python-chess. White moves go through `variant_py/moves_white.py`, which filters python-chess pseudo-legal moves with a **Chessckers-correct check predicate** (`_is_white_in_chessckers_check`) — python-chess's own `is_check` is wrong here because it treats the Black-King encoding as a standard 8-direction king.

Key invariant: for every Black square, `stacks[sq]`'s top piece matches the bitboard top piece (King = `Black-King`, Stone = `Black-Pawn`). Bitboards are truth for the top piece; the overlay is truth for everything below.

**Native acceleration:** the C++ extension `engine/cpp/` reimplements the move-gen + check predicate (`black_can_capture_white_king`), White gen, apply, status, search, and NN inference, and MUST stay equivalent to PyVariant. A move-gen rule change now has **three** landing sites to keep in sync: PyVariant, `engine/cpp/` (rebuild with `cpp/build.sh`, verify `tests/test_cpp_*.py`), and the `akshay-chessckers-0` fork's own copy under `src/chessckers/` (the actual production player).

**One Move per chain:** a full diagonal capture chain is computed inside the generator and emitted as a single move with the complete bitboard + overlay delta applied; `waypoints`/`chainHops` carry the path for disambiguation/display.

### In-process API (PyVariantClient)

`variant_py/client.py` exposes `PyVariantClient` — an in-process API whose methods (`new_game`, `make_move`, `moves_at`, plus the MCTS fast-path `parse`/`apply_known`/`status_and_legal`) return the same JSON-shaped dicts the old Scala server returned: `fen`, `turn`, `check`, `status`, `winner`, `legalMoves` (with UCI + from/to/piece/waypoints), and a `stacks` map. This replaced the HTTP `ServerClient`; the engine is stateless (each call carries the full FEN).

### Terminal rendering

`chessckers_engine/render_board.py` renders the 10×10 board (the 8×8 grid plus the rim ring) into text, with stacks shown as `s`/`k` strings (rightmost = top, White uppercase) and an optional numbered path overlay. This is the debugging surface that replaced the browser UI — it is LLM-readable, so move-gen bugs can be reproduced and inspected from a FEN without a human in the loop.

### AlphaZero engine

`model.py` (`ChesskersScorer`: policy + value heads), `encoding.py` (board/move tensors), `mcts_puct.py` (PUCT MCTS), `selfplay_az.py` (`play_az_game` — the reference self-play loop + parity oracle, used by `train_az`/`replay_buffer`/`native_search`) / `selfplay_az_loop.py` (synchronous collect-then-train loop), `inference_server.py` + `cross_inference.py` (batched GPU eval), `replay_buffer.py`, `train_az.py`, `trainer_loop.py`. **Production self-play runs outside this repo**: the lc0-split cutover retired the Python self-play engine (`selfplay_worker_async`/`selfplay_workers_only`/`selfplay_az_async`); every fleet game is now played by the `akshay-chessckers-0` lc0 fork (run by `lczero-client`), which uploads ccz1 games to `lczero-server` for `train_continuous` to consume. (The intermediate `cc_selfplay`-owned-by-`fleet_client` setup is itself now superseded.) Self-play correctness depends on the move-gen + check detection above; `n_sims` should be ≥ 50 (lower yields degenerate visit distributions).

### FEN Extension

Chessckers FEN appends a bracketed stack overlay after the standard board field: `[a6:s,a7:k,a8:Sks,...]`. Each entry is `square:pieces` where pieces are bottom-to-top: `s`=Stone(unmoved), `S`=Stone(moved), `k`=King.

An optional trailing `{wm:N,r8:N}` block after the six standard fields carries Chessckers turn/win state: `wm` = White sub-moves left this turn (2 only at the opening double-move; default 1) and `r8` = the rank-8 win counter (0–2; default 0). It is omitted when both are at their defaults (so ordinary FENs are unchanged); `STARTING_FEN` carries `{wm:2}`. Parsed/serialized in `variant_py/state.py` and the C++ `parse_fen`; encoded as position-tensor channel 14 (`r8/3`).

## Game Rules Reference

The formal spec is in `chessckers.md` (monorepo root). Key points (v3 terms):
- White plays standard FIDE chess (ranks 1-2). **Opening double-move:** on White's first turn of the game, White plays two moves in succession (carried as FEN `{wm:2}`).
- Black has 24 checker pieces (Stones + Kings) on ranks 6-8, organized as **Towers** (stacks). Initial setup: Stones on ranks 6 and 8 plus a7 & h7; Kings on b7-g7.
- Black moves: diagonal movement (range = stack height), deploys (stacking/unstacking), back-rank sprint, diagonal **capture hops/chains**, and orthogonal **Charges** (King-top towers).
- §3B capture rules: a hop walks one diagonal and captures Whites in transit; a chain is several hops sharing a **cadence** (the first hop's length). Paths **never bounce off the rim** — a hop whose cadence landing overshoots the 10×10 grid settles on the last on-board square and ends the turn (legal only if it captured ≥1 White). Intermediate chain stops are optional. Notation: `c<N>:<from>~<hops>-><rest>` (cadence leading; `<rest>` always on-board).
- **Mandate** (mandatory capture): if any Black tower has a normal-landing capture available, Black must capture this turn.
- **Ram** (suicide capture): landing on an enemy destroys the tower but captures all enemies on the path; never mandatory; does not capture the landing piece (so a ram onto the King does NOT capture it — only path-captures do).
- Win: White wins by eliminating all Black towers **or by holding its king on rank 8 for three of White's turns without being in check** (FEN `r8` counter; any check resets it); Black wins by checkmating (capturing) White's king.

## Code Style

- **Python**: type hints throughout, dict shapes preserved at the `PyVariantClient` boundary (these mirror what the old Scala server returned). Keep changes surgical.
- **C++** (`cpp/src/`): mirrors PyVariant and must stay equivalent — any rule change goes in both, with the `tests/test_cpp_*.py` parity verified before committing.
