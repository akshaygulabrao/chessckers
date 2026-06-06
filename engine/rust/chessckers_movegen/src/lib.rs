//! Native Rust implementation of Chessckers Black-side capture move generation.
//!
//! Mirrors `chessckers_engine.variant_py.moves_black.black_diagonal_capture_moves`
//! line-for-line. The Python version was the dominant cost in MCTS (~75% of
//! `status_and_legal`); this is a 1:1 port with the bouncer-path geometry
//! precomputed at first call.
//!
//! Surface to Python:
//!   black_diagonal_capture_moves(occupied: u64, occupied_white: u64,
//!                                turn_is_black: bool, stacks: dict[int, str])
//!     -> list[dict]
//!   black_mandatory_capture_active(occupied, occupied_white, stacks) -> bool
//!
//! `stacks` is a Python dict from chess.Square (0..63) to a pieces string
//! (e.g. "ssk", bottom-to-top, alphabet {s, S, k}).
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};
use std::collections::HashMap;
use std::sync::OnceLock;

// -------- Geometry --------

const FORWARD_DIAGS: [(i8, i8); 2] = [(-1, -1), (1, -1)];
const ALL_DIAGS: [(i8, i8); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];

#[inline(always)]
fn on_board(f: i8, r: i8) -> bool {
    (0..=7).contains(&f) && (0..=7).contains(&r)
}

#[inline(always)]
fn on_grid(f: i8, r: i8) -> bool {
    // 8×8 board plus the one-square rim ring.
    (-1..=8).contains(&f) && (-1..=8).contains(&r)
}

#[inline(always)]
fn sq_idx(f: i8, r: i8) -> u8 {
    ((r as u8) << 3) | (f as u8)
}

/// 64 square names ("a1".."h8"), populated once.
fn sq_names() -> &'static [String; 64] {
    static SQ: OnceLock<[String; 64]> = OnceLock::new();
    SQ.get_or_init(|| {
        let mut out: [String; 64] = std::array::from_fn(|_| String::new());
        for sq in 0u8..64 {
            let f = (sq & 7) as i8;
            let r = (sq >> 3) as i8;
            out[sq as usize] = format!(
                "{}{}",
                (b'a' + f as u8) as char,
                (r + 1)
            );
        }
        out
    })
}

/// 10×10 grid → 2-char key. Same convention as Python: rim files 'z'/'i',
/// rim ranks '0'/'9'. Key format: "<file_char><rank_digit>".
fn coord_key(f: i8, r: i8) -> String {
    let fc = match f {
        -1 => 'z',
        0..=7 => (b'a' + f as u8) as char,
        8 => 'i',
        _ => '?',
    };
    // rank digit: r+1 mapped to '0'.. '9' (rim 9 covers rank 8)
    let rn = (r + 1) as i32;
    format!("{}{}", fc, rn)
}

/// Inverse of coord_key — only used for parsing rim landings on chain
/// continuation. Returns (f, r) or None on parse failure.
fn parse_waypoint_key(s: &str) -> Option<(i8, i8)> {
    if s.len() != 2 {
        return None;
    }
    let mut chars = s.chars();
    let fc = chars.next()?;
    let rc = chars.next()?;
    let f: i8 = match fc {
        'z' => -1,
        'a'..='h' => (fc as u8 - b'a') as i8,
        'i' => 8,
        _ => return None,
    };
    let rd: i8 = (rc as u8).checked_sub(b'0')? as i8 - 1;
    Some((f, rd))
}

// -------- Capture-path table --------
//
// Precomputed per-(start_f, start_r, df0, dr0) straight-diagonal trajectory.
// Same as the Python `_CAPTURE_PATHS` table — pure geometry, no board state.
// Per spec §3B step 3 (no-bounce rule), the path is a straight diagonal; if
// a step would go off the 10×10 grid the trace just terminates. The
// `df`/`dr` fields always match the original direction; `did_bounce` is
// always false (kept on PathStep for ABI shape parity with consumers that
// destructure the step tuple).

#[derive(Clone, Debug)]
struct PathStep {
    f: i8,
    r: i8,
    sq: i16, // -1 if rim
    key: String,
    df: i8,
    dr: i8,
    did_bounce: bool,
}

const MAX_HOP_STEPS: usize = 26;

fn capture_paths() -> &'static HashMap<(i8, i8, i8, i8), Vec<PathStep>> {
    static PATHS: OnceLock<HashMap<(i8, i8, i8, i8), Vec<PathStep>>> = OnceLock::new();
    PATHS.get_or_init(|| {
        let mut paths = HashMap::new();
        for f0 in -1..=8 {
            for r0 in -1..=8 {
                for &df0 in &[-1i8, 1] {
                    for &dr0 in &[-1i8, 1] {
                        let mut steps = Vec::with_capacity(MAX_HOP_STEPS);
                        let mut f = f0;
                        let mut r = r0;
                        for _ in 0..MAX_HOP_STEPS {
                            let nf = f + df0;
                            let nr = r + dr0;
                            if nf < -1 || nf > 8 || nr < -1 || nr > 8 {
                                break;
                            }
                            f = nf;
                            r = nr;
                            let on_now = on_board(f, r);
                            let sq: i16 = if on_now { sq_idx(f, r) as i16 } else { -1 };
                            steps.push(PathStep {
                                f,
                                r,
                                sq,
                                key: coord_key(f, r),
                                df: df0,
                                dr: dr0,
                                did_bounce: false,
                            });
                        }
                        paths.insert((f0, r0, df0, dr0), steps);
                    }
                }
            }
        }
        paths
    })
}

// -------- Square ownership (bitboard fast path) --------

const SQ_EMPTY: u8 = 0;
const SQ_WHITE: u8 = 1;
const SQ_BLACK: u8 = 2;

#[inline(always)]
fn owner(occupied: u64, occupied_white: u64, sq: u8) -> u8 {
    let mask = 1u64 << sq;
    if occupied & mask == 0 {
        SQ_EMPTY
    } else if occupied_white & mask != 0 {
        SQ_WHITE
    } else {
        SQ_BLACK
    }
}

// -------- Pure-Rust move generator --------

/// Equivalent of Python CaptureHop dataclass.
#[derive(Clone, Debug)]
struct CaptureHop {
    direction: (i8, i8),
    landing_key: String,
    landing_square: Option<u8>,
    captures: Vec<u8>,
    waypoints: Vec<String>,
    is_suicide: bool,
    crossed_rank1: bool,
    // §3B: landing distance k (= waypoints.len() for on-grid landings; one past
    // the last on-grid step for an off-grid overshoot, so it can exceed it).
    cadence: usize,
    // True when the cadence landing fell off the 10x10 grid: captured its path
    // Whites, can't land, settles on the last on-board square and ends the chain.
    is_overshoot: bool,
}

fn find_capture_hops(
    occupied: u64,
    occupied_white: u64,
    stacks: &HashMap<u8, Vec<u8>>,
    f0: i8,
    r0: i8,
    df0: i8,
    dr0: i8,
    n: usize,
) -> Vec<CaptureHop> {
    let mut options: Vec<CaptureHop> = Vec::new();
    let mut captures_so_far: Vec<u8> = Vec::new();
    let mut captured_set: u64 = 0;
    let mut waypoints_so_far: Vec<String> = Vec::new();

    let mut crossed_rank1 = false;
    let mut friendly_blocked = false;

    let path = match capture_paths().get(&(f0, r0, df0, dr0)) {
        Some(p) => p,
        None => return options,
    };
    let max_step = n + 1;

    for (step_idx, step) in path.iter().enumerate() {
        if step_idx >= max_step {
            break;
        }
        let cur_key = step.key.clone();
        waypoints_so_far.push(cur_key.clone());
        if step.r == 0 {
            crossed_rank1 = true;
        }
        let step_num = step_idx + 1; // 1-based landing distance k

        if step.sq >= 0 {
            let sq = step.sq as u8;
            let cap_mask = 1u64 << sq;
            if captured_set & cap_mask != 0 {
                // Revisit of an already-captured (now empty) square.
                if !captures_so_far.is_empty() {
                    options.push(CaptureHop {
                        direction: (step.df, step.dr),
                        landing_key: cur_key.clone(),
                        landing_square: Some(sq),
                        captures: captures_so_far.clone(),
                        waypoints: waypoints_so_far.clone(),
                        is_suicide: false,
                        crossed_rank1,
                        cadence: step_num,
                        is_overshoot: false,
                    });
                }
            } else {
                let o = owner(occupied, occupied_white, sq);
                match o {
                    SQ_EMPTY => {
                        if !captures_so_far.is_empty() {
                            options.push(CaptureHop {
                                direction: (step.df, step.dr),
                                landing_key: cur_key.clone(),
                                landing_square: Some(sq),
                                captures: captures_so_far.clone(),
                                waypoints: waypoints_so_far.clone(),
                                is_suicide: false,
                                crossed_rank1,
                                cadence: step_num,
                                is_overshoot: false,
                            });
                        }
                    }
                    SQ_BLACK if stacks.contains_key(&sq) => {
                        // Friendly tower blocks the trace (NOT an off-grid exit).
                        friendly_blocked = true;
                        break;
                    }
                    _ => {
                        // White piece. Rams require k > d (a path capture must
                        // already exist). Emit ram BEFORE adding this white to
                        // captures so the landing White isn't double-counted.
                        if !captures_so_far.is_empty() {
                            options.push(CaptureHop {
                                direction: (step.df, step.dr),
                                landing_key: cur_key.clone(),
                                landing_square: Some(sq),
                                captures: captures_so_far.clone(),
                                waypoints: waypoints_so_far.clone(),
                                is_suicide: true,
                                crossed_rank1,
                                cadence: step_num,
                                is_overshoot: false,
                            });
                        }
                        captures_so_far.push(sq);
                        captured_set |= cap_mask;
                    }
                }
            }
        } else {
            // Rim square (T) — never friendly.
            if !captures_so_far.is_empty() {
                options.push(CaptureHop {
                    direction: (step.df, step.dr),
                    landing_key: cur_key.clone(),
                    landing_square: None,
                    captures: captures_so_far.clone(),
                    waypoints: waypoints_so_far.clone(),
                    is_suicide: false,
                    crossed_rank1,
                    cadence: step_num,
                    is_overshoot: false,
                });
            }
        }
    }

    // §3B off-grid overshoot: the straight path left the 10x10 grid before the
    // cadence limit (path shorter than n+1, not friendly-blocked) AND >= 1 White
    // was captured. The hop can't land off-grid, so it settles on the last
    // on-board square (resolved in build_final_move) and ends the chain. Cadence
    // is one past the last on-grid square; distinct from the rim landing at the
    // same key, so dedup must keep both.
    if !friendly_blocked && !captures_so_far.is_empty() && path.len() < max_step {
        options.push(CaptureHop {
            direction: (df0, dr0),
            landing_key: waypoints_so_far.last().unwrap().clone(),
            landing_square: None,
            captures: captures_so_far.clone(),
            waypoints: waypoints_so_far.clone(),
            is_suicide: false,
            crossed_rank1,
            cadence: path.len() + 1,
            is_overshoot: true,
        });
    }
    options
}

#[inline]
fn dirs_for_top(top: u8) -> &'static [(i8, i8)] {
    if top == b'k' {
        &ALL_DIAGS
    } else {
        &FORWARD_DIAGS
    }
}

fn next_capture_options(
    occupied: u64,
    occupied_white: u64,
    stacks: &HashMap<u8, Vec<u8>>,
    cf: i8,
    cr: i8,
    cur_stack: &[u8],
    last_dir: Option<(i8, i8)>,
    n: usize,
    cadence: Option<usize>,
    include_suicide: bool,
) -> Vec<CaptureHop> {
    if cur_stack.is_empty() {
        return Vec::new();
    }
    let dirs = dirs_for_top(*cur_stack.last().unwrap());
    let mut options: Vec<CaptureHop> = Vec::new();
    for &(df, dr) in dirs {
        if let Some((ldf, ldr)) = last_dir {
            if df == -ldf && dr == -ldr {
                continue;
            }
        }
        for hop in find_capture_hops(occupied, occupied_white, stacks, cf, cr, df, dr, n) {
            options.push(hop);
        }
    }
    if !include_suicide {
        options.retain(|h| !h.is_suicide);
    }
    if let Some(c) = cadence {
        options.retain(|h| h.cadence == c);
    }
    // Dedup by (direction, landing_key, captures, is_suicide, is_overshoot,
    // cadence) — a rim landing and an off-grid overshoot can share the first
    // four but are distinct moves, so cadence/is_overshoot are part of identity.
    let mut seen: std::collections::HashSet<(i8, i8, String, Vec<u8>, bool, bool, usize)> =
        std::collections::HashSet::new();
    let mut deduped: Vec<CaptureHop> = Vec::with_capacity(options.len());
    for h in options {
        let key = (
            h.direction.0,
            h.direction.1,
            h.landing_key.clone(),
            h.captures.clone(),
            h.is_suicide,
            h.is_overshoot,
            h.cadence,
        );
        if seen.insert(key) {
            deduped.push(h);
        }
    }
    deduped
}

#[inline]
fn hop_promotes(hop: &CaptureHop) -> bool {
    if hop.crossed_rank1 {
        return true;
    }
    if let Some(sq) = hop.landing_square {
        if (sq >> 3) == 0 {
            return true;
        }
    }
    false
}

#[inline]
fn promote_all_stones(stack: &[u8]) -> Vec<u8> {
    stack
        .iter()
        .map(|&c| if c == b's' || c == b'S' { b'k' } else { c })
        .collect()
}

/// Compact representation we materialize into Python dicts at the boundary.
#[derive(Clone)]
struct ChainMove {
    uci: String,
    from_name: String,
    to_name: String,
    piece: &'static str,
    capture: Option<String>,
    waypoints: Option<Vec<String>>,
    chain_hops: Vec<String>,
    chain_all_captures: Vec<String>,
    is_suicide: bool,
    chain_promotes: bool,
    cadence: usize,
}

fn build_final_move(
    chain_start: u8,
    orig_stack: &[u8],
    hops: &[CaptureHop],
) -> ChainMove {
    let is_suicide_chain = !hops.is_empty() && hops.last().unwrap().is_suicide;
    let mut all_captures: Vec<u8> = Vec::new();
    let mut all_waypoints: Vec<String> = Vec::new();
    let mut hop_keys: Vec<String> = Vec::new();
    for h in hops {
        all_captures.extend(&h.captures);
        all_waypoints.extend(h.waypoints.iter().cloned());
        hop_keys.push(h.landing_key.clone());
    }

    // Final landing.
    let last_landing = hops.last().and_then(|h| h.landing_square);
    let final_landing: u8 = match last_landing {
        Some(sq) => sq,
        None => {
            // End-of-turn fallback: walk waypoints backwards for last on-board key.
            let mut fallback = chain_start;
            for wp in all_waypoints.iter().rev() {
                if let Some((f, r)) = parse_waypoint_key(wp) {
                    if on_board(f, r) {
                        fallback = sq_idx(f, r);
                        break;
                    }
                }
            }
            fallback
        }
    };

    // Final top piece.
    let final_top: u8 = if is_suicide_chain {
        *orig_stack.last().unwrap()
    } else {
        let mut stack_thru: Vec<u8> = orig_stack.to_vec();
        for h in hops {
            if hop_promotes(h) {
                stack_thru = promote_all_stones(&stack_thru);
            }
        }
        *stack_thru.last().unwrap()
    };

    let sq_n = sq_names();
    let from_name = sq_n[chain_start as usize].clone();
    let dest_name = sq_n[final_landing as usize].clone();

    let capture: Option<String> = if !all_captures.is_empty() {
        Some(sq_n[all_captures[0] as usize].clone())
    } else if is_suicide_chain {
        Some(sq_n[final_landing as usize].clone())
    } else {
        None
    };

    // §3B notation: c<N>:<from>~<hop landings>-><rest>. Cadence (the first
    // hop's k) leads; <rest> (always on-board) always shown; hop keys are the
    // on-grid landing keys (for an overshoot, the last on-grid square reached).
    let cadence = hops[0].cadence;
    let uci = format!("c{}:{}~{}->{}", cadence, from_name, hop_keys.join("~"), dest_name);
    let waypoints_field = if hops.len() > 1 { Some(all_waypoints.clone()) } else { None };

    let all_cap_names: Vec<String> = all_captures
        .iter()
        .map(|&sq| sq_n[sq as usize].clone())
        .collect();
    let chain_promotes_any = hops.iter().any(hop_promotes);

    ChainMove {
        uci,
        from_name,
        to_name: dest_name,
        piece: if final_top == b'k' { "king" } else { "pawn" },
        capture,
        waypoints: waypoints_field,
        chain_hops: hop_keys,
        chain_all_captures: all_cap_names,
        is_suicide: is_suicide_chain,
        chain_promotes: chain_promotes_any,
        cadence,
    }
}

/// Apply a hop to (occupied, occupied_white, stacks) — returns new state.
/// Mirrors the Python explore() inner block that mutates new_board / new_stacks.
fn apply_hop(
    mut occupied: u64,
    mut occupied_white: u64,
    mut stacks: HashMap<u8, Vec<u8>>,
    cf: i8,
    cr: i8,
    cur_stack: &[u8],
    hop: &CaptureHop,
) -> (u64, u64, HashMap<u8, Vec<u8>>, Vec<u8>) {
    // Remove moving tower from current square (only if on board).
    if on_board(cf, cr) {
        let cur_sq = sq_idx(cf, cr);
        let m = !(1u64 << cur_sq);
        occupied &= m;
        occupied_white &= m;
        stacks.remove(&cur_sq);
    }
    // Capture path Whites.
    for &cap_sq in &hop.captures {
        let m = !(1u64 << cap_sq);
        occupied &= m;
        occupied_white &= m;
    }
    // Promote in-transit if path touched rank 1.
    let land_stack: Vec<u8> = if hop_promotes(hop) {
        promote_all_stones(cur_stack)
    } else {
        cur_stack.to_vec()
    };
    // Place tower at landing.
    if let Some(sq) = hop.landing_square {
        let mask = 1u64 << sq;
        occupied |= mask;
        occupied_white &= !mask; // Black
        stacks.insert(sq, land_stack.clone());
    }
    (occupied, occupied_white, stacks, land_stack)
}

fn enumerate_chains_recursive(
    occupied: u64,
    occupied_white: u64,
    king_sq: i64,
    stacks: &HashMap<u8, Vec<u8>>,
    chain_start: u8,
    cf: i8,
    cr: i8,
    cur_stack: &[u8],
    last_dir: Option<(i8, i8)>,
    hops_so_far: Vec<CaptureHop>,
    cadence: Option<usize>,
    n: usize,
    orig_stack: &[u8],
    results: &mut Vec<ChainMove>,
) {
    // White-king-captured short-circuit: if a prior hop already captured the
    // White king, the game is over — stop the chain (mirrors Python explore).
    // Required for correctness once chains can continue past a king capture.
    if king_sq >= 0 && (occupied_white & (1u64 << king_sq)) == 0 {
        return;
    }
    let options = next_capture_options(
        occupied,
        occupied_white,
        stacks,
        cf,
        cr,
        cur_stack,
        last_dir,
        n,
        cadence,
        false,
    );
    if options.is_empty() {
        // Nothing further; whatever chain reached here was already emitted as a
        // "stop" by the parent loop below (every hop emits its stop-move).
        return;
    }
    for hop in &options {
        let mut hops_next = hops_so_far.clone();
        hops_next.push(hop.clone());
        // Emit the chain ending at this hop. For an off-grid overshoot this is
        // the settle move and the chain ENDS. Otherwise it's the optional stop
        // (§3B: continuing is optional) and we also recurse to extend it.
        results.push(build_final_move(chain_start, orig_stack, &hops_next));
        if hop.is_overshoot {
            continue;
        }
        let (new_occ, new_occ_w, new_stacks, land_stack) = apply_hop(
            occupied,
            occupied_white,
            stacks.clone(),
            cf,
            cr,
            cur_stack,
            hop,
        );
        let (nf, nr) = match hop.landing_square {
            Some(sq) => ((sq & 7) as i8, (sq >> 3) as i8),
            None => parse_waypoint_key(&hop.landing_key).unwrap_or((cf, cr)),
        };
        let next_cadence = cadence.or(Some(hop.cadence));
        enumerate_chains_recursive(
            new_occ,
            new_occ_w,
            king_sq,
            &new_stacks,
            chain_start,
            nf,
            nr,
            &land_stack,
            Some(hop.direction),
            hops_next,
            next_cadence,
            n,
            orig_stack,
            results,
        );
    }
}

fn enumerate_chains(
    occupied: u64,
    occupied_white: u64,
    king_sq: i64,
    stacks: &HashMap<u8, Vec<u8>>,
    chain_start: u8,
) -> Vec<ChainMove> {
    let orig_stack = match stacks.get(&chain_start) {
        Some(s) if !s.is_empty() => s.clone(),
        _ => return Vec::new(),
    };
    let n = orig_stack.len();
    let cf0 = (chain_start & 7) as i8;
    let cr0 = (chain_start >> 3) as i8;
    let mut results: Vec<ChainMove> = Vec::new();
    enumerate_chains_recursive(
        occupied,
        occupied_white,
        king_sq,
        stacks,
        chain_start,
        cf0,
        cr0,
        &orig_stack,
        None,
        Vec::new(),
        None,
        n,
        &orig_stack,
        &mut results,
    );
    results
}

fn first_hop_suicides(
    occupied: u64,
    occupied_white: u64,
    stacks: &HashMap<u8, Vec<u8>>,
    chain_start: u8,
) -> Vec<ChainMove> {
    let pieces = match stacks.get(&chain_start) {
        Some(p) if !p.is_empty() => p.clone(),
        _ => return Vec::new(),
    };
    let n = pieces.len();
    let dirs = dirs_for_top(*pieces.last().unwrap());
    let cf = (chain_start & 7) as i8;
    let cr = (chain_start >> 3) as i8;
    let mut moves: Vec<ChainMove> = Vec::new();
    for &(df, dr) in dirs {
        for hop in find_capture_hops(occupied, occupied_white, stacks, cf, cr, df, dr, n) {
            if hop.is_suicide {
                moves.push(build_final_move(chain_start, &pieces, &[hop]));
            }
        }
    }
    moves
}

fn black_diagonal_capture_moves_native(
    occupied: u64,
    occupied_white: u64,
    king_sq: i64,
    stacks: &HashMap<u8, Vec<u8>>,
) -> Vec<ChainMove> {
    let mut moves: Vec<ChainMove> = Vec::new();
    // Iterate stable order (matches Python dict iteration insertion order).
    let mut keys: Vec<u8> = stacks.keys().copied().collect();
    keys.sort_unstable();
    for sq in keys {
        if let Some(pieces) = stacks.get(&sq) {
            if pieces.is_empty() {
                continue;
            }
            moves.extend(enumerate_chains(occupied, occupied_white, king_sq, stacks, sq));
            moves.extend(first_hop_suicides(occupied, occupied_white, stacks, sq));
        }
    }
    moves
}

// -------- King-capture detection (bool early-exit) --------
//
// Used by white check detection (_is_white_in_chessckers_check), which only
// needs to know whether ANY Black diagonal hop/chain/overshoot/ram captures the
// White king — not the full move list. Mirrors black_diagonal_capture_moves_native
// exactly (same chain enumeration + first-hop suicides), but returns true on the
// first hop whose path-captures include `king`, builds no ChainMove/strings, and
// prunes the search. This is the hot path in self-play (one call per White
// candidate move); avoiding the per-move dict marshalling is a large win.

fn chain_captures_king_rec(
    occupied: u64,
    occupied_white: u64,
    stacks: &HashMap<u8, Vec<u8>>,
    cf: i8,
    cr: i8,
    cur_stack: &[u8],
    last_dir: Option<(i8, i8)>,
    cadence: Option<usize>,
    n: usize,
    king: u8,
) -> bool {
    // King already gone (a prior hop captured it): caller returns true before
    // recursing, so this only guards against a stale state — stop the chain.
    if (occupied_white & (1u64 << king)) == 0 {
        return false;
    }
    let options = next_capture_options(
        occupied,
        occupied_white,
        stacks,
        cf,
        cr,
        cur_stack,
        last_dir,
        n,
        cadence,
        false,
    );
    for hop in &options {
        if hop.captures.contains(&king) {
            return true;
        }
        if hop.is_overshoot {
            continue;
        }
        let (new_occ, new_occ_w, new_stacks, land_stack) = apply_hop(
            occupied,
            occupied_white,
            stacks.clone(),
            cf,
            cr,
            cur_stack,
            hop,
        );
        let (nf, nr) = match hop.landing_square {
            Some(sq) => ((sq & 7) as i8, (sq >> 3) as i8),
            None => parse_waypoint_key(&hop.landing_key).unwrap_or((cf, cr)),
        };
        let next_cadence = cadence.or(Some(hop.cadence));
        if chain_captures_king_rec(
            new_occ,
            new_occ_w,
            &new_stacks,
            nf,
            nr,
            &land_stack,
            Some(hop.direction),
            next_cadence,
            n,
            king,
        ) {
            return true;
        }
    }
    false
}

fn black_can_capture_white_king_native(
    occupied: u64,
    occupied_white: u64,
    king_sq: i64,
    stacks: &HashMap<u8, Vec<u8>>,
) -> bool {
    if king_sq < 0 || (occupied_white & (1u64 << (king_sq as u8))) == 0 {
        return false;
    }
    let king = king_sq as u8;
    for (sq, pieces) in stacks.iter() {
        if pieces.is_empty() {
            continue;
        }
        let n = pieces.len();
        let cf = (sq & 7) as i8;
        let cr = (sq >> 3) as i8;
        // Non-suicide chains.
        if chain_captures_king_rec(
            occupied,
            occupied_white,
            stacks,
            cf,
            cr,
            pieces,
            None,
            None,
            n,
            king,
        ) {
            return true;
        }
        // First-hop suicides (rams) — a ram captures its path Whites in transit,
        // which may include the king.
        let dirs = dirs_for_top(*pieces.last().unwrap());
        for &(df, dr) in dirs {
            for hop in find_capture_hops(occupied, occupied_white, stacks, cf, cr, df, dr, n) {
                if hop.is_suicide && hop.captures.contains(&king) {
                    return true;
                }
            }
        }
    }
    false
}

// -------- Quiet moves (Phase 2A + sprint) --------

/// Output for a quiet (non-capturing) Black move.
#[derive(Clone)]
struct QuietMove {
    uci: String,
    from_name: String,
    to_name: String,
    piece: &'static str,
}

fn build_quiet(from_name: &str, to_sq: u8, top: u8) -> QuietMove {
    let to_name = sq_names()[to_sq as usize].clone();
    let piece = if top == b'k' { "king" } else { "pawn" };
    QuietMove {
        uci: format!("{}{}", from_name, to_name),
        from_name: from_name.to_string(),
        to_name,
        piece,
    }
}

fn black_diagonal_quiet_moves_native(
    occupied: u64,
    occupied_white: u64,
    stacks: &HashMap<u8, Vec<u8>>,
) -> Vec<QuietMove> {
    let mut moves: Vec<QuietMove> = Vec::new();
    let mut keys: Vec<u8> = stacks.keys().copied().collect();
    keys.sort_unstable();
    for from_sq in keys {
        let pieces = match stacks.get(&from_sq) {
            Some(p) if !p.is_empty() => p,
            _ => continue,
        };
        let height = pieces.len();
        let top = *pieces.last().unwrap();
        let dirs = dirs_for_top(top);
        let from_file = (from_sq & 7) as i8;
        let from_rank = (from_sq >> 3) as i8;
        let from_name = sq_names()[from_sq as usize].clone();

        for &(df, dr) in dirs {
            for k in 1..=(height as i8) {
                let tf = from_file + k * df;
                let tr = from_rank + k * dr;
                if !on_board(tf, tr) {
                    break;
                }
                let to_sq = sq_idx(tf, tr);
                let o = owner(occupied, occupied_white, to_sq);
                if o == SQ_EMPTY {
                    moves.push(build_quiet(&from_name, to_sq, top));
                    continue;
                }
                if o == SQ_BLACK && stacks.contains_key(&to_sq) {
                    // Friendly merge — emit and stop.
                    moves.push(build_quiet(&from_name, to_sq, top));
                }
                break;
            }
        }
        // Sprint: height-1 unmoved Stone-top at rank 8, two squares forward.
        if height == 1 && top == b's' && from_rank == 7 {
            for &(df, dr) in &FORWARD_DIAGS {
                let int_f = from_file + df;
                let int_r = from_rank + dr;
                if !on_board(int_f, int_r) {
                    continue;
                }
                let int_sq = sq_idx(int_f, int_r);
                if owner(occupied, occupied_white, int_sq) != SQ_EMPTY {
                    continue;
                }
                let tf = from_file + 2 * df;
                let tr = from_rank + 2 * dr;
                if !on_board(tf, tr) {
                    continue;
                }
                let to_sq = sq_idx(tf, tr);
                let o = owner(occupied, occupied_white, to_sq);
                if o == SQ_EMPTY {
                    moves.push(build_quiet(&from_name, to_sq, top));
                } else if o == SQ_BLACK && stacks.contains_key(&to_sq) {
                    moves.push(build_quiet(&from_name, to_sq, top));
                }
            }
        }
    }
    moves
}

// -------- Deploy moves (Phase 2B) --------

/// A deploy emits a top-s sub-tower into a diagonal landing.
#[derive(Clone)]
struct DeployMove {
    uci: String,
    from_name: String,
    to_name: String,
    piece: &'static str,
    deploy_count: usize,
}

fn build_deploy(from_name: &str, to_sq: u8, top: u8, s: usize) -> DeployMove {
    let to_name = sq_names()[to_sq as usize].clone();
    DeployMove {
        uci: format!("{}{}[{}]", from_name, to_name, s),
        from_name: from_name.to_string(),
        to_name,
        piece: if top == b'k' { "king" } else { "pawn" },
        deploy_count: s,
    }
}

fn black_deploy_moves_native(
    occupied: u64,
    occupied_white: u64,
    stacks: &HashMap<u8, Vec<u8>>,
) -> Vec<DeployMove> {
    let mut moves: Vec<DeployMove> = Vec::new();
    let mut keys: Vec<u8> = stacks.keys().copied().collect();
    keys.sort_unstable();
    for from_sq in keys {
        let pieces = match stacks.get(&from_sq) {
            Some(p) if !p.is_empty() => p,
            _ => continue,
        };
        let n = pieces.len();
        if n < 2 {
            continue;
        }
        let top = *pieces.last().unwrap();
        let dirs = dirs_for_top(top);
        let from_file = (from_sq & 7) as i8;
        let from_rank = (from_sq >> 3) as i8;
        let from_name = sq_names()[from_sq as usize].clone();

        for s in 1..n {
            for &(df, dr) in dirs {
                for k in 1..=(s as i8) {
                    let tf = from_file + k * df;
                    let tr = from_rank + k * dr;
                    if !on_board(tf, tr) {
                        break;
                    }
                    let to_sq = sq_idx(tf, tr);
                    let o = owner(occupied, occupied_white, to_sq);
                    if o == SQ_EMPTY {
                        moves.push(build_deploy(&from_name, to_sq, top, s));
                        continue;
                    }
                    if o == SQ_BLACK && stacks.contains_key(&to_sq) {
                        moves.push(build_deploy(&from_name, to_sq, top, s));
                    }
                    break;
                }
            }
        }
    }
    moves
}

// -------- Charge moves (Phase 2E) --------

const ORTHO_DIRS: [(i8, i8); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

/// A charge: orthogonal King-top tower move, with optional demotion choice.
#[derive(Clone)]
struct ChargeMove {
    uci: String,
    from_name: String,
    to_name: String,
    piece: &'static str,
    capture: Option<String>,
    // Some([rim_key]) for an overshoot charge (rim landing, fall back to
    // to_name); None for an on-board landing. Mirrors the Python charge dict:
    // both the apply flag and the policy-encoding key.
    waypoints: Option<Vec<String>>,
    demoted_kings: Option<Vec<usize>>,
    demotions_required: Option<usize>,
    source_king_positions: Option<Vec<usize>>,
}

/// In-place generation of all r-combinations of [0, n) — used to enumerate
/// king-demotion choices when n_kings > d.
fn combinations(items: &[usize], r: usize) -> Vec<Vec<usize>> {
    let n = items.len();
    if r == 0 || r > n {
        return Vec::new();
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut idx: Vec<usize> = (0..r).collect();
    loop {
        out.push(idx.iter().map(|&i| items[i]).collect());
        // Find rightmost index that can be incremented.
        let mut i = r;
        while i > 0 {
            i -= 1;
            if idx[i] != i + n - r {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        idx[i] += 1;
        for j in (i + 1)..r {
            idx[j] = idx[j - 1] + 1;
        }
        if idx[0] > n - r {
            break;
        }
    }
    out
}

fn black_charge_moves_native(
    occupied: u64,
    occupied_white: u64,
    stacks: &HashMap<u8, Vec<u8>>,
) -> Vec<ChargeMove> {
    let mut moves: Vec<ChargeMove> = Vec::new();
    let mut keys: Vec<u8> = stacks.keys().copied().collect();
    keys.sort_unstable();
    let sq_n = sq_names();
    for from_sq in keys {
        let pieces = match stacks.get(&from_sq) {
            Some(p) if !p.is_empty() && *p.last().unwrap() == b'k' => p,
            _ => continue,
        };
        let n_kings = pieces.iter().filter(|&&p| p == b'k').count();
        if n_kings == 0 {
            continue;
        }
        let from_file = (from_sq & 7) as i8;
        let from_rank = (from_sq >> 3) as i8;
        let from_name = sq_n[from_sq as usize].clone();
        let king_positions: Vec<usize> =
            pieces.iter().enumerate().filter(|(_, &p)| p == b'k').map(|(i, _)| i + 1).collect();

        for &(df, dr) in &ORTHO_DIRS {
            let mut stop_after = false;
            for d in 1..=n_kings {
                if stop_after {
                    break;
                }
                // Path scan 1..d-1. Allow rim squares (no pieces, no
                // captures). Off-grid (file/rank outside [-1, 8]) ends
                // the charge for this and higher d.
                let mut blocked = false;
                let mut off_grid = false;
                let mut path_captures: Vec<String> = Vec::new();
                let mut last_on_board_sq: Option<u8> = None;
                for k in 1..d {
                    let pf = from_file + (k as i8) * df;
                    let pr = from_rank + (k as i8) * dr;
                    if pf < -1 || pf > 8 || pr < -1 || pr > 8 {
                        off_grid = true;
                        break;
                    }
                    if (0..=7).contains(&pf) && (0..=7).contains(&pr) {
                        let psq = sq_idx(pf, pr);
                        let powner = owner(occupied, occupied_white, psq);
                        if powner == SQ_BLACK && stacks.contains_key(&psq) {
                            blocked = true;
                            break;
                        }
                        if powner == SQ_WHITE {
                            path_captures.push(sq_n[psq as usize].clone());
                        }
                        last_on_board_sq = Some(psq);
                    }
                    // else: rim square, no action
                }
                if off_grid {
                    break;
                }
                if blocked {
                    break;
                }
                let tf = from_file + (d as i8) * df;
                let tr = from_rank + (d as i8) * dr;
                if tf < -1 || tf > 8 || tr < -1 || tr > 8 {
                    break; // off-grid landing
                }
                // Landing classification: on-board or rim-with-fallback.
                // `rim_landing_key` = the on-grid key of the actual rim landing
                // for an overshoot charge (None for an on-board land).
                let (to_sq, to_name, towner, is_ram, is_friendly_merge, rim_landing_key) =
                    if (0..=7).contains(&tf) && (0..=7).contains(&tr) {
                        let s = sq_idx(tf, tr);
                        let n = sq_n[s as usize].clone();
                        let o = owner(occupied, occupied_white, s);
                        let r = o == SQ_WHITE;
                        let m = o == SQ_BLACK && stacks.contains_key(&s);
                        (s, n, o, r, m, None)
                    } else {
                        // Rim landing → fallback to last on-board square.
                        // If no on-board path step exists (d=1 rim), skip.
                        match last_on_board_sq {
                            Some(s) => {
                                let n = sq_n[s as usize].clone();
                                (s, n, SQ_EMPTY, false, false, Some(coord_key(tf, tr)))
                            }
                            None => continue,
                        }
                    };

                // Notation: an overshoot charge spells out the rim landing it
                // aimed at, then `->` its resting square (`e2e0->e1`), so the
                // intent never reads as a ram. `waypoints` carries the rim key.
                let (landing_repr, charge_waypoints): (String, Option<Vec<String>>) =
                    match &rim_landing_key {
                        None => (to_name.clone(), None),
                        Some(k) => (format!("{}->{}", k, to_name), Some(vec![k.clone()])),
                    };

                let capture_field: Option<String> = if !path_captures.is_empty() {
                    Some(path_captures[0].clone())
                } else if is_ram {
                    Some(to_name.clone())
                } else {
                    None
                };

                if is_ram {
                    // Per §3C (revised): rams require ≥1 path capture —
                    // the charge must overshoot at least one enemy before
                    // crashing. A dist-1 charge has 0 intermediate squares
                    // and is illegal as a ram.
                    if !path_captures.is_empty() {
                        moves.push(ChargeMove {
                            uci: format!("{}{}", from_name, to_name),
                            from_name: from_name.clone(),
                            to_name,
                            piece: "king",
                            capture: capture_field,
                            waypoints: None,
                            demoted_kings: None,
                            demotions_required: None,
                            source_king_positions: None,
                        });
                    }
                    continue;
                }

                if n_kings == d {
                    // Forced demotion — null choice fields.
                    let mut new_pieces = pieces.clone();
                    for &pos in &king_positions {
                        new_pieces[pos - 1] = b'S';
                    }
                    let resulting_top = *new_pieces.last().unwrap();
                    moves.push(ChargeMove {
                        uci: format!("{}{}", from_name, landing_repr),
                        from_name: from_name.clone(),
                        to_name,
                        piece: if resulting_top == b'k' { "king" } else { "pawn" },
                        capture: capture_field,
                        waypoints: charge_waypoints,
                        demoted_kings: None,
                        demotions_required: None,
                        source_king_positions: None,
                    });
                } else {
                    for choice in combinations(&king_positions, d) {
                        let mut new_pieces = pieces.clone();
                        for &pos in &choice {
                            new_pieces[pos - 1] = b'S';
                        }
                        let resulting_top = *new_pieces.last().unwrap();
                        let choice_str: String = choice
                            .iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        moves.push(ChargeMove {
                            uci: format!("{}{}{{{}}}", from_name, landing_repr, choice_str),
                            from_name: from_name.clone(),
                            to_name: to_name.clone(),
                            piece: if resulting_top == b'k' { "king" } else { "pawn" },
                            capture: capture_field.clone(),
                            waypoints: charge_waypoints.clone(),
                            demoted_kings: Some(choice),
                            demotions_required: Some(d),
                            source_king_positions: Some(king_positions.clone()),
                        });
                    }
                }

                if is_friendly_merge {
                    stop_after = true;
                }
            }
        }
    }
    moves
}

// -------- Combined entry --------

/// Tagged move output. Build PyDicts with a single match at the boundary.
enum AnyMove {
    Quiet(QuietMove),
    Deploy(DeployMove),
    Charge(ChargeMove),
    Chain(ChainMove),
}

fn any_to_pydict<'py>(py: Python<'py>, m: &AnyMove) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("color", "black")?;
    match m {
        AnyMove::Quiet(q) => {
            d.set_item("uci", &q.uci)?;
            d.set_item("from", &q.from_name)?;
            d.set_item("to", &q.to_name)?;
            d.set_item("piece", q.piece)?;
            d.set_item("capture", py.None())?;
            d.set_item("waypoints", py.None())?;
            d.set_item("chainHops", py.None())?;
            d.set_item("promotion", py.None())?;
            d.set_item("demotedKings", py.None())?;
            d.set_item("demotionsRequired", py.None())?;
            d.set_item("sourceKingPositions", py.None())?;
            d.set_item("deployCount", py.None())?;
        }
        AnyMove::Deploy(dm) => {
            d.set_item("uci", &dm.uci)?;
            d.set_item("from", &dm.from_name)?;
            d.set_item("to", &dm.to_name)?;
            d.set_item("piece", dm.piece)?;
            d.set_item("capture", py.None())?;
            d.set_item("waypoints", py.None())?;
            d.set_item("chainHops", py.None())?;
            d.set_item("promotion", py.None())?;
            d.set_item("demotedKings", py.None())?;
            d.set_item("demotionsRequired", py.None())?;
            d.set_item("sourceKingPositions", py.None())?;
            d.set_item("deployCount", dm.deploy_count)?;
        }
        AnyMove::Charge(c) => {
            d.set_item("uci", &c.uci)?;
            d.set_item("from", &c.from_name)?;
            d.set_item("to", &c.to_name)?;
            d.set_item("piece", c.piece)?;
            d.set_item("capture", c.capture.as_ref())?;
            match &c.waypoints {
                Some(v) => d.set_item("waypoints", v)?,
                None => d.set_item("waypoints", py.None())?,
            }
            d.set_item("chainHops", py.None())?;
            d.set_item("promotion", py.None())?;
            match &c.demoted_kings {
                Some(v) => d.set_item("demotedKings", v)?,
                None => d.set_item("demotedKings", py.None())?,
            }
            match c.demotions_required {
                Some(v) => d.set_item("demotionsRequired", v)?,
                None => d.set_item("demotionsRequired", py.None())?,
            }
            match &c.source_king_positions {
                Some(v) => d.set_item("sourceKingPositions", v)?,
                None => d.set_item("sourceKingPositions", py.None())?,
            }
            d.set_item("deployCount", py.None())?;
        }
        AnyMove::Chain(cm) => {
            d.set_item("uci", &cm.uci)?;
            d.set_item("from", &cm.from_name)?;
            d.set_item("to", &cm.to_name)?;
            d.set_item("piece", cm.piece)?;
            d.set_item("capture", cm.capture.as_ref())?;
            match &cm.waypoints {
                Some(v) => d.set_item("waypoints", v)?,
                None => d.set_item("waypoints", py.None())?,
            }
            d.set_item("chainHops", &cm.chain_hops)?;
            d.set_item("promotion", py.None())?;
            d.set_item("demotedKings", py.None())?;
            d.set_item("demotionsRequired", py.None())?;
            d.set_item("sourceKingPositions", py.None())?;
            d.set_item("deployCount", py.None())?;
            d.set_item("_chain_all_captures", &cm.chain_all_captures)?;
            d.set_item("cadence", cm.cadence)?;
            d.set_item("_is_suicide", cm.is_suicide)?;
            d.set_item("_chain_promotes", cm.chain_promotes)?;
        }
    }
    Ok(d)
}

/// Compute the full Black legal-move list, mandate filter applied.
/// This replaces ~75% of `status_and_legal`'s Python work in one call —
/// minimizes Python<->Rust crossings in the MCTS hot path.
fn all_black_legal_moves_native(
    occupied: u64,
    occupied_white: u64,
    king_sq: i64,
    stacks: &HashMap<u8, Vec<u8>>,
) -> Vec<AnyMove> {
    let mut quiet = black_diagonal_quiet_moves_native(occupied, occupied_white, stacks);
    let mut deploy = black_deploy_moves_native(occupied, occupied_white, stacks);
    let mut charge = black_charge_moves_native(occupied, occupied_white, stacks);
    let chain = black_diagonal_capture_moves_native(occupied, occupied_white, king_sq, stacks);

    let mandate = black_mandatory_capture_active_native(occupied, occupied_white, stacks);

    let mut out: Vec<AnyMove> = Vec::with_capacity(
        quiet.len() + deploy.len() + charge.len() + chain.len(),
    );
    if mandate {
        // Capture moves only: charges with non-null `capture`, plus all chain moves.
        for c in charge.drain(..) {
            if c.capture.is_some() {
                out.push(AnyMove::Charge(c));
            }
        }
        for cm in chain {
            out.push(AnyMove::Chain(cm));
        }
    } else {
        for q in quiet.drain(..) {
            out.push(AnyMove::Quiet(q));
        }
        for dm in deploy.drain(..) {
            out.push(AnyMove::Deploy(dm));
        }
        for c in charge.drain(..) {
            out.push(AnyMove::Charge(c));
        }
        for cm in chain {
            out.push(AnyMove::Chain(cm));
        }
    }
    out
}

/// True iff some Black stack can capture a White piece at `target_sq` next
/// turn under Chessckers attack rules. Mirrors moves_white.py's
/// `_square_attacked_by_black_chessckers_py` (walk-based, doesn't model
/// rim-bounce diagonals — those rare cases would require enumerating the
/// bouncer-path here too).
fn square_attacked_by_black_chessckers_native(
    occupied: u64,
    occupied_white: u64,
    stacks: &HashMap<u8, Vec<u8>>,
    target_sq: u8,
) -> bool {
    for (&from_sq, pieces) in stacks {
        if pieces.is_empty() {
            continue;
        }
        let n = pieces.len();
        let top = *pieces.last().unwrap();
        let is_king_top = top == b'k';
        let n_kings: usize = if is_king_top {
            pieces.iter().filter(|&&p| p == b'k').count()
        } else {
            0
        };
        let sf = (from_sq & 7) as i8;
        let sr = (from_sq >> 3) as i8;

        // Diagonal walk: target reachable in 1..n diagonal squares without a
        // friendly Black tower blocking. White pieces in path are free
        // path-captures and don't block.
        let diag_dirs: &[(i8, i8)] = if is_king_top { &ALL_DIAGS } else { &FORWARD_DIAGS };
        for &(df, dr) in diag_dirs {
            for k in 1..=(n as i8) {
                let nf = sf + k * df;
                let nr = sr + k * dr;
                if !on_board(nf, nr) {
                    break;
                }
                let nsq = sq_idx(nf, nr);
                if nsq == target_sq {
                    return true;
                }
                let o = owner(occupied, occupied_white, nsq);
                if o == SQ_BLACK && stacks.contains_key(&nsq) {
                    break;
                }
            }
        }

        // Orthogonal charge (king-top only, n_kings ≥ 2 to have any path
        // square — charge of length 1 to white is a ram = no capture).
        if is_king_top && n_kings >= 2 {
            for &(df, dr) in &ORTHO_DIRS {
                for k in 1..(n_kings as i8) {
                    let nf = sf + k * df;
                    let nr = sr + k * dr;
                    if !on_board(nf, nr) {
                        break;
                    }
                    let nsq = sq_idx(nf, nr);
                    if nsq == target_sq {
                        // Charge to k+1 must land on the grid (board OR rim).
                        // A rim landing means the charge overshoots, captures
                        // the target in transit, and falls back — so the
                        // target is still attacked (e.g. a King-top tower
                        // checking a White king on a board edge ahead of it).
                        // Only an off-grid landing (past the rim) is illegal.
                        let nf2 = sf + (k + 1) * df;
                        let nr2 = sr + (k + 1) * dr;
                        if on_grid(nf2, nr2) {
                            return true;
                        }
                        break;
                    }
                    let o = owner(occupied, occupied_white, nsq);
                    if o == SQ_BLACK && stacks.contains_key(&nsq) {
                        break;
                    }
                }
            }
        }
    }
    false
}

fn black_mandatory_capture_active_native(
    occupied: u64,
    occupied_white: u64,
    stacks: &HashMap<u8, Vec<u8>>,
) -> bool {
    for (&from_sq, pieces) in stacks {
        if pieces.is_empty() {
            continue;
        }
        let n = pieces.len();
        let dirs = dirs_for_top(*pieces.last().unwrap());
        let from_file = (from_sq & 7) as i8;
        let from_rank = (from_sq >> 3) as i8;
        for &(df, dr) in dirs {
            let adj_f = from_file + df;
            let adj_r = from_rank + dr;
            if !on_board(adj_f, adj_r) {
                continue;
            }
            let adj_sq = sq_idx(adj_f, adj_r);
            if owner(occupied, occupied_white, adj_sq) != SQ_WHITE {
                continue;
            }
            for hop in find_capture_hops(
                occupied, occupied_white, stacks, from_file, from_rank, df, dr, n,
            ) {
                if !hop.is_suicide && hop.landing_square.is_some() {
                    return true;
                }
            }
        }
    }
    false
}

// -------- White move generation (FIDE chess + Chessckers check filter) --------
//
// Mirrors `chessckers_engine.variant_py.moves_white.white_legal_moves`. We
// hand-roll python-chess's pseudo-legal White move set (pawn/knight/bishop/
// rook/queen/king/castling, with promotions + en-passant) and filter each
// candidate by the composite Chessckers check predicate on the post-move
// position (`black_can_capture_white_king_native` OR
// `square_attacked_by_black_chessckers_native`). Castling additionally rejects
// when any square the king crosses (origin + intermediate) is attacked under
// the Chessckers attack model — matching `_castling_path_attacked_chessckers`.

const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
];
const KING_DELTAS: [(i8, i8); 8] = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
];
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

#[inline(always)]
fn bit(sq: u8) -> u64 {
    1u64 << sq
}

/// FIDE attacker test (literal piece bitboards, python-chess `attackers_mask`
/// semantics). True iff a Black piece attacks `target` given `occupied`.
/// Used only to replicate python-chess's castling pseudo-legal conditions.
#[allow(clippy::too_many_arguments)]
fn black_attacks_square_fide(
    occupied: u64,
    occupied_white: u64,
    pawns: u64,
    knights: u64,
    bishops: u64,
    rooks: u64,
    queens: u64,
    kings: u64,
    target: u8,
    occupied_co_black: u64,
) -> bool {
    let tf = (target & 7) as i8;
    let tr = (target >> 3) as i8;

    // Knights.
    for &(df, dr) in &KNIGHT_DELTAS {
        let nf = tf + df;
        let nr = tr + dr;
        if on_board(nf, nr) {
            let s = sq_idx(nf, nr);
            if (knights & occupied_co_black) & bit(s) != 0 {
                return true;
            }
        }
    }
    // King (adjacent).
    for &(df, dr) in &KING_DELTAS {
        let nf = tf + df;
        let nr = tr + dr;
        if on_board(nf, nr) {
            let s = sq_idx(nf, nr);
            if (kings & occupied_co_black) & bit(s) != 0 {
                return true;
            }
        }
    }
    // Black pawns: a black pawn on (tf±1, tr-1) attacks `target` (black pawns
    // capture toward decreasing rank). Matches BB_PAWN_ATTACKS[WHITE][target]
    // & black pawns (python-chess: attackers of color BLACK use
    // BB_PAWN_ATTACKS[not BLACK] = [WHITE]).
    for &df in &[-1i8, 1] {
        let nf = tf + df;
        let nr = tr - 1;
        if on_board(nf, nr) {
            let s = sq_idx(nf, nr);
            if (pawns & occupied_co_black) & bit(s) != 0 {
                return true;
            }
        }
    }
    // Sliders: bishops/queens on diagonals, rooks/queens on orthogonals.
    let bq = (bishops | queens) & occupied_co_black;
    for &(df, dr) in &BISHOP_DIRS {
        let mut nf = tf + df;
        let mut nr = tr + dr;
        while on_board(nf, nr) {
            let s = sq_idx(nf, nr);
            let m = bit(s);
            if occupied & m != 0 {
                if bq & m != 0 {
                    return true;
                }
                break;
            }
            nf += df;
            nr += dr;
        }
    }
    let rq = (rooks | queens) & occupied_co_black;
    for &(df, dr) in &ROOK_DIRS {
        let mut nf = tf + df;
        let mut nr = tr + dr;
        while on_board(nf, nr) {
            let s = sq_idx(nf, nr);
            let m = bit(s);
            if occupied & m != 0 {
                if rq & m != 0 {
                    return true;
                }
                break;
            }
            nf += df;
            nr += dr;
        }
    }
    false
}

#[derive(Clone, Copy, PartialEq)]
enum WPiece {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl WPiece {
    fn name(self) -> &'static str {
        match self {
            WPiece::Pawn => "pawn",
            WPiece::Knight => "knight",
            WPiece::Bishop => "bishop",
            WPiece::Rook => "rook",
            WPiece::Queen => "queen",
            WPiece::King => "king",
        }
    }
}

/// A pseudo-legal White candidate move (pre check-filter).
struct WCandidate {
    from_sq: u8,
    to_sq: u8,
    piece: WPiece,
    promotion: Option<&'static str>,
    /// Square whose stack/piece is captured (= to_sq for normal captures, the
    /// captured-pawn square for en-passant), or None for a non-capture.
    capture_sq: Option<u8>,
    is_en_passant: bool,
    /// Castling form: None = not castling. Some((kingside, rook_sq)).
    castling: Option<(bool, u8)>,
}

/// Board-state inputs as a bundle (literal python-chess bitboards).
#[derive(Clone, Copy)]
struct WhiteBoard {
    occupied: u64,
    occupied_white: u64,
    pawns: u64,
    knights: u64,
    bishops: u64,
    rooks: u64,
    queens: u64,
    kings: u64,
    castling_rights: u64,
    ep_square: i64,
}

impl WhiteBoard {
    #[inline]
    fn occupied_black(&self) -> u64 {
        self.occupied & !self.occupied_white
    }
    #[inline]
    fn white_king_sq(&self) -> Option<u8> {
        let wk = self.kings & self.occupied_white;
        if wk == 0 {
            None
        } else {
            Some(wk.trailing_zeros() as u8)
        }
    }
}

/// Generate python-chess's pseudo-legal White move set (no Chessckers filter).
fn white_pseudo_legal(b: &WhiteBoard) -> Vec<WCandidate> {
    let mut out: Vec<WCandidate> = Vec::new();
    let occ = b.occupied;
    let own = b.occupied_white;
    let enemy = b.occupied_black();

    // --- Non-pawn piece moves (knight/bishop/rook/queen/king). ---
    let push_slider = |out: &mut Vec<WCandidate>, from: u8, piece: WPiece, dirs: &[(i8, i8)]| {
        let ff = (from & 7) as i8;
        let fr = (from >> 3) as i8;
        for &(df, dr) in dirs {
            let mut nf = ff + df;
            let mut nr = fr + dr;
            while on_board(nf, nr) {
                let s = sq_idx(nf, nr);
                let m = bit(s);
                if own & m != 0 {
                    break;
                }
                let cap = if enemy & m != 0 { Some(s) } else { None };
                out.push(WCandidate {
                    from_sq: from,
                    to_sq: s,
                    piece,
                    promotion: None,
                    capture_sq: cap,
                    is_en_passant: false,
                    castling: None,
                });
                if occ & m != 0 {
                    break;
                }
                nf += df;
                nr += dr;
            }
        }
    };
    let push_step = |out: &mut Vec<WCandidate>, from: u8, piece: WPiece, deltas: &[(i8, i8)]| {
        let ff = (from & 7) as i8;
        let fr = (from >> 3) as i8;
        for &(df, dr) in deltas {
            let nf = ff + df;
            let nr = fr + dr;
            if !on_board(nf, nr) {
                continue;
            }
            let s = sq_idx(nf, nr);
            let m = bit(s);
            if own & m != 0 {
                continue;
            }
            let cap = if enemy & m != 0 { Some(s) } else { None };
            out.push(WCandidate {
                from_sq: from,
                to_sq: s,
                piece,
                promotion: None,
                capture_sq: cap,
                is_en_passant: false,
                castling: None,
            });
        }
    };

    let mut bbw = (b.knights & own) | (b.bishops & own) | (b.rooks & own)
        | (b.queens & own) | (b.kings & own);
    while bbw != 0 {
        let from = bbw.trailing_zeros() as u8;
        bbw &= bbw - 1;
        let m = bit(from);
        if b.knights & m != 0 {
            push_step(&mut out, from, WPiece::Knight, &KNIGHT_DELTAS);
        } else if b.kings & m != 0 {
            push_step(&mut out, from, WPiece::King, &KING_DELTAS);
        } else if b.queens & m != 0 {
            push_slider(&mut out, from, WPiece::Queen, &BISHOP_DIRS);
            push_slider(&mut out, from, WPiece::Queen, &ROOK_DIRS);
        } else if b.bishops & m != 0 {
            push_slider(&mut out, from, WPiece::Bishop, &BISHOP_DIRS);
        } else if b.rooks & m != 0 {
            push_slider(&mut out, from, WPiece::Rook, &ROOK_DIRS);
        }
    }

    // --- Castling (python-chess generate_castling_moves, non-chess960). ---
    white_castling_pseudo(b, &mut out);

    // --- Pawn moves. ---
    let pawns_w = b.pawns & own;
    // Captures (incl. promotions).
    let mut pw = pawns_w;
    while pw != 0 {
        let from = pw.trailing_zeros() as u8;
        pw &= pw - 1;
        let ff = (from & 7) as i8;
        let fr = (from >> 3) as i8;
        for &df in &[-1i8, 1] {
            let nf = ff + df;
            let nr = fr + 1;
            if !on_board(nf, nr) {
                continue;
            }
            let s = sq_idx(nf, nr);
            if enemy & bit(s) == 0 {
                continue;
            }
            if nr == 7 {
                for promo in ["queen", "rook", "bishop", "knight"] {
                    out.push(WCandidate {
                        from_sq: from, to_sq: s, piece: WPiece::Pawn,
                        promotion: Some(promo), capture_sq: Some(s),
                        is_en_passant: false, castling: None,
                    });
                }
            } else {
                out.push(WCandidate {
                    from_sq: from, to_sq: s, piece: WPiece::Pawn,
                    promotion: None, capture_sq: Some(s),
                    is_en_passant: false, castling: None,
                });
            }
        }
    }
    // Single + double pushes (incl. promotions).
    let single = (pawns_w << 8) & !occ;
    let double = (single << 8) & !occ & 0x0000_0000_FF00_0000u64; // BB_RANK_4
    let mut sm = single;
    while sm != 0 {
        let to = sm.trailing_zeros() as u8;
        sm &= sm - 1;
        let from = to - 8;
        let nr = (to >> 3) as i8;
        if nr == 7 {
            for promo in ["queen", "rook", "bishop", "knight"] {
                out.push(WCandidate {
                    from_sq: from, to_sq: to, piece: WPiece::Pawn,
                    promotion: Some(promo), capture_sq: None,
                    is_en_passant: false, castling: None,
                });
            }
        } else {
            out.push(WCandidate {
                from_sq: from, to_sq: to, piece: WPiece::Pawn,
                promotion: None, capture_sq: None,
                is_en_passant: false, castling: None,
            });
        }
    }
    let mut dm = double;
    while dm != 0 {
        let to = dm.trailing_zeros() as u8;
        dm &= dm - 1;
        let from = to - 16;
        out.push(WCandidate {
            from_sq: from, to_sq: to, piece: WPiece::Pawn,
            promotion: None, capture_sq: None,
            is_en_passant: false, castling: None,
        });
    }
    // En passant.
    if b.ep_square >= 0 {
        let ep = b.ep_square as u8;
        if occ & bit(ep) == 0 {
            let ef = (ep & 7) as i8;
            let er = (ep >> 3) as i8;
            // Capturers: white pawns on rank 5 (index 4) that attack ep_square,
            // i.e. on (ef±1, er-1).
            for &df in &[-1i8, 1] {
                let cf = ef + df;
                let cr = er - 1;
                if !on_board(cf, cr) || cr != 4 {
                    continue;
                }
                let cs = sq_idx(cf, cr);
                if pawns_w & bit(cs) == 0 {
                    continue;
                }
                // Captured pawn sits on (ef, er-1).
                let cap_sq = sq_idx(ef, er - 1);
                out.push(WCandidate {
                    from_sq: cs, to_sq: ep, piece: WPiece::Pawn,
                    promotion: None, capture_sq: Some(cap_sq),
                    is_en_passant: true, castling: None,
                });
            }
        }
    }

    out
}

/// python-chess castling pseudo-legal (non-chess960, White to move).
fn white_castling_pseudo(b: &WhiteBoard, out: &mut Vec<WCandidate>) {
    let own = b.occupied_white;
    let king_bb = b.kings & own & 0x0000_0000_0000_00FFu64; // BB_RANK_1
    // Non-chess960: king must be on e1.
    let e1: u8 = 4;
    if king_bb == 0 || (king_bb & bit(e1)) == 0 {
        return;
    }
    let king = e1;
    // clean_castling_rights: rights on a1/h1 with a white rook present.
    let mut candidates = b.castling_rights & b.rooks & own & 0x0000_0000_0000_00FFu64;
    candidates &= bit(0) | bit(7); // BB_A1 | BB_H1
    let occ_black = b.occupied_black();
    let mut cand = candidates;
    while cand != 0 {
        let rook = cand.trailing_zeros() as u8;
        cand &= cand - 1;
        let a_side = rook < king;
        let (king_to, rook_to) = if a_side { (2u8, 3u8) } else { (6u8, 5u8) }; // c1/d1 or g1/f1
        // Paths (exclusive between endpoints).
        let king_path = between_mask(king, king_to);
        let rook_path = between_mask(rook, rook_to);
        let kingm = bit(king);
        let rookm = bit(rook);
        let to_kingm = bit(king_to);
        let to_rookm = bit(rook_to);

        // Square-occupancy check: (occupied ^ king ^ rook) must not intersect
        // (king_path | rook_path | king_to | rook_to).
        if (b.occupied ^ kingm ^ rookm) & (king_path | rook_path | to_kingm | to_rookm) != 0 {
            continue;
        }
        // King not attacked along (king_path | king) with occ = occupied ^ king.
        let occ1 = b.occupied ^ kingm;
        if mask_attacked_by_black_fide(b, king_path | kingm, occ1, occ_black) {
            continue;
        }
        // King destination not attacked with occ = occupied ^ king ^ rook ^ rook_to.
        let occ2 = b.occupied ^ kingm ^ rookm ^ to_rookm;
        if mask_attacked_by_black_fide(b, to_kingm, occ2, occ_black) {
            continue;
        }
        out.push(WCandidate {
            from_sq: king,
            to_sq: king_to,
            piece: WPiece::King,
            promotion: None,
            capture_sq: None,
            is_en_passant: false,
            castling: Some((!a_side, rook)),
        });
    }
}

/// FIDE attack test over a mask of squares (any square attacked by Black with
/// the given hypothetical `occupied`). occupied_co_black is the static black
/// occupancy (rooks/king moving during castling don't change which squares are
/// black-owned, matching python-chess which only swaps the `occupied` arg).
fn mask_attacked_by_black_fide(b: &WhiteBoard, mut path: u64, occupied: u64, occ_black: u64) -> bool {
    while path != 0 {
        let sq = path.trailing_zeros() as u8;
        path &= path - 1;
        if black_attacks_square_fide(
            occupied, b.occupied_white, b.pawns, b.knights, b.bishops,
            b.rooks, b.queens, b.kings, sq, occ_black,
        ) {
            return true;
        }
    }
    false
}

/// Inclusive-exclusive "between" mask: bits strictly between a and b along
/// their shared rank (the only case castling needs). a,b on the same rank.
fn between_mask(a: u8, b: u8) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let mut m = 0u64;
    let mut s = lo + 1;
    while s < hi {
        m |= bit(s);
        s += 1;
    }
    m
}

/// True iff the White king is in Chessckers check on this (post-move) position.
/// Composite of `black_can_capture_white_king_native` (chains/overshoots) and
/// `square_attacked_by_black_chessckers_native` (single diagonals + charges).
fn white_in_chessckers_check(
    occupied: u64,
    occupied_white: u64,
    white_king: Option<u8>,
    stacks: &HashMap<u8, Vec<u8>>,
) -> bool {
    let king = match white_king {
        Some(k) => k,
        None => return false, // king already captured
    };
    if black_can_capture_white_king_native(occupied, occupied_white, king as i64, stacks) {
        return true;
    }
    square_attacked_by_black_chessckers_native(occupied, occupied_white, stacks, king)
}

/// Apply a White candidate on bitboards; return (post occupied, post
/// occupied_white, post white-king-sq, post stacks with captured square gone).
fn apply_white_candidate(
    b: &WhiteBoard,
    c: &WCandidate,
    stacks: &HashMap<u8, Vec<u8>>,
) -> (u64, u64, Option<u8>, HashMap<u8, Vec<u8>>) {
    let from_m = bit(c.from_sq);
    let to_m = bit(c.to_sq);
    let mut occ = b.occupied;
    let mut own = b.occupied_white;
    let mut white_king = b.white_king_sq();

    // Captured square removal from occupancy + stacks overlay.
    let mut new_stacks = stacks.clone();
    if let Some(cap) = c.capture_sq {
        let cap_m = bit(cap);
        occ &= !cap_m;
        own &= !cap_m; // cap is a black square, but clearing is harmless
        new_stacks.remove(&cap);
    }

    // Move the white piece.
    occ &= !from_m;
    own &= !from_m;
    occ |= to_m;
    own |= to_m;

    if c.piece == WPiece::King {
        white_king = Some(c.to_sq);
    }

    // Castling: also relocate the rook.
    if let Some((kingside, rook_sq)) = c.castling {
        let rook_to = if kingside { 5u8 } else { 3u8 }; // f1 / d1
        let rm = bit(rook_sq);
        let rtm = bit(rook_to);
        occ &= !rm;
        own &= !rm;
        occ |= rtm;
        own |= rtm;
    }

    (occ, own, white_king, new_stacks)
}

/// Full White legal move list (Chessckers-filtered), as candidate structs +
/// metadata for dict building. Mirrors `white_legal_moves` ordering semantics
/// loosely (the test compares sets, so order is irrelevant).
fn white_legal_moves_native(
    b: &WhiteBoard,
    stacks: &HashMap<u8, Vec<u8>>,
) -> Vec<WCandidate> {
    let mut out: Vec<WCandidate> = Vec::new();
    for c in white_pseudo_legal(b) {
        let (occ, own, wk, post_stacks) = apply_white_candidate(b, &c, stacks);
        if white_in_chessckers_check(occ, own, wk, &post_stacks) {
            continue;
        }
        if let Some((kingside, _)) = c.castling {
            // Reject if any square the king crosses (origin + intermediate) is
            // attacked under the Chessckers attack model. Kingside: e1,f1;
            // queenside: e1,d1. (Destination handled by the post-move filter.)
            let cross = if kingside { [4u8, 5u8] } else { [4u8, 3u8] };
            let mut attacked = false;
            for &sq in &cross {
                if square_attacked_by_black_chessckers_native(
                    b.occupied, b.occupied_white, stacks, sq,
                ) {
                    attacked = true;
                    break;
                }
            }
            if attacked {
                continue;
            }
        }
        out.push(c);
    }
    out
}

/// UCI for a White candidate (standard form; castling uses e1g1/e1c1).
fn white_uci(c: &WCandidate) -> String {
    let names = sq_names();
    let mut s = format!("{}{}", names[c.from_sq as usize], names[c.to_sq as usize]);
    if let Some(p) = c.promotion {
        s.push(match p {
            "queen" => 'q',
            "rook" => 'r',
            "bishop" => 'b',
            "knight" => 'n',
            _ => '?',
        });
    }
    s
}

fn white_move_to_pydict<'py>(
    py: Python<'py>,
    c: &WCandidate,
) -> PyResult<Bound<'py, PyDict>> {
    let names = sq_names();
    let d = PyDict::new_bound(py);
    d.set_item("uci", white_uci(c))?;
    d.set_item("from", &names[c.from_sq as usize])?;
    d.set_item("to", &names[c.to_sq as usize])?;
    d.set_item("piece", c.piece.name())?;
    d.set_item("color", "white")?;
    match c.capture_sq {
        Some(cap) => d.set_item("capture", &names[cap as usize])?,
        None => d.set_item("capture", py.None())?,
    }
    d.set_item("waypoints", py.None())?;
    d.set_item("chainHops", py.None())?;
    match c.promotion {
        Some(p) => d.set_item("promotion", p)?,
        None => d.set_item("promotion", py.None())?,
    }
    d.set_item("demotedKings", py.None())?;
    d.set_item("demotionsRequired", py.None())?;
    d.set_item("sourceKingPositions", py.None())?;
    d.set_item("deployCount", py.None())?;
    Ok(d)
}

/// Build the king-to-rook castling alt form dict (e1h1 / e1a1).
fn white_castling_alt_to_pydict<'py>(
    py: Python<'py>,
    c: &WCandidate,
    rook_sq: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let names = sq_names();
    let d = PyDict::new_bound(py);
    d.set_item("uci", format!("{}{}", names[c.from_sq as usize], names[rook_sq as usize]))?;
    d.set_item("from", &names[c.from_sq as usize])?;
    d.set_item("to", &names[rook_sq as usize])?;
    d.set_item("piece", "king")?;
    d.set_item("color", "white")?;
    d.set_item("capture", py.None())?;
    d.set_item("waypoints", py.None())?;
    d.set_item("chainHops", py.None())?;
    d.set_item("promotion", py.None())?;
    d.set_item("demotedKings", py.None())?;
    d.set_item("demotionsRequired", py.None())?;
    d.set_item("sourceKingPositions", py.None())?;
    d.set_item("deployCount", py.None())?;
    Ok(d)
}

// -------- Python boundary --------

fn parse_stacks(py_stacks: &Bound<'_, PyDict>) -> PyResult<HashMap<u8, Vec<u8>>> {
    let mut out: HashMap<u8, Vec<u8>> = HashMap::with_capacity(py_stacks.len());
    for (k, v) in py_stacks.iter() {
        let sq: u8 = k.extract()?;
        let s: String = v.extract()?;
        out.insert(sq, s.into_bytes());
    }
    Ok(out)
}

fn chain_move_to_pydict<'py>(py: Python<'py>, m: &ChainMove) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("uci", &m.uci)?;
    d.set_item("from", &m.from_name)?;
    d.set_item("to", &m.to_name)?;
    d.set_item("piece", m.piece)?;
    d.set_item("color", "black")?;
    d.set_item("capture", m.capture.as_ref())?;
    match &m.waypoints {
        Some(wps) => d.set_item("waypoints", wps)?,
        None => d.set_item("waypoints", py.None())?,
    }
    d.set_item("chainHops", &m.chain_hops)?;
    d.set_item("promotion", py.None())?;
    d.set_item("demotedKings", py.None())?;
    d.set_item("demotionsRequired", py.None())?;
    d.set_item("sourceKingPositions", py.None())?;
    d.set_item("deployCount", py.None())?;
    d.set_item("_chain_all_captures", &m.chain_all_captures)?;
    d.set_item("cadence", m.cadence)?;
    d.set_item("_is_suicide", m.is_suicide)?;
    d.set_item("_chain_promotes", m.chain_promotes)?;
    Ok(d)
}

#[pyfunction]
fn ping() -> &'static str {
    "ok"
}

#[pyfunction]
fn black_diagonal_capture_moves<'py>(
    py: Python<'py>,
    occupied: u64,
    occupied_white: u64,
    king_sq: i64,
    stacks: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let stacks_rs = parse_stacks(stacks)?;
    let moves = black_diagonal_capture_moves_native(occupied, occupied_white, king_sq, &stacks_rs);
    let out = PyList::empty_bound(py);
    for m in &moves {
        out.append(chain_move_to_pydict(py, m)?)?;
    }
    Ok(out)
}

#[pyfunction]
fn black_mandatory_capture_active(
    occupied: u64,
    occupied_white: u64,
    stacks: &Bound<'_, PyDict>,
) -> PyResult<bool> {
    let stacks_rs = parse_stacks(stacks)?;
    Ok(black_mandatory_capture_active_native(
        occupied,
        occupied_white,
        &stacks_rs,
    ))
}

#[pyfunction]
fn all_black_legal_moves<'py>(
    py: Python<'py>,
    occupied: u64,
    occupied_white: u64,
    king_sq: i64,
    stacks: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let stacks_rs = parse_stacks(stacks)?;
    let moves = all_black_legal_moves_native(occupied, occupied_white, king_sq, &stacks_rs);
    let out = PyList::empty_bound(py);
    for m in &moves {
        out.append(any_to_pydict(py, m)?)?;
    }
    Ok(out)
}

#[pyfunction]
fn square_attacked_by_black_chessckers(
    occupied: u64,
    occupied_white: u64,
    stacks: &Bound<'_, PyDict>,
    target_sq: u8,
) -> PyResult<bool> {
    let stacks_rs = parse_stacks(stacks)?;
    Ok(square_attacked_by_black_chessckers_native(
        occupied,
        occupied_white,
        &stacks_rs,
        target_sq,
    ))
}

#[pyfunction]
fn black_can_capture_white_king(
    occupied: u64,
    occupied_white: u64,
    king_sq: i64,
    stacks: &Bound<'_, PyDict>,
) -> PyResult<bool> {
    let stacks_rs = parse_stacks(stacks)?;
    Ok(black_can_capture_white_king_native(
        occupied,
        occupied_white,
        king_sq,
        &stacks_rs,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn white_legal_moves<'py>(
    py: Python<'py>,
    occupied: u64,
    occupied_white: u64,
    pawns: u64,
    knights: u64,
    bishops: u64,
    rooks: u64,
    queens: u64,
    kings: u64,
    castling_rights: u64,
    ep_square: i64,
    stacks: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let stacks_rs = parse_stacks(stacks)?;
    let b = WhiteBoard {
        occupied,
        occupied_white,
        pawns,
        knights,
        bishops,
        rooks,
        queens,
        kings,
        castling_rights,
        ep_square,
    };
    let moves = white_legal_moves_native(&b, &stacks_rs);
    let out = PyList::empty_bound(py);
    for c in &moves {
        out.append(white_move_to_pydict(py, c)?)?;
        if let Some((_, rook_sq)) = c.castling {
            out.append(white_castling_alt_to_pydict(py, c, rook_sq)?)?;
        }
    }
    Ok(out)
}

// -------- Position / move tensor encodings --------
// Byte-for-byte equivalent to chessckers_engine.encoding.encode_position(fen)
// and encode_move(dict). The Python versions build the (14,8,8)/(240,) tensors
// with per-element assignment; these build a flat Vec<f32> the caller wraps
// with torch.tensor(...).view(...). See encoding.py for the channel/dim spec.

const ENC_POS_C: usize = 15;
const ENC_MOVE_D: usize = 240;

#[inline(always)]
fn pos_idx(ch: usize, y: usize, x: usize) -> usize {
    ch * 64 + y * 8 + x
}

fn white_piece_ch(c: u8) -> Option<usize> {
    match c {
        b'P' => Some(0),
        b'N' => Some(1),
        b'B' => Some(2),
        b'R' => Some(3),
        b'Q' => Some(4),
        b'K' => Some(5),
        _ => None,
    }
}

// 10x10 grid file/rank chars -> 0..9 (rim 'z'/'i' and '0'/'9'). Mirrors
// encoding._FILE10 / _RANK10.
fn file10(c: u8) -> Option<usize> {
    match c {
        b'z' => Some(0),
        b'a'..=b'h' => Some((c - b'a' + 1) as usize),
        b'i' => Some(9),
        _ => None,
    }
}

fn rank10(c: u8) -> Option<usize> {
    if c.is_ascii_digit() {
        Some((c - b'0') as usize)
    } else {
        None
    }
}

fn sq_index(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.len() < 2 {
        return None;
    }
    let file = b[0].wrapping_sub(b'a') as usize;
    let rank = b[1].wrapping_sub(b'1') as usize;
    if file >= 8 || rank >= 8 {
        return None;
    }
    Some(rank * 8 + file)
}

fn promo_index(p: &str) -> usize {
    match p {
        "q" => 1,
        "r" => 2,
        "b" => 3,
        "n" => 4,
        _ => 0,
    }
}

// Dict lookup that returns Some only when the key is present AND not None
// (mirrors `move.get(key) is not None` / `move.get(key) or default`).
fn dget<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v)),
        _ => Ok(None),
    }
}

// Set channels 8-12 for one tower at (x,y) from its pieces (bottom-to-top bytes
// in {s,S,k}). f64-divide-then-cast matches Python's float division + float32
// tensor-assignment double-rounding (bit-exact). Shared by both encoders.
fn apply_tower_channels(out: &mut [f32], x: usize, y: usize, pieces: &[u8]) {
    let height = pieces.len();
    if height == 0 {
        return;
    }
    let kings = pieces.iter().filter(|&&p| p == b'k').count();
    let stones = pieces.iter().filter(|&&p| p == b's' || p == b'S').count();
    out[pos_idx(8, y, x)] = (height as f64 / 24.0) as f32;
    out[pos_idx(9, y, x)] = (stones as f64 / 24.0) as f32;
    out[pos_idx(10, y, x)] = (kings as f64 / 24.0) as f32;
    if pieces[height - 1] == b's' {
        out[pos_idx(11, y, x)] = 1.0;
    }
    if height >= 2 && pieces[height - 2] == b'k' {
        out[pos_idx(12, y, x)] = 1.0;
    }
}

// Set channel `ch` to 1.0 at every square in the bitboard.
fn set_bits_channel(out: &mut [f32], mut bb: u64, ch: usize) {
    while bb != 0 {
        let sq = bb.trailing_zeros() as usize;
        out[pos_idx(ch, sq >> 3, sq & 7)] = 1.0;
        bb &= bb - 1;
    }
}

// White's rank-8 win counter from the FEN's trailing {..,r8:N} block. "r8:"
// occurs only there (no board square is named r8), so a plain search is
// unambiguous — mirrors the Python `_FEN_R8` regex.
fn parse_r8(fen: &str) -> Option<u32> {
    let idx = fen.find("r8:")?;
    fen[idx + 3..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

fn encode_position_native(fen: &str) -> Result<Vec<f32>, String> {
    let bytes = fen.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i] != b'[' && bytes[i] != b' ' {
        i += 1;
    }
    let board = &fen[..i];
    let mut overlay: &str = "";
    if i < bytes.len() && bytes[i] == b'[' {
        let start = i + 1;
        let mut j = start;
        while j < bytes.len() && bytes[j] != b']' {
            j += 1;
        }
        if j >= bytes.len() {
            return Err(format!("unrecognized Chessckers FEN: {:?}", fen));
        }
        overlay = &fen[start..j];
        i = j + 1;
    }
    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }
    if i >= bytes.len() || (bytes[i] != b'w' && bytes[i] != b'b') {
        return Err(format!("unrecognized Chessckers FEN: {:?}", fen));
    }
    let turn = bytes[i];

    let mut out = vec![0.0f32; ENC_POS_C * 64];

    let ranks: Vec<&str> = board.split('/').collect();
    if ranks.len() != 8 {
        return Err(format!("FEN board must have 8 ranks: {:?}", board));
    }
    for (fen_rank_idx, rank_str) in ranks.iter().enumerate() {
        let y = 7 - fen_rank_idx;
        let mut x = 0usize;
        for &ch in rank_str.as_bytes() {
            if ch.is_ascii_digit() {
                x += (ch - b'0') as usize;
                continue;
            }
            if let Some(c) = white_piece_ch(ch) {
                if x < 8 {
                    out[pos_idx(c, y, x)] = 1.0;
                }
            } else if ch == b'p' {
                if x < 8 {
                    out[pos_idx(6, y, x)] = 1.0;
                }
            } else if ch == b'k' {
                if x < 8 {
                    out[pos_idx(7, y, x)] = 1.0;
                }
            }
            x += 1;
        }
    }

    for entry in overlay.split(',') {
        let mut it = entry.splitn(2, ':');
        let sq = match it.next() {
            Some(s) => s,
            None => continue,
        };
        let pieces = match it.next() {
            Some(p) => p,
            None => continue,
        };
        if pieces.is_empty() {
            continue;
        }
        let sb = sq.as_bytes();
        if sb.len() < 2 {
            continue;
        }
        let x = sb[0].wrapping_sub(b'a') as usize;
        let y = sb[1].wrapping_sub(b'1') as usize;
        if x >= 8 || y >= 8 {
            continue;
        }
        apply_tower_channels(&mut out, x, y, pieces.as_bytes());
    }

    if turn == b'b' {
        for v in out[13 * 64..14 * 64].iter_mut() {
            *v = 1.0;
        }
    }
    // Rank-8 win counter (#3): channel 14 = r8 / 3, constant-filled.
    if let Some(r8) = parse_r8(fen) {
        let v = r8 as f32 / 3.0;
        for x in out[14 * 64..15 * 64].iter_mut() {
            *x = v;
        }
    }
    Ok(out)
}

// Return raw f32 bytes (native-endian) so Python wraps them with
// torch.frombuffer (one buffer) instead of materializing N PyFloats — the dense
// list return was ~0.7-0.85x Python; bytes makes the Rust path a real win.
fn f32_to_pybytes<'py>(py: Python<'py>, v: &[f32]) -> Bound<'py, PyBytes> {
    let buf: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
    PyBytes::new_bound(py, &buf)
}

#[pyfunction]
fn encode_position<'py>(py: Python<'py>, fen: &str) -> PyResult<Bound<'py, PyBytes>> {
    let out = encode_position_native(fen).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(f32_to_pybytes(py, &out))
}

#[pyfunction]
fn encode_move<'py>(py: Python<'py>, mv: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyBytes>> {
    let mut out = vec![0.0f32; ENC_MOVE_D];

    let from: String = mv
        .get_item("from")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("from"))?
        .extract()?;
    let to: String = mv
        .get_item("to")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("to"))?
        .extract()?;
    if let Some(idx) = sq_index(&from) {
        out[idx] = 1.0;
    }
    if let Some(idx) = sq_index(&to) {
        out[64 + idx] = 1.0;
    }

    if dget(mv, "capture")?.is_some() {
        out[128] = 1.0;
    }
    let waypoints: Vec<String> = match dget(mv, "waypoints")? {
        Some(v) => v.extract().unwrap_or_default(),
        None => Vec::new(),
    };
    if !waypoints.is_empty() {
        out[129] = 1.0;
    }
    let deploy_count: Option<i64> = match dget(mv, "deployCount")? {
        Some(v) => Some(v.extract()?),
        None => None,
    };
    if deploy_count.is_some() {
        out[130] = 1.0;
    }
    let demotions: Option<i64> = match dget(mv, "demotionsRequired")? {
        Some(v) => Some(v.extract()?),
        None => None,
    };
    if demotions.is_some() {
        out[131] = 1.0;
    }

    out[132] = (waypoints.len() as f64 / 8.0) as f32;
    out[133] = (deploy_count.unwrap_or(0) as f64 / 24.0) as f32;
    out[134] = (demotions.unwrap_or(0) as f64 / 8.0) as f32;

    let promo: Option<String> = match dget(mv, "promotion")? {
        Some(v) => Some(v.extract()?),
        None => None,
    };
    out[135 + promo.as_deref().map(promo_index).unwrap_or(0)] = 1.0;

    for w in &waypoints {
        let wb = w.as_bytes();
        if wb.len() != 2 {
            continue;
        }
        if let (Some(f10), Some(r10)) = (file10(wb[0]), rank10(wb[1])) {
            out[140 + r10 * 10 + f10] = 1.0;
        }
    }
    Ok(f32_to_pybytes(py, &out))
}

// Hot-path leaf encoder: same (14,8,8) output as encode_position(fen), but
// straight from the State's piece bitboards (mirrors encode_position_state,
// which the MCTS uses per leaf). White P/N/B/R/Q/K -> ch 0-5, Black Stone-top
// (pawn bb) -> ch6, King-top (king bb) -> ch7; stacks -> ch 8-12; side -> ch13.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn encode_position_bb<'py>(
    py: Python<'py>,
    wp: u64,
    wn: u64,
    wb: u64,
    wr: u64,
    wq: u64,
    wk: u64,
    bp: u64,
    bk: u64,
    stacks: &Bound<'_, PyDict>,
    turn_is_black: bool,
    rank8_count: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut out = vec![0.0f32; ENC_POS_C * 64];
    for (bb, ch) in [(wp, 0), (wn, 1), (wb, 2), (wr, 3), (wq, 4), (wk, 5), (bp, 6), (bk, 7)] {
        set_bits_channel(&mut out, bb, ch);
    }
    let stacks_rs = parse_stacks(stacks)?;
    for (sq, pieces) in &stacks_rs {
        apply_tower_channels(&mut out, (*sq & 7) as usize, (*sq >> 3) as usize, pieces);
    }
    if turn_is_black {
        for v in out[13 * 64..14 * 64].iter_mut() {
            *v = 1.0;
        }
    }
    if rank8_count > 0 {
        let v = rank8_count as f32 / 3.0;
        for x in out[14 * 64..15 * 64].iter_mut() {
            *x = v;
        }
    }
    Ok(f32_to_pybytes(py, &out))
}

#[pymodule]
fn chessckers_movegen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(black_diagonal_capture_moves, m)?)?;
    m.add_function(wrap_pyfunction!(black_mandatory_capture_active, m)?)?;
    m.add_function(wrap_pyfunction!(all_black_legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(square_attacked_by_black_chessckers, m)?)?;
    m.add_function(wrap_pyfunction!(black_can_capture_white_king, m)?)?;
    m.add_function(wrap_pyfunction!(white_legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(encode_position, m)?)?;
    m.add_function(wrap_pyfunction!(encode_move, m)?)?;
    m.add_function(wrap_pyfunction!(encode_position_bb, m)?)?;
    Ok(())
}
