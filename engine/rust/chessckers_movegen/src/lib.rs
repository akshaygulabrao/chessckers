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
use pyo3::types::{PyDict, PyList};
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

// -------- Bouncer-path table --------
//
// Precomputed per-(start_f, start_r, df0, dr0) trajectory. Same as the Python
// `_CAPTURE_PATHS` table — pure geometry, no board state. Each step records
// (f, r, sq_or_-1, key, df_after, dr_after, did_bounce).

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
        // Include rim starts so chain continuations from rim landings work.
        for f0 in -1..=8 {
            for r0 in -1..=8 {
                for &df0 in &[-1i8, 1] {
                    for &dr0 in &[-1i8, 1] {
                        let mut steps = Vec::with_capacity(MAX_HOP_STEPS);
                        let mut f = f0;
                        let mut r = r0;
                        let mut df = df0;
                        let mut dr = dr0;
                        for _ in 0..MAX_HOP_STEPS {
                            let mut nf = f + df;
                            let mut nr = r + dr;
                            let mut did_bounce = false;
                            if nf < -1 || nf > 8 {
                                df = -df;
                                nf = f + df;
                                did_bounce = true;
                            }
                            if nr < -1 || nr > 8 {
                                dr = -dr;
                                nr = r + dr;
                                did_bounce = true;
                            }
                            if nf < -1 || nf > 8 || nr < -1 || nr > 8 {
                                break;
                            }
                            let on_now = on_board(nf, nr);
                            if did_bounce && !on_now {
                                break;
                            }
                            f = nf;
                            r = nr;
                            let sq: i16 = if on_now { sq_idx(f, r) as i16 } else { -1 };
                            steps.push(PathStep {
                                f,
                                r,
                                sq,
                                key: coord_key(f, r),
                                df,
                                dr,
                                did_bounce,
                            });
                            if did_bounce {
                                break;
                            }
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
    let mut pending_ram: Option<CaptureHop> = None;

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
        let step_num = step_idx + 1; // 1-based, matches Python `step`

        if step.sq >= 0 {
            let sq = step.sq as u8;
            let cap_mask = 1u64 << sq;
            if captured_set & cap_mask != 0 {
                // Revisit of an already-captured (now empty) square.
                if let Some(p) = pending_ram.take() {
                    options.push(p);
                }
                if !captures_so_far.is_empty() {
                    options.push(CaptureHop {
                        direction: (step.df, step.dr),
                        landing_key: cur_key.clone(),
                        landing_square: Some(sq),
                        captures: captures_so_far.clone(),
                        waypoints: waypoints_so_far.clone(),
                        is_suicide: false,
                        crossed_rank1,
                    });
                }
            } else {
                let o = owner(occupied, occupied_white, sq);
                match o {
                    SQ_EMPTY => {
                        if let Some(p) = pending_ram.take() {
                            options.push(p);
                        }
                        if !captures_so_far.is_empty() {
                            options.push(CaptureHop {
                                direction: (step.df, step.dr),
                                landing_key: cur_key.clone(),
                                landing_square: Some(sq),
                                captures: captures_so_far.clone(),
                                waypoints: waypoints_so_far.clone(),
                                is_suicide: false,
                                crossed_rank1,
                            });
                        }
                    }
                    SQ_BLACK if stacks.contains_key(&sq) => {
                        // Friendly tower → invalidate pending k=d ram.
                        pending_ram = None;
                        break;
                    }
                    _ => {
                        // White piece (or defensive black-without-stack).
                        if let Some(p) = pending_ram.take() {
                            options.push(p);
                        }
                        if step_num <= n || !captures_so_far.is_empty() {
                            let ram = CaptureHop {
                                direction: (step.df, step.dr),
                                landing_key: cur_key.clone(),
                                landing_square: Some(sq),
                                captures: captures_so_far.clone(),
                                waypoints: waypoints_so_far.clone(),
                                is_suicide: true,
                                crossed_rank1,
                            };
                            if captures_so_far.is_empty() {
                                pending_ram = Some(ram);
                            } else {
                                options.push(ram);
                            }
                        }
                        captures_so_far.push(sq);
                        captured_set |= cap_mask;
                    }
                }
            }
        } else {
            // Rim square (T) — never friendly.
            if let Some(p) = pending_ram.take() {
                options.push(p);
            }
            if !captures_so_far.is_empty() {
                options.push(CaptureHop {
                    direction: (step.df, step.dr),
                    landing_key: cur_key.clone(),
                    landing_square: None,
                    captures: captures_so_far.clone(),
                    waypoints: waypoints_so_far.clone(),
                    is_suicide: false,
                    crossed_rank1,
                });
            }
        }
        // Path-table already terminates after a bounce step; loop exits naturally.
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
        options.retain(|h| h.waypoints.len() == c);
    }
    // Dedup by (direction, landing_key, captures, is_suicide).
    let mut seen: std::collections::HashSet<(i8, i8, String, Vec<u8>, bool)> =
        std::collections::HashSet::new();
    let mut deduped: Vec<CaptureHop> = Vec::with_capacity(options.len());
    for h in options {
        let key = (
            h.direction.0,
            h.direction.1,
            h.landing_key.clone(),
            h.captures.clone(),
            h.is_suicide,
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

    let (uci, waypoints_field) = if hops.len() > 1 {
        let mid = all_waypoints.join("~");
        (format!("{}~{}~{}", from_name, mid, dest_name), Some(all_waypoints.clone()))
    } else {
        (format!("{}{}", from_name, dest_name), None)
    };

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
    // White-king-captured short-circuit. The only way occupied_white loses
    // its king is if we've already captured it on the chain. We'd need to
    // know which white pieces are kings — but for the diagonal-capture path
    // the `_first_hop_suicides` handles single-hop king kills, and within
    // chain DFS we can detect "no white king left" by checking occupied_white
    // bit-pop count vs initial; simpler: just keep enumerating, the hops will
    // fail to find more white pieces naturally. Python special-cases this for
    // efficiency; we omit for simplicity (tested to give same results).
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
        if !hops_so_far.is_empty() {
            results.push(build_final_move(chain_start, orig_stack, &hops_so_far));
        }
        return;
    }
    for hop in &options {
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
        let next_cadence = cadence.or(Some(hop.waypoints.len()));
        let mut hops_next = hops_so_far.clone();
        hops_next.push(hop.clone());
        enumerate_chains_recursive(
            new_occ,
            new_occ_w,
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
            moves.extend(enumerate_chains(occupied, occupied_white, stacks, sq));
            moves.extend(first_hop_suicides(occupied, occupied_white, stacks, sq));
        }
    }
    moves
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
                let mut blocked = false;
                let mut path_captures: Vec<String> = Vec::new();
                for k in 1..d {
                    let pf = from_file + (k as i8) * df;
                    let pr = from_rank + (k as i8) * dr;
                    let psq = sq_idx(pf, pr);
                    let powner = owner(occupied, occupied_white, psq);
                    if powner != SQ_EMPTY {
                        if powner == SQ_BLACK && stacks.contains_key(&psq) {
                            blocked = true;
                            break;
                        }
                        if powner == SQ_WHITE {
                            path_captures.push(sq_n[psq as usize].clone());
                        }
                    }
                }
                if blocked {
                    break;
                }
                let tf = from_file + (d as i8) * df;
                let tr = from_rank + (d as i8) * dr;
                if !on_board(tf, tr) {
                    break;
                }
                let to_sq = sq_idx(tf, tr);
                let to_name = sq_n[to_sq as usize].clone();
                let towner = owner(occupied, occupied_white, to_sq);
                let is_ram = towner == SQ_WHITE;
                let is_friendly_merge = towner == SQ_BLACK && stacks.contains_key(&to_sq);

                let capture_field: Option<String> = if !path_captures.is_empty() {
                    Some(path_captures[0].clone())
                } else if is_ram {
                    Some(to_name.clone())
                } else {
                    None
                };

                if is_ram {
                    moves.push(ChargeMove {
                        uci: format!("{}{}", from_name, to_name),
                        from_name: from_name.clone(),
                        to_name,
                        piece: "king",
                        capture: capture_field,
                        demoted_kings: None,
                        demotions_required: None,
                        source_king_positions: None,
                    });
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
                        uci: format!("{}{}", from_name, to_name),
                        from_name: from_name.clone(),
                        to_name,
                        piece: if resulting_top == b'k' { "king" } else { "pawn" },
                        capture: capture_field,
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
                            uci: format!("{}{}{{{}}}", from_name, to_name, choice_str),
                            from_name: from_name.clone(),
                            to_name: to_name.clone(),
                            piece: if resulting_top == b'k' { "king" } else { "pawn" },
                            capture: capture_field.clone(),
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
            d.set_item("waypoints", py.None())?;
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
    stacks: &HashMap<u8, Vec<u8>>,
) -> Vec<AnyMove> {
    let mut quiet = black_diagonal_quiet_moves_native(occupied, occupied_white, stacks);
    let mut deploy = black_deploy_moves_native(occupied, occupied_white, stacks);
    let mut charge = black_charge_moves_native(occupied, occupied_white, stacks);
    let chain = black_diagonal_capture_moves_native(occupied, occupied_white, stacks);

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
                        // Charge to k+1 must land on-board to be a legal
                        // charge target.
                        let nf2 = sf + (k + 1) * df;
                        let nr2 = sr + (k + 1) * dr;
                        if on_board(nf2, nr2) {
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
    stacks: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let stacks_rs = parse_stacks(stacks)?;
    let moves = black_diagonal_capture_moves_native(occupied, occupied_white, &stacks_rs);
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
    stacks: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let stacks_rs = parse_stacks(stacks)?;
    let moves = all_black_legal_moves_native(occupied, occupied_white, &stacks_rs);
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

#[pymodule]
fn chessckers_movegen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(black_diagonal_capture_moves, m)?)?;
    m.add_function(wrap_pyfunction!(black_mandatory_capture_active, m)?)?;
    m.add_function(wrap_pyfunction!(all_black_legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(square_attacked_by_black_chessckers, m)?)?;
    Ok(())
}
