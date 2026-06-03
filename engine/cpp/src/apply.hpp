// Chessckers C++ engine — Slice 5a: apply-move + status.
//
// State advancement for the search tree. Black apply is a 1:1 port of PyVariant's
// _apply_* helpers (manual bitboard/stack mutation; turn flips to White; castling
// rights, ep and clocks are untouched — Black is a White-only-castling variant
// and PyVariant never pushes a Black move through python-chess). White apply
// (the python-chess board.push port) lands in a follow-up.
//
// detect_status mirrors client._detect_status + the move-gen-derived terminal
// states in _state_to_dict / status_and_legal.
#pragma once

#include <cstdlib>
#include <string>
#include <vector>

#include "board.hpp"
#include "movegen.hpp"
#include "movegen_white.hpp"

namespace cc {

// -------- Board mutation primitives --------

inline void bb_remove_piece(Board& b, int sq) {
    const uint64_t m = ~(1ULL << sq);
    b.pawns &= m;
    b.knights &= m;
    b.bishops &= m;
    b.rooks &= m;
    b.queens &= m;
    b.kings &= m;
    b.occupied_white &= m;
    b.occupied_black &= m;
}

inline void bb_set_black_pawn(Board& b, int sq) {
    bb_remove_piece(b, sq);
    const uint64_t m = 1ULL << sq;
    b.pawns |= m;
    b.occupied_black |= m;
}

inline void bb_set_black_king(Board& b, int sq) {
    bb_remove_piece(b, sq);
    const uint64_t m = 1ULL << sq;
    b.kings |= m;
    b.occupied_black |= m;
}

// Sync the bitboard piece at sq with a stack's top char (k -> black king,
// s/S -> black pawn). Mirrors _set_top_piece_on_board.
inline void set_top_piece_on_board(Board& b, int sq, char top) {
    if (top == 'k') bb_set_black_king(b, sq);
    else if (top == 's' || top == 'S') bb_set_black_pawn(b, sq);
    else bb_remove_piece(b, sq);
}

// Move the whole tower from->to, merging onto a friendly destination (incoming
// on top). `override_pieces` substitutes the moving stack (sprint / demotion).
inline void move_full_tower(Board& b, int from_sq, int to_sq,
                            const std::string* override_pieces = nullptr) {
    const std::string moving = override_pieces ? *override_pieces : b.stacks.at((uint8_t)from_sq);
    b.stacks.erase((uint8_t)from_sq);
    bb_remove_piece(b, from_sq);
    std::string existing;
    const auto it = b.stacks.find((uint8_t)to_sq);
    if (it != b.stacks.end()) existing = it->second;
    const std::string new_stack = existing + moving;  // incoming on top
    b.stacks[(uint8_t)to_sq] = new_stack;
    set_top_piece_on_board(b, to_sq, new_stack.back());
}

inline bool is_orthogonal(int from, int to) {
    return ((from & 7) == (to & 7) || (from >> 3) == (to >> 3)) && from != to;
}

// -------- Black move apply --------
//
// The fields the PyVariant apply path reads off a move dict, extracted once.
struct BlackMove {
    int from_sq, to_sq;
    bool has_deploy_count = false;
    int deploy_count = 0;
    bool has_chain_hops = false;
    bool has_capture = false;
    bool has_waypoints = false;  // overshoot charge
    std::vector<std::string> chain_all_captures;
    bool is_suicide = false;
    bool chain_promotes = false;
    std::vector<int> demoted_kings;  // empty -> forced (all kings)
};

inline void apply_quiet_or_sprint(Board& b, const BlackMove& mv) {
    const std::string pieces = b.stacks.at((uint8_t)mv.from_sq);
    const bool is_sprint = pieces == "s" && (mv.from_sq >> 3) == 7 &&
                           std::abs((mv.to_sq >> 3) - (mv.from_sq >> 3)) == 2;
    if (is_sprint) {
        const std::string ov = "S";
        move_full_tower(b, mv.from_sq, mv.to_sq, &ov);
    } else {
        move_full_tower(b, mv.from_sq, mv.to_sq);
    }
    if ((mv.to_sq >> 3) == 0) {  // rank-1 promotion (after the merge)
        const std::string promoted = promote_all_stones(b.stacks.at((uint8_t)mv.to_sq));
        b.stacks[(uint8_t)mv.to_sq] = promoted;
        set_top_piece_on_board(b, mv.to_sq, promoted.back());
    }
}

inline void apply_deploy(Board& b, const BlackMove& mv) {
    const std::string pieces = b.stacks.at((uint8_t)mv.from_sq);
    const int s = mv.deploy_count;
    const std::string sub = pieces.substr(pieces.size() - s);
    const std::string remainder = pieces.substr(0, pieces.size() - s);
    b.stacks[(uint8_t)mv.from_sq] = remainder;
    set_top_piece_on_board(b, mv.from_sq, remainder.back());
    std::string existing;
    const auto it = b.stacks.find((uint8_t)mv.to_sq);
    if (it != b.stacks.end()) existing = it->second;
    std::string new_stack = existing + sub;
    if ((mv.to_sq >> 3) == 0) new_stack = promote_all_stones(new_stack);
    b.stacks[(uint8_t)mv.to_sq] = new_stack;
    set_top_piece_on_board(b, mv.to_sq, new_stack.back());
}

inline void apply_charge(Board& b, const BlackMove& mv) {
    const std::string pieces = b.stacks.at((uint8_t)mv.from_sq);
    const int ff = mv.from_sq & 7, fr = mv.from_sq >> 3, tf = mv.to_sq & 7, tr = mv.to_sq >> 3;
    const int df = tf - ff, dr = tr - fr;
    const int d = std::max(std::abs(df), std::abs(dr));
    const int dfs = d ? df / d : 0, drs = d ? dr / d : 0;
    const bool is_rim_overshoot = mv.has_waypoints;
    const int last_step = is_rim_overshoot ? d : d - 1;
    for (int k = 1; k <= last_step; ++k) {
        const int sq = sq_idx(ff + k * dfs, fr + k * drs);
        if (owner(b.occupied(), b.occupied_white, sq) == SQ_WHITE) bb_remove_piece(b, sq);
    }
    if (!is_rim_overshoot && owner(b.occupied(), b.occupied_white, mv.to_sq) == SQ_WHITE) {
        b.stacks.erase((uint8_t)mv.from_sq);  // ram: tower destroyed, landing White stays
        bb_remove_piece(b, mv.from_sq);
        return;
    }
    std::vector<int> king_positions;
    for (int i = 0; i < (int)pieces.size(); ++i)
        if (pieces[i] == 'k') king_positions.push_back(i + 1);
    const std::vector<int>& chosen = mv.demoted_kings.empty() ? king_positions : mv.demoted_kings;
    std::string new_pieces = pieces;
    for (int pos : chosen) new_pieces[pos - 1] = 'S';
    move_full_tower(b, mv.from_sq, mv.to_sq, &new_pieces);
}

inline void apply_diagonal_capture(Board& b, const BlackMove& mv) {
    const int ff = mv.from_sq & 7, fr = mv.from_sq >> 3, tf = mv.to_sq & 7, tr = mv.to_sq >> 3;
    const int df = tf - ff, dr = tr - fr;
    const int d = std::max(std::abs(df), std::abs(dr));
    const int dfs = d ? df / d : 0, drs = d ? dr / d : 0;
    for (int k = 1; k < d; ++k) {
        const int sq = sq_idx(ff + k * dfs, fr + k * drs);
        if (owner(b.occupied(), b.occupied_white, sq) == SQ_WHITE) bb_remove_piece(b, sq);
    }
    if (owner(b.occupied(), b.occupied_white, mv.to_sq) == SQ_WHITE) {
        b.stacks.erase((uint8_t)mv.from_sq);
        bb_remove_piece(b, mv.from_sq);
        return;
    }
    move_full_tower(b, mv.from_sq, mv.to_sq);
}

inline void apply_chain_move(Board& b, const BlackMove& mv) {
    const std::string orig_stack = b.stacks.at((uint8_t)mv.from_sq);
    for (const std::string& cap : mv.chain_all_captures) bb_remove_piece(b, parse_square(cap));
    b.stacks.erase((uint8_t)mv.from_sq);
    bb_remove_piece(b, mv.from_sq);
    if (mv.is_suicide) return;
    const std::string final_stack = mv.chain_promotes ? promote_all_stones(orig_stack) : orig_stack;
    b.stacks[(uint8_t)mv.to_sq] = final_stack;
    if (final_stack.back() == 'k') bb_set_black_king(b, mv.to_sq);
    else bb_set_black_pawn(b, mv.to_sq);
}

// Dispatch mirrors apply_black_move_known. Mutates b in place; flips turn to
// White. castling/ep/clocks are deliberately left unchanged.
inline void apply_black_move(Board& b, const BlackMove& mv) {
    if (mv.has_deploy_count) apply_deploy(b, mv);
    else if (is_orthogonal(mv.from_sq, mv.to_sq)) apply_charge(b, mv);
    else if (mv.has_chain_hops) apply_chain_move(b, mv);
    else if (mv.has_capture) apply_diagonal_capture(b, mv);
    else apply_quiet_or_sprint(b, mv);
    b.turn_white = true;
}

// -------- Status detection --------

struct Status {
    std::string status;  // "" == None
    std::string winner;  // "" == None
};

inline Status detect_status(const Board& b) {
    if (b.stacks.empty()) return {"variantEnd", "white"};  // Black eliminated
    const uint64_t wk_bb = b.kings & b.occupied_white;
    if (wk_bb == 0) return {"variantEnd", "black"};  // White king captured
    const int wk_sq = __builtin_ctzll(wk_bb);
    if (b.turn_white) {
        const WhiteBoard wb{b.occupied(),  b.occupied_white, b.pawns,           b.knights,
                            b.bishops,     b.rooks,          b.queens,          b.kings,
                            b.castling_rights, (long)b.ep_square};
        if (white_legal_moves(wb, b.stacks).empty()) {
            const bool check =
                white_in_chessckers_check(b.occupied(), b.occupied_white, wk_sq, b.stacks);
            return check ? Status{"mate", "black"} : Status{"stalemate", ""};
        }
    } else {
        if (all_black_legal_moves(b.occupied(), b.occupied_white, wk_sq, b.stacks).empty())
            return {"variantEnd", "white"};
    }
    return {"", ""};
}

}  // namespace cc
