// pybind11 module `chessckers_cpp` — the Python-facing surface of the C++ engine.
// Slice 0 exposes the Board struct (read-only bb fields, for later oracle tests)
// plus parse_fen / serialize_fen.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <memory>
#include <type_traits>
#include <variant>

#include "apply.hpp"
#include "board.hpp"
#include "movegen.hpp"
#include "movegen_white.hpp"
#include "search.hpp"

namespace py = pybind11;

static cc::WPiece wpiece_from_name(const std::string& n) {
    if (n == "pawn") return cc::WPiece::Pawn;
    if (n == "knight") return cc::WPiece::Knight;
    if (n == "bishop") return cc::WPiece::Bishop;
    if (n == "rook") return cc::WPiece::Rook;
    if (n == "queen") return cc::WPiece::Queen;
    return cc::WPiece::King;
}

// Reconstruct a White move from its dict. Castling is a White king from e1 to a
// corner/2-away square (covers both the e1g1 and the e1h1 notation forms).
static cc::WhiteMove parse_white_move(const py::dict& move) {
    cc::WhiteMove mv;
    int from = cc::parse_square(move["from"].cast<std::string>());
    int to = cc::parse_square(move["to"].cast<std::string>());
    mv.piece = wpiece_from_name(move["piece"].cast<std::string>());
    mv.has_promotion = !move["promotion"].is_none();
    if (mv.has_promotion) mv.promotion = wpiece_from_name(move["promotion"].cast<std::string>());
    if (!move["capture"].is_none()) mv.capture_sq = cc::parse_square(move["capture"].cast<std::string>());
    if (mv.piece == cc::WPiece::King && from == 4 && (to == 0 || to == 2 || to == 6 || to == 7)) {
        mv.is_castling = true;
        mv.castling_kingside = (to == 6 || to == 7);
        mv.castling_rook_sq = mv.castling_kingside ? 7 : 0;
        mv.capture_sq = -1;
        to = mv.castling_kingside ? 6 : 2;  // king destination
    }
    mv.from_sq = from;
    mv.to_sq = to;
    return mv;
}

// Extract the fields PyVariant's Black apply reads off a move dict.
static cc::BlackMove parse_black_move(const py::dict& move) {
    cc::BlackMove mv;
    mv.from_sq = cc::parse_square(move["from"].cast<std::string>());
    mv.to_sq = cc::parse_square(move["to"].cast<std::string>());
    mv.has_deploy_count = !move["deployCount"].is_none();
    if (mv.has_deploy_count) mv.deploy_count = move["deployCount"].cast<int>();
    mv.has_chain_hops = !move["chainHops"].is_none();
    mv.has_capture = !move["capture"].is_none();
    mv.has_waypoints = !move["waypoints"].is_none();
    if (move.contains("_chain_all_captures") && !move["_chain_all_captures"].is_none())
        mv.chain_all_captures = move["_chain_all_captures"].cast<std::vector<std::string>>();
    if (move.contains("_is_suicide"))
        mv.is_suicide = move["_is_suicide"].cast<bool>();
    if (move.contains("_chain_promotes"))
        mv.chain_promotes = move["_chain_promotes"].cast<bool>();
    if (!move["demotedKings"].is_none())
        mv.demoted_kings = move["demotedKings"].cast<std::vector<int>>();
    return mv;
}

static py::dict white_move_to_dict(const cc::WCandidate& c) {
    py::dict d;
    d["uci"] = cc::white_uci(c);
    d["from"] = cc::square_name(c.from_sq);
    d["to"] = cc::square_name(c.to_sq);
    d["piece"] = cc::wpiece_name(c.piece);
    d["color"] = "white";
    if (c.capture_sq >= 0) d["capture"] = cc::square_name(c.capture_sq);
    else d["capture"] = py::none();
    d["waypoints"] = py::none();
    d["chainHops"] = py::none();
    if (c.promotion) d["promotion"] = *c.promotion;
    else d["promotion"] = py::none();
    d["demotedKings"] = py::none();
    d["demotionsRequired"] = py::none();
    d["sourceKingPositions"] = py::none();
    d["deployCount"] = py::none();
    return d;
}

// King-to-rook alternate castling form (e1h1 / e1a1) the Rust also emits.
static py::dict white_castling_alt_to_dict(const cc::WCandidate& c) {
    py::dict d;
    d["uci"] = cc::square_name(c.from_sq) + cc::square_name(c.castling_rook_sq);
    d["from"] = cc::square_name(c.from_sq);
    d["to"] = cc::square_name(c.castling_rook_sq);
    d["piece"] = "king";
    d["color"] = "white";
    d["capture"] = py::none();
    d["waypoints"] = py::none();
    d["chainHops"] = py::none();
    d["promotion"] = py::none();
    d["demotedKings"] = py::none();
    d["demotionsRequired"] = py::none();
    d["sourceKingPositions"] = py::none();
    d["deployCount"] = py::none();
    return d;
}

static py::dict chain_to_dict(const cc::ChainMove& m) {
    py::dict d;
    d["uci"] = m.uci;
    d["from"] = m.from_name;
    d["to"] = m.to_name;
    d["piece"] = m.piece;
    d["color"] = "black";
    if (m.capture) d["capture"] = *m.capture;
    else d["capture"] = py::none();
    if (m.waypoints) d["waypoints"] = *m.waypoints;
    else d["waypoints"] = py::none();
    d["chainHops"] = m.chain_hops;
    d["promotion"] = py::none();
    d["demotedKings"] = py::none();
    d["demotionsRequired"] = py::none();
    d["sourceKingPositions"] = py::none();
    d["deployCount"] = py::none();
    d["_chain_all_captures"] = m.chain_all_captures;
    d["cadence"] = m.cadence;
    d["_is_suicide"] = m.is_suicide;
    d["_chain_promotes"] = m.chain_promotes;
    return d;
}

// Quiet/deploy share the null-filled key block; only deployCount differs.
static py::dict simple_move_dict(const std::string& uci, const std::string& from,
                                 const std::string& to, const std::string& piece,
                                 py::object deploy_count) {
    py::dict d;
    d["uci"] = uci;
    d["from"] = from;
    d["to"] = to;
    d["piece"] = piece;
    d["color"] = "black";
    d["capture"] = py::none();
    d["waypoints"] = py::none();
    d["chainHops"] = py::none();
    d["promotion"] = py::none();
    d["demotedKings"] = py::none();
    d["demotionsRequired"] = py::none();
    d["sourceKingPositions"] = py::none();
    d["deployCount"] = std::move(deploy_count);
    return d;
}

static py::dict charge_to_dict(const cc::ChargeMove& c) {
    py::dict d;
    d["uci"] = c.uci;
    d["from"] = c.from_name;
    d["to"] = c.to_name;
    d["piece"] = c.piece;
    d["color"] = "black";
    if (c.capture) d["capture"] = *c.capture;
    else d["capture"] = py::none();
    if (c.waypoints) d["waypoints"] = *c.waypoints;
    else d["waypoints"] = py::none();
    d["chainHops"] = py::none();
    d["promotion"] = py::none();
    if (c.demoted_kings) d["demotedKings"] = *c.demoted_kings;
    else d["demotedKings"] = py::none();
    if (c.demotions_required) d["demotionsRequired"] = *c.demotions_required;
    else d["demotionsRequired"] = py::none();
    if (c.source_king_positions) d["sourceKingPositions"] = *c.source_king_positions;
    else d["sourceKingPositions"] = py::none();
    d["deployCount"] = py::none();
    return d;
}

static py::dict hop_to_dict(const cc::CaptureHop& h) {
    py::dict d;
    d["direction"] = py::make_tuple(h.df, h.dr);
    d["landing_key"] = h.landing_key;
    if (h.landing_square < 0) d["landing_square"] = py::none();
    else d["landing_square"] = h.landing_square;
    d["captures"] = h.captures;
    d["waypoints"] = h.waypoints;
    d["is_suicide"] = h.is_suicide;
    d["crossed_rank1"] = h.crossed_rank1;
    d["cadence"] = h.cadence;
    d["is_overshoot"] = h.is_overshoot;
    return d;
}

// -------- Slice 5b/5c: native PUCT search with a Python NN eval bridge --------

// Generate the side-to-move's legal moves as Python dicts (same content + order
// as the move-gen bindings, incl. both White castling forms) — what the leaf NN
// eval encodes.
static py::list gen_legal_dicts(const cc::Board& b) {
    py::list out;
    if (b.turn_white) {
        cc::WhiteBoard wb{b.occupied(),      b.occupied_white, b.pawns, b.knights,
                          b.bishops,         b.rooks,          b.queens, b.kings,
                          b.castling_rights, (long)b.ep_square};
        for (const auto& c : cc::white_legal_moves(wb, b.stacks)) {
            out.append(white_move_to_dict(c));
            if (c.is_castling) out.append(white_castling_alt_to_dict(c));
        }
    } else {
        const uint64_t wk = b.kings & b.occupied_white;
        const long king_sq = wk ? __builtin_ctzll(wk) : -1;
        for (const auto& mv : cc::all_black_legal_moves(b.occupied(), b.occupied_white, king_sq, b.stacks)) {
            std::visit(
                [&](auto&& x) {
                    using T = std::decay_t<decltype(x)>;
                    if constexpr (std::is_same_v<T, cc::QuietMove>)
                        out.append(simple_move_dict(x.uci, x.from_name, x.to_name, x.piece, py::none()));
                    else if constexpr (std::is_same_v<T, cc::DeployMove>)
                        out.append(simple_move_dict(x.uci, x.from_name, x.to_name, x.piece,
                                                    py::cast(x.deploy_count)));
                    else if constexpr (std::is_same_v<T, cc::ChargeMove>)
                        out.append(charge_to_dict(x));
                    else if constexpr (std::is_same_v<T, cc::ChainMove>)
                        out.append(chain_to_dict(x));
                },
                mv);
        }
    }
    return out;
}

static cc::Board apply_dict(cc::Board b, const py::dict& mv) {
    if (b.turn_white) cc::apply_white_move(b, parse_white_move(mv));
    else cc::apply_black_move(b, parse_black_move(mv));
    return b;
}

// Expand a leaf: native legal-move gen -> Python eval(fen, dicts) -> (value,
// priors) -> create a child per move via native apply + native status. Returns
// the leaf's value.
static double mcts_expand(cc::PuctNode* node, const py::function& eval_fn) {
    py::list legal = gen_legal_dicts(node->board);
    const std::string fen = cc::serialize_fen(node->board);
    const py::tuple res = eval_fn(fen, legal).cast<py::tuple>();
    const double value = res[0].cast<double>();
    const size_t n = py::len(legal);
    node->expanded = true;
    if (n == 0) return value;
    const std::vector<double> priors = res[1].cast<std::vector<double>>();
    for (size_t i = 0; i < n; ++i) {
        const py::dict mv = legal[i].cast<py::dict>();
        auto child = std::make_unique<cc::PuctNode>();
        child->board = apply_dict(node->board, mv);
        child->uci = mv["uci"].cast<std::string>();
        child->prior = priors[i];
        const auto st = cc::detect_status(child->board);
        child->is_terminal = !st.status.empty();
        child->terminal_status = st.status;
        node->children.push_back(std::move(child));
    }
    return value;
}

static void mcts_simulate(cc::PuctNode* root, const py::function& eval_fn, double c_puct,
                          double gamma) {
    const auto path = cc::select_to_leaf(root, c_puct);
    cc::PuctNode* leaf = path.back();
    double value;
    if (leaf->is_terminal) {
        value = cc::terminal_value(*leaf);
    } else if (!leaf->expanded) {
        value = mcts_expand(leaf, eval_fn);
    } else {  // expanded but childless — value-head fallback (mirrors Python)
        const py::tuple t = eval_fn(cc::serialize_fen(leaf->board), py::list()).cast<py::tuple>();
        value = t[0].cast<double>();
    }
    cc::backup(path, value, gamma);
}

PYBIND11_MODULE(chessckers_cpp, m) {
    m.doc() = "Chessckers C++ engine (Slice 0: board + FEN; Slice 1: §3B capture hops)";

    py::class_<cc::Board>(m, "Board")
        .def_readonly("pawns", &cc::Board::pawns)
        .def_readonly("knights", &cc::Board::knights)
        .def_readonly("bishops", &cc::Board::bishops)
        .def_readonly("rooks", &cc::Board::rooks)
        .def_readonly("queens", &cc::Board::queens)
        .def_readonly("kings", &cc::Board::kings)
        .def_readonly("occupied_white", &cc::Board::occupied_white)
        .def_readonly("occupied_black", &cc::Board::occupied_black)
        .def_property_readonly("occupied", [](const cc::Board& b) { return b.occupied(); })
        .def_readonly("castling_rights", &cc::Board::castling_rights)
        .def_readonly("ep_square", &cc::Board::ep_square)
        .def_readonly("turn_white", &cc::Board::turn_white)
        .def_readonly("halfmove", &cc::Board::halfmove)
        .def_readonly("fullmove", &cc::Board::fullmove)
        .def_readonly("stacks", &cc::Board::stacks);

    m.def("parse_fen", &cc::parse_fen, py::arg("fen"),
          "Parse a Chessckers FEN into a Board (mirrors variant_py.state.parse_fen).");
    m.def("serialize_fen", &cc::serialize_fen, py::arg("board"),
          "Serialize a Board back to a Chessckers FEN (mirrors variant_py.state.serialize_fen).");

    m.def(
        "find_capture_hops",
        [](uint64_t occupied, uint64_t occupied_white, std::map<uint8_t, std::string> stacks,
           int f0, int r0, int df0, int dr0, int n) {
            py::list out;
            for (const auto& h :
                 cc::find_capture_hops(occupied, occupied_white, stacks, f0, r0, df0, dr0, n))
                out.append(hop_to_dict(h));
            return out;
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("stacks"), py::arg("f0"),
        py::arg("r0"), py::arg("df0"), py::arg("dr0"), py::arg("n"),
        "Slice 1: §3B capture hops from (f0,r0) along (df0,dr0), up to n+1 steps. "
        "Mirrors variant_py.moves_black._find_capture_hops.");

    m.def(
        "black_diagonal_capture_moves",
        [](uint64_t occupied, uint64_t occupied_white, long king_sq,
           std::map<uint8_t, std::string> stacks) {
            py::list out;
            for (const auto& m :
                 cc::black_diagonal_capture_moves(occupied, occupied_white, king_sq, stacks))
                out.append(chain_to_dict(m));
            return out;
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("king_sq"), py::arg("stacks"),
        "Slice 2a: Black diagonal capture moves (chains + first-hop rams). king_sq is "
        "the White king square (-1 if none). Mirrors moves_black.black_diagonal_capture_moves.");

    m.def(
        "black_diagonal_quiet_moves",
        [](uint64_t occupied, uint64_t occupied_white, std::map<uint8_t, std::string> stacks) {
            py::list out;
            for (const auto& q : cc::black_diagonal_quiet_moves(occupied, occupied_white, stacks))
                out.append(simple_move_dict(q.uci, q.from_name, q.to_name, q.piece, py::none()));
            return out;
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("stacks"),
        "Slice 2b: Black quiet diagonal moves + back-rank sprint. Mirrors "
        "moves_black.black_diagonal_quiet_moves.");

    m.def(
        "black_deploy_moves",
        [](uint64_t occupied, uint64_t occupied_white, std::map<uint8_t, std::string> stacks) {
            py::list out;
            for (const auto& dm : cc::black_deploy_moves(occupied, occupied_white, stacks))
                out.append(simple_move_dict(dm.uci, dm.from_name, dm.to_name, dm.piece,
                                            py::cast(dm.deploy_count)));
            return out;
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("stacks"),
        "Slice 2b: Black deploy moves (sub-tower diagonal deploys). Mirrors "
        "moves_black.black_deploy_moves.");

    m.def(
        "black_charge_moves",
        [](uint64_t occupied, uint64_t occupied_white, std::map<uint8_t, std::string> stacks) {
            py::list out;
            for (const auto& c : cc::black_charge_moves(occupied, occupied_white, stacks))
                out.append(charge_to_dict(c));
            return out;
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("stacks"),
        "Slice 2c: Black charges (orthogonal King-top tower moves with demotion "
        "choices + overshoot). Mirrors moves_black.black_charge_moves.");

    m.def(
        "black_mandatory_capture_active",
        [](uint64_t occupied, uint64_t occupied_white, std::map<uint8_t, std::string> stacks) {
            return cc::black_mandatory_capture_active(occupied, occupied_white, stacks);
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("stacks"),
        "Slice 2d: §4 mandate trigger. Mirrors moves_black.black_mandatory_capture_active.");

    m.def(
        "all_black_legal_moves",
        [](uint64_t occupied, uint64_t occupied_white, long king_sq,
           std::map<uint8_t, std::string> stacks) {
            py::list out;
            for (const auto& mv :
                 cc::all_black_legal_moves(occupied, occupied_white, king_sq, stacks)) {
                std::visit(
                    [&](auto&& x) {
                        using T = std::decay_t<decltype(x)>;
                        if constexpr (std::is_same_v<T, cc::QuietMove>)
                            out.append(simple_move_dict(x.uci, x.from_name, x.to_name, x.piece,
                                                        py::none()));
                        else if constexpr (std::is_same_v<T, cc::DeployMove>)
                            out.append(simple_move_dict(x.uci, x.from_name, x.to_name, x.piece,
                                                        py::cast(x.deploy_count)));
                        else if constexpr (std::is_same_v<T, cc::ChargeMove>)
                            out.append(charge_to_dict(x));
                        else if constexpr (std::is_same_v<T, cc::ChainMove>)
                            out.append(chain_to_dict(x));
                    },
                    mv);
            }
            return out;
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("king_sq"), py::arg("stacks"),
        "Slice 2d: full Black legal move list with mandate applied, in the authoritative "
        "(Rust) order. Mirrors moves_black._all_black_legal / Rust all_black_legal_moves.");

    m.def(
        "black_can_capture_white_king",
        [](uint64_t occupied, uint64_t occupied_white, long king_sq,
           std::map<uint8_t, std::string> stacks) {
            return cc::black_can_capture_white_king(occupied, occupied_white, king_sq, stacks);
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("king_sq"), py::arg("stacks"),
        "Slice 3a: can Black capture the White king (diagonal chains + rams)?");

    m.def(
        "square_attacked_by_black_chessckers",
        [](uint64_t occupied, uint64_t occupied_white, std::map<uint8_t, std::string> stacks,
           int target_sq) {
            return cc::square_attacked_by_black_chessckers(occupied, occupied_white, stacks,
                                                           target_sq);
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("stacks"), py::arg("target_sq"),
        "Slice 3a: walk-based Black attack test on a target square.");

    m.def(
        "white_in_chessckers_check",
        [](uint64_t occupied, uint64_t occupied_white, long white_king,
           std::map<uint8_t, std::string> stacks) {
            return cc::white_in_chessckers_check(occupied, occupied_white, white_king, stacks);
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("white_king"), py::arg("stacks"),
        "Slice 3a: is White in Chessckers check? black_can_capture_white_king OR "
        "square_attacked_by_black_chessckers on the king square.");

    m.def(
        "white_legal_moves",
        [](uint64_t occupied, uint64_t occupied_white, uint64_t pawns, uint64_t knights,
           uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings,
           uint64_t castling_rights, long ep_square, std::map<uint8_t, std::string> stacks) {
            cc::WhiteBoard b{occupied, occupied_white, pawns,          knights,  bishops, rooks,
                             queens,   kings,          castling_rights, ep_square};
            py::list out;
            for (const auto& c : cc::white_legal_moves(b, stacks)) {
                out.append(white_move_to_dict(c));
                if (c.is_castling) out.append(white_castling_alt_to_dict(c));  // alt form too
            }
            return out;
        },
        py::arg("occupied"), py::arg("occupied_white"), py::arg("pawns"), py::arg("knights"),
        py::arg("bishops"), py::arg("rooks"), py::arg("queens"), py::arg("kings"),
        py::arg("castling_rights"), py::arg("ep_square"), py::arg("stacks"),
        "Slice 3b: full White legal move list (FIDE pseudo-legal + Chessckers check filter), "
        "with the king-to-rook castling alt form. Mirrors Rust white_legal_moves.");

    m.def(
        "apply_black_move",
        [](cc::Board board, const py::dict& move) {
            cc::apply_black_move(board, parse_black_move(move));
            return board;
        },
        py::arg("board"), py::arg("move"),
        "Slice 5a: apply a Black move dict to a Board, returning the new Board "
        "(turn flips to White). Mirrors moves_black.apply_black_move_known.");

    m.def(
        "apply_white_move",
        [](cc::Board board, const py::dict& move) {
            cc::apply_white_move(board, parse_white_move(move));
            return board;
        },
        py::arg("board"), py::arg("move"),
        "Slice 5a: apply a White move dict to a Board, returning the new Board "
        "(turn flips to Black). Ports python-chess board.push for the search-relevant fields.");

    m.def(
        "detect_status",
        [](const cc::Board& b) {
            const auto s = cc::detect_status(b);
            py::object status = s.status.empty() ? py::none() : py::cast(s.status);
            py::object winner = s.winner.empty() ? py::none() : py::cast(s.winner);
            return py::make_tuple(status, winner);
        },
        py::arg("board"),
        "Slice 5a: (status, winner) for a Board — terminal detection mirroring "
        "client._detect_status + the move-gen-derived mate/stalemate/variantEnd.");

    m.def(
        "run_mcts",
        [](cc::Board board, const py::function& eval_fn, int n_sims, double c_puct) {
            const char* g = std::getenv("CHESSCKERS_VALUE_DISCOUNT");
            const double gamma = g ? std::atof(g) : 1.0;
            auto root = std::make_unique<cc::PuctNode>();
            root->board = std::move(board);
            // First sim expands the root; then search to the n_sims visit budget.
            if (n_sims > 0 && !(root->expanded && !root->children.empty()))
                mcts_simulate(root.get(), eval_fn, c_puct, gamma);
            const int remaining = std::max(0, n_sims - root->visits);
            for (int i = 0; i < remaining; ++i)
                mcts_simulate(root.get(), eval_fn, c_puct, gamma);

            py::dict visit_dist;
            std::string chosen;
            int best = -1;
            for (auto& c : root->children) {
                visit_dist[py::str(c->uci)] = c->visits;
                if (c->visits > best) {
                    best = c->visits;
                    chosen = c->uci;
                }
            }
            return py::make_tuple(chosen, visit_dist);
        },
        py::arg("board"), py::arg("eval_fn"), py::arg("n_sims") = 100, py::arg("c_puct") = 1.5,
        "Slice 5b/5c: native PUCT search. eval_fn(fen, legal_move_dicts) -> (value, priors); "
        "only the NN forward crosses into Python. Returns (chosen_uci, {uci: visits}). "
        "No Dirichlet -> deterministic, for parity with mcts_puct.run_mcts.");
}
