// pybind11 module `chessckers_cpp` — the Python-facing surface of the C++ engine.
// Slice 0 exposes the Board struct (read-only bb fields, for later oracle tests)
// plus parse_fen / serialize_fen.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <type_traits>
#include <variant>

#include "apply.hpp"
#include "board.hpp"
#include "chunk.hpp"
#include "client.hpp"
#include "encode.hpp"
#include "fleet_http.hpp"
#include "movegen.hpp"
#include "movegen_white.hpp"
#include "native_move.hpp"
#include "nn.hpp"
#include "search.hpp"
#include "selfplay.hpp"

namespace py = pybind11;

// Reconstruct a White move from its dict. Castling is a White king from e1 to a
// corner/2-away square (covers both the e1g1 and the e1h1 notation forms).
static cc::WhiteMove parse_white_move(const py::dict& move) {
    cc::WhiteMove mv;
    int from = cc::parse_square(move["from"].cast<std::string>());
    int to = cc::parse_square(move["to"].cast<std::string>());
    mv.piece = cc::wpiece_from_name(move["piece"].cast<std::string>());
    mv.has_promotion = !move["promotion"].is_none();
    if (mv.has_promotion) mv.promotion = cc::wpiece_from_name(move["promotion"].cast<std::string>());
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

// Native encode of a move dict's features (no torch) — used by the native eval.
static std::vector<float> encode_move_dict(const py::dict& mv) {
    const int from_sq = cc::parse_square(mv["from"].cast<std::string>());
    const int to_sq = cc::parse_square(mv["to"].cast<std::string>());
    std::vector<std::string> wps;
    if (!mv["waypoints"].is_none()) wps = mv["waypoints"].cast<std::vector<std::string>>();
    const bool has_deploy = !mv["deployCount"].is_none();
    const bool has_dem = !mv["demotionsRequired"].is_none();
    const std::string promo = mv["promotion"].is_none() ? "" : mv["promotion"].cast<std::string>();
    return cc::encode_move(from_sq, to_sq, !mv["capture"].is_none(), wps, has_deploy,
                           has_deploy ? mv["deployCount"].cast<int>() : 0, has_dem,
                           has_dem ? mv["demotionsRequired"].cast<int>() : 0, promo);
}

// Native V2 move-feature encode (114-dim gather features) — mirrors
// encoding.encode_move_v2 reading the same dict keys.
static std::vector<float> encode_move_v2_dict(const py::dict& mv) {
    const int from_sq = cc::parse_square(mv["from"].cast<std::string>());
    const int to_sq = cc::parse_square(mv["to"].cast<std::string>());
    std::vector<std::string> wps;
    if (!mv["waypoints"].is_none()) wps = mv["waypoints"].cast<std::vector<std::string>>();
    const bool has_deploy = !mv["deployCount"].is_none();
    const bool has_dem = !mv["demotionsRequired"].is_none();
    const std::string promo = mv["promotion"].is_none() ? "" : mv["promotion"].cast<std::string>();
    return cc::encode_move_v2(from_sq, to_sq, wps, !mv["capture"].is_none(), has_deploy,
                              has_deploy ? mv["deployCount"].cast<int>() : 0, has_dem,
                              has_dem ? mv["demotionsRequired"].cast<int>() : 0, promo);
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

// Pick the version-correct encoders for a net: V1 (15ch 8x8 / 240-dim) vs V2
// (16ch 10x10 / 114-dim gather features). net.is_v2 is set at load from the keys.
static std::vector<float> encode_pos_for(const cc::ChesskersNet& net, const cc::Board& b) {
    return net.is_v2 ? cc::encode_position_v2(b) : cc::encode_position(b);
}
static std::vector<float> encode_move_for(const cc::ChesskersNet& net, const py::dict& mv) {
    return net.is_v2 ? encode_move_v2_dict(mv) : encode_move_dict(mv);
}

// Fully-native expansion: native legal-move gen + native encode + native forward
// (cc::ChesskersNet) — NOTHING crosses into Python. This is the eval that makes
// the search fast (the per-leaf Python round-trip was the ~1x bottleneck).
static double mcts_expand_native(cc::PuctNode* node, const cc::ChesskersNet& net) {
    py::list legal = gen_legal_dicts(node->board);
    const size_t n = py::len(legal);
    node->expanded = true;
    if (n == 0) {
        // No legal moves => a no-moves terminal (mate / stalemate / Black-stuck)
        // that the cheap child-creation checks below couldn't resolve. Resolve
        // its exact status now (lazy terminal detection) and back up the true
        // terminal value, not the value head. detect_status re-runs one move-gen,
        // but only for genuine no-moves nodes, which are rare.
        const auto st = cc::detect_status(node->board);
        node->is_terminal = true;
        node->terminal_status = st.status;
        return cc::terminal_value(*node);
    }
    const auto pos_enc = encode_pos_for(net, node->board);
    std::vector<std::vector<float>> move_encs;
    move_encs.reserve(n);
    for (size_t i = 0; i < n; ++i)
        move_encs.push_back(encode_move_for(net, legal[i].cast<py::dict>()));
    const auto r = net.eval(pos_enc, move_encs);
    for (size_t i = 0; i < n; ++i) {
        const py::dict mv = legal[i].cast<py::dict>();
        auto child = std::make_unique<cc::PuctNode>();
        child->board = apply_dict(node->board, mv);
        child->uci = mv["uci"].cast<std::string>();
        child->prior = r.second[i];
        // Cheap-only terminal checks (O(1) popcounts), mirroring detect_status's
        // first two branches: Black eliminated, or White king captured -- the
        // DOMINANT Chessckers terminals. The no-moves terminals are detected
        // lazily when this child is itself expanded (n==0 above), so we never run
        // full move-gen for the ~80-90% of children that are never expanded. This
        // is behavior-identical to eager detection (the terminal value enters
        // backup only when the node is first selected) but generates far fewer
        // move lists -- the per-child move-gen was the native search's bottleneck.
        if (child->board.stacks.empty() ||
            (child->board.kings & child->board.occupied_white) == 0) {
            child->is_terminal = true;
            child->terminal_status = "variantEnd";
        }
        node->children.push_back(std::move(child));
    }
    return r.first;
}

static void mcts_simulate_native(cc::PuctNode* root, const cc::ChesskersNet& net, double c_puct,
                                 double gamma) {
    const auto path = cc::select_to_leaf(root, c_puct);
    cc::PuctNode* leaf = path.back();
    double value;
    if (leaf->is_terminal) value = cc::terminal_value(*leaf);
    else if (!leaf->expanded) value = mcts_expand_native(leaf, net);
    else value = net.eval(encode_pos_for(net, leaf->board), {}).first;
    cc::backup(path, value, gamma);
}

// Opaque holder so a native PUCT tree can outlive a single run_mcts_native call
// and be handed back as `reuse` next ply (Lc0 tree reuse). Python holds it via
// the native_search _RootShim; child(uci) detaches the played move's subtree.
struct NativeTree {
    std::unique_ptr<cc::PuctNode> root;
};

// One move's search on `root`: expand it (first sim), mix Dirichlet root noise
// (alpha>0, drawn from `rng`), then search to the n_sims visit budget (carried
// reuse visits count, so only the shortfall runs). Shared by run_mcts_native and
// play_game_native so the per-move search can never drift between them.
static void expand_dirichlet_search(cc::PuctNode* root, const cc::ChesskersNet& net, int n_sims,
                                    double c_puct, double gamma, double dirichlet_alpha,
                                    double dirichlet_eps, std::mt19937_64& rng) {
    if (n_sims > 0 && !(root->expanded && !root->children.empty()))
        mcts_simulate_native(root, net, c_puct, gamma);
    if (dirichlet_alpha > 0.0 && !root->children.empty()) {
        std::gamma_distribution<double> gd(dirichlet_alpha, 1.0);
        const int nch = (int)root->children.size();
        std::vector<double> noise(nch);
        double s = 0.0;
        for (int i = 0; i < nch; ++i) {
            noise[i] = gd(rng);
            s += noise[i];
        }
        if (s > 0.0) {
            int i = 0;
            for (auto& c : root->children) {
                c->prior = (1.0 - dirichlet_eps) * c->prior + dirichlet_eps * (noise[i] / s);
                ++i;
            }
        }
    }
    const int remaining = std::max(0, n_sims - root->visits);
    for (int i = 0; i < remaining; ++i)
        mcts_simulate_native(root, net, c_puct, gamma);
}

// Mirror selfplay_az._sample_move_index_from_visits: temp<=0 -> argmax (first max,
// matching Python's max(range, key=...)); temp>0 -> sample ∝ visits^(1/temp).
static int sample_move_index(const std::vector<int>& visits, double temperature,
                             std::mt19937_64& rng) {
    const int n = (int)visits.size();
    if (n == 0) return 0;
    if (temperature <= 0.0) {
        int best = 0;
        for (int i = 1; i < n; ++i)
            if (visits[i] > visits[best]) best = i;
        return best;
    }
    long total = 0;
    for (int v : visits) total += v;
    if (total == 0) return 0;
    std::vector<double> probs(n);
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::pow((double)visits[i], 1.0 / temperature);
        s += probs[i];
    }
    std::uniform_real_distribution<double> u(0.0, s);
    const double r = u(rng);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += probs[i];
        if (r <= acc) return i;
    }
    return n - 1;
}

// Mirror selfplay_az._outcome_from_state: winner takes precedence, else map status.
static std::string outcome_from(const std::string& status, const std::string& winner) {
    if (winner == "white") return "white";
    if (winner == "black") return "black";
    if (status == "mate") return "black";
    if (status == "variantEnd") return "white";
    return "draw";
}

// Phase 1: fully-native self-play game loop — search+sample+apply+record per ply
// with Lc0 tree reuse, ZERO Python in the hot loop. A 1:1 port of
// selfplay_az.play_az_game running the native search. Returns
// (records, outcome, final_status) where records = [(fen, legal_move_dicts,
// visit_counts, side)] so Python builds AZExample exactly as before.
static py::object play_game_native(cc::Board board, const cc::ChesskersNet& net, int n_sims,
                                   double c_puct, double temperature, int temp_cutoff_plies,
                                   int max_plies, double dirichlet_alpha, double dirichlet_eps,
                                   uint64_t seed, double resign_threshold,
                                   double resign_no_resign_frac, int resign_consecutive,
                                   int resign_min_ply) {
    const char* g = std::getenv("CHESSCKERS_VALUE_DISCOUNT");
    const double gamma = g ? std::atof(g) : 1.0;
    std::mt19937_64 rng(seed);

    const bool resign_enabled = resign_threshold > 0.0;
    // A fraction of games never resign (lc0 false-positive calibration). Short-
    // circuits when disabled so the rng stream is untouched (deterministic path).
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const bool no_resign_game = !resign_enabled || (u01(rng) < resign_no_resign_frac);
    int consec_resign = 0;

    py::list records;
    std::unique_ptr<cc::PuctNode> reuse;  // detached subtree from the previous ply
    int ply = 0;
    std::string final_status, final_winner, outcome;
    bool resigned = false;

    while (ply < max_plies) {
        const auto st = cc::detect_status(board);
        if (!st.status.empty()) {
            final_status = st.status;
            final_winner = st.winner;
            break;
        }
        py::list legal = gen_legal_dicts(board);
        const int nlegal = (int)py::len(legal);
        if (nlegal == 0) break;  // no moves but not flagged terminal — end as draw

        // Re-root the reused subtree iff it matches this position (it always does:
        // it's the child of the move we just applied); else search fresh.
        std::unique_ptr<cc::PuctNode> root;
        if (reuse && cc::serialize_fen(reuse->board) == cc::serialize_fen(board)) {
            root = std::move(reuse);
            root->uci.clear();
        } else {
            root = std::make_unique<cc::PuctNode>();
            root->board = board;
        }
        expand_dirichlet_search(root.get(), net, n_sims, c_puct, gamma, dirichlet_alpha,
                                dirichlet_eps, rng);

        // Visit counts aligned to `legal` (by uci — robust to any gen-order detail).
        std::map<std::string, int> vmap;
        for (auto& c : root->children) vmap[c->uci] = c->visits;
        std::vector<int> visits(nlegal, 0);
        py::list vc;
        for (int i = 0; i < nlegal; ++i) {
            const std::string u = legal[i].cast<py::dict>()["uci"].cast<std::string>();
            const auto it = vmap.find(u);
            visits[i] = (it != vmap.end()) ? it->second : 0;
            vc.append(visits[i]);
        }
        const double root_value = root->q();

        records.append(py::make_tuple(cc::serialize_fen(board), legal, vc,
                                      std::string(board.turn_white ? "white" : "black")));

        if (resign_enabled && !no_resign_game && ply >= resign_min_ply) {
            if (root_value <= -resign_threshold) {
                if (++consec_resign >= resign_consecutive) {
                    outcome = board.turn_white ? "black" : "white";  // STM resigns
                    resigned = true;
                    break;
                }
            } else {
                consec_resign = 0;
            }
        }

        const double eff_temp = (ply < temp_cutoff_plies) ? temperature : 0.0;
        const int idx = sample_move_index(visits, eff_temp, rng);
        py::dict chosen = legal[idx].cast<py::dict>();
        const std::string chosen_uci = chosen["uci"].cast<std::string>();

        board = apply_dict(board, chosen);

        // Lc0 tree reuse: detach the chosen child to re-root next ply.
        reuse.reset();
        for (auto& c : root->children) {
            if (c && c->uci == chosen_uci) {
                reuse = std::move(c);
                break;
            }
        }
        ++ply;
    }

    if (!resigned) outcome = outcome_from(final_status, final_winner);
    return py::make_tuple(records, outcome,
                          final_status.empty() ? py::none() : py::cast(final_status));
}

// Phase 2: reconstruct the Python move dict for a NativeMove (cold path — runs
// once per recorded legal move under the GIL, after the threads join). Reuses the
// exact gen_legal_dicts builders off the stored original move, so the dicts are
// byte-identical to the Phase-1 path.
static py::dict native_move_to_dict(const cc::NativeMove& m) {
    if (m.is_white) {
        return m.is_castling_alt ? white_castling_alt_to_dict(m.white_src)
                                 : white_move_to_dict(m.white_src);
    }
    return std::visit(
        [&](auto&& x) -> py::dict {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, cc::QuietMove>)
                return simple_move_dict(x.uci, x.from_name, x.to_name, x.piece, py::none());
            else if constexpr (std::is_same_v<T, cc::DeployMove>)
                return simple_move_dict(x.uci, x.from_name, x.to_name, x.piece,
                                        py::cast(x.deploy_count));
            else if constexpr (std::is_same_v<T, cc::ChargeMove>)
                return charge_to_dict(x);
            else
                return chain_to_dict(x);
        },
        m.black_src);
}

// Phase 2: batched, multi-threaded native self-play. Plays `num_games` games
// across `num_threads` worker threads with the GIL released (the hot loop is pure
// C++ — see selfplay.hpp); re-acquires the GIL only to convert the games to the
// Python record shape. Game i is seeded base_seed+i, so the output is independent
// of num_threads and each game equals a single-threaded play_game_native(seed=
// base_seed+i). Returns [ (records, outcome, final_status) ] — one entry per game,
// each shaped exactly like play_game_native's return.
static py::object play_games_native(cc::Board board, const cc::ChesskersNet& net, int num_games,
                                    int num_threads, int n_sims, double c_puct, double temperature,
                                    int temp_cutoff_plies, int max_plies, double dirichlet_alpha,
                                    double dirichlet_eps, uint64_t base_seed,
                                    double resign_threshold, double resign_no_resign_frac,
                                    int resign_consecutive, int resign_min_ply) {
    const char* g = std::getenv("CHESSCKERS_VALUE_DISCOUNT");
    const double gamma = g ? std::atof(g) : 1.0;
    std::vector<cc::PureGame> games;
    {
        py::gil_scoped_release release;
        games = cc::play_games_pure(board, net, num_games, num_threads, n_sims, c_puct, temperature,
                                    temp_cutoff_plies, max_plies, dirichlet_alpha, dirichlet_eps,
                                    base_seed, resign_threshold, resign_no_resign_frac,
                                    resign_consecutive, resign_min_ply, gamma);
    }
    py::list out;
    for (auto& game : games) {
        py::list records;
        for (auto& rec : game.records) {
            py::list legal;
            for (auto& mv : rec.legal) legal.append(native_move_to_dict(mv));
            py::list vc;
            for (int v : rec.visits) vc.append(v);
            records.append(py::make_tuple(rec.fen, legal, vc,
                                          std::string(rec.side_white ? "white" : "black")));
        }
        out.append(py::make_tuple(
            records, game.outcome,
            game.final_status.empty() ? py::none() : py::cast(game.final_status)));
    }
    return out;
}

// Phase 3A: play one native game and encode it to a ccz1 training chunk (gzipped
// JSON) entirely in C++ — the self-play CLIENT primitive (play -> chunk -> upload).
// seed S here matches play_game_native(seed=S) / play_games_native(base_seed=S)[0].
static py::bytes play_game_chunk(cc::Board board, const cc::ChesskersNet& net, int n_sims,
                                 double c_puct, double temperature, int temp_cutoff_plies,
                                 int max_plies, double dirichlet_alpha, double dirichlet_eps,
                                 uint64_t seed, double resign_threshold, double resign_no_resign_frac,
                                 int resign_consecutive, int resign_min_ply) {
    const char* g = std::getenv("CHESSCKERS_VALUE_DISCOUNT");
    const double gamma = g ? std::atof(g) : 1.0;
    std::string bytes;
    {
        py::gil_scoped_release release;
        cc::PureGame game = cc::play_game_pure(
            board, net, n_sims, c_puct, temperature, temp_cutoff_plies, max_plies, dirichlet_alpha,
            dirichlet_eps, seed, resign_threshold, resign_no_resign_frac, resign_consecutive,
            resign_min_ply, gamma);
        bytes = cc::encode_chunk(game);
    }
    return py::bytes(bytes);
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
        "encode_position", [](const cc::Board& b) { return cc::encode_position(b); },
        py::arg("board"),
        "Slice 6c: encode a Board to the flat 14*8*8 NN position planes. Mirrors "
        "encoding.encode_position / Rust encode_position_bb.");

    m.def(
        "encode_move", [](const py::dict& mv) { return encode_move_dict(mv); }, py::arg("move"),
        "Slice 6c: encode a move dict to the flat 240-dim NN move features. Mirrors "
        "encoding.encode_move / Rust encode_move.");

    m.def(
        "encode_position_v2", [](const cc::Board& b) { return cc::encode_position_v2(b); },
        py::arg("board"),
        "V2: encode a Board to the flat 16*10*10 gather-head planes. Mirrors "
        "encoding.encode_position_v2 / encode_position_state_v2.");

    m.def(
        "encode_move_v2", [](const py::dict& mv) { return encode_move_v2_dict(mv); }, py::arg("move"),
        "V2: encode a move dict to the flat 114-dim gather move features "
        "([from_idx, to_idx, path_mask(100), scalars(12)]). Mirrors encoding.encode_move_v2.");

    py::class_<cc::ChesskersNet>(m, "ChesskersNet")
        .def(py::init<const std::string&>(), py::arg("weights_path"),
             "Slice 6: native NN forward, loaded from native_net.export_state_dict.")
        .def(
            "eval",
            [](const cc::ChesskersNet& net, const std::vector<float>& position,
               const std::vector<std::vector<float>>& moves) {
                const auto r = net.eval(position, moves);
                return py::make_tuple(r.first, r.second);
            },
            py::arg("position"), py::arg("moves"),
            "(value, priors): position is the flat 14*8*8 encoded planes; moves is a list of "
            "240-dim move features. Mirrors model.policy_and_value (WDL->Q + softmax priors).")
        .def(
            "eval_batch",
            [](const cc::ChesskersNet& net, const std::vector<std::vector<float>>& positions,
               const std::vector<std::vector<std::vector<float>>>& moves_per) {
                std::vector<std::pair<float, std::vector<float>>> r;
                {
                    py::gil_scoped_release rel;
                    r = net.eval_batch(positions, moves_per);
                }
                py::list out;
                for (auto& p : r) out.append(py::make_tuple(p.first, p.second));
                return out;
            },
            py::arg("positions"), py::arg("moves_per"),
            "Batched eval over K leaves (batched conv trunk). Byte-equivalent to K eval() calls; "
            "V2 only batches, V1 falls back to a serial loop.");

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

    // Opaque native-tree handle for Lc0 tree reuse across plies.
    py::class_<NativeTree>(m, "NativeTree")
        .def("child", [](NativeTree& self, const std::string& uci) -> std::unique_ptr<NativeTree> {
            if (self.root) {
                for (auto& c : self.root->children) {
                    if (c && c->uci == uci) {
                        auto t = std::make_unique<NativeTree>();
                        t->root = std::move(c);  // detach; siblings freed with the parent tree
                        return t;
                    }
                }
            }
            return nullptr;  // not found / unexpanded -> None in Python (fresh search next ply)
        })
        .def("fen", [](const NativeTree& self) {
            return self.root ? cc::serialize_fen(self.root->board) : std::string();
        })
        .def("visits", [](const NativeTree& self) { return self.root ? self.root->visits : 0; });

    m.def(
        "run_mcts_native",
        [](cc::Board board, const cc::ChesskersNet& net, int n_sims, double c_puct,
           double dirichlet_alpha, double dirichlet_eps, uint64_t seed, py::object reuse) {
            const char* g = std::getenv("CHESSCKERS_VALUE_DISCOUNT");
            const double gamma = g ? std::atof(g) : 1.0;
            std::unique_ptr<cc::PuctNode> root;
            // Tree reuse (Lc0): adopt the passed subtree iff its position matches this
            // search position; its carried visits then count toward the n_sims budget
            // (only the shortfall is searched anew). Any mismatch -> fresh tree.
            if (!reuse.is_none()) {
                NativeTree* rt = nullptr;
                try { rt = reuse.cast<NativeTree*>(); } catch (const std::exception&) { rt = nullptr; }
                if (rt && rt->root && cc::serialize_fen(rt->root->board) == cc::serialize_fen(board)) {
                    root = std::move(rt->root);
                    root->uci.clear();  // it is the root now
                }
            }
            if (!root) {
                root = std::make_unique<cc::PuctNode>();
                root->board = std::move(board);
            }
            // Expand root + Dirichlet noise + search to budget (shared with
            // play_game_native's per-ply search so the two can't drift).
            std::mt19937_64 rng(seed);
            expand_dirichlet_search(root.get(), net, n_sims, c_puct, gamma, dirichlet_alpha,
                                    dirichlet_eps, rng);
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
            const double root_value = root->q();
            auto handle = std::make_unique<NativeTree>();
            handle->root = std::move(root);
            return py::make_tuple(chosen, visit_dist, root_value, py::cast(std::move(handle)));
        },
        py::arg("board"), py::arg("net"), py::arg("n_sims") = 100, py::arg("c_puct") = 1.5,
        py::arg("dirichlet_alpha") = 0.0, py::arg("dirichlet_eps") = 0.25, py::arg("seed") = 0,
        py::arg("reuse") = py::none(),
        "Slice 6d/8: FULLY-NATIVE PUCT search — native move-gen, apply, encode, NN forward. "
        "Optional Dirichlet root noise (alpha>0) for self-play exploration. Returns "
        "(chosen_uci, {uci: visits}, root_value, tree): root_value=root->q() is the side-to-"
        "move's expected outcome at the root (negamax Q in [-1,1], for resignation); `tree` is a "
        "NativeTree — pass tree.child(uci) back as `reuse` next ply for Lc0 tree reuse (carried "
        "visits count toward n_sims).");

    m.def(
        "play_game_native", &play_game_native, py::arg("board"), py::arg("net"),
        py::arg("n_sims") = 100, py::arg("c_puct") = 1.5, py::arg("temperature") = 1.0,
        py::arg("temp_cutoff_plies") = 30, py::arg("max_plies") = 400,
        py::arg("dirichlet_alpha") = 0.0, py::arg("dirichlet_eps") = 0.25, py::arg("seed") = 0,
        py::arg("resign_threshold") = 0.0, py::arg("resign_no_resign_frac") = 0.1,
        py::arg("resign_consecutive") = 2, py::arg("resign_min_ply") = 8,
        "Phase 1: fully-native self-play game loop (search+sample+apply+record per ply, Lc0 tree "
        "reuse, zero Python in the hot loop). Mirrors selfplay_az.play_az_game on the native "
        "search. Returns (records, outcome, final_status) where records = "
        "[(fen, legal_move_dicts, visit_counts, side)] -> Python builds AZExample as before. "
        "Set temperature=0 + dirichlet_alpha=0 + resign_threshold=0 for a deterministic game "
        "(parity-checkable against play_az_game).");

    m.def(
        "play_games_native", &play_games_native, py::arg("board"), py::arg("net"),
        py::arg("num_games"), py::arg("num_threads") = 1, py::arg("n_sims") = 100,
        py::arg("c_puct") = 1.5, py::arg("temperature") = 1.0, py::arg("temp_cutoff_plies") = 30,
        py::arg("max_plies") = 400, py::arg("dirichlet_alpha") = 0.0, py::arg("dirichlet_eps") = 0.25,
        py::arg("base_seed") = 0, py::arg("resign_threshold") = 0.0,
        py::arg("resign_no_resign_frac") = 0.1, py::arg("resign_consecutive") = 2,
        py::arg("resign_min_ply") = 8,
        "Phase 2: batched, multi-threaded native self-play. Plays num_games games across "
        "num_threads std::threads with the GIL released (pure-C++ hot loop); game i is seeded "
        "base_seed+i so the result is thread-count-independent and game i is byte-identical to "
        "play_game_native(seed=base_seed+i). Returns [(records, outcome, final_status)] — one "
        "tuple per game, each shaped exactly like play_game_native's return.");

    m.def(
        "play_game_chunk", &play_game_chunk, py::arg("board"), py::arg("net"),
        py::arg("n_sims") = 100, py::arg("c_puct") = 1.5, py::arg("temperature") = 1.0,
        py::arg("temp_cutoff_plies") = 30, py::arg("max_plies") = 400,
        py::arg("dirichlet_alpha") = 0.0, py::arg("dirichlet_eps") = 0.25, py::arg("seed") = 0,
        py::arg("resign_threshold") = 0.0, py::arg("resign_no_resign_frac") = 0.1,
        py::arg("resign_consecutive") = 2, py::arg("resign_min_ply") = 8,
        "Phase 3A: play one native game (seed) and encode it to a ccz1 training chunk (gzipped "
        "JSON bytes) — the self-play client primitive. Decodable by training_chunk.decode_chunk; "
        "tensor-identical to az_game_to_examples(play_game_native(seed)).");

    m.def(
        "play_match_game",
        [](const cc::ChesskersNet& white_net, const cc::ChesskersNet& black_net,
           const std::string& start_fen, int n_sims, double c_puct, double dirichlet_alpha,
           double dirichlet_eps, int max_plies, uint64_t dir_seed_base) {
            const char* g = std::getenv("CHESSCKERS_VALUE_DISCOUNT");
            const double gamma = g ? std::atof(g) : 1.0;
            cc::MatchGame mg;
            {
                py::gil_scoped_release release;
                mg = cc::play_match_game(white_net, black_net, start_fen, n_sims, c_puct,
                                         dirichlet_alpha, dirichlet_eps, max_plies, dir_seed_base,
                                         gamma);
            }
            return py::make_tuple(mg.outcome, py::cast(mg.moves));
        },
        py::arg("white_net"), py::arg("black_net"), py::arg("start_fen"), py::arg("n_sims") = 160,
        py::arg("c_puct") = 1.5, py::arg("dirichlet_alpha") = 0.0, py::arg("dirichlet_eps") = 0.25,
        py::arg("max_plies") = 200, py::arg("dir_seed_base") = 0,
        "Phase 4b: play one keep-best GATE game (white_net vs black_net) from start_fen and "
        "return (outcome, [ucis]). Mirrors fleet_arena._play_from driven by _native_picker: "
        "per-move fresh-tree native PUCT, argmax-visit pick, per-move Dirichlet seed "
        "(dir_seed_base+ply). Parity-checkable against the Python gate path.");

    // Phase 3B-2a: the self-play client's HTTP surface (cpp-httplib). The native
    // socket I/O runs with the GIL released. Each returns (status, bytes).
    m.def(
        "fleet_get_network",
        [](const std::string& base_url, const std::string& sha) {
            std::pair<int, std::string> r;
            {
                py::gil_scoped_release release;
                r = cc::fleet_get_network(base_url, sha);
            }
            return py::make_tuple(r.first, py::bytes(r.second));
        },
        py::arg("base_url"), py::arg("sha"),
        "GET /get_network?sha= — content-addressed net fetch. Returns (status, bytes).");
    m.def(
        "fleet_upload_game",
        [](const std::string& base_url, const std::string& filename, const py::bytes& chunk,
           const py::bytes& meta) {
            std::string c = chunk, mt = meta;
            std::pair<int, std::string> r;
            {
                py::gil_scoped_release release;
                r = cc::fleet_upload_game(base_url, filename, c, mt);
            }
            return py::make_tuple(r.first, py::bytes(r.second));
        },
        py::arg("base_url"), py::arg("filename"), py::arg("chunk"), py::arg("meta") = py::bytes(),
        "POST /upload_game (multipart) — land a ccz chunk in the server buffer/. Returns (status, body).");
    m.def(
        "fleet_next_game",
        [](const std::string& base_url) {
            std::pair<int, std::string> r;
            {
                py::gil_scoped_release release;
                r = cc::fleet_next_game(base_url);
            }
            return py::make_tuple(r.first, py::bytes(r.second));
        },
        py::arg("base_url"),
        "POST /next_game — claim a job. Returns (status, raw job JSON bytes).");

    // Phase 3B-2b: job-JSON parse (unit-testable vs json.loads) + the full client loop.
    m.def(
        "parse_job",
        [](const std::string& body) {
            cc::Job j = cc::parse_job(body);
            py::dict params;
            params["n_sims"] = j.params.n_sims;
            params["c_puct"] = j.params.c_puct;
            params["temperature"] = j.params.temperature;
            params["temp_cutoff_plies"] = j.params.temp_cutoff_plies;
            params["max_plies"] = j.params.max_plies;
            params["dirichlet_alpha"] = j.params.dirichlet_alpha;
            params["dirichlet_eps"] = j.params.dirichlet_eps;
            params["resign_threshold"] = j.params.resign_threshold;
            params["resign_no_resign_frac"] = j.params.resign_no_resign_frac;
            params["resign_consecutive"] = j.params.resign_consecutive;
            params["resign_min_ply"] = j.params.resign_min_ply;
            py::dict d;
            d["type"] = j.type;
            d["sha"] = j.sha;
            d["bin_sha"] = j.bin_sha;
            d["params"] = params;
            if (j.type == "match") {
                py::dict mm;
                mm["match_id"] = j.match.match_id;
                mm["candidate_bin_sha"] = j.match.candidate_bin_sha;
                mm["opponent_bin_sha"] = j.match.opponent_bin_sha;
                mm["opponent"] = j.match.opponent;
                mm["seed"] = j.match.seed_fen;
                mm["cand_white"] = j.match.cand_white;
                mm["sims"] = j.match.sims;
                mm["c_puct"] = j.match.c_puct;
                mm["dir_alpha"] = j.match.dir_alpha;
                mm["dir_eps"] = j.match.dir_eps;
                mm["max_plies"] = j.match.max_plies;
                d["match"] = mm;
            }
            return d;
        },
        py::arg("body"),
        "Parse a next_game job JSON into {type, sha, bin_sha, params(, match)}.");
    m.def(
        "run_selfplay_client",
        [](const std::string& base_url, const std::string& start_fen, int num_games, int worker_id,
           const std::string& machine, uint64_t base_seed, const std::string& net_cache_dir,
           long seq_start) {
            int n;
            {
                py::gil_scoped_release release;
                n = cc::run_selfplay_client(base_url, start_fen, num_games, worker_id, machine,
                                            base_seed, net_cache_dir, seq_start);
            }
            return n;
        },
        py::arg("base_url"), py::arg("start_fen"), py::arg("num_games"), py::arg("worker_id") = 400,
        py::arg("machine") = "cpp-client", py::arg("base_seed") = 0, py::arg("net_cache_dir") = ".",
        py::arg("seq_start") = 1,
        "Run the standalone self-play client loop (next_game -> get_network -> play -> upload) "
        "against base_url. Train jobs only (match = Phase 4). Returns games uploaded.");
    m.def(
        "run_jobs_local",
        [](const std::string& run_dir, const std::string& start_fen, int worker_id,
           const std::string& machine, uint64_t base_seed, int max_jobs, long seq_start) {
            int n;
            {
                py::gil_scoped_release release;
                n = cc::run_jobs_local(run_dir, start_fen, worker_id, machine, base_seed, max_jobs,
                                       seq_start);
            }
            return n;
        },
        py::arg("run_dir"), py::arg("start_fen"), py::arg("worker_id") = 400,
        py::arg("machine") = "cpp-client", py::arg("base_seed") = 0, py::arg("max_jobs") = 0,
        py::arg("seq_start") = 1,
        "Run the local-job ENGINE loop (NO HTTP): claim jobs from run_dir/jobs/ (atomic), play "
        "the native engine, write train chunks to run_dir/buffer/ + gate outcomes to "
        "run_dir/match_out/. The train net is run_dir/weights.bin (mtime hot-reload); match nets "
        "are the job's local cand_bin/opp_bin paths. max_jobs<=0 runs until run_dir/STOP. Returns "
        "the count handled.");
}
