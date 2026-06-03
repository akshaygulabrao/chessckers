// pybind11 module `chessckers_cpp` — the Python-facing surface of the C++ engine.
// Slice 0 exposes the Board struct (read-only bb fields, for later oracle tests)
// plus parse_fen / serialize_fen.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "board.hpp"
#include "movegen.hpp"

namespace py = pybind11;

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
}
