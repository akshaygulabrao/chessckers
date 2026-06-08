#pragma once
// Phase 3B-2b of the lc0-split migration: the standalone C++ self-play client loop.
// Ties the pieces built in 3A/3B-2a into the lc0 next_game cycle, with ZERO Python:
//
//   POST /next_game -> parse job -> (train) GET /get_network?sha=bin_sha (cache by
//   sha) -> play_game_pure -> encode_chunk -> POST /upload_game.
//
// Match jobs (gate) are skipped here — gate play in C++ is Phase 4. This header is
// pybind-free so it compiles into BOTH the Python extension (for in-process tests)
// and the standalone executable (client_main.cpp, no Python at all).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "board.hpp"
#include "chunk.hpp"
#include "fleet_http.hpp"
#include "json_parse.hpp"
#include "nn.hpp"
#include "selfplay.hpp"

namespace cc {

struct ClientPlayParams {
    int n_sims = 100;
    double c_puct = 1.5;
    double temperature = 1.0;
    int temp_cutoff_plies = 30;
    int max_plies = 400;
    double dirichlet_alpha = 0.3;
    double dirichlet_eps = 0.25;
    double resign_threshold = 0.0;
    double resign_no_resign_frac = 0.1;
    int resign_consecutive = 2;
    int resign_min_ply = 8;
};

struct Job {
    std::string type;       // "train" | "match"
    std::string sha;        // .pt net sha (Python clients)
    std::string bin_sha;    // .bin net sha (this client fetches THIS)
    ClientPlayParams params;
};

// Parse the next_game job JSON. params keys mirror what the Python worker reads off
// the job (sims/c_puct/temperature/max_plies/dirichlet_*; selfplay.json-sourced);
// absent keys keep the defaults above (resign off, temp_cutoff 30).
inline Job parse_job(const std::string& body) {
    JsonValue v = parse_json(body);
    Job j;
    j.type = v.get_str("type");
    j.sha = v.get_str("sha");
    j.bin_sha = v.get_str("bin_sha");
    const JsonValue* p = v.find("params");
    if (p && p->is_obj()) {
        ClientPlayParams& c = j.params;
        c.n_sims = (int)p->get_num("sims", c.n_sims);
        c.c_puct = p->get_num("c_puct", c.c_puct);
        c.temperature = p->get_num("temperature", c.temperature);
        c.temp_cutoff_plies = (int)p->get_num("temp_cutoff_plies", c.temp_cutoff_plies);
        c.max_plies = (int)p->get_num("max_plies", c.max_plies);
        c.dirichlet_alpha = p->get_num("dirichlet_alpha", c.dirichlet_alpha);
        c.dirichlet_eps = p->get_num("dirichlet_eps", c.dirichlet_eps);
        c.resign_threshold = p->get_num("resign_threshold", c.resign_threshold);
        c.resign_no_resign_frac = p->get_num("resign_no_resign_frac", c.resign_no_resign_frac);
        c.resign_consecutive = (int)p->get_num("resign_consecutive", c.resign_consecutive);
        c.resign_min_ply = (int)p->get_num("resign_min_ply", c.resign_min_ply);
    }
    return j;
}

// Upload filename: <worker_id>_<10-digit seq>.pkl — matches fleet_server._NAME_RE.
inline std::string client_filename(int worker_id, long seq) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%d_%010ld.pkl", worker_id, seq);
    return std::string(buf);
}

// .pkl.meta sidecar — same shape the Python worker writes (per-machine/seed/W-B-D
// dashboards on ingest): {worker_id, machine, outcome, plies, seed_fen}.
inline std::string client_meta(int worker_id, const std::string& machine, const PureGame& game,
                               const std::string& seed_fen) {
    std::string o = "{";
    o += "\"worker_id\":" + std::to_string(worker_id);
    o += ",\"machine\":";
    json_escape(o, machine);
    o += ",\"outcome\":";
    json_escape(o, game.outcome);
    o += ",\"plies\":" + std::to_string(game.records.size());
    o += ",\"seed_fen\":";
    json_escape(o, seed_fen);
    o += "}";
    return o;
}

// Run the self-play client loop against `base_url`. Plays up to num_games train
// games (num_games<=0 => until a transport error). Returns the count uploaded.
// game i is seeded base_seed+i (deterministic); the net is fetched by content
// address and re-fetched only when bin_sha changes. net_cache_dir holds the
// downloaded .bin(s). Stops on the first /next_game transport failure.
inline int run_selfplay_client(const std::string& base_url, const std::string& start_fen,
                               int num_games, int worker_id, const std::string& machine,
                               uint64_t base_seed, const std::string& net_cache_dir,
                               long seq_start = 1) {
    const char* g = std::getenv("CHESSCKERS_VALUE_DISCOUNT");
    const double gamma = g ? std::atof(g) : 1.0;
    Board start = parse_fen(start_fen);
    std::unique_ptr<ChesskersNet> net;
    std::string net_sha;
    int played = 0;
    long seq = seq_start;
    for (int i = 0; num_games <= 0 || i < num_games; ++i) {
        auto [st, body] = fleet_next_game(base_url);
        if (st != 200) break;  // transport / server gone
        Job job = parse_job(body);
        if (job.type != "train") continue;  // match jobs = Phase 4
        if (job.bin_sha.empty()) continue;   // no C++ net published yet (gate-only run)
        if (net_sha != job.bin_sha) {
            auto [ns, netbytes] = fleet_get_network(base_url, job.bin_sha);
            if (ns != 200 || netbytes.empty()) continue;
            std::string netpath = net_cache_dir + "/net-" + job.bin_sha + ".bin";
            FILE* f = std::fopen(netpath.c_str(), "wb");
            if (!f) continue;
            std::fwrite(netbytes.data(), 1, netbytes.size(), f);
            std::fclose(f);
            net = std::make_unique<ChesskersNet>(netpath);
            net_sha = job.bin_sha;
        }
        const ClientPlayParams& p = job.params;
        PureGame game = play_game_pure(start, *net, p.n_sims, p.c_puct, p.temperature,
                                       p.temp_cutoff_plies, p.max_plies, p.dirichlet_alpha,
                                       p.dirichlet_eps, base_seed + (uint64_t)i, p.resign_threshold,
                                       p.resign_no_resign_frac, p.resign_consecutive,
                                       p.resign_min_ply, gamma);
        std::string chunk = encode_chunk(game);
        std::string seed_fen = game.records.empty() ? start_fen : game.records.front().fen;
        std::string fn = client_filename(worker_id, seq++);
        std::string meta = client_meta(worker_id, machine, game, seed_fen);
        auto [us, ubody] = fleet_upload_game(base_url, fn, chunk, meta);
        (void)ubody;
        if (us == 200) ++played;
    }
    return played;
}

}  // namespace cc
