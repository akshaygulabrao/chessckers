#pragma once
// Phase 3B-2b of the lc0-split migration: the standalone C++ self-play client loop.
// Ties the pieces built in 3A/3B-2a into the lc0 next_game cycle, with ZERO Python:
//
//   POST /next_game -> parse job -> (train) GET /get_network?sha=bin_sha (cache by
//   sha) -> play_game_pure -> encode_chunk -> POST /upload_game.
//
// Phase 4b adds GATE (match) jobs: fetch BOTH nets by content address, play one gate
// game (play_match_game), POST the outcome to /match_result. This header is pybind-
// free so it compiles into BOTH the Python extension (for in-process tests) and the
// standalone executable (client_main.cpp, no Python at all).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

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

// A "match" gate job (Phase 4b): two nets fetched by content address, one plays
// White and one Black per `cand_white`, the outcome POSTed to /match_result. params
// keys differ from train (sims/c_puct/dir_alpha/dir_eps/max_plies — mirrors
// fleet_arena's match.json, NOT the train job's dirichlet_*).
struct MatchSpec {
    int match_id = 0;
    std::string candidate_bin_sha;
    std::string opponent_bin_sha;
    std::string opponent;      // panel opponent id ("best" | "net-<id>")
    std::string seed_fen;      // the job's "seed" — a start position FEN
    bool cand_white = true;
    int sims = 160;
    double c_puct = 1.5;
    double dir_alpha = 0.0;
    double dir_eps = 0.25;
    int max_plies = 200;
};

struct Job {
    std::string type;       // "train" | "match"
    std::string sha;        // .pt net sha (Python clients)
    std::string bin_sha;    // .bin net sha (this client fetches THIS)
    ClientPlayParams params;
    MatchSpec match;        // populated only when type == "match"
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
    if (j.type == "match") {
        MatchSpec& mm = j.match;
        mm.match_id = (int)v.get_num("match_id", 0);
        mm.candidate_bin_sha = v.get_str("candidate_bin_sha");
        mm.opponent_bin_sha = v.get_str("opponent_bin_sha");
        mm.opponent = v.get_str("opponent", "best");
        mm.seed_fen = v.get_str("seed");
        const JsonValue* cw = v.find("cand_white");
        mm.cand_white = cw ? cw->b : true;
        if (p && p->is_obj()) {  // match params: sims/c_puct/dir_alpha/dir_eps/max_plies
            mm.sims = (int)p->get_num("sims", mm.sims);
            mm.c_puct = p->get_num("c_puct", mm.c_puct);
            mm.dir_alpha = p->get_num("dir_alpha", mm.dir_alpha);
            mm.dir_eps = p->get_num("dir_eps", mm.dir_eps);
            mm.max_plies = (int)p->get_num("max_plies", mm.max_plies);
        }
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

// The /match_result body the server tallies — same shape the Python worker drops for
// fleet_client to POST: {match_id, seed, opp, cand_white, outcome}.
inline std::string match_result_json(const MatchSpec& mm, const std::string& outcome) {
    std::string o = "{";
    o += "\"match_id\":" + std::to_string(mm.match_id);
    o += ",\"seed\":";
    json_escape(o, mm.seed_fen);
    o += ",\"opp\":";
    json_escape(o, mm.opponent);
    o += ",\"cand_white\":";
    o += (mm.cand_white ? "true" : "false");
    o += ",\"outcome\":";
    json_escape(o, outcome);
    o += "}";
    return o;
}

// Fetch the net with content address `bin_sha`, caching the loaded ChesskersNet by
// sha (re-downloaded only on a cache miss). The .bin is written under net_cache_dir.
// Returns nullptr on transport / load failure (the caller skips the job).
inline std::shared_ptr<ChesskersNet> fetch_net(
    const std::string& base_url, const std::string& bin_sha, const std::string& net_cache_dir,
    std::map<std::string, std::shared_ptr<ChesskersNet>>& cache) {
    auto it = cache.find(bin_sha);
    if (it != cache.end()) return it->second;
    auto [ns, netbytes] = fleet_get_network(base_url, bin_sha);
    if (ns != 200 || netbytes.empty()) return nullptr;
    std::string netpath = net_cache_dir + "/net-" + bin_sha + ".bin";
    FILE* f = std::fopen(netpath.c_str(), "wb");
    if (!f) return nullptr;
    std::fwrite(netbytes.data(), 1, netbytes.size(), f);
    std::fclose(f);
    auto net = std::make_shared<ChesskersNet>(netpath);
    cache[bin_sha] = net;
    return net;
}

// Run the self-play / gate client loop against `base_url`. Handles up to num_games
// jobs (num_games<=0 => until a transport error). Returns the count handled (train
// games uploaded + gate games reported). game i is seeded base_seed+i for train
// (deterministic); each gate game gets a non-overlapping Dirichlet-seed range. Nets
// are fetched by content address and cached by sha (re-downloaded only on change).
// net_cache_dir holds the downloaded .bin(s). Stops on the first /next_game failure.
inline int run_selfplay_client(const std::string& base_url, const std::string& start_fen,
                               int num_games, int worker_id, const std::string& machine,
                               uint64_t base_seed, const std::string& net_cache_dir,
                               long seq_start = 1) {
    const char* g = std::getenv("CHESSCKERS_VALUE_DISCOUNT");
    const double gamma = g ? std::atof(g) : 1.0;
    Board start = parse_fen(start_fen);
    std::map<std::string, std::shared_ptr<ChesskersNet>> nets;  // bin_sha -> loaded net
    int handled = 0;
    long seq = seq_start;
    long match_idx = 0;  // distinct Dirichlet-seed range per gate game
    for (int i = 0; num_games <= 0 || i < num_games; ++i) {
        auto [st, body] = fleet_next_game(base_url);
        if (st != 200) break;  // transport / server gone
        Job job = parse_job(body);
        if (job.type == "match") {
            const MatchSpec& mm = job.match;
            if (mm.candidate_bin_sha.empty() || mm.opponent_bin_sha.empty())
                continue;  // no .bin twins yet — the Python gate path still handles it
            auto cand = fetch_net(base_url, mm.candidate_bin_sha, net_cache_dir, nets);
            auto opp = fetch_net(base_url, mm.opponent_bin_sha, net_cache_dir, nets);
            if (!cand || !opp) continue;
            const ChesskersNet& white = mm.cand_white ? *cand : *opp;
            const ChesskersNet& black = mm.cand_white ? *opp : *cand;
            const uint64_t dseed = base_seed + (uint64_t)(match_idx++) * (uint64_t)(mm.max_plies + 1);
            MatchGame mg = play_match_game(white, black, mm.seed_fen, mm.sims, mm.c_puct,
                                           mm.dir_alpha, mm.dir_eps, mm.max_plies, dseed, gamma);
            auto [rs, rbody] = fleet_post_match_result(base_url, match_result_json(mm, mg.outcome));
            (void)rbody;
            if (rs == 200) ++handled;
            continue;
        }
        if (job.type != "train") continue;
        if (job.bin_sha.empty()) continue;  // no C++ net published yet (gate-only run)
        auto net = fetch_net(base_url, job.bin_sha, net_cache_dir, nets);
        if (!net) continue;
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
        if (us == 200) ++handled;
    }
    return handled;
}

// ---- Phase 3B-3: local-job ENGINE mode (the lc0 engine half — NO HTTP) ----
//
// In the orchestrator+engine split (true lc0: a thin Go-style client drives a heavy
// engine binary), fleet_client.py is the orchestrator — it owns ALL HTTP (next_game /
// get_network / upload_game / match_result / self-update / STOP), mints jobs into
// run_dir/jobs/, syncs run_dir/weights.bin, and pre-fetches gate nets to local .bin
// paths. cc_selfplay --jobs-local is the ENGINE: it claims one job at a time from the
// queue, plays it, and writes the output to run_dir/buffer/ (train chunk) or
// run_dir/match_out/ (gate outcome) for the orchestrator to ship. It never touches the
// network. This is a 1:1 mirror of selfplay_worker_async.play_jobs_forever, swapping the
// torch engine for the native one (play_game_pure / play_match_game / encode_chunk).

namespace fsx = std::filesystem;

// Read a whole file into a string; returns false on open failure.
inline bool read_file_all(const std::string& path, std::string& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out.clear();
    char buf[8192];
    size_t r;
    while ((r = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, r);
    std::fclose(f);
    return true;
}

// Write `data` to `path` atomically (write .tmp, then rename) — so the orchestrator
// never reads a half-written chunk/outcome. Returns false on any failure.
inline bool write_file_atomic(const std::string& path, const std::string& data) {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const size_t w = std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
    if (w != data.size()) return false;
    std::error_code ec;
    fsx::rename(tmp, path, ec);
    return !ec;
}

// Atomically claim one queued job (mirror of selfplay_worker_async._claim_job): of the
// sorted jobs/*.json, rename the first claimable to <name>.c<wid> (POSIX rename is
// atomic — were this ever multi-process, exactly one racer wins; the losers' rename
// errors and they fall through). On success fills (seq=the file's stem, claimed_path,
// body) and returns true. A file that vanishes mid-claim is skipped. Returns false when
// nothing is claimable right now.
inline bool claim_job_local(const std::string& jobs_dir, int worker_id, std::string& seq,
                            std::string& claimed_path, std::string& body) {
    std::error_code ec;
    std::vector<fsx::path> candidates;
    for (auto it = fsx::directory_iterator(jobs_dir, ec);
         !ec && it != fsx::directory_iterator(); it.increment(ec)) {
        const auto& p = it->path();
        if (p.extension() == ".json") candidates.push_back(p);
    }
    std::sort(candidates.begin(), candidates.end());
    for (const auto& src : candidates) {
        fsx::path dst = src;
        dst += ".c" + std::to_string(worker_id);
        fsx::rename(src, dst, ec);
        if (ec) continue;  // lost the race / vanished — next candidate
        if (!read_file_all(dst.string(), body)) {
            fsx::remove(dst, ec);  // unreadable — drop so it can't wedge the queue
            continue;
        }
        seq = src.stem().string();  // "<seq>.json" -> "<seq>"
        claimed_path = dst.string();
        return true;
    }
    return false;
}

// Run the local-job engine loop in run_dir. Claims jobs until `max_jobs` are handled
// (>0) or — for the live fleet (max_jobs<=0) — until run_dir/STOP appears. Train games
// are seeded base_seed + (train index) so a given job is reproducible regardless of
// interleaved match jobs; each gate game gets a non-overlapping Dirichlet-seed range.
// The train net is run_dir/weights.bin (hot-reloaded on mtime, exactly like the Python
// worker's weights.pt poll); match nets are the job's local cand_bin/opp_bin paths
// (the orchestrator pre-fetched them). Returns the count handled.
inline int run_jobs_local(const std::string& run_dir, const std::string& start_fen,
                          int worker_id, const std::string& machine, uint64_t base_seed,
                          int max_jobs = 0, long seq_start = 1) {
    const char* g = std::getenv("CHESSCKERS_VALUE_DISCOUNT");
    const double gamma = g ? std::atof(g) : 1.0;
    const std::string jobs_dir = run_dir + "/jobs";
    const std::string buffer_dir = run_dir + "/buffer";
    const std::string match_out = run_dir + "/match_out";
    const std::string weights_bin = run_dir + "/weights.bin";
    const std::string stop_path = run_dir + "/STOP";
    std::error_code ec;
    fsx::create_directories(buffer_dir, ec);
    fsx::create_directories(match_out, ec);

    Board start = parse_fen(start_fen);
    std::map<std::string, std::shared_ptr<ChesskersNet>> match_nets;  // path -> loaded net
    std::shared_ptr<ChesskersNet> train_net;
    fsx::file_time_type last_wt{};
    bool have_wt = false;
    int handled = 0, train_count = 0, match_count = 0;
    long seq = seq_start;

    while (max_jobs <= 0 || handled < max_jobs) {
        if (fsx::exists(stop_path)) break;
        // Hot-reload the train net when the orchestrator syncs a fresher weights.bin
        // (mtime poll; a mid-write read just throws and we retry next tick).
        {
            auto wt = fsx::last_write_time(weights_bin, ec);
            if (!ec && (!have_wt || wt != last_wt)) {
                try {
                    train_net = std::make_shared<ChesskersNet>(weights_bin);
                    last_wt = wt;
                    have_wt = true;
                } catch (...) { /* mid-write — retry */ }
            }
        }
        std::string jseq, claimed, body;
        if (!claim_job_local(jobs_dir, worker_id, jseq, claimed, body)) {
            if (max_jobs > 0) break;  // bounded mode: an empty queue means drained
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        Job job = parse_job(body);
        bool ok = false;
        if (job.type == "match") {
            const MatchSpec& mm = job.match;
            JsonValue v = parse_json(body);
            const std::string cand_bin = v.get_str("cand_bin");
            const std::string opp_bin = v.get_str("opp_bin");
            auto load = [&](const std::string& path) -> std::shared_ptr<ChesskersNet> {
                if (path.empty()) return nullptr;
                auto it = match_nets.find(path);
                if (it != match_nets.end()) return it->second;
                std::shared_ptr<ChesskersNet> n;
                try { n = std::make_shared<ChesskersNet>(path); } catch (...) { return nullptr; }
                match_nets[path] = n;
                return n;
            };
            auto cand = load(cand_bin), opp = load(opp_bin);
            if (cand && opp) {
                const ChesskersNet& white = mm.cand_white ? *cand : *opp;
                const ChesskersNet& black = mm.cand_white ? *opp : *cand;
                const uint64_t dseed =
                    base_seed + (uint64_t)(match_count) * (uint64_t)(mm.max_plies + 1);
                MatchGame mg = play_match_game(white, black, mm.seed_fen, mm.sims, mm.c_puct,
                                               mm.dir_alpha, mm.dir_eps, mm.max_plies, dseed, gamma);
                ok = write_file_atomic(match_out + "/" + jseq + ".json",
                                       match_result_json(mm, mg.outcome));
                ++match_count;
            }
        } else if (job.type == "train" && train_net) {
            const ClientPlayParams& p = job.params;
            PureGame game = play_game_pure(
                start, *train_net, p.n_sims, p.c_puct, p.temperature, p.temp_cutoff_plies,
                p.max_plies, p.dirichlet_alpha, p.dirichlet_eps, base_seed + (uint64_t)train_count,
                p.resign_threshold, p.resign_no_resign_frac, p.resign_consecutive, p.resign_min_ply,
                gamma);
            const std::string seed_fen = game.records.empty() ? start_fen : game.records.front().fen;
            const std::string fn = client_filename(worker_id, seq++);
            ok = write_file_atomic(buffer_dir + "/" + fn, encode_chunk(game));
            // .meta best-effort (the chunk is the training payload; meta is dashboards).
            write_file_atomic(buffer_dir + "/" + fn + ".meta",
                              client_meta(worker_id, machine, game, seed_fen));
            ++train_count;
        }
        fsx::remove(claimed, ec);  // release the slot (job done, skipped, or net not ready)
        if (ok) ++handled;
    }
    return handled;
}

}  // namespace cc
