// Phase 3B-2b: the standalone C++ self-play client executable — lc0-shaped, NO
// Python. Loops the next_game cycle against a fleet_server and uploads ccz chunks.
//
//   cc_selfplay --server http://127.0.0.1:8000 --games 50 --worker-id 400 \
//               --machine leena --seed 0 --cache-dir /tmp/ccnets \
//               --start-fen "<FEN>"
//
// --games <=0 runs until the server goes away. --start-fen defaults to env
// CHESSCKERS_START_FEN, else the standard opening.
//
// Phase 3B-3 (orchestrator+engine, true lc0): with --jobs-local --run-dir <dir> the
// binary is the ENGINE half — NO HTTP. fleet_client.py (the orchestrator) owns all
// network I/O, mints jobs into <dir>/jobs/, and syncs <dir>/weights.bin; this loop
// claims jobs, plays them, and writes <dir>/buffer/ chunks + <dir>/match_out/ outcomes
// for the orchestrator to ship. Runs until <dir>/STOP appears.
//
//   cc_selfplay --jobs-local --run-dir /run --worker-id 400 --machine leena --seed 0
#include <cstdlib>
#include <iostream>
#include <string>

#include "client.hpp"

namespace {

std::string arg_val(int argc, char** argv, const std::string& flag, const std::string& dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (flag == argv[i]) return argv[i + 1];
    return dflt;
}

const char* kStartFenDefault =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

}  // namespace

bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (flag == argv[i]) return true;
    return false;
}

int main(int argc, char** argv) {
    int worker_id = std::atoi(arg_val(argc, argv, "--worker-id", "400").c_str());
    std::string machine = arg_val(argc, argv, "--machine", "cpp-client");
    auto base_seed = (uint64_t)std::strtoull(arg_val(argc, argv, "--seed", "0").c_str(), nullptr, 10);
    const char* env_fen = std::getenv("CHESSCKERS_START_FEN");
    std::string start_fen =
        arg_val(argc, argv, "--start-fen", env_fen ? env_fen : kStartFenDefault);

    if (has_flag(argc, argv, "--jobs-local")) {
        // Engine half (Phase 3B-3): no HTTP — fleet_client.py orchestrates.
        std::string run_dir = arg_val(argc, argv, "--run-dir", "");
        if (run_dir.empty()) {
            std::cerr << "cc_selfplay --jobs-local: --run-dir is required" << std::endl;
            return 2;
        }
        int batch_size = std::atoi(arg_val(argc, argv, "--batch-size", "1").c_str());
        bool use_gpu = has_flag(argc, argv, "--gpu");
        std::cout << "cc_selfplay: jobs-local run-dir=" << run_dir << " worker=" << worker_id
                  << " machine=" << machine << " batch-size=" << batch_size
                  << " gpu=" << (use_gpu ? "on" : "off") << std::endl;
        int handled = cc::run_jobs_local(run_dir, start_fen, worker_id, machine, base_seed,
                                         /*max_jobs*/ 0, /*seq_start*/ 1, batch_size, use_gpu);
        std::cout << "cc_selfplay: handled " << handled << " jobs" << std::endl;
        return 0;
    }

    std::string server = arg_val(argc, argv, "--server", "http://127.0.0.1:8000");
    int games = std::atoi(arg_val(argc, argv, "--games", "0").c_str());
    std::string cache_dir = arg_val(argc, argv, "--cache-dir", ".");

    std::cout << "cc_selfplay: server=" << server << " games=" << games
              << " worker=" << worker_id << " machine=" << machine << std::endl;
    int played = cc::run_selfplay_client(server, start_fen, games, worker_id, machine, base_seed,
                                         cache_dir, /*seq_start*/ 1);
    std::cout << "cc_selfplay: uploaded " << played << " games" << std::endl;
    return 0;
}
