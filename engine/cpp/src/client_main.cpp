// Phase 3B-2b: the standalone C++ self-play client executable — lc0-shaped, NO
// Python. Loops the next_game cycle against a fleet_server and uploads ccz chunks.
//
//   cc_selfplay --server http://127.0.0.1:8000 --games 50 --worker-id 400 \
//               --machine leena --seed 0 --cache-dir /tmp/ccnets \
//               --start-fen "<FEN>"
//
// --games <=0 runs until the server goes away. --start-fen defaults to env
// CHESSCKERS_START_FEN, else the standard opening.
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

int main(int argc, char** argv) {
    std::string server = arg_val(argc, argv, "--server", "http://127.0.0.1:8000");
    int games = std::atoi(arg_val(argc, argv, "--games", "0").c_str());
    int worker_id = std::atoi(arg_val(argc, argv, "--worker-id", "400").c_str());
    std::string machine = arg_val(argc, argv, "--machine", "cpp-client");
    auto base_seed = (uint64_t)std::strtoull(arg_val(argc, argv, "--seed", "0").c_str(), nullptr, 10);
    std::string cache_dir = arg_val(argc, argv, "--cache-dir", ".");
    const char* env_fen = std::getenv("CHESSCKERS_START_FEN");
    std::string start_fen =
        arg_val(argc, argv, "--start-fen", env_fen ? env_fen : kStartFenDefault);

    std::cout << "cc_selfplay: server=" << server << " games=" << games
              << " worker=" << worker_id << " machine=" << machine << std::endl;
    int played = cc::run_selfplay_client(server, start_fen, games, worker_id, machine, base_seed,
                                         cache_dir, /*seq_start*/ 1);
    std::cout << "cc_selfplay: uploaded " << played << " games" << std::endl;
    return 0;
}
