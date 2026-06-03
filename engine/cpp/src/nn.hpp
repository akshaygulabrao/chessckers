// Chessckers C++ engine — Slice 6: native NN forward.
//
// Hand-rolled forward of ChesskersScorer (model.py) off the exported PyTorch
// weights (native_net.export_state_dict). Plain-C++ ops (the net is tiny: 14->96
// conv trunk, 4 residual blocks on an 8x8 board, ~2.5M params); Accelerate/Eigen
// can replace the matmuls later. Held within ~1e-4 of PyTorch by
// tests/test_cpp_nn_parity.py. Accumulates in double for stability; PyTorch
// computes in float32, so we are if anything slightly more accurate.
//
// Forward (per leaf):
//   pos_emb = position_trunk(pos[14,8,8])
//   value   = wdl[0]-wdl[2],  wdl = softmax(value_head(pos_emb))
//   logit_i = head(cat[pos_emb, move_encoder(move_i)]);  priors = softmax(logits)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cc {

struct NTensor {
    std::vector<int> shape;
    std::vector<float> data;
};

struct WeightStore {
    std::map<std::string, NTensor> tensors;
    const std::vector<float>& at(const std::string& k) const {
        const auto it = tensors.find(k);
        if (it == tensors.end()) throw std::runtime_error("missing weight: " + k);
        return it->second.data;
    }
};

inline WeightStore load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open weights: " + path);
    auto ri = [&]() {
        int32_t v;
        f.read(reinterpret_cast<char*>(&v), 4);
        return v;
    };
    WeightStore ws;
    const int n = ri();
    for (int i = 0; i < n; ++i) {
        const int nl = ri();
        std::string name(nl, '\0');
        f.read(&name[0], nl);
        const int nd = ri();
        NTensor t;
        long prod = 1;
        for (int d = 0; d < nd; ++d) {
            const int dim = ri();
            t.shape.push_back(dim);
            prod *= dim;
        }
        t.data.resize(prod);
        f.read(reinterpret_cast<char*>(t.data.data()), prod * 4);
        if (!f) throw std::runtime_error("truncated weights for " + name);
        ws.tensors[name] = std::move(t);
    }
    return ws;
}

// --- ops (8x8 board; channel-major flat layout c*64 + y*8 + x) ---

inline std::vector<float> conv3x3(const std::vector<float>& in, int Cin,
                                  const std::vector<float>& w, int Cout) {
    constexpr int H = 8, Wd = 8;
    std::vector<float> out(Cout * H * Wd, 0.0f);
    for (int co = 0; co < Cout; ++co)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < Wd; ++x) {
                double acc = 0.0;
                for (int ci = 0; ci < Cin; ++ci) {
                    const float* ip = &in[ci * 64];
                    const float* wp = &w[((co * Cin) + ci) * 9];
                    for (int ky = 0; ky < 3; ++ky) {
                        const int iy = y + ky - 1;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < 3; ++kx) {
                            const int ix = x + kx - 1;
                            if (ix < 0 || ix >= Wd) continue;
                            acc += static_cast<double>(ip[iy * 8 + ix]) * wp[ky * 3 + kx];
                        }
                    }
                }
                out[co * 64 + y * 8 + x] = static_cast<float>(acc);
            }
    return out;
}

inline void groupnorm_(std::vector<float>& x, int C, int G, const std::vector<float>& w,
                       const std::vector<float>& b, float eps = 1e-5f) {
    constexpr int HW = 64;
    const int cg = C / G;
    for (int g = 0; g < G; ++g) {
        const int c0 = g * cg, c1 = (g + 1) * cg;
        const int cnt = cg * HW;
        double mean = 0.0;
        for (int c = c0; c < c1; ++c)
            for (int i = 0; i < HW; ++i) mean += x[c * HW + i];
        mean /= cnt;
        double var = 0.0;
        for (int c = c0; c < c1; ++c)
            for (int i = 0; i < HW; ++i) {
                const double d = x[c * HW + i] - mean;
                var += d * d;
            }
        var /= cnt;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int c = c0; c < c1; ++c)
            for (int i = 0; i < HW; ++i)
                x[c * HW + i] = static_cast<float>((x[c * HW + i] - mean) * inv * w[c] + b[c]);
    }
}

inline std::vector<float> layernorm(const std::vector<float>& x, const std::vector<float>& w,
                                    const std::vector<float>& b, float eps = 1e-5f) {
    const int D = (int)x.size();
    double mean = 0.0;
    for (float v : x) mean += v;
    mean /= D;
    double var = 0.0;
    for (float v : x) {
        const double d = v - mean;
        var += d * d;
    }
    var /= D;
    const double inv = 1.0 / std::sqrt(var + eps);
    std::vector<float> out(D);
    for (int i = 0; i < D; ++i) out[i] = static_cast<float>((x[i] - mean) * inv * w[i] + b[i]);
    return out;
}

inline std::vector<float> linear(const std::vector<float>& x, const std::vector<float>& w,
                                 const std::vector<float>& b, int out_dim, int in_dim) {
    std::vector<float> y(out_dim);
    for (int o = 0; o < out_dim; ++o) {
        double acc = b[o];
        const float* wp = &w[o * in_dim];
        for (int i = 0; i < in_dim; ++i) acc += static_cast<double>(x[i]) * wp[i];
        y[o] = static_cast<float>(acc);
    }
    return y;
}

inline void relu_(std::vector<float>& x) {
    for (float& v : x)
        if (v < 0) v = 0;
}

struct ChesskersNet {
    WeightStore w;
    int c_in = 14, c_filters = 96, d_hidden = 256, d_move = 240;

    explicit ChesskersNet(const std::string& path) : w(load_weights(path)) {}

    std::vector<float> trunk(const std::vector<float>& pos) const {
        auto x = conv3x3(pos, c_in, w.at("position_trunk.0.weight"), c_filters);
        groupnorm_(x, c_filters, 8, w.at("position_trunk.1.weight"), w.at("position_trunk.1.bias"));
        relu_(x);
        for (int blk : {3, 4, 5, 6}) {
            const std::string p = "position_trunk." + std::to_string(blk) + ".";
            auto c1 = conv3x3(x, c_filters, w.at(p + "conv1.weight"), c_filters);
            groupnorm_(c1, c_filters, 8, w.at(p + "bn1.weight"), w.at(p + "bn1.bias"));
            relu_(c1);
            auto c2 = conv3x3(c1, c_filters, w.at(p + "conv2.weight"), c_filters);
            groupnorm_(c2, c_filters, 8, w.at(p + "bn2.weight"), w.at(p + "bn2.bias"));
            for (size_t i = 0; i < x.size(); ++i) c2[i] += x[i];
            relu_(c2);
            x = std::move(c2);
        }
        auto e = linear(x, w.at("position_trunk.8.weight"), w.at("position_trunk.8.bias"), d_hidden,
                        c_filters * 64);
        e = layernorm(e, w.at("position_trunk.9.weight"), w.at("position_trunk.9.bias"));
        relu_(e);
        return e;  // pos_emb [d_hidden]
    }

    float value(const std::vector<float>& pos_emb) const {
        auto v = linear(pos_emb, w.at("value_head.0.weight"), w.at("value_head.0.bias"), d_hidden / 2,
                        d_hidden);
        v = layernorm(v, w.at("value_head.1.weight"), w.at("value_head.1.bias"));
        relu_(v);
        const auto wdl = linear(v, w.at("value_head.3.weight"), w.at("value_head.3.bias"), 3,
                                d_hidden / 2);
        const float mx = std::max({wdl[0], wdl[1], wdl[2]});
        const double e0 = std::exp(wdl[0] - mx), e1 = std::exp(wdl[1] - mx), e2 = std::exp(wdl[2] - mx);
        return static_cast<float>((e0 - e2) / (e0 + e1 + e2));  // P(win) - P(loss)
    }

    float policy_logit(const std::vector<float>& pos_emb, const std::vector<float>& move) const {
        auto me = linear(move, w.at("move_encoder.0.weight"), w.at("move_encoder.0.bias"), d_hidden,
                         d_move);
        me = layernorm(me, w.at("move_encoder.1.weight"), w.at("move_encoder.1.bias"));
        relu_(me);
        std::vector<float> combined(2 * d_hidden);
        std::copy(pos_emb.begin(), pos_emb.end(), combined.begin());
        std::copy(me.begin(), me.end(), combined.begin() + d_hidden);
        auto h = linear(combined, w.at("head.0.weight"), w.at("head.0.bias"), d_hidden, 2 * d_hidden);
        h = layernorm(h, w.at("head.1.weight"), w.at("head.1.bias"));
        relu_(h);
        return linear(h, w.at("head.3.weight"), w.at("head.3.bias"), 1, d_hidden)[0];
    }

    // (value, priors) for a position + its candidate move features.
    std::pair<float, std::vector<float>> eval(const std::vector<float>& pos,
                                              const std::vector<std::vector<float>>& moves) const {
        const auto pe = trunk(pos);
        const float v = value(pe);
        std::vector<float> priors(moves.size());
        if (moves.empty()) return {v, priors};
        std::vector<float> logits(moves.size());
        for (size_t i = 0; i < moves.size(); ++i) logits[i] = policy_logit(pe, moves[i]);
        const float mx = *std::max_element(logits.begin(), logits.end());
        double s = 0.0;
        for (float l : logits) s += std::exp(l - mx);
        for (size_t i = 0; i < logits.size(); ++i)
            priors[i] = static_cast<float>(std::exp(logits[i] - mx) / s);
        return {v, priors};
    }
};

}  // namespace cc
