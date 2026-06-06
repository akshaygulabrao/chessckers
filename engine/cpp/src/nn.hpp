// Chessckers C++ engine — Slice 6/7: native NN forward (Accelerate BLAS).
//
// Hand-rolled forward of ChesskersScorer off the exported PyTorch weights
// (native_net.export_state_dict). Linear/conv go through Accelerate cblas_sgemm
// (conv via im2col); the per-leaf policy head is batched over ALL the leaf's
// moves in single GEMMs. GroupNorm/LayerNorm stay as (cheap) loops, in double.
// Held within ~1e-4 of PyTorch by tests/test_cpp_nn_parity.py.
//
//   pos_emb = position_trunk(pos[14,8,8])
//   value   = wdl[0]-wdl[2],  wdl = softmax(value_head(pos_emb))
//   logits  = head(cat[pos_emb, move_encoder(M)]) over M=[N,240];  priors = softmax
#pragma once

#include <Accelerate/Accelerate.h>

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

// Y[N,out] = X[N,in] @ W[out,in]^T + b   (W is PyTorch Linear weight, row-major).
inline std::vector<float> linear_batch(const std::vector<float>& X, int N,
                                       const std::vector<float>& W, const std::vector<float>& b,
                                       int out_dim, int in_dim) {
    std::vector<float> Y((size_t)N * out_dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, out_dim, in_dim, 1.0f, X.data(), in_dim,
                W.data(), in_dim, 0.0f, Y.data(), out_dim);
    for (int n = 0; n < N; ++n) {
        float* yp = &Y[(size_t)n * out_dim];
        for (int o = 0; o < out_dim; ++o) yp[o] += b[o];
    }
    return Y;
}

// conv k3p1 bias=false via im2col + GEMM. in: [Cin,8,8], w: [Cout,Cin*9] -> [Cout,8,8].
inline std::vector<float> conv3x3(const std::vector<float>& in, int Cin,
                                  const std::vector<float>& w, int Cout) {
    constexpr int HW = 64;
    std::vector<float> col((size_t)Cin * 9 * HW, 0.0f);
    for (int ci = 0; ci < Cin; ++ci)
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
                float* cp = &col[(size_t)(ci * 9 + ky * 3 + kx) * HW];
                const float* ip = &in[(size_t)ci * HW];
                for (int y = 0; y < 8; ++y) {
                    const int iy = y + ky - 1;
                    for (int x = 0; x < 8; ++x) {
                        const int ix = x + kx - 1;
                        cp[y * 8 + x] =
                            (iy >= 0 && iy < 8 && ix >= 0 && ix < 8) ? ip[iy * 8 + ix] : 0.0f;
                    }
                }
            }
    std::vector<float> out((size_t)Cout * HW);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Cout, HW, Cin * 9, 1.0f, w.data(),
                Cin * 9, col.data(), HW, 0.0f, out.data(), HW);
    return out;
}

inline void groupnorm_(std::vector<float>& x, int C, int G, const std::vector<float>& w,
                       const std::vector<float>& b, float eps = 1e-5f) {
    constexpr int HW = 64;
    const int cg = C / G;
    for (int g = 0; g < G; ++g) {
        const int c0 = g * cg, c1 = (g + 1) * cg, cnt = cg * HW;
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

// per-row LayerNorm over [N, D].
inline void layernorm_(std::vector<float>& x, int N, int D, const std::vector<float>& w,
                       const std::vector<float>& b, float eps = 1e-5f) {
    for (int n = 0; n < N; ++n) {
        float* row = &x[(size_t)n * D];
        double mean = 0.0;
        for (int i = 0; i < D; ++i) mean += row[i];
        mean /= D;
        double var = 0.0;
        for (int i = 0; i < D; ++i) {
            const double d = row[i] - mean;
            var += d * d;
        }
        var /= D;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int i = 0; i < D; ++i) row[i] = static_cast<float>((row[i] - mean) * inv * w[i] + b[i]);
    }
}

inline void relu_(std::vector<float>& x) {
    for (float& v : x)
        if (v < 0) v = 0;
}

struct ChesskersNet {
    WeightStore w;
    int c_in = 15, c_filters = 96, d_hidden = 256, d_move = 240;

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
        auto e = linear_batch(x, 1, w.at("position_trunk.8.weight"), w.at("position_trunk.8.bias"),
                              d_hidden, c_filters * 64);
        layernorm_(e, 1, d_hidden, w.at("position_trunk.9.weight"), w.at("position_trunk.9.bias"));
        relu_(e);
        return e;  // pos_emb [d_hidden]
    }

    float value(const std::vector<float>& pos_emb) const {
        auto v = linear_batch(pos_emb, 1, w.at("value_head.0.weight"), w.at("value_head.0.bias"),
                              d_hidden / 2, d_hidden);
        layernorm_(v, 1, d_hidden / 2, w.at("value_head.1.weight"), w.at("value_head.1.bias"));
        relu_(v);
        const auto wdl = linear_batch(v, 1, w.at("value_head.3.weight"), w.at("value_head.3.bias"),
                                      3, d_hidden / 2);
        const float mx = std::max({wdl[0], wdl[1], wdl[2]});
        const double e0 = std::exp(wdl[0] - mx), e1 = std::exp(wdl[1] - mx), e2 = std::exp(wdl[2] - mx);
        return static_cast<float>((e0 - e2) / (e0 + e1 + e2));
    }

    // Policy logits for N moves (features stacked into M[N, d_move]), batched.
    std::vector<float> policy_logits(const std::vector<float>& pos_emb,
                                     const std::vector<float>& M, int N) const {
        auto me = linear_batch(M, N, w.at("move_encoder.0.weight"), w.at("move_encoder.0.bias"),
                               d_hidden, d_move);
        layernorm_(me, N, d_hidden, w.at("move_encoder.1.weight"), w.at("move_encoder.1.bias"));
        relu_(me);
        std::vector<float> comb((size_t)N * 2 * d_hidden);
        for (int i = 0; i < N; ++i) {
            float* r = &comb[(size_t)i * 2 * d_hidden];
            std::copy(pos_emb.begin(), pos_emb.end(), r);
            std::copy(&me[(size_t)i * d_hidden], &me[(size_t)(i + 1) * d_hidden], r + d_hidden);
        }
        auto h = linear_batch(comb, N, w.at("head.0.weight"), w.at("head.0.bias"), d_hidden,
                              2 * d_hidden);
        layernorm_(h, N, d_hidden, w.at("head.1.weight"), w.at("head.1.bias"));
        relu_(h);
        return linear_batch(h, N, w.at("head.3.weight"), w.at("head.3.bias"), 1, d_hidden);  // [N]
    }

    std::pair<float, std::vector<float>> eval(const std::vector<float>& pos,
                                              const std::vector<std::vector<float>>& moves) const {
        const auto pe = trunk(pos);
        const float v = value(pe);
        const int N = (int)moves.size();
        std::vector<float> priors(N);
        if (N == 0) return {v, priors};
        std::vector<float> M((size_t)N * d_move);
        for (int i = 0; i < N; ++i)
            std::copy(moves[i].begin(), moves[i].end(), &M[(size_t)i * d_move]);
        const auto logits = policy_logits(pe, M, N);
        const float mx = *std::max_element(logits.begin(), logits.end());
        double s = 0.0;
        for (float l : logits) s += std::exp(l - mx);
        for (int i = 0; i < N; ++i) priors[i] = static_cast<float>(std::exp(logits[i] - mx) / s);
        return {v, priors};
    }
};

}  // namespace cc
