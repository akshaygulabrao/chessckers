// Apple-only Metal (MPSGraph) NN backend for the native engine — Phase 6.
//
// The CPU forward (nn.hpp, cblas_sgemm) is the portable default + the parity oracle. On Apple
// this Metal backend runs the SAME forward on the GPU with FULL leaf batches, where the win is
// large: a GIL-free full-batch MPS trunk forward measured ~10x/board vs batch-1 (the lc0 reason
// batching exists — it's a GPU play). Compiled only when CC_HAVE_METAL is defined (APPLE).
#pragma once

namespace cc {

// 6a toolchain spike: C = A[M,K] @ B[K,N] via MPSGraph on the default Metal device; returns the
// max abs diff vs a CPU reference (≈0 ⇒ MPSGraph links + runs + agrees with BLAS). -1 if no GPU.
float metal_matmul_selftest(int M, int K, int N);

}  // namespace cc
