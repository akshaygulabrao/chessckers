// Metal (MPSGraph) NN backend — Phase 6a toolchain spike. See nn_metal.h.
#include "nn_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <vector>

namespace cc {

float metal_matmul_selftest(int M, int K, int N) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return -1.0f;  // no Metal GPU (e.g. headless CI)
        id<MTLCommandQueue> q = [dev newCommandQueue];

        std::vector<float> A((size_t)M * K), B((size_t)K * N);
        for (size_t i = 0; i < A.size(); ++i) A[i] = std::sin(0.1 * (double)i);
        for (size_t i = 0; i < B.size(); ++i) B[i] = std::cos(0.07 * (double)i);

        // CPU reference (the parity oracle).
        std::vector<float> ref((size_t)M * N, 0.0f);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                const float a = A[(size_t)m * K + k];
                for (int n = 0; n < N; ++n) ref[(size_t)m * N + n] += a * B[(size_t)k * N + n];
            }

        MPSGraph* g = [MPSGraph new];
        MPSGraphTensor* ta = [g placeholderWithShape:@[ @(M), @(K) ]
                                            dataType:MPSDataTypeFloat32
                                                name:@"A"];
        MPSGraphTensor* tb = [g placeholderWithShape:@[ @(K), @(N) ]
                                            dataType:MPSDataTypeFloat32
                                                name:@"B"];
        MPSGraphTensor* tc = [g matrixMultiplicationWithPrimaryTensor:ta secondaryTensor:tb name:nil];

        id<MTLBuffer> ba = [dev newBufferWithBytes:A.data()
                                            length:A.size() * sizeof(float)
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb = [dev newBufferWithBytes:B.data()
                                            length:B.size() * sizeof(float)
                                           options:MTLResourceStorageModeShared];
        MPSGraphTensorData* da = [[MPSGraphTensorData alloc] initWithMTLBuffer:ba
                                                                        shape:@[ @(M), @(K) ]
                                                                     dataType:MPSDataTypeFloat32];
        MPSGraphTensorData* db = [[MPSGraphTensorData alloc] initWithMTLBuffer:bb
                                                                        shape:@[ @(K), @(N) ]
                                                                     dataType:MPSDataTypeFloat32];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* res =
            [g runWithMTLCommandQueue:q
                                feeds:@{ta : da, tb : db}
                        targetTensors:@[ tc ]
                     targetOperations:nil];
        MPSGraphTensorData* dc = res[tc];

        std::vector<float> out((size_t)M * N);
        [[dc mpsndarray] readBytes:out.data() strideBytes:nil];

        float md = 0.0f;
        for (size_t i = 0; i < out.size(); ++i) md = std::max(md, std::fabs(out[i] - ref[i]));
        return md;
    }
}

}  // namespace cc
