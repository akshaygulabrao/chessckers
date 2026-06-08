"""Phase 6a: Metal/MPSGraph backend toolchain spike.

Confirms the Apple-only MPSGraph backend links, runs on the GPU, and agrees with the CPU
BLAS forward (the parity oracle). Skips on non-Apple / headless boxes (no Metal device, or
the module built without CC_HAVE_METAL).
"""
from __future__ import annotations

import sys

import pytest

import chessckers_cpp as cpp


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal backend is Apple-only")
@pytest.mark.parametrize("shape", [(64, 96, 256), (800, 96, 288), (1600, 864, 96)])
def test_metal_matmul_matches_cpu(shape):
    m, k, n = shape
    diff = cpp.metal_matmul_selftest(m, k, n)
    if diff == -2.0:
        pytest.skip("built without CC_HAVE_METAL")
    if diff == -1.0:
        pytest.skip("no Metal GPU on this box")
    assert diff < 1e-3, f"MPSGraph matmul diverged from CPU: max|diff|={diff}"
