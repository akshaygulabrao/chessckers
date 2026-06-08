cmake_minimum_required(VERSION 3.18)
project(chessckers_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)  # std::thread for Phase-2 multi-threaded self-play
find_package(ZLIB REQUIRED)     # gzip for Phase-3A training-chunk encoding

# BLAS backend for the native NN forward (nn.hpp). The forward is portable cblas_sgemm, so the
# engine is cross-platform (lc0-style). Apple: the Accelerate framework (system, no install;
# ACCELERATE_NEW_LAPACK = current cblas headers, included via <Accelerate/Accelerate.h>).
# Elsewhere (Linux volunteer / GCP boxes): a standard CBLAS via <cblas.h> — prefer OpenBLAS,
# else the generic CMake BLAS find. To build on Linux: apt-get install libopenblas-dev.
# The BLAS *impl* is chosen by platform and is independent of which NN *backend* (cpu/metal/...)
# the engine selects at runtime: the `cpu` backend always uses this BLAS; `metal` adds a GPU path.
set(CC_BLAS_DEFS "")
set(CC_BLAS_INCLUDES "")
if(APPLE)
  set(CC_BLAS_LIBS "-framework Accelerate")
  set(CC_BLAS_DEFS ACCELERATE_NEW_LAPACK)
else()
  find_library(OPENBLAS_LIB NAMES openblas)
  find_path(CBLAS_INCLUDE_DIR NAMES cblas.h PATH_SUFFIXES openblas)
  if(OPENBLAS_LIB)
    set(CC_BLAS_LIBS ${OPENBLAS_LIB})
    if(CBLAS_INCLUDE_DIR)
      set(CC_BLAS_INCLUDES ${CBLAS_INCLUDE_DIR})
    endif()
  else()
    find_package(BLAS REQUIRED)  # reference BLAS / MKL / ATLAS — must export the cblas_* symbols
    set(CC_BLAS_LIBS ${BLAS_LIBRARIES})
  endif()
endif()

# ---- Pluggable NN backends (lc0-style; selected at runtime via --backend / auto) ----
# Each OPTIONAL backend contributes sources + libs + a CC_HAVE_* define that gates its
# registration in nn_backend.hpp's ctor. The `cpu` (BLAS) backend is always built. To add e.g.
# CUDA: add `option(CC_USE_CUDA ...)` + an if() block below that appends src/nn_cuda.cu to
# CC_BACKEND_SRC, its libs to CC_BACKEND_LIBS, and CC_HAVE_CUDA to CC_BACKEND_DEFS; then register
# it with one add({...}) line in nn_backend.hpp under #ifdef CC_HAVE_CUDA. Search/self-play never
# change. Both targets (pybind module + standalone cc_selfplay) get every built backend.
set(CC_BACKEND_SRC "")
set(CC_BACKEND_LIBS "")
set(CC_BACKEND_DEFS "")
option(CC_USE_METAL "Build the Metal (MPSGraph) GPU NN backend (Apple only)" ${APPLE})
if(CC_USE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "CC_USE_METAL requires Apple (Metal frameworks); reconfigure with -DCC_USE_METAL=OFF.")
  endif()
  enable_language(OBJCXX)  # nn_metal.mm is Objective-C++
  list(APPEND CC_BACKEND_SRC src/nn_metal.mm)
  list(APPEND CC_BACKEND_LIBS "-framework Metal" "-framework MetalPerformanceShadersGraph" "-framework Foundation")
  list(APPEND CC_BACKEND_DEFS CC_HAVE_METAL)
endif()

# Both targets share the same compile/link surface (only the entry source differs): the same BLAS,
# the same built NN backends (CC_BACKEND_*), threads, and zlib.
function(cc_configure_target tgt)
  target_include_directories(${tgt} PRIVATE src third_party ${CC_BLAS_INCLUDES})  # third_party = vendored cpp-httplib
  target_compile_options(${tgt} PRIVATE -O3 -Wall -Wextra)
  if(CC_BLAS_DEFS)
    target_compile_definitions(${tgt} PRIVATE ${CC_BLAS_DEFS})
  endif()
  if(CC_BACKEND_DEFS)
    target_compile_definitions(${tgt} PRIVATE ${CC_BACKEND_DEFS})
  endif()
  target_link_libraries(${tgt} PRIVATE ${CC_BLAS_LIBS} ${CC_BACKEND_LIBS} Threads::Threads ZLIB::ZLIB)
endfunction()

pybind11_add_module(chessckers_cpp src/bindings.cpp ${CC_BACKEND_SRC})
cc_configure_target(chessckers_cpp)

# Phase 3B-2b: standalone self-play client executable — NO Python (lc0-shaped, a volunteer box
# runs just this binary). Shares the header-only core TUs + every built NN backend (so
# `cc_selfplay --jobs-local --backend metal` can GPU-batch self-play where the backend is built).
add_executable(cc_selfplay src/client_main.cpp ${CC_BACKEND_SRC})
cc_configure_target(cc_selfplay)
