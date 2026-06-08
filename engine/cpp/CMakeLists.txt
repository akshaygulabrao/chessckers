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

# Both targets share the same compile/link surface (only the source differs).
function(cc_configure_target tgt)
  target_include_directories(${tgt} PRIVATE src third_party ${CC_BLAS_INCLUDES})  # third_party = vendored cpp-httplib
  target_compile_options(${tgt} PRIVATE -O3 -Wall -Wextra)
  if(CC_BLAS_DEFS)
    target_compile_definitions(${tgt} PRIVATE ${CC_BLAS_DEFS})
  endif()
  target_link_libraries(${tgt} PRIVATE ${CC_BLAS_LIBS} Threads::Threads ZLIB::ZLIB)
endfunction()

pybind11_add_module(chessckers_cpp src/bindings.cpp)
cc_configure_target(chessckers_cpp)

# Phase 3B-2b: standalone self-play client executable — NO Python (lc0-shaped, a
# volunteer box runs just this binary). Shares the header-only core TUs.
add_executable(cc_selfplay src/client_main.cpp)
cc_configure_target(cc_selfplay)
