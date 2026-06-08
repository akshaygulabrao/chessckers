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

pybind11_add_module(chessckers_cpp src/bindings.cpp)
target_include_directories(chessckers_cpp PRIVATE src)
target_compile_options(chessckers_cpp PRIVATE -O3 -Wall -Wextra)
# Accelerate (BLAS) for the native NN forward — macOS system framework, no install.
# ACCELERATE_NEW_LAPACK uses the current (non-deprecated) cblas headers.
target_compile_definitions(chessckers_cpp PRIVATE ACCELERATE_NEW_LAPACK)
target_link_libraries(chessckers_cpp PRIVATE "-framework Accelerate" Threads::Threads ZLIB::ZLIB)
