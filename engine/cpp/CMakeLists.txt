cmake_minimum_required(VERSION 3.18)
project(chessckers_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(chessckers_cpp src/bindings.cpp)
target_include_directories(chessckers_cpp PRIVATE src)
target_compile_options(chessckers_cpp PRIVATE -O3 -Wall -Wextra)
