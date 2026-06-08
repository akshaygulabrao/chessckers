#!/usr/bin/env bash
# Build the C++ engine extension and install it into the venv. Run after editing
# anything under cpp/src/.
#
#   cpp/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENG="$(cd "$HERE/.." && pwd)"
PY="$ENG/.venv/bin/python"
BUILD="$HERE/build"

PYBIND_DIR="$("$PY" -c 'import pybind11; print(pybind11.get_cmake_dir())')"

cmake -S "$HERE" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PY" \
  -Dpybind11_DIR="$PYBIND_DIR" >/dev/null
cmake --build "$BUILD" -j >/dev/null

SO="$(ls "$BUILD"/chessckers_cpp*.so)"
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
cp "$SO" "$SITE/"
echo "installed $(basename "$SO") -> $SITE"
