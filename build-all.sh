#!/bin/bash

set -e

export CMAKE_BUILD_PARALLEL_LEVEL="$(getconf _NPROCESSORS_ONLN)"

cmake -DCMAKE_BUILD_TYPE=Release "$@" -B build src
cmake --build build --config Release --target build_external_libraries
cmake build
cmake --build build --config Release
