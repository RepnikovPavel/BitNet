#!/usr/bin/env bash
# One-button build (and rebuild) of bitnet.cpp.
#
#   scripts/build.sh          build inside Docker (default, reproducible)
#   scripts/build.sh --local  build on the host (needs clang + cmake)
#
# Rebuilds are incremental: the build directory is kept between runs,
# so after touching sources just rerun the same command.
set -euo pipefail
cd "$(dirname "$0")/.."

CMAKE_FLAGS=(
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_BUILD_TYPE=Release
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_EXAMPLES=ON
    -DLLAMA_BUILD_COMMON=ON
)

if [[ "${1:-}" == "--local" ]]; then
    cmake -S . -B build "${CMAKE_FLAGS[@]}"
    cmake --build build -j "$(nproc)"
    echo "OK: build/bin/llama-cli"
    exit 0
fi

IMAGE=bitnet-cpp:dev
docker build -t "$IMAGE" -f Dockerfile .
docker run --rm \
    -v "$PWD":/src \
    -v /mnt:/mnt \
    -w /src \
    "$IMAGE" \
    bash -c "cmake -S . -B build-docker ${CMAKE_FLAGS[*]} && cmake --build build-docker -j \$(nproc)"
echo "OK: build-docker/bin/llama-cli"
