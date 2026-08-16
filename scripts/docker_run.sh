#!/usr/bin/env bash
# Run any repo script/binary inside the build container.
# Mounts the repo at /src and host /mnt at /mnt (models, data).
# --privileged is used so RAPL power counters are visible for benchmarks.
#
#   scripts/docker_run.sh ./scripts/benchmark.sh build-docker
#   scripts/docker_run.sh build-docker/bin/llama-cli -m models/... -p "hi" -n 32
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=bitnet-cpp:dev
docker run --rm --privileged \
    -v "$PWD":/src \
    -v /mnt:/mnt \
    -w /src \
    "$IMAGE" \
    "$@"
