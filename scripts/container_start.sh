#!/usr/bin/env bash
# Start the persistent BitNet dev container (detached), with the repo and
# the host /mnt mounted. After this, use scripts/container_attach.sh.
#
#   scripts/container_start.sh
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=bitnet-cpp:dev
NAME=bitnet-dev

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container '$NAME' is already running"
    exit 0
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker start "$NAME"
    echo "container '$NAME' started (existing)"
    exit 0
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" -f Dockerfile .
fi
docker run -d --name "$NAME" --privileged \
    -v "$PWD":/src \
    -v /mnt:/mnt \
    -w /src \
    "$IMAGE" \
    sleep infinity
echo "container '$NAME' started; attach with: scripts/container_attach.sh"
