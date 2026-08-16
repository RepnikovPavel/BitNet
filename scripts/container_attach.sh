#!/usr/bin/env bash
# Attach an interactive shell to the running dev container.
# Start it first with scripts/container_start.sh.
#
#   scripts/container_attach.sh
set -euo pipefail

NAME=bitnet-dev
docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
    echo "container '$NAME' is not running; start it: scripts/container_start.sh" >&2
    exit 1
}
exec docker exec -it "$NAME" bash
