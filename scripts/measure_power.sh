#!/usr/bin/env bash
# Measure average CPU package power (W) during a fixed inference run via RAPL.
# Must see /sys/class/powercap: run as root, or via scripts/docker_run.sh.
#
#   scripts/measure_power.sh [build_dir] [n_tokens]
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="${1:-build}"
[[ -x "$BUILD_DIR/bin/llama-cli" ]] || BUILD_DIR=build-docker
N="${2:-256}"
MODEL="$(ls models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf 2>/dev/null || true)"
[[ -n "$MODEL" ]] || MODEL=/mnt/nvme/bitnetmodels/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
[[ -r "$RAPL" ]] || { echo "RAPL unreadable; run as root or via scripts/docker_run.sh" >&2; exit 1; }

E0=$(cat "$RAPL")
T0=$(date +%s.%N)
"$BUILD_DIR/bin/llama-cli" -m "$MODEL" -p "A large language model is" \
    -n "$N" -t "$(nproc)" --temp 0.6 -ngl 0 -c 4096 -st >/dev/null 2>&1
T1=$(date +%s.%N)
E1=$(cat "$RAPL")

python3 - "$E0" "$E1" "$T0" "$T1" <<'EOF'
import sys
e0, e1, t0, t1 = (float(x) for x in sys.argv[1:5])
w = (e1 - e0) / 1e6 / (t1 - t0)
print(f"avg package power: {w:.1f} W over {t1-t0:.1f}s")
EOF
