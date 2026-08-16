#!/usr/bin/env bash
# Interactive chat with a BINARY (1-bit, original BitNet architecture) model.
#
# Microsoft has NOT published binary BitNet weights (the 2023 paper models
# were never released) - all official checkpoints are ternary b1.58.
# If you have a binary BitNet GGUF, drop it into models/binary/ and this
# script will pick it up:
#
#   models/binary/<name>.gguf
#   scripts/chat_binary.sh ["prompt"]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$(ls models/binary/*.gguf 2>/dev/null | head -1 || true)"
if [[ -z "$MODEL" ]]; then
    cat >&2 <<'EOF'
no binary BitNet model found in models/binary/*.gguf

Binary (1-bit) BitNet weights were never officially published by Microsoft;
all public checkpoints (2B-4T, 3B, large, Falcon-E) are ternary b1.58.
For the ternary model use:  scripts/chat_ternary.sh
If you converted a binary checkpoint yourself, put it at models/binary/*.gguf.
EOF
    exit 1
fi

CLI=build/bin/llama-cli
[[ -x "$CLI" ]] || CLI=build-docker/bin/llama-cli
[[ -x "$CLI" ]] || { echo "no build found; run scripts/build.sh" >&2; exit 1; }

echo "binary model: $MODEL"
if [[ $# -gt 0 ]]; then
    exec "$CLI" -m "$MODEL" -p "$1" -n 512 -t "$(nproc)" --temp 0.7 --top-p 0.9 \
        -ngl 0 -c 4096 -cnv -st
else
    exec "$CLI" -m "$MODEL" -t "$(nproc)" --temp 0.7 --top-p 0.9 \
        -ngl 0 -c 4096 -cnv
fi
