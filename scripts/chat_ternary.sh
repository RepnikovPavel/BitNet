#!/usr/bin/env bash
# Interactive chat with the TERIMARY BitNet b1.58 2B-4T model (I2_S).
# Uses the fixed chat template (assets/chat_template_bitnet_b1_58.jinja).
# Works both on the host (build/bin) and inside the container (build-docker/bin).
#
#   scripts/chat_ternary.sh ["prompt"]     # without prompt: interactive chat
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
[[ -f "$MODEL" ]] || MODEL=/mnt/nvme/bitnetmodels/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
[[ -f "$MODEL" ]] || { echo "model not found; run scripts/download_data.sh" >&2; exit 1; }

CLI=build/bin/llama-cli
[[ -x "$CLI" ]] || CLI=build-docker/bin/llama-cli
[[ -x "$CLI" ]] || { echo "no build found; run scripts/build.sh" >&2; exit 1; }

TEMPLATE=assets/chat_template_bitnet_b1_58.jinja

if [[ $# -gt 0 ]]; then
    # one-shot question
    exec "$CLI" -m "$MODEL" -p "$1" -n 512 -t "$(nproc)" --temp 0.7 --top-p 0.9 \
        -ngl 0 -c 4096 -cnv -st --chat-template-file "$TEMPLATE"
else
    # interactive chat loop (type /exit or Ctrl+C to quit)
    exec "$CLI" -m "$MODEL" -t "$(nproc)" --temp 0.7 --top-p 0.9 \
        -ngl 0 -c 4096 -cnv --chat-template-file "$TEMPLATE"
fi
