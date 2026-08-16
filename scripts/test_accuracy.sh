#!/usr/bin/env bash
# One-button accuracy test: perplexity of BitNet-b1.58-2B-4T (I2_S) on wikitext-2 test.
#
#   scripts/test_accuracy.sh [build_dir]
#
# Reference values (see README): official bitnet.cpp reference PPL ~ 17.109,
# broken SiLU build scores ~ 99.8 (microsoft/BitNet issue #588).
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="${1:-build}"
[[ -x "$BUILD_DIR/bin/llama-perplexity" ]] || BUILD_DIR=build-docker
[[ -x "$BUILD_DIR/bin/llama-perplexity" ]] || { echo "no build found; run scripts/build.sh first" >&2; exit 1; }

MODEL="$(ls models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf 2>/dev/null || true)"
[[ -n "$MODEL" ]] || MODEL=/mnt/nvme/bitnetmodels/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
[[ -f "$MODEL" ]] || { echo "model not found; run scripts/download_data.sh first" >&2; exit 1; }
[[ -s data/wiki.test.raw ]] || { echo "data/wiki.test.raw missing; run scripts/download_data.sh first" >&2; exit 1; }

echo "model: $MODEL"
echo "running llama-perplexity on wikitext-2 test (ctx 512, this takes a while)..."
"$BUILD_DIR/bin/llama-perplexity" -m "$MODEL" -f data/wiki.test.raw -c 512 -ngl 0 2>&1 | tee results/accuracy_$(hostname).log | tail -5
