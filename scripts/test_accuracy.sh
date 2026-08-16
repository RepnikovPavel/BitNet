#!/usr/bin/env bash
# One-button accuracy test: perplexity of BitNet-b1.58-2B-4T (I2_S) on wikitext-2 test.
#
#   scripts/test_accuracy.sh [build_dir]
#
# Reference values (see README): official bitnet.cpp reference PPL ~ 17.109,
# broken SiLU build scores ~ 99.8 (microsoft/BitNet issue #588).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
mkdir -p results

BUILD_DIR="${1:-build}"
[[ -x "$BUILD_DIR/bin/llama-perplexity" ]] || BUILD_DIR=build-docker
[[ -x "$BUILD_DIR/bin/llama-perplexity" ]] || { echo "no build found; run scripts/build.sh first" >&2; exit 1; }

MODEL="$BITNET_MODEL"
[[ -f "$MODEL" ]] || { echo "model not found; run scripts/download_data.sh first" >&2; exit 1; }
RAW="$BITNET_DATA_DIR/wiki.test.raw"
[[ -s "$RAW" ]] || { echo "$RAW missing; run scripts/download_data.sh first" >&2; exit 1; }

echo "model: $MODEL"
echo "running llama-perplexity on wikitext-2 test (ctx 512, this takes a while)..."
"$BUILD_DIR/bin/llama-perplexity" -m "$MODEL" -f "$RAW" -c 512 -ngl 0 2>&1 | tee "results/accuracy_$(hostname).log" | tail -5
