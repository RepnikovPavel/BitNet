#!/usr/bin/env bash
# Shared paths for all BitNet scripts. Models and data live OUTSIDE the repo.
# Override via env if needed:
#   BITNET_MODELS_DIR=/path/to/models BITNET_DATA_DIR=/path/to/data scripts/...
# Without overrides, the first existing base dir wins: /mnt/nvme, then /mnt/nvme2.
if [[ -z "${BITNET_MODELS_DIR:-}" ]]; then
    for base in /mnt/nvme /mnt/nvme2; do
        [[ -d "$base" ]] && { BITNET_MODELS_DIR="$base/bitnetmodels"; break; }
    done
    BITNET_MODELS_DIR="${BITNET_MODELS_DIR:-/mnt/nvme/bitnetmodels}"
fi
if [[ -z "${BITNET_DATA_DIR:-}" ]]; then
    for base in /mnt/nvme /mnt/nvme2; do
        [[ -d "$base" ]] && { BITNET_DATA_DIR="$base/bitnetdata"; break; }
    done
    BITNET_DATA_DIR="${BITNET_DATA_DIR:-/mnt/nvme/bitnetdata}"
fi
BITNET_MODEL="$BITNET_MODELS_DIR/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
