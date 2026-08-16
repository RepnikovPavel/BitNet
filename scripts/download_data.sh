#!/usr/bin/env bash
# One-button download of all pinned data: model weights + accuracy test set.
# Every file is verified against manifests/weights.json (size + sha256).
#
#   scripts/download_data.sh
#
# Override the model directory with BITNET_MODELS_DIR (default: ./models).
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST=manifests/weights.json
MODELS_DIR="${BITNET_MODELS_DIR:-models}"

read -r W_URL W_FILE W_SIZE W_SHA <<<"$(python3 - "$MANIFEST" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))["weights"]
print(m["download_url"], m["file"], m["size_bytes"], m["sha256"])
EOF
)"
read -r T_URL T_SIZE T_SHA <<<"$(python3 - "$MANIFEST" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))["test_data"]["wikitext2_test_parquet"]
print(t["url"], t["size_bytes"], t["sha256"])
EOF
)"

check() { # path size sha256
    [[ -f "$1" ]] || return 1
    [[ "$(stat -c%s "$1")" == "$2" ]] || return 1
    echo "$3  $1" | sha256sum -c - >/dev/null 2>&1
}

MODEL_DIR="$MODELS_DIR/BitNet-b1.58-2B-4T"
MODEL_PATH="$MODEL_DIR/$W_FILE"
mkdir -p "$MODEL_DIR" data

if check "$MODEL_PATH" "$W_SIZE" "$W_SHA"; then
    echo "weights OK (pinned revision, sha256 verified): $MODEL_PATH"
else
    echo "downloading weights -> $MODEL_PATH"
    curl -L -C - -o "$MODEL_PATH.part" "$W_URL"
    mv "$MODEL_PATH.part" "$MODEL_PATH"
    check "$MODEL_PATH" "$W_SIZE" "$W_SHA" || { echo "ERROR: weights checksum mismatch" >&2; exit 1; }
    echo "weights OK (sha256 verified)"
fi

PARQUET=data/wikitext-2-test.parquet
if check "$PARQUET" "$T_SIZE" "$T_SHA"; then
    echo "test data OK (sha256 verified): $PARQUET"
else
    echo "downloading wikitext-2 test set -> $PARQUET"
    curl -L -o "$PARQUET.part" "$T_URL"
    mv "$PARQUET.part" "$PARQUET"
    check "$PARQUET" "$T_SIZE" "$T_SHA" || { echo "ERROR: test data checksum mismatch" >&2; exit 1; }
    echo "test data OK (sha256 verified)"
fi

# convert parquet -> plain text for llama-perplexity
if [[ ! -s data/wiki.test.raw ]]; then
    python3 scripts/parquet_to_raw.py "$PARQUET" data/wiki.test.raw
fi
echo "raw text OK: data/wiki.test.raw"
echo "DONE"
