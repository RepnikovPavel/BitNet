#!/usr/bin/env bash
# Compare local inference with the official online demo on the same prompt.
#
#   scripts/compare_with_demo.sh ["prompt"] [n_tokens] [build_dir]
#
# Default prompt is the regression probe:
#   "write spconv kernel with nvidia cuda assembly and tensor cores"
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

PROMPT="${1:-write spconv kernel with nvidia cuda assembly and tensor cores}"
N="${2:-256}"
BUILD_DIR="${3:-build}"
[[ -x "$BUILD_DIR/bin/llama-cli" ]] || BUILD_DIR=build-docker
[[ -x "$BUILD_DIR/bin/llama-cli" ]] || { echo "no build found; run scripts/build.sh first" >&2; exit 1; }

MODEL="$(ls models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf 2>/dev/null || true)"
[[ -n "$MODEL" ]] || MODEL=/mnt/nvme/bitnetmodels/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
[[ -f "$MODEL" ]] || { echo "model not found; run scripts/download_data.sh" >&2; exit 1; }

DEMO_URL="https://demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net/completion"

echo "== prompt: $PROMPT"
echo "== demo (online reference)..."
python3 - "$DEMO_URL" "$PROMPT" > results/demo_answer.txt <<'EOF'
import json, sys, urllib.request
url, prompt = sys.argv[1], sys.argv[2]
body = json.dumps({
    "messages": [{"role": "user", "content": prompt}],
    "userId": "bitnet-cpp-check", "chatId": "check-1", "device": "cpu",
}).encode()
req = urllib.request.Request(url, data=body,
                             headers={"Content-Type": "application/json"})
out = []
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        line = line.decode("utf-8", "replace").strip()
        if line.startswith("data: "):
            chunk = json.loads(line[6:])
            out.append(chunk.get("content", ""))
            if chunk.get("finished"):
                break
print("".join(out))
EOF
head -c 800 results/demo_answer.txt; echo; echo

echo "== local ($MODEL)..."
"$BUILD_DIR/bin/llama-cli" -m "$MODEL" -p "$PROMPT" -n "$N" -t "$(nproc)" \
    --temp 0.6 -ngl 0 -c 4096 2>/dev/null | tee results/local_answer.txt | head -c 800
echo; echo
echo "full answers: results/demo_answer.txt, results/local_answer.txt"
