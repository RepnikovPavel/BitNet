#!/usr/bin/env bash
# Probe the max "adequate" context size: hide a needle fact at the start of a
# filler context of increasing length and ask about it. Reports the largest
# context where the model still retrieves the needle.
#
#   scripts/probe_context.sh [build_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

BUILD_DIR="${1:-build}"
[[ -x "$BUILD_DIR/bin/llama-cli" ]] || BUILD_DIR=build-docker
MODEL="$(ls models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf 2>/dev/null || true)"
[[ -n "$MODEL" ]] || MODEL=/mnt/nvme/bitnetmodels/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

python3 - "$BUILD_DIR" "$MODEL" <<'EOF'
import subprocess, sys

build, model = sys.argv[1], sys.argv[2]
NEEDLE = "The secret codeword is PINEAPPLE-42."
QUESTION = "What is the secret codeword?"
# ~1 token per 4 chars of English filler
FILLER = ("The history of computing spans mechanical calculators, vacuum tubes, "
          "transistors, integrated circuits and modern microprocessors. Each era "
          "brought smaller, faster and cheaper machines. ")

def probe(ctx):
    n_fill = max(0, (ctx - 120) // len(FILLER) * len(FILLER))
    text = NEEDLE + " " + (FILLER * (n_fill // len(FILLER) + 1))[:n_fill]
    prompt = (f"System: You are a helpful assistant.<|eot_id|>"
              f"User: {text}\n\n{QUESTION}<|eot_id|>Assistant:")
    r = subprocess.run([f"{build}/bin/llama-cli", "-m", model, "-p", prompt,
                        "-n", "32", "-t", "16", "--temp", "0", "-ngl", "0",
                        "-c", str(ctx), "-no-cnv", "-st", "--special"],
                       capture_output=True, text=True, timeout=3600)
    ok = "PINEAPPLE-42" in r.stdout
    print(f"ctx={ctx}: {'OK (needle found)' if ok else 'FAIL'}")
    return ok

last_ok = 0
for ctx in (512, 1024, 2048, 3072, 4096):
    if probe(ctx):
        last_ok = ctx
print(f"max adequate context: {last_ok} (model hard limit: 4096)")
EOF
