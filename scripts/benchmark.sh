#!/usr/bin/env bash
# One-button benchmark: generation speed (tok/s), prefill speed, CPU package
# power (RAPL, if readable), and adequacy probes (context / generation length).
# Appends a row set to results/benchmark_<hostname>.md
#
#   scripts/benchmark.sh [build_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

BUILD_DIR="${1:-build}"
[[ -x "$BUILD_DIR/bin/llama-cli" ]] || BUILD_DIR=build-docker
[[ -x "$BUILD_DIR/bin/llama-cli" ]] || { echo "no build found; run scripts/build.sh first" >&2; exit 1; }

MODEL="$(ls models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf 2>/dev/null || true)"
[[ -n "$MODEL" ]] || MODEL=/mnt/nvme/bitnetmodels/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
[[ -f "$MODEL" ]] || { echo "model not found; run scripts/download_data.sh first" >&2; exit 1; }

HOST="$(hostname)"
CPU="$(lscpu | sed -n 's/^Model name:\s*//p' | head -1)"
NTHREADS="$(nproc)"
OUT="results/benchmark_${HOST}.md"
echo "# benchmark: $HOST ($CPU, $NTHREADS threads)" | tee "$OUT"
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT"
echo "model: $MODEL" | tee -a "$OUT"

rapl_read() { cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null || echo ""; }

# --- speed + power: fixed-length generation run ---
GEN_TOKENS=256
PROMPT="A large language model is"
E0="$(rapl_read)"
T0=$(date +%s.%N)
LOG=$("$BUILD_DIR/bin/llama-cli" -m "$MODEL" -p "$PROMPT" -n "$GEN_TOKENS" -t "$NTHREADS" --temp 0.6 -ngl 0 -c 4096 2>&1)
T1=$(date +%s.%N)
E1="$(rapl_read)"
echo "$LOG" > "results/bench_gen_${HOST}.log"

TG="$(echo "$LOG" | grep -oE 'eval time.*' | tail -1)"
echo "$TG" | tee -a "$OUT"

WATTS="N/A (RAPL unreadable: run as root or use --privileged docker)"
if [[ -n "$E0" && -n "$E1" ]]; then
    WATTS="$(python3 -c "print(f'{($E1-$E0)/1e6/($T1-$T0):.1f}')")"
fi
echo "avg package power during generation: ${WATTS} W" | tee -a "$OUT"

# --- llama-bench standard table ---
if [[ -x "$BUILD_DIR/bin/llama-bench" ]]; then
    "$BUILD_DIR/bin/llama-bench" -m "$MODEL" -t "$NTHREADS" -ngl 0 2>&1 | tee -a "results/llama_bench_${HOST}.log" | tail -8
fi

# --- adequacy probes: repetition detector on long generation ---
python3 - "$BUILD_DIR" "$MODEL" "$NTHREADS" "$OUT" <<'EOF'
import subprocess, sys, re
build, model, threads, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def gen(prompt, n, ctx):
    r = subprocess.run([f"{build}/bin/llama-cli", "-m", model, "-p", prompt,
                        "-n", str(n), "-t", threads, "--temp", "0.6",
                        "-ngl", "0", "-c", str(ctx)],
                       capture_output=True, text=True, timeout=3600)
    return r.stdout

def rep_ratio(text):
    # fraction of duplicate 8-word shingles: ~0 = healthy, ->1 = degenerate loop
    words = text.split()
    if len(words) < 16:
        return 0.0
    shingles = [" ".join(words[i:i+8]) for i in range(len(words)-8)]
    return 1.0 - len(set(shingles)) / len(shingles)

lines = []
for n in (128, 512, 1024):
    txt = gen("Write a short story about a robot learning to paint.", n, 4096)
    body = txt.split("learning to paint.", 1)[-1]
    lines.append(f"gen {n} tokens: repetition ratio {rep_ratio(body):.3f}")
for l in lines:
    print(l)
with open(out, "a") as f:
    f.write("\n" + "\n".join(lines) + "\n")
EOF

echo "saved: $OUT"
