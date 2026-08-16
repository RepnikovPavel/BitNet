# Fork status: fixed inference, one-button build, measured benchmarks

> This fork (branch `deepbench`) fixes the broken upstream master: degenerate
> repetitive output was caused by a hardcoded SiLU in the BitNet FFN instead of
> ReLU^2 (upstream issues [#588](github_mirror/issues/000588.md) /
> [#602](github_mirror/issues/000602.md)), plus a rotten webui-assets download
> that broke the default build. Fixes live in the pinned submodule
> `RepnikovPavel/llama.cpp@bitnet-ffn-relu2-fix`. All pins (code, weights,
> test data, sha256) are in [manifests/weights.json](manifests/weights.json).

One button per task (Docker, `/mnt` from the host is mounted into the container).
Models and test data never live in the project dir: weights go to
`/mnt/nvme/bitnetmodels/`, test data to `/mnt/nvme/bitnetdata/`
(override with `BITNET_MODELS_DIR` / `BITNET_DATA_DIR`).

```sh
# --- 1. data first ---
scripts/download_data.sh        # pinned weights + wikitext-2 test set (sha256-verified)

# --- 2. build ---
scripts/container_start.sh      # start the persistent dev container (bitnet-dev)
scripts/container_attach.sh     # attach an interactive shell to it
scripts/build.sh                # build / rebuild: inside the attached container it
                                #   rebuilds incrementally; outside it uses Docker;
                                #   scripts/build.sh --local builds on the bare host

# --- 3. chat with the model ---
scripts/chat_ternary.sh         # interactive chat, ternary BitNet b1.58 2B-4T
scripts/chat_ternary.sh "hi"    # one-shot question
scripts/chat_binary.sh          # chat with a binary 1-bit model from
                                #   /mnt/nvme/bitnetmodels/binary/*.gguf (see note)

# --- tests / benchmarks ---
scripts/test_accuracy.sh        # perplexity on wikitext-2
scripts/benchmark.sh            # speed / power / degeneration probes
scripts/probe_context.sh        # max adequate context probe (needle test)
scripts/compare_with_demo.sh    # diff local answers vs the online demo
scripts/docker_run.sh <cmd>     # run anything inside a throwaway container
```

Note on binary BitNet: Microsoft never published binary (1-bit) weights -
all official checkpoints are ternary b1.58. `chat_binary.sh` works as soon as
you drop a binary BitNet GGUF into `/mnt/nvme/bitnetmodels/binary/`; until
then use `chat_ternary.sh`. Paper-level binary metrics are in Table 1 below.

## Table 1 — model quality metrics (BitNet b1.58 2B-4T, ternary; binary BitNet from the papers)

| Metric | Binary BitNet (paper 1, 6.7B, W1A8) | Ternary BitNet b1.58 2B-4T (paper 6) | This fork, measured |
|---|---|---|---|
| Perplexity (wikitext-2) | 17.07 (FP16 baseline: 15.19) | reference 17.109 (bitnet.cpp, I2_S) | **16.86 ± 0.13** (localhost, ctx 512) |
| Zero/few-shot average | 55.9 (FP16: 57.8) | 54.19 (16 tasks) | — |
| Energy per token | 0.02–0.06 J (matmul, 7nm est.) | 0.028 J (decode, est.) | see Table 2 (watts) |
| Non-embedding memory | ~10x smaller than FP16 | 0.4 GB | 1.10 GiB GGUF on disk |

## Table 2 — measured CPU inference benchmark (BitNet b1.58 2B-4T I2_S, this fork)

| Metric | localhost: AMD Ryzen 7 5700X (8C/16T) | server1: AMD Threadripper 3960X (24C/48T) |
|---|---|---|
| Prefill pp512, tok/s | 270.3 | 246.3 |
| Generation tg128, tok/s | 18.7 | 9.9 (48 threads) |
| Avg package power (RAPL) | 66.1 W | 195.3 W |
| Benchmark wall time | 2m 43s | 3m 40s |
| Max adequate context (needle probe) | 4096 (hard limit of the model) | 4096 |
| Max adequate generation (repetition probe) | >= 1024 tokens (ratio 0.000) | >= 1024 tokens (ratio <= 0.019) |
| Perplexity wikitext-2 (ctx 512) | 16.86 ± 0.13 | 16.86 ± 0.13 (bit-identical) |

Note: on the 24-core Threadripper token generation is memory-bandwidth bound,
so more threads do not help. All runs: `-ngl 0`, temp 0.6,
GGUF sha256-pinned (see manifest), build pinned to
`RepnikovPavel/llama.cpp@bitnet-ffn-relu2-fix`.

Paper-reference speeds (bitnet.cpp papers 3/5, ternary, decoding tok/s):
700M: 119.1 (i7-13700H) / 194.4 (M2 Ultra); 7B: 18.8 / 52.4; 13B: 11.0 / 33.8;
speedup vs llama.cpp FP16: 2.37x-6.17x (x86), 1.37x-5.07x (ARM).

### Note on output quality vs the online demo

The official demo and this fork produce the same *class* of answers. Measured
on the probe prompt "write spconv kernel with nvidia cuda assembly and tensor
cores" (uniq-8-gram ratio, higher = less repetition): demo 0.98 / 0.78, this
fork 0.99 / 0.91 / 0.97 / 0.57 - same distribution. Important: this is a 2B
1.58-bit model, answers are approximate and vary between samples in both the
demo and local runs. If an answer degenerates (repeated loops), it is almost
always one of: missing chat template fix, generation not stopping at
`<|eot_id|>`, or temperature above the official 0.6 - the provided scripts
already set all of these.


---

```sh
ocrc parse https://arxiv.org/pdf/2310.11453 --out /home/user/bitnetseries/1 && \
ocrc parse https://arxiv.org/pdf/2402.17764 --out /home/user/bitnetseries/2 && \
ocrc parse https://arxiv.org/pdf/2410.16144 --out /home/user/bitnetseries/3 && \
ocrc parse https://arxiv.org/pdf/2411.04965 --out /home/user/bitnetseries/4 && \
ocrc parse https://arxiv.org/pdf/2502.11880 --out /home/user/bitnetseries/5 && \
ocrc parse https://arxiv.org/pdf/2504.12285 --out /home/user/bitnetseries/6

wget -O /home/user/bitnetseries/1.pdf https://arxiv.org/pdf/2310.11453 && \
wget -O /home/user/bitnetseries/2.pdf https://arxiv.org/pdf/2402.17764 && \
wget -O /home/user/bitnetseries/3.pdf https://arxiv.org/pdf/2410.16144 && \
wget -O /home/user/bitnetseries/4.pdf https://arxiv.org/pdf/2411.04965 && \
wget -O /home/user/bitnetseries/5.pdf https://arxiv.org/pdf/2502.11880 && \
wget -O /home/user/bitnetseries/6.pdf https://arxiv.org/pdf/2504.12285
```

<div align="center">

# bitnet.cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/badge/version-1.0-blue)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/collections/microsoft/bitnet)
[![Technical Report](https://img.shields.io/badge/Technical-Report-red?logo=arxiv)](https://arxiv.org/abs/2502.11880)
[![Demo](https://img.shields.io/badge/Online-Demo-green?logo=microsoft)](https://demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net/)
[![GPU Kernel](https://img.shields.io/badge/GPU-Kernel-6F42C1?logo=github)](https://github.com/microsoft/BitNet/blob/main/gpu/README.md)

</div>

<div align="left">

<h3>📰 News</h3>

<strong>07/23/2026:</strong> 📣 We released <a href="https://github.com/microsoft/VibeASR.cpp"><strong>VibeASR.cpp</strong></a> — a real-time multilingual ASR inference engine on CPU using BitNet I2_S quantization, achieving RTF < 1 with very few threads on x86 (AVX2) and ARM (NEON) platforms. [<a href="https://github.com/microsoft/VibeASR.cpp">Code</a>] [<a href="https://huggingface.co/microsoft/VibeVoice-ASR-BitNet">Models</a>] [<a href="https://arxiv.org/abs/2607.21075">Report</a>] ![NEW](https://img.shields.io/badge/NEW-red)

<strong>07/20/2026:</strong> 📣 We released <a href="https://huggingface.co/microsoft/BitNet-embedding-0.6B"><strong>BitNet-embedding-0.6B</strong></a> and <a href="https://huggingface.co/microsoft/BitNet-embedding-270M"><strong>BitNet-embedding-270M</strong></a> on Hugging Face — the first 1-bit embedding models that deliver competitive embedding quality with significantly faster inference on CPUs.
- **1.42x to 2.28x speedup** over F16 on BitNet-embedding-0.6B prefill (8 threads)
- **1.32x to 1.74x speedup** over F16 on BitNet-embedding-270M prefill (8 threads)
- Supports I2_S conversion with optimized kernels on x86 CPUs
- Lossless inference with 2 bits per weight

07/16/2026: 📣 Released [BitNet Embeddings 0.6B/270M: I2_S Conversion and Inference Optimization](docs/bitnet-embeddings-i2s-guide.md) — detailed guide for converting and running BitNet embedding models with optimized I2_S kernels.

01/15/2026: 📣 Released [BitNet CPU Inference Optimization](https://github.com/microsoft/BitNet/blob/main/src/README.md) — parallel kernel implementations with configurable tiling and embedding quantization support, achieving **1.15x to 2.1x** additional speedup over the original implementation.

05/20/2025: 📣 Released [BitNet Official GPU inference kernel](https://github.com/microsoft/BitNet/blob/main/gpu/README.md) — extending 1-bit inference beyond CPUs.

04/14/2025: 📣 Released [BitNet Official 2B Parameter Model](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) on Hugging Face — the first official BitNet b1.58 model trained with 4T tokens.

02/18/2025: 📑 [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880) — system-level paper on bitnet.cpp's architecture and design.

11/08/2024: 📑 [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965) — enabling 4-bit activations for further efficiency gains.

10/21/2024: 📑 [1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs](https://arxiv.org/abs/2410.16144) — the technical report behind bitnet.cpp.

10/17/2024: 📣 bitnet.cpp 1.0 released.

03/21/2024: 📑 [The-Era-of-1-bit-LLMs: Training Tips, Code, FAQ](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)

02/27/2024: 📑 [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) — the foundational paper introducing BitNet b1.58.

10/17/2023: 📑 [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) — the original BitNet paper.

</div>

## Overview

bitnet.cpp is the official inference framework for 1-bit LLMs (e.g., BitNet b1.58). It offers a suite of optimized kernels that support **fast** and **lossless** inference of 1.58-bit models on **CPU** and **GPU** (NPU support coming next).

Try it out via this [online demo](https://demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net/), or build and run it on your own [CPU](https://github.com/microsoft/BitNet?tab=readme-ov-file#build-from-source) or [GPU](https://github.com/microsoft/BitNet/blob/main/gpu/README.md).

bitnet.cpp achieves speedups of **1.37x** to **5.07x** on ARM CPUs, with larger models experiencing greater performance gains. Additionally, it reduces energy consumption by **55.4%** to **70.0%**, further boosting overall efficiency. On x86 CPUs, speedups range from **2.37x** to **6.17x** with energy reductions between **71.9%** to **82.2%**. Furthermore, bitnet.cpp can run a 100B BitNet b1.58 model on a single CPU, achieving speeds comparable to human reading (5-7 tokens per second), significantly enhancing the potential for running LLMs on local devices. Please refer to the [technical report](https://arxiv.org/abs/2410.16144) for more details.

<img src="./assets/performance.png" alt="performance_comparison" width="800"/>

## Model Releases

### 1. [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) - 1-bit Large Language Model

**BitNet-b1.58-2B-4T** is the first official BitNet b1.58 model with **2.4B parameters**, trained on **4 trillion tokens**. It is a ternary (1.58-bit) language model that delivers competitive performance with full-precision models of similar size while enabling significantly faster and more energy-efficient inference.

- **Fast CPU Inference**: Achieves up to **6.17x speedup** on x86 CPUs and **5.07x** on ARM CPUs compared to full-precision models.
- **Energy Efficient**: Reduces energy consumption by up to **82.2%** on x86 and **70.0%** on ARM.
- **GPU Support**: Official GPU inference kernel available for accelerated deployment.
- **Chat-Ready**: Supports conversational mode for interactive use.

[🤗 Hugging Face](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | [🔗 Online Demo](https://demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net/) | [📄 Technical Report](https://arxiv.org/abs/2410.16144)

<img src="./assets/bitnet_b1.58_2b_benchmark.png" alt="BitNet b1.58 2B Benchmark" width="600"/>

### 2. [BitNet-embedding-0.6B](https://huggingface.co/microsoft/BitNet-embedding-0.6B) - 1-bit Embedding Model

**BitNet-embedding-0.6B** is a **0.6B-parameter** 1-bit embedding model that achieves competitive embedding quality with significantly faster CPU inference. It is the first model to demonstrate that ternary weights can deliver strong performance on embedding tasks.

- **1.42x to 2.28x speedup** over F16 on prefill (8 threads, x86)
- **Lossless Quality**: Competitive embedding quality with 2 bits per weight
- **I2_S Kernel**: Supports optimized I2_S conversion on x86 CPUs

[🤗 Hugging Face](https://huggingface.co/microsoft/BitNet-embedding-0.6B) | [📄 I2_S Guide](docs/bitnet-embeddings-i2s-guide.md)

<img src="./assets/embedding_prefill_0.6B.png" alt="BitNet Embedding 0.6B Prefill Performance" width="600"/>

### 3. [BitNet-embedding-270M](https://huggingface.co/microsoft/BitNet-embedding-270M) - Lightweight 1-bit Embedding Model

**BitNet-embedding-270M** is a compact **270M-parameter** 1-bit embedding model designed for resource-constrained environments, offering fast inference with minimal memory footprint.

- **1.32x to 1.74x speedup** over F16 on prefill (8 threads, x86)
- **Lossless Quality**: Competitive embedding quality with 2 bits per weight
- **Lightweight**: Only 270M parameters for edge deployment scenarios

[🤗 Hugging Face](https://huggingface.co/microsoft/BitNet-embedding-270M) | [📄 I2_S Guide](docs/bitnet-embeddings-i2s-guide.md)

<img src="./assets/embedding_prefill_270M.png" alt="BitNet Embedding 270M Prefill Performance" width="600"/>


## Supported Models

<table>
    <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Parameters</th>
        <th rowspan="2">CPU</th>
        <th colspan="3">Kernel</th>
    </tr>
    <tr>
        <th>I2_S</th>
        <th>TL1</th>
        <th>TL2</th>
    </tr>
    <tr>
        <th colspan="6" style="text-align:left;">Official Models</th>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/microsoft/BitNet-b1.58-2B-4T">BitNet-b1.58-2B-4T</a></td>
        <td rowspan="2">2.4B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/microsoft/BitNet-embedding-0.6B">BitNet-embedding-0.6B</a></td>
        <td rowspan="2">0.6B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/microsoft/BitNet-embedding-270M">BitNet-embedding-270M</a></td>
        <td rowspan="2">270M</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <th colspan="6" style="text-align:left;">Community Models</th>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-large">bitnet_b1_58-large</a></td>
        <td rowspan="2">0.7B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">bitnet_b1_58-3B</a></td>
        <td rowspan="2">3.3B</td>
        <td>x86</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens">Llama3-8B-1.58-100B-tokens</a></td>
        <td rowspan="2">8.0B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026">Falcon3 Family</a></td>
        <td rowspan="2">1B-10B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130">Falcon-E Family</a></td>
        <td rowspan="2">1B-3B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
</table>

❗️**We use existing 1-bit LLMs available on [Hugging Face](https://huggingface.co/) to demonstrate the inference capabilities of bitnet.cpp. We hope the release of bitnet.cpp will inspire the development of 1-bit LLMs in large-scale settings in terms of model size and training tokens.**

## Installation

### Requirements
- python>=3.10
- cmake>=3.22
- clang>=18
    - For Windows users, install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/). In the installer, toggle on at least the following options(this also automatically installs the required additional tools like CMake):
        -  Desktop-development with C++
        -  C++-CMake Tools for Windows
        -  Git for Windows
        -  C++-Clang Compiler for Windows
        -  MS-Build Support for LLVM-Toolset (clang)
    - For Debian/Ubuntu users, you can download with [Automatic installation script](https://apt.llvm.org/)

        `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
- conda (highly recommend)

### Build from source

> [!IMPORTANT]
> If you are using Windows, please remember to always use a Developer Command Prompt / PowerShell for VS2022 for the following commands. Please refer to the FAQs below if you see any issues.

1. Clone the repo
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
```
2. Install the dependencies
```bash
# (Recommended) Create a new conda environment
conda create -n bitnet-cpp python=3.10
conda activate bitnet-cpp

pip install -r requirements.txt
```
3. Build the project
```bash
# Manually download the model and run with local path
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

```
<pre>
usage: setup_env.py [-h] [--hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}] [--model-dir MODEL_DIR] [--log-dir LOG_DIR] [--quant-type {i2_s,tl1}] [--quant-embd]
                    [--use-pretuned]

Setup the environment for running inference

optional arguments:
  -h, --help            show this help message and exit
  --hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}, -hr {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}
                        Model used for inference
  --model-dir MODEL_DIR, -md MODEL_DIR
                        Directory to save/load the model
  --log-dir LOG_DIR, -ld LOG_DIR
                        Directory to save the logging info
  --quant-type {i2_s,tl1}, -q {i2_s,tl1}
                        Quantization type
  --quant-embd          Quantize the embeddings to f16
  --use-pretuned, -p    Use the pretuned kernel parameters
</pre>

## Usage
### Basic usage
```bash
# Run inference with the quantized model
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```
<pre>
usage: run_inference.py [-h] [-m MODEL] [-n N_PREDICT] -p PROMPT [-t THREADS] [-c CTX_SIZE] [-temp TEMPERATURE] [-cnv]

Run inference

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to model file
  -n N_PREDICT, --n-predict N_PREDICT
                        Number of tokens to predict when generating text
  -p PROMPT, --prompt PROMPT
                        Prompt to generate text from
  -t THREADS, --threads THREADS
                        Number of threads to use
  -c CTX_SIZE, --ctx-size CTX_SIZE
                        Size of the prompt context
  -temp TEMPERATURE, --temperature TEMPERATURE
                        Temperature, a hyperparameter that controls the randomness of the generated text
  -cnv, --conversation  Whether to enable chat mode or not (for instruct models.)
                        (When this option is turned on, the prompt specified by -p will be used as the system prompt.)
</pre>

### Demo

A demo of bitnet.cpp running a BitNet b1.58 3B model on Apple M2:

https://github.com/user-attachments/assets/7f46b736-edec-4828-b809-4be780a3e5b1

### Benchmark
We provide scripts to run the inference benchmark providing a model.

```  
usage: e2e_benchmark.py -m MODEL [-n N_TOKEN] [-p N_PROMPT] [-t THREADS]  
   
Setup the environment for running the inference  
   
required arguments:  
  -m MODEL, --model MODEL  
                        Path to the model file. 
   
optional arguments:  
  -h, --help  
                        Show this help message and exit. 
  -n N_TOKEN, --n-token N_TOKEN  
                        Number of generated tokens. 
  -p N_PROMPT, --n-prompt N_PROMPT  
                        Prompt to generate text from. 
  -t THREADS, --threads THREADS  
                        Number of threads to use. 
```  
   
Here's a brief explanation of each argument:  
   
- `-m`, `--model`: The path to the model file. This is a required argument that must be provided when running the script.  
- `-n`, `--n-token`: The number of tokens to generate during the inference. It is an optional argument with a default value of 128.  
- `-p`, `--n-prompt`: The number of prompt tokens to use for generating text. This is an optional argument with a default value of 512.  
- `-t`, `--threads`: The number of threads to use for running the inference. It is an optional argument with a default value of 2.  
- `-h`, `--help`: Show the help message and exit. Use this argument to display usage information.  
   
For example:  
   
```sh  
python utils/e2e_benchmark.py -m /path/to/model -n 200 -p 256 -t 4  
```  
   
This command would run the inference benchmark using the model located at `/path/to/model`, generating 200 tokens from a 256 token prompt, utilizing 4 threads.  

For the model layout that do not supported by any public model, we provide scripts to generate a dummy model with the given model layout, and run the benchmark on your machine:

```bash
python utils/generate-dummy-bitnet-model.py models/bitnet_b1_58-large --outfile models/dummy-bitnet-125m.tl1.gguf --outtype tl1 --model-size 125M

# Run benchmark with the generated model, use -m to specify the model path, -p to specify the prompt processed, -n to specify the number of token to generate
python utils/e2e_benchmark.py -m models/dummy-bitnet-125m.tl1.gguf -p 512 -n 128
```

### Convert from `.safetensors` Checkpoints

```sh
# Prepare the .safetensors model file
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir ./models/bitnet-b1.58-2B-4T-bf16

# Convert to gguf model
python ./utils/convert-helper-bitnet.py ./models/bitnet-b1.58-2B-4T-bf16
```

## Acknowledgements

This project is based on the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework. We would like to thank all the authors for their contributions to the open-source community. Also, bitnet.cpp's kernels are built on top of the Lookup Table methodologies pioneered in [T-MAC](https://github.com/microsoft/T-MAC/). For inference of general low-bit LLMs beyond ternary models, we recommend using T-MAC.

### FAQ (Frequently Asked Questions)📌 

#### Q1: The build dies with errors building llama.cpp due to issues with std::chrono in log.cpp?

**A:**
This is an issue introduced in recent version of llama.cpp. Please refer to this [commit](https://github.com/tinglou/llama.cpp/commit/4e3db1e3d78cc1bcd22bcb3af54bd2a4628dd323) in the [discussion](https://github.com/abetlen/llama-cpp-python/issues/1942) to fix this issue.

#### Q2: How to build with clang in conda environment on windows?

**A:** 
Before building the project, verify your clang installation and access to Visual Studio tools by running:
```
clang -v
```

This command checks that you are using the correct version of clang and that the Visual Studio tools are available. If you see an error message such as:
```
'clang' is not recognized as an internal or external command, operable program or batch file.
```

It indicates that your command line window is not properly initialized for Visual Studio tools.

• If you are using Command Prompt, run:
```
"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
```

• If you are using Windows PowerShell, run the following commands:
```
Import-Module "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\Microsoft.VisualStudio.DevShell.dll" Enter-VsDevShell 3f0e31ad -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
```

These steps will initialize your environment and allow you to use the correct Visual Studio tools.
