# BitNet.cpp one-button build environment (CPU inference, x86_64/ARM).
# Build image:   docker build -t bitnet-cpp:dev .
# Usually you do not need to run this manually: scripts/build.sh does it.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        clang \
        cmake \
        make \
        git \
        curl \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
        libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# pinned webui assets version for the llama.cpp server build
# (the default derived version b9919 no longer exists on the HF bucket)
ENV HF_UI_VERSION=b9918

WORKDIR /src
