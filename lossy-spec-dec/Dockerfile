# Build from the repository root:
#   docker build -f lossy-spec-dec/Dockerfile -t sglang-lossy-specdec .
#
# Run with GPUs, for example:
#   docker run --gpus all --ipc=host --network=host -it sglang-lossy-specdec

FROM lmsysorg/sglang:latest

ARG SGLANG_BUILD_TYPE=all
ARG RUST_TOOLCHAIN=1.90

ENV DEBIAN_FRONTEND=noninteractive \
    SGLANG_ENABLE_SPEC_V2=True \
    PATH="/root/.cargo/bin:${PATH}"

WORKDIR /sgl-workspace

# Keep the CUDA/Torch stack from lmsysorg/sglang, but make sure the final
# runtime image can rebuild this fork's editable Python package and Rust module.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN if ! command -v rustup >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"; \
    else \
        rustup toolchain install "${RUST_TOOLCHAIN}" --profile minimal; \
    fi \
    && rustup default "${RUST_TOOLCHAIN}" \
    && rustc --version \
    && cargo --version

# Replace the source tree from the base image with this checkout's fork.
RUN rm -rf /sgl-workspace/sglang && mkdir -p /sgl-workspace/sglang
WORKDIR /sgl-workspace/sglang
COPY README.md LICENSE ./
COPY python ./python
COPY rust ./rust
COPY proto ./proto
COPY lossy-spec-dec ./lossy-spec-dec

# Install the local fork without reinstalling the heavyweight CUDA/Torch deps
# that are already present in the SGLang base image.
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-deps -e "python[${SGLANG_BUILD_TYPE}]" \
    && python3 -m pip install PyYAML datasets requests tqdm

EXPOSE 30000

CMD ["/bin/bash"]
