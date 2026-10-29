# knn-builder's image. A build stage makes the binary with the gpu feature
# against the conda environment of environment.yml (libcuvs 26.08.01, CUDA
# 12.9); the image holds that binary and the libraries it loads from the
# environment, alone, on CUDA 12.9's base image for Ubuntu 24.04: the
# upstream service's, at its latest patch release (12.9.2, where upstream
# has 12.9.0), which asks the same of the host's driver. It runs as a user
# of its own, on port 1025:
#
#   docker build -t knn-builder .
#   docker run --gpus all -p 80:1025 knn-builder

ARG CUDA_IMAGE=nvidia/cuda:12.9.2-base-ubuntu24.04@sha256:bb0d3cf98bade70e5bf3e15d92355d1418b9628f59b7ed853e83ea66ff20fa49

FROM ${CUDA_IMAGE} AS build

# knn-gpu's C++ shim builds with the system g++ and make, as in WSL.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl g++ make \
    && rm -rf /var/lib/apt/lists/*

# Miniforge, then the environment, at the path the binary's rpath names.
ARG MINIFORGE=26.9.4-1
ARG MINIFORGE_SHA256=571b410272cc434f28bc6c5c49ca8ffd4a4d1bcba9f57f1b6baccdb46446cfcc
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE}/Miniforge3-${MINIFORGE}-Linux-x86_64.sh" \
    && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum -c - \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
COPY environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda env create -q -f /tmp/environment.yml -p /opt/knn-builder/env \
    && /opt/conda/bin/conda clean -afy
# As `conda activate` would: its CMake first on the path.
ENV CONDA_PREFIX=/opt/knn-builder/env PATH=/opt/knn-builder/env/bin:${PATH}

# Rust: rustup, then the toolchain rust-toolchain.toml names.
ARG RUSTUP=1.29.2
ARG RUSTUP_SHA256=5be191e8c70a5007074254cb74d78dbfdd25cd5a61c27de7b8e5410122ec24f4
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:${PATH}
RUN curl -fsSL -o /tmp/rustup-init \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP}/x86_64-unknown-linux-gnu/rustup-init" \
    && echo "${RUSTUP_SHA256}  /tmp/rustup-init" | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y -q --no-modify-path --profile minimal --default-toolchain none \
    && rm /tmp/rustup-init
WORKDIR /src
COPY rust-toolchain.toml .
# The toolchain it names, with its components, in this stage only.
RUN rustup toolchain install --no-self-update

COPY . .
# The binary, and every library it loads from the environment, gathered
# for the image; the caches outlive the build.
RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p knn-builder --features gpu \
    && mkdir -p /out/lib \
    && cp target/release/knn-builder /out/ \
    && ldd /out/knn-builder \
        | awk '$3 ~ "^/opt/knn-builder/env/" { print $3 }' \
        | xargs -r cp -L -t /out/lib

FROM ${CUDA_IMAGE}

RUN useradd --create-home --shell /usr/sbin/nologin knn-builder
COPY --from=build /out/lib/ /opt/knn-builder/env/lib/
COPY --from=build /out/knn-builder /usr/local/bin/knn-builder

USER knn-builder
EXPOSE 1025
ENTRYPOINT ["/usr/local/bin/knn-builder"]
