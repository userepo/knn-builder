# knn-builder

A remote vector index build service for [OpenSearch k-NN](https://github.com/opensearch-project/k-NN), written in Rust. It is meant as a drop-in replacement for [remote-vector-index-builder](https://github.com/opensearch-project/remote-vector-index-builder): the same HTTP API, the same index files, and the same cuVS CAGRA build on the GPU, with less host memory per build and a GPU that stays busy while vectors download and indexes upload.

**Status:** early development. `knn-gpu` builds the CAGRA graph on the GPU with the upstream service's parameters, and `knn-write` writes the index file the service writes, identical to it on every test file; `knn-store` downloads the vectors from S3 and streams the index file back up, with no temporary file; and the library half of `knn-builder` runs the three as a pipeline, the next job downloading while one builds and the last one uploading, within a budget of host memory. The `knn-builder` binary serves the HTTP API on port 1025, answering as the upstream service does; it is not yet packaged as a container image.

## Building

The workspace builds on Linux and Windows with the Rust toolchain pinned in `rust-toolchain.toml`:

```sh
cargo build --workspace
cargo test --workspace
```

### GPU build

The cuVS code in `knn-gpu` is behind its `gpu` feature, which builds a small C++ shim over cuVS's C++ API with CMake. It needs Linux (WSL2 works), an NVIDIA GPU, a C++20 compiler (the system g++ will do), and libcuvs with the CUDA runtime and CMake, which the conda environment in `environment.yml` provides at the versions the upstream service uses (libcuvs 26.08.01, CUDA 12.9):

```sh
conda env create -f environment.yml
conda activate knn-builder
cargo test -p knn-gpu --features gpu
```

Under WSL, the first build's CMake step can take minutes while CMake searches the Windows directories WSL appends to `PATH`; leaving them out of `PATH` for the build avoids it.

### Configuration

The service reads the upstream service's environment variables, with the same defaults: `REQUEST_STORE_MAX_SIZE` (10000), `REQUEST_STORE_TTL_SECONDS` (1800, counted from when a job ends, so a running job never expires), `GPU_MEMORY_LIMIT` and `CPU_MEMORY_LIMIT` (24 and 32, in GB; the CPU limit also bounds the host memory of the jobs in flight), `LOG_LEVEL` (`INFO`; `DEBUG` adds each job's stage times), `S3_ENDPOINT_URL` and `AWS_DEFAULT_REGION`, with credentials from the AWS SDK's default chain. `REQUEST_STORE_TYPE` takes only `memory`; `MAX_WORKERS` and `INDEX_SERIALIZATION_MODE` are read and ignored. `BASIC_AUTH_USERNAME` and `BASIC_AUTH_PASSWORD`, set together, turn on Basic Auth: every request must then carry them.

### S3 tests

The tests of `knn-store` and `knn-builder` that need S3 are ignored by default. CI runs them against LocalStack 4.14, pinned by digest as in `tools/README.md`; locally, start the same image and point the service's own environment variables at it:

```sh
docker run -d --rm --name knn-store-localstack -p 4566:4566 localstack/localstack@sha256:3ebc37595918b8accb852f8048fef2aff047d465167edd655528065b07bc364a
S3_ENDPOINT_URL=http://localhost:4566 AWS_DEFAULT_REGION=us-east-1 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test cargo test -p knn-store -p knn-builder -- --ignored
```

Each test creates a bucket of its own and removes it when it passes.

## License

Apache-2.0. See [LICENSE](LICENSE).
