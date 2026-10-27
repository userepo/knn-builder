# knn-builder

A remote vector index build service for [OpenSearch k-NN](https://github.com/opensearch-project/k-NN), written in Rust. It is meant as a drop-in replacement for [remote-vector-index-builder](https://github.com/opensearch-project/remote-vector-index-builder): the same HTTP API, the same index files, and the same cuVS CAGRA build on the GPU, with less host memory per build and a GPU that stays busy while vectors download and indexes upload.

**Status:** early development. `knn-write` writes the index file the upstream service writes, identical to it on every test file; the GPU build, the storage transfers and the service itself do not exist yet.

## Building

The workspace builds on Linux and Windows with the Rust toolchain pinned in `rust-toolchain.toml`:

```sh
cargo build --workspace
cargo test --workspace
```

### GPU build

The cuVS code in `knn-gpu` is behind its `gpu` feature. It needs Linux (WSL2 works), an NVIDIA GPU, and libcuvs with the CUDA runtime, which the conda environment in `environment.yml` provides at the versions the upstream service uses (libcuvs 26.08.01, CUDA 12.9):

```sh
conda env create -f environment.yml
conda activate knn-builder
cargo test -p knn-gpu --features gpu
```

## License

Apache-2.0. See [LICENSE](LICENSE).
