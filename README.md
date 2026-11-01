# knn-builder

A remote vector index build service for [OpenSearch k-NN](https://github.com/opensearch-project/k-NN), written in Rust. It is meant as a drop-in replacement for [remote-vector-index-builder](https://github.com/opensearch-project/remote-vector-index-builder): the same HTTP API, the same index files, and the same cuVS CAGRA build on the GPU, with less host memory per build and a GPU that stays busy while vectors download and indexes upload.

**Status:** `knn-gpu` builds the CAGRA graph on the GPU with the upstream service's parameters, and `knn-write` writes the index file the service writes, identical to it on every test file; `knn-store` downloads the vectors from S3 and streams the index file back up, with no temporary file; and the library half of `knn-builder` runs the three as a pipeline, the next job downloading while one builds and the last one uploading, within a budget of host memory. The `knn-builder` binary serves the HTTP API on port 1025, answering as the upstream service does, and runs from a container image as the upstream one does; an OpenSearch 3.9.0 node with k-NN's remote index build turned on builds its float, half_float, byte and binary segments through it.

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

### Container image

The `Dockerfile` builds the service with the `gpu` feature against the conda environment of `environment.yml`, then copies the binary and the CUDA and cuVS libraries it loads onto `nvidia/cuda:12.9.2-base-ubuntu24.04`, the latest patch release of the base the upstream image uses (12.9.0); it runs as its own user, on port 1025. It needs BuildKit, Docker's default builder, and runs as the upstream user guide starts the upstream image, on a host with the NVIDIA container toolkit:

```sh
docker build -t knn-builder .
docker run --gpus all -p 80:1025 knn-builder
```

### Configuration

The service reads the upstream service's environment variables, with the same defaults: `REQUEST_STORE_MAX_SIZE` (10000), `REQUEST_STORE_TTL_SECONDS` (1800, counted from when a job ends, so a running job never expires), `GPU_MEMORY_LIMIT` and `CPU_MEMORY_LIMIT` (24 and 32, in GB; the CPU limit also bounds the host memory of the jobs in flight), `LOG_LEVEL` (`INFO`; `DEBUG` adds each job's stage times), `S3_ENDPOINT_URL` and `AWS_DEFAULT_REGION`, with credentials from the AWS SDK's default chain; a bucket in another region than `AWS_DEFAULT_REGION` is reached through S3's redirect to it, as the upstream service reaches it. `REQUEST_STORE_TYPE` takes only `memory`; `MAX_WORKERS` and `INDEX_SERIALIZATION_MODE` are read and ignored. `BASIC_AUTH_USERNAME` and `BASIC_AUTH_PASSWORD`, set together, turn on Basic Auth: every request but `GET /_heart_beat`, which stays open for health checks, must then carry them.

Three more describe the data nodes that call the service. `CLIENT_TIMEOUT_SECONDS` (3600) and `CLIENT_POLL_INTERVAL_SECONDS` (5) are k-NN's `knn.remote_index_build.client.timeout` and `knn.remote_index_build.poll.interval`, in seconds, and should match the cluster's. `CPU_BUILD_THREADS`, unset by default, is the data node's `knn.algo_param.index_thread_qty`, 1 or 4. The service forecasts when each new job would end, from the jobs ahead of it and what earlier jobs took, and answers 507 at once, rather than take the job, when the GPU's free memory cannot hold its build, when the data node would see it ended only past its timeout, or, with `CPU_BUILD_THREADS` set, when the data node would build the segment sooner itself. k-NN does not retry a 507: the data node builds that segment on its CPU right away, instead of after the timeout. A job whose build no longer fits the GPU's free memory when its turn comes fails at once, for the same reason. And a job the data node has stopped waiting for, its timeout counted from the request, is dropped: one still queued never starts, a transfer stops, and a build under way, which cannot be stopped, stores nothing, so the jobs still waited for move up. Whenever a job ends, the jobs not yet building are forecast again with what it taught, and those that would now miss are dropped at once, so their data nodes build the segments themselves at their next poll.

### S3 tests

The tests of `knn-store` and `knn-builder` that need S3 are ignored by default. CI runs them against LocalStack 4.14, pinned by digest as in `tools/README.md`; locally, start the same image and point the service's own environment variables at it:

```sh
docker run -d --rm --name knn-store-localstack -p 4566:4566 localstack/localstack@sha256:3ebc37595918b8accb852f8048fef2aff047d465167edd655528065b07bc364a
S3_ENDPOINT_URL=http://localhost:4566 AWS_DEFAULT_REGION=us-east-1 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test cargo test -p knn-store -p knn-builder -- --ignored
```

Each test creates a bucket of its own and removes it when it passes.

## Benchmarks

Both services, measured the same way on the same machine over the datasets the upstream `benchmarking/` tool names, 1M vectors each: sift-128, ms-marco-384, cohere-768 and gist-960, as float inputs in L2 and inner product, the first three also as byte inputs (each component scaled over the dataset's range onto -128..127) and as binary inputs in Hamming space (a bit per positive component). The services ran from their container images under Docker Desktop on WSL2 with LocalStack for S3, `GPU_MEMORY_LIMIT=8`, on an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB, driver 617.42), an Intel Core i9-14900HX and 32 GB of memory, 15 GB of it the Docker VM's; knn-builder at commit d4af6b8 with libcuvs 26.08.01 and CUDA 12.9, the upstream service as its `api-snapshot` image at digest `78ced9d0…` (libcuvs 26.08.01, Faiss 1.14.3, the same CUDA) on `MAX_WORKERS=1`, since two builds at once crash it under WSL2; knn-builder builds one job on the GPU while the next downloads and the last uploads. Every number is the median of five builds, or counted over five rounds, except the bursts and the hour-long queues, run once.

The metrics, per data type in the tables below:

- **Latency** is a build of the input alone, from the request to the status that shows it completed, asked every 0.2 s. **Steps** is the same job on the service's own clock, the sum of its steps, which its `DEBUG` log gives: **Download**, **Build** (for the upstream service, its build, conversion and write together) and **Upload**. The two differ where the client saw the completed status late: 20 to 29 s after the service logged the job's end on 7 of knn-builder's 85 builds, with no status request reaching the service in between, and 5 to 31 s after the last step on 14 of the upstream's, all of files of 1.5 GB or more. Those waits, between the client and the container on this host (the requests pass through Docker Desktop's port forwarding), are in the latencies and not in the steps.
- **Host peak** is the service process's peak resident memory over the job (`VmHWM`, reset before it), **Device peak** the GPU's memory at its peak during the job less the least seen, sampled every 50 ms.
- **Recall@10** is the first build's index searched for the dataset's first 100 test queries with [knn-inspect](https://github.com/userepo/knn-inspect) 0.17.0, against the exact neighbors; the GPU build is not deterministic, so another build's recall differs in its last digits.
- **Concurrent**: five rounds of eight requests sent at once, as k-NN's data nodes send them (one retry after a second of a 408, 409, 429, 500, 502, 503 or 509, a first status request after three poll intervals of 5 s and one an interval after), over the data type's inputs in turn; the latency each data node saw, its p50 and p99 over the 40 jobs.
- **Burst**: 48 requests at once over the data type's L2 inputs (Hamming for binary), the service restarted and warmed up before each, with the data nodes' timeout set to 90 s and the services told so (`CLIENT_TIMEOUT_SECONDS`): how many were built in time, refused at once (a 507, after which the data node builds the segment itself), failed (knn-builder drops a job it forecasts to miss the timeout as the jobs before it end, so the data node starts its own build at the next poll instead of after the timeout) or timed out; **CPU wait** is the seconds the 48 data nodes waited in all before building on their own CPUs.
- **Sustained**: four jobs kept in flight for an hour over the data type's inputs in turn, the next sent as one completes; **Jobs per hour** counts the jobs seen completed within the hour.

### binary

| Input | Service | Builds | Latency s | Steps s | Download s | Build s | Upload s | Host peak GB | Device peak GB | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cohere-hamming-1000000x768-binary` | knn-builder | 5 | 23.8 | 23.5 | 0.2 | 22.7 | 0.7 | 3.06 | 0.49 | 0.963 |
| `cohere-hamming-1000000x768-binary` | upstream | 5 | 25.1 | 24.9 | 0.2 | 23.8 | 0.8 | 2.98 | 0.62 | 0.966 |
| `ms-marco-hamming-1000000x384-binary` | knn-builder | 5 | 16.8 | 16.7 | 0.1 | 16.0 | 0.6 | 2.94 | 0.62 | 0.989 |
| `ms-marco-hamming-1000000x384-binary` | upstream | 5 | 18.2 | 18.1 | 0.2 | 17.4 | 0.6 | 2.78 | 0.71 | 0.988 |
| `sift-hamming-1000000x128-binary` | knn-builder | 5 | 11.4 | 11.3 | 0.1 | 10.7 | 0.5 | 2.89 | 0.41 | 1.000 |
| `sift-hamming-1000000x128-binary` | upstream | 5 | 12.9 | 12.7 | 0.3 | 12.0 | 0.4 | 2.83 | 0.54 | 1.000 |

| Service | Concurrent jobs | Built | Latency p50 s | p99 s | Burst | CPU wait s | Sustained | Jobs per hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn-builder | 8 x 5 | 40 of 40 | 76.4 | 135.1 | 48: 4 built, 44 refused, 0 failed, 0 timed out | 1.4 | 4 in flight for 3600 s | 228.0 |
| upstream | 8 x 5 | 40 of 40 | 72.7 | 156.6 | 48: 3 built, 0 refused, 0 failed, 45 timed out | 4181.9 | 4 in flight for 3600 s | 201.0 |

### byte

| Input | Service | Builds | Latency s | Steps s | Download s | Build s | Upload s | Host peak GB | Device peak GB | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cohere-ip-1000000x768-byte` | knn-builder | 5 | 20.1 | 20.1 | 2.0 | 15.6 | 2.7 | 2.03 | 1.57 | 0.948 |
| `cohere-ip-1000000x768-byte` | upstream | 5 | 24.2 | 24.1 | 1.8 | 17.0 | 5.1 | 3.25 | 1.47 | 0.959 |
| `cohere-l2-1000000x768-byte` | knn-builder | 5 | 19.8 | 19.7 | 2.0 | 15.6 | 2.6 | 2.02 | 1.57 | 0.957 |
| `cohere-l2-1000000x768-byte` | upstream | 5 | 23.7 | 23.6 | 2.0 | 17.6 | 4.0 | 3.28 | 1.58 | 0.964 |
| `ms-marco-ip-1000000x384-byte` | knn-builder | 5 | 11.0 | 10.9 | 1.0 | 8.6 | 1.3 | 1.57 | 0.78 | 0.980 |
| `ms-marco-ip-1000000x384-byte` | upstream | 5 | 14.2 | 14.1 | 1.1 | 10.1 | 3.1 | 2.36 | 1.09 | 0.973 |
| `ms-marco-l2-1000000x384-byte` | knn-builder | 5 | 11.2 | 11.0 | 1.0 | 8.7 | 1.3 | 1.57 | 0.78 | 0.981 |
| `ms-marco-l2-1000000x384-byte` | upstream | 5 | 14.0 | 13.9 | 1.0 | 10.2 | 2.6 | 2.39 | 1.09 | 0.982 |
| `sift-ip-1000000x128-byte` | knn-builder | 5 | 5.1 | 5.0 | 0.4 | 3.9 | 0.7 | 1.31 | 0.53 | 0.679 |
| `sift-ip-1000000x128-byte` | upstream | 5 | 6.0 | 5.9 | 0.4 | 4.6 | 0.8 | 1.94 | 0.70 | 0.609 |
| `sift-l2-1000000x128-byte` | knn-builder | 5 | 5.1 | 5.0 | 0.4 | 3.9 | 0.8 | 1.31 | 0.53 | 0.993 |
| `sift-l2-1000000x128-byte` | upstream | 5 | 5.8 | 5.7 | 0.4 | 4.5 | 0.8 | 1.95 | 0.70 | 0.996 |

| Service | Concurrent jobs | Built | Latency p50 s | p99 s | Burst | CPU wait s | Sustained | Jobs per hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn-builder | 8 x 5 | 40 of 40 | 58.6 | 97.5 | 48: 11 built, 36 refused, 1 failed, 0 timed out | 15.4 | 4 in flight for 3600 s | 399.0 |
| upstream | 8 x 5 | 40 of 40 | 70.8 | 159.2 | 48: 4 built, 0 refused, 0 failed, 44 timed out | 4085.6 | 4 in flight for 3600 s | 270.0 |

### float

| Input | Service | Builds | Latency s | Steps s | Download s | Build s | Upload s | Host peak GB | Device peak GB | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cohere-ip-1000000x768` | knn-builder | 5 | 56.1 | 32.9 | 7.8 | 14.3 | 10.6 | 4.33 | 3.87 | 0.958 |
| `cohere-ip-1000000x768` | upstream | 5 | 75.9 | 71.5 | 7.4 | 17.9 | 47.2 | 7.86 | 3.88 | 0.951 |
| `cohere-l2-1000000x768` | knn-builder | 5 | 33.1 | 31.1 | 8.0 | 15.0 | 7.7 | 4.33 | 3.87 | 0.969 |
| `cohere-l2-1000000x768` | upstream | 5 | 110.6 | 88.7 | 7.3 | 19.9 | 62.9 | 7.89 | 3.88 | 0.966 |
| `gist-ip-1000000x960` | knn-builder | 5 | 50.4 | 50.2 | 10.8 | 17.9 | 20.5 | 5.08 | 4.79 | 0.985 |
| `gist-ip-1000000x960` | upstream | 5 | 94.4 | 84.5 | 9.2 | 21.6 | 54.4 | 9.37 | 4.80 | 0.955 |
| `gist-l2-1000000x960` | knn-builder | 5 | 58.8 | 58.8 | 10.3 | 20.7 | 28.1 | 5.02 | 4.79 | 0.953 |
| `gist-l2-1000000x960` | upstream | 5 | 110.2 | 99.9 | 9.5 | 24.6 | 65.4 | 9.22 | 4.80 | 0.943 |
| `ms-marco-ip-1000000x384` | knn-builder | 5 | 17.6 | 17.6 | 4.1 | 8.6 | 5.1 | 2.72 | 1.93 | 0.984 |
| `ms-marco-ip-1000000x384` | upstream | 5 | 38.9 | 31.3 | 3.9 | 9.7 | 17.6 | 4.64 | 2.24 | 0.983 |
| `ms-marco-l2-1000000x384` | knn-builder | 5 | 16.4 | 16.3 | 4.0 | 8.7 | 3.4 | 2.72 | 1.93 | 0.975 |
| `ms-marco-l2-1000000x384` | upstream | 5 | 43.7 | 43.1 | 3.6 | 10.4 | 29.5 | 4.68 | 2.24 | 0.966 |
| `sift-ip-1000000x128` | knn-builder | 5 | 7.5 | 7.3 | 1.3 | 3.8 | 2.1 | 1.71 | 0.91 | 0.992 |
| `sift-ip-1000000x128` | upstream | 5 | 9.0 | 8.8 | 1.3 | 4.9 | 2.4 | 2.69 | 1.08 | 0.990 |
| `sift-l2-1000000x128` | knn-builder | 5 | 7.2 | 7.0 | 1.3 | 3.9 | 1.7 | 1.69 | 0.91 | 0.996 |
| `sift-l2-1000000x128` | upstream | 5 | 9.3 | 9.1 | 1.3 | 4.9 | 2.9 | 2.71 | 1.08 | 0.995 |

| Service | Concurrent jobs | Built | Latency p50 s | p99 s | Burst | CPU wait s | Sustained | Jobs per hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn-builder | 8 x 5 | 40 of 40 | 162.1 | 314.6 | 48: 2 built, 42 refused, 3 failed, 1 timed out | 228.4 | 4 in flight for 3600 s | 104.0 |
| upstream | 8 x 5 | 40 of 40 | 318.6 | 714.4 | 48: 0 built, 0 refused, 0 failed, 48 timed out | 4438.6 | 4 in flight for 3600 s | 45.0 |

Reading across the tables: knn-builder's build step is 5 to 25% shorter than the upstream service's (the least on binary, the most on float), its upload of a float index of 1.5 GB or more, streamed as the file is written, an eighth to two fifths of the upstream's, and its host memory a third to almost half less on float inputs (4.3 GB against 7.9 for cohere-768, 5.0 against 9.2 and 9.4 for gist-960), a third less on byte, and the same on binary, where it is 0.1 GB higher; the device peaks are the same on float, and 0.1 to 0.3 GB lower on byte and binary but for cohere-768 byte in inner product, 0.1 GB higher. Under load it completes more jobs an hour (104 against 45 on float, 399 against 270 on byte, 228 against 201 on binary) with a lower p99 (the binary p50 is 4 s higher), and in a burst it refuses at once what it cannot build in time, so the 48 data nodes wait at most a few minutes in all rather than the whole timeout each. The recalls match within 0.02, except gist-960 in inner product (0.985 against 0.955) and SIFT's byte input in inner product, where both are low (0.68 and 0.61): scaling SIFT's counts onto -128..127 shifts every component, and the inner product then ranks mostly by the shift.

The runs' records are in `fixtures/bench/` (`run.json` and `recall.json` per service, `datasets.json`, and `environment.json` with the versions), and `tools/bench/` reproduces them: `datasets.py` fetches the datasets and writes the inputs, `run.py` runs one service through the phases, and `report.py` prints the tables above from the run directories. From the repository root, with Docker, `nvidia-smi` and Python on the host, as `tools/README.md` describes in full:

```sh
python tools/bench/datasets.py ~/knn-bench/hdf5 target/bench/inputs
docker build -t knn-builder:dev .
python tools/bench/run.py target/bench/knn-builder knn-builder
python tools/bench/run.py target/bench/upstream upstream
python tools/bench/report.py ../knn-inspect/target/release/knn-inspect target/bench/knn-builder target/bench/upstream
```

## License

Apache-2.0. See [LICENSE](LICENSE).
