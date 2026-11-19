# knn-builder

A remote vector index build service for [OpenSearch k-NN](https://github.com/opensearch-project/k-NN), written in Rust. It is meant as a drop-in replacement for [remote-vector-index-builder](https://github.com/opensearch-project/remote-vector-index-builder): the same HTTP API, the same index files, and the same cuVS CAGRA build on the GPU, with less host memory per build and a GPU that stays busy while vectors download and indexes upload.

**Status:** released. `knn-gpu` builds the CAGRA graph on the GPU with the upstream service's parameters, and `knn-write` writes the index file the service writes, identical to it on every test file; `knn-store` downloads the vectors from S3 and streams the index file back up, with no temporary file; and the library half of `knn-builder` runs the three as a pipeline, the next job downloading while one builds and the last one uploading, within a budget of host memory. The `knn-builder` binary serves the HTTP API on port 1025, answering as the upstream service does, and runs from a container image as the upstream one does; an OpenSearch 3.9.0 node with k-NN's remote index build turned on builds its float, half_float, byte and binary segments through it. `project_link_to_base_layer: true`, an option of the upstream service's next, unreleased version that k-NN does not send yet, is taken as that version takes it: a float, half_float or byte graph whose clusters a search could not leave is linked through an HNSW layer over a sample of its vectors before it is written (`docs/params.md`), which on clustered inputs restores the recall as the upstream service's does (the projection, under Benchmarks).

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

Each release's image is on Docker Hub as `userepo/knn-builder:<version>`, the newest also as `latest`. It runs as the upstream user guide starts the upstream image, on a host with the NVIDIA container toolkit and a driver for CUDA 12.9:

```sh
docker run --gpus all -p 80:1025 userepo/knn-builder:1.1.0
```

The `Dockerfile` builds the service with the `gpu` feature against the conda environment of `environment.yml`, then copies the binary and the CUDA and cuVS libraries it loads onto `nvidia/cuda:12.9.2-base-ubuntu24.04`, the latest patch release of the base the upstream image uses (12.9.0); it runs as its own user, on port 1025. The release workflow builds it from the tag; to build it yourself, with BuildKit, Docker's default builder:

```sh
docker build -t knn-builder .
docker run --gpus all -p 80:1025 knn-builder
```

### Configuration

The service reads the upstream service's environment variables, with the same defaults: `REQUEST_STORE_MAX_SIZE` (10000), `REQUEST_STORE_TTL_SECONDS` (1800, counted from when a job ends, so a running job never expires), `GPU_MEMORY_LIMIT` and `CPU_MEMORY_LIMIT` (24 and 32, in GB; the CPU limit also bounds the host memory of the jobs in flight), `LOG_LEVEL` (`INFO`; `DEBUG` adds each job's stage times and what its projection did), `S3_ENDPOINT_URL` and `AWS_DEFAULT_REGION`, with credentials from the AWS SDK's default chain; a bucket in another region than `AWS_DEFAULT_REGION` is reached through S3's redirect to it, as the upstream service reaches it. `REQUEST_STORE_TYPE` takes only `memory`; `MAX_WORKERS` and `INDEX_SERIALIZATION_MODE` are read and ignored. `BASIC_AUTH_USERNAME` and `BASIC_AUTH_PASSWORD`, set together, turn on Basic Auth: every request but `GET /_heart_beat`, which stays open for health checks, must then carry them.

Three more describe the data nodes that call the service. `CLIENT_TIMEOUT_SECONDS` (3600) and `CLIENT_POLL_INTERVAL_SECONDS` (5) are k-NN's `knn.remote_index_build.client.timeout` and `knn.remote_index_build.poll.interval`, in seconds, and should match the cluster's. `CPU_BUILD_THREADS`, unset by default, is the data node's `knn.algo_param.index_thread_qty`, 1 or 4. The service forecasts when each new job would end, from the jobs ahead of it and what earlier jobs took, and answers 507 at once, rather than take the job, when the GPU's free memory cannot hold its build, when the data node would see it ended only past its timeout, or, with `CPU_BUILD_THREADS` set, when the data node would build the segment sooner itself. k-NN does not retry a 507: the data node builds that segment on its CPU right away, instead of after the timeout. A job whose build no longer fits the GPU's free memory when its turn comes fails at once, for the same reason. And a job the data node has stopped waiting for, its timeout counted from the request, is dropped: one still queued never starts, a transfer stops, and a build under way, which cannot be stopped, stores nothing, so the jobs still waited for move up. Whenever a job ends, the jobs not yet building are forecast again with what it taught, and those that would now miss are dropped at once, so their data nodes build the segments themselves at their next poll.

### S3 tests

The tests of `knn-store` and `knn-builder` that need S3 are ignored by default. CI runs them against LocalStack 4.14, pinned by digest as in `tools/README.md`; locally, start the same image and point the service's own environment variables at it:

```sh
docker run -d --rm --name knn-store-localstack -p 4566:4566 localstack/localstack@sha256:3ebc37595918b8accb852f8048fef2aff047d465167edd655528065b07bc364a
S3_ENDPOINT_URL=http://localhost:4566 AWS_DEFAULT_REGION=us-east-1 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test cargo test -p knn-store -p knn-builder -- --ignored
```

Each test creates a bucket of its own and removes it when it passes.

## Benchmarks

Both services, measured the same way on the same machine over the datasets the upstream `benchmarking/` tool names, 1M vectors each: sift-128, ms-marco-384, cohere-768 and gist-960, as float inputs in L2 and inner product, the first three also as byte inputs (each component scaled over the dataset's range onto -128..127) and as binary inputs in Hamming space (a bit per positive component). The services ran from their container images under Docker Desktop on WSL2 with LocalStack for S3, `GPU_MEMORY_LIMIT=8`, on an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB), an Intel Core i9-14900HX and 32 GB of memory, 15 GB of it the Docker VM's: knn-builder with libcuvs 26.08.01 and CUDA 12.9, the upstream service as its `api-snapshot` image (libcuvs 26.08.01, Faiss 1.14.3, the same CUDA) on `MAX_WORKERS=1`, since two builds at once crashed it under WSL2; knn-builder builds one job on the GPU while the next downloads and the last uploads. Every number is the median of five builds, or counted over five rounds, except the bursts and the hour-long queues, run once.

The tables hold two rounds of runs. The single builds of the float and byte inputs are the later one: knn-builder at commit 3a4eb45 and the upstream image at digest `e8fecaef…` (commit ea7db9f), on driver 621.24. The binary tables and every row of concurrent, burst and sustained jobs are the earlier one: knn-builder at commit d4af6b8 and the upstream image at digest `78ced9d0…` (commit 3a02e64), on driver 617.42. Between the two, both services came to build float, half_float and byte graphs from the vectors in host memory instead of a copy of them on the GPU, the upstream service at ea7db9f, which knn-builder follows. A float or byte build's device peak fell by up to the size of its vectors (cohere-768 as float, 3.07 GB of them, peaked at 3.70 and 3.74 GB in the earlier round and peaks at 0.82 and 0.87 GB), so those single builds were run again. Binary graphs build as they did, and the jobs under load were not run again: their rows are of the earlier build.

The metrics, per data type in the tables below:

- **Latency** is a build of the input alone, from the request to the status that shows it completed, asked every 0.2 s. **Steps** is the same job on the service's own clock, the sum of its steps, which its `DEBUG` log gives: **Download**, **Build** (for the upstream service, its build, conversion and write together) and **Upload**. The two differ where the client saw the completed status late: 18 to 31 s after the service logged the job's end on 5 of knn-builder's 85 builds, with no status request reaching the service in between, and 1 to 26 s after the last step on 16 of the upstream's, all of float files of 1.8 GB or more; on every other build the two are within 0.3 s (knn-builder) and 0.6 s (upstream). Those waits, between the client and the container on this host (the requests pass through Docker Desktop's port forwarding), are in the latencies and not in the steps.
- **Host peak** is the service process's peak resident memory over the job (`VmHWM`, reset before it), **Device peak** the GPU's memory at its peak during the job, sampled every 50 ms, less the level the samples last stayed at before it, since `nvidia-smi` under WSL2 reads no memory used once the GPU has idled for about ten seconds. The earlier round's runs, first measured against the least seen, are recomputed from their samples: 0.14 to 0.18 GB lower where the GPU had idled.
- **Recall@10** is the first build's index searched for the dataset's first 100 test queries with [knn-inspect](https://github.com/userepo/knn-inspect) 0.17.0, against the exact neighbors; the GPU build is not deterministic, so another build's recall differs in its last digits.
- **Concurrent**: five rounds of eight requests sent at once, as k-NN's data nodes send them (one retry after a second of a 408, 409, 429, 500, 502, 503 or 509, a first status request after three poll intervals of 5 s and one an interval after), over the data type's inputs in turn; the latency each data node saw, its p50 and p99 over the 40 jobs.
- **Burst**: 48 requests at once over the data type's L2 inputs (Hamming for binary), the service restarted and warmed up before each, with the data nodes' timeout set to 90 s and the services told so (`CLIENT_TIMEOUT_SECONDS`): how many were built in time, refused at once (a 507, after which the data node builds the segment itself), failed (knn-builder drops a job it forecasts to miss the timeout as the jobs before it end, so the data node starts its own build at the next poll instead of after the timeout) or timed out; **CPU wait** is the seconds the 48 data nodes waited in all before building on their own CPUs.
- **Sustained**: four jobs kept in flight for an hour over the data type's inputs in turn, the next sent as one completes; **Jobs per hour** counts the jobs seen completed within the hour.

### binary

| Input | Service | Builds | Latency s | Steps s | Download s | Build s | Upload s | Host peak GB | Device peak GB | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cohere-hamming-1000000x768-binary` | knn-builder | 5 | 23.8 | 23.5 | 0.2 | 22.7 | 0.7 | 3.06 | 0.49 | 0.963 |
| `cohere-hamming-1000000x768-binary` | upstream | 5 | 25.1 | 24.9 | 0.2 | 23.8 | 0.8 | 2.98 | 0.62 | 0.966 |
| `ms-marco-hamming-1000000x384-binary` | knn-builder | 5 | 16.8 | 16.7 | 0.1 | 16.0 | 0.6 | 2.94 | 0.44 | 0.989 |
| `ms-marco-hamming-1000000x384-binary` | upstream | 5 | 18.2 | 18.1 | 0.2 | 17.4 | 0.6 | 2.78 | 0.57 | 0.988 |
| `sift-hamming-1000000x128-binary` | knn-builder | 5 | 11.4 | 11.3 | 0.1 | 10.7 | 0.5 | 2.89 | 0.41 | 1.000 |
| `sift-hamming-1000000x128-binary` | upstream | 5 | 12.9 | 12.7 | 0.3 | 12.0 | 0.4 | 2.83 | 0.54 | 1.000 |

| Service | Concurrent jobs | Built | Latency p50 s | p99 s | Burst | CPU wait s | Sustained | Jobs per hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn-builder | 8 x 5 | 40 of 40 | 76.4 | 135.1 | 48: 4 built, 44 refused, 0 failed, 0 timed out | 1.4 | 4 in flight for 3600 s | 228.0 |
| upstream | 8 x 5 | 40 of 40 | 72.7 | 156.6 | 48: 3 built, 0 refused, 0 failed, 45 timed out | 4181.9 | 4 in flight for 3600 s | 201.0 |

### byte

| Input | Service | Builds | Latency s | Steps s | Download s | Build s | Upload s | Host peak GB | Device peak GB | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cohere-ip-1000000x768-byte` | knn-builder | 5 | 20.1 | 20.0 | 2.0 | 15.0 | 2.0 | 1.58 | 0.67 | 0.950 |
| `cohere-ip-1000000x768-byte` | upstream | 5 | 23.6 | 23.4 | 1.9 | 17.6 | 3.8 | 3.23 | 0.72 | 0.959 |
| `cohere-l2-1000000x768-byte` | knn-builder | 5 | 19.7 | 19.6 | 2.0 | 15.2 | 2.2 | 1.58 | 0.67 | 0.961 |
| `cohere-l2-1000000x768-byte` | upstream | 5 | 23.3 | 23.2 | 1.9 | 17.8 | 3.2 | 3.14 | 0.72 | 0.962 |
| `ms-marco-ip-1000000x384-byte` | knn-builder | 5 | 11.1 | 11.0 | 1.0 | 8.7 | 1.3 | 1.19 | 0.40 | 0.981 |
| `ms-marco-ip-1000000x384-byte` | upstream | 5 | 14.6 | 14.4 | 1.1 | 10.3 | 3.0 | 2.33 | 0.57 | 0.983 |
| `ms-marco-l2-1000000x384-byte` | knn-builder | 5 | 11.3 | 11.1 | 1.1 | 8.7 | 1.3 | 1.19 | 0.40 | 0.985 |
| `ms-marco-l2-1000000x384-byte` | upstream | 5 | 13.8 | 13.7 | 1.0 | 10.3 | 2.2 | 2.35 | 0.57 | 0.980 |
| `sift-ip-1000000x128-byte` | knn-builder | 5 | 5.3 | 5.1 | 0.3 | 4.0 | 0.8 | 0.93 | 0.40 | 0.672 |
| `sift-ip-1000000x128-byte` | upstream | 5 | 6.0 | 5.9 | 0.4 | 4.7 | 0.8 | 1.86 | 0.57 | 0.721 |
| `sift-l2-1000000x128-byte` | knn-builder | 5 | 5.3 | 5.3 | 0.3 | 4.1 | 0.8 | 0.93 | 0.40 | 0.998 |
| `sift-l2-1000000x128-byte` | upstream | 5 | 5.9 | 5.7 | 0.4 | 4.5 | 0.8 | 1.87 | 0.57 | 0.994 |

| Service | Concurrent jobs | Built | Latency p50 s | p99 s | Burst | CPU wait s | Sustained | Jobs per hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn-builder | 8 x 5 | 40 of 40 | 58.6 | 97.5 | 48: 11 built, 36 refused, 1 failed, 0 timed out | 15.4 | 4 in flight for 3600 s | 399.0 |
| upstream | 8 x 5 | 40 of 40 | 70.8 | 159.2 | 48: 4 built, 0 refused, 0 failed, 44 timed out | 4085.6 | 4 in flight for 3600 s | 270.0 |

### float

| Input | Service | Builds | Latency s | Steps s | Download s | Build s | Upload s | Host peak GB | Device peak GB | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cohere-ip-1000000x768` | knn-builder | 5 | 33.6 | 30.4 | 8.4 | 15.2 | 7.2 | 3.88 | 0.82 | 0.955 |
| `cohere-ip-1000000x768` | upstream | 5 | 74.0 | 63.2 | 7.5 | 18.1 | 37.7 | 7.83 | 0.87 | 0.945 |
| `cohere-l2-1000000x768` | knn-builder | 5 | 32.7 | 32.5 | 8.2 | 15.6 | 7.6 | 3.88 | 0.82 | 0.964 |
| `cohere-l2-1000000x768` | upstream | 5 | 69.7 | 69.1 | 7.5 | 19.2 | 42.4 | 7.75 | 0.87 | 0.941 |
| `gist-ip-1000000x960` | knn-builder | 5 | 59.1 | 40.8 | 10.4 | 20.0 | 10.2 | 4.65 | 1.02 | 0.961 |
| `gist-ip-1000000x960` | upstream | 5 | 92.8 | 82.6 | 9.3 | 21.6 | 51.8 | 9.29 | 1.07 | 0.998 |
| `gist-l2-1000000x960` | knn-builder | 5 | 47.5 | 44.3 | 10.2 | 22.1 | 11.0 | 4.65 | 1.02 | 0.951 |
| `gist-l2-1000000x960` | upstream | 5 | 90.5 | 86.5 | 9.0 | 22.6 | 55.0 | 9.20 | 1.07 | 0.948 |
| `ms-marco-ip-1000000x384` | knn-builder | 5 | 18.9 | 18.8 | 4.0 | 10.5 | 4.1 | 2.34 | 0.42 | 0.980 |
| `ms-marco-ip-1000000x384` | upstream | 5 | 39.8 | 32.4 | 3.8 | 11.8 | 16.9 | 4.62 | 0.57 | 0.980 |
| `ms-marco-l2-1000000x384` | knn-builder | 5 | 18.0 | 17.7 | 4.0 | 9.5 | 4.3 | 2.34 | 0.42 | 0.964 |
| `ms-marco-l2-1000000x384` | upstream | 5 | 33.8 | 31.2 | 3.8 | 10.7 | 16.9 | 4.64 | 0.57 | 0.978 |
| `sift-ip-1000000x128` | knn-builder | 5 | 7.4 | 7.3 | 1.4 | 4.2 | 1.8 | 1.32 | 0.40 | 0.991 |
| `sift-ip-1000000x128` | upstream | 5 | 8.8 | 8.6 | 1.3 | 5.0 | 2.4 | 2.61 | 0.57 | 0.990 |
| `sift-l2-1000000x128` | knn-builder | 5 | 7.4 | 7.2 | 1.3 | 4.1 | 1.8 | 1.32 | 0.40 | 0.997 |
| `sift-l2-1000000x128` | upstream | 5 | 8.5 | 8.3 | 1.3 | 5.0 | 2.0 | 2.63 | 0.57 | 0.998 |

| Service | Concurrent jobs | Built | Latency p50 s | p99 s | Burst | CPU wait s | Sustained | Jobs per hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn-builder | 8 x 5 | 40 of 40 | 162.1 | 314.6 | 48: 2 built, 42 refused, 3 failed, 1 timed out | 228.4 | 4 in flight for 3600 s | 104.0 |
| upstream | 8 x 5 | 40 of 40 | 318.6 | 714.4 | 48: 0 built, 0 refused, 0 failed, 48 timed out | 4438.6 | 4 in flight for 3600 s | 45.0 |

Reading across the tables: knn-builder's build step is 2 to 18% shorter than the upstream service's (the least on gist-960 in L2, the most on cohere-768 in L2), its upload of a float index of 1.5 GB or more, streamed as the file is written, 18 to 25% of the upstream's at the median (on this host one file's five uploads differ by 1.6 to 3.7 times between the fastest and the slowest, for both services), and its host memory half the upstream's on float and byte inputs (3.88 GB against 7.75 and 7.83 for cohere-768, 4.65 against 9.20 and 9.29 for gist-960), and the same on binary, where it is 0.1 GB higher; its device peaks are 0.05 GB lower on cohere-768 and gist-960 as float and byte inputs, and 0.13 to 0.17 GB lower on the rest. Under load, as the earlier round measured it, it completes more jobs an hour (104 against 45 on float, 399 against 270 on byte, 228 against 201 on binary) with a lower p99 (the binary p50 is 4 s higher), and in a burst it refuses at once what it cannot build in time, so the 48 data nodes wait at most a few minutes in all rather than the whole timeout each. The recalls, each of one build over 100 queries, match within 0.025, except gist-960 in inner product (0.961 against 0.998) and SIFT's byte input in inner product, where both are low (0.67 and 0.72): scaling SIFT's counts onto -128..127 shifts every component, and the inner product then ranks mostly by the shift. Gist's gap is that build's and not the service's. Under inner product its queries share their neighbors, 54 vectors filling all 1,000 places, the most common of them three times the median vector's length, so a build that leaves one of them hard to reach at `ef_search` 100 loses it for dozens of queries at once: 37 of the 100 miss one neighbor each in that file. Five more builds by each service recall 0.973 to 0.999 (knn-builder) and 0.960 to 0.997 (upstream), the upstream's lowest two missing the same way, and every one 0.989 or more at `ef_search` 400 (`fixtures/ea7db9f/bench-single/gist-ip.json`).

The runs' records are in `fixtures/bench/` (the earlier round: `run.json` and `recall.json` per service, `datasets.json`, and `environment.json` with the versions) and `fixtures/ea7db9f/bench-single/` (the later one, the same files but for the datasets), and `tools/bench/` reproduces them: `datasets.py` fetches the datasets and writes the inputs, `run.py` runs one service through the phases, or through those named over the inputs named (`--phases single` and the float and byte inputs' names for the later round), and `report.py` prints a round's tables from its run directories. From the repository root, with Docker, `nvidia-smi` and Python on the host, as `tools/README.md` describes in full:

```sh
python tools/bench/datasets.py ~/knn-bench/hdf5 target/bench/inputs
docker build -t knn-builder:dev .
python tools/bench/run.py target/bench/knn-builder knn-builder
python tools/bench/run.py target/bench/upstream upstream
python tools/bench/report.py ../knn-inspect/target/release/knn-inspect target/bench/knn-builder target/bench/upstream
```

### The projection

`project_link_to_base_layer` is for vectors in clusters so far apart that the CAGRA build links none of them to another, so a search never leaves the clusters its entry points fall in ([issue 158](https://github.com/opensearch-project/remote-vector-index-builder/issues/158) of the upstream service). Both services built six such inputs without it and with it, five builds each: Gaussian mixtures of 300,000 unit vectors of dimension 256 under inner product, in 200 clusters at sigma 0.3 (as a float, a half_float and a byte field) and at sigma 1.0, in 1,000 clusters at sigma 0.3, and, as a control whose graph builds connected, in 200 overlapping clusters at sigma 2.0. They ran as the builds above on the same machine (driver 621.24), knn-builder at commit 3a4eb45, which keeps layer 1's threads from one batch to the next, and the upstream service as its `api-snapshot` image at digest `e8fecaef…` (commit ea7db9f). A cell holds the mean of the five builds without the projection, then with it:

- **Components** counts the strongly connected components of the file's level 0: 1 where a search can get from any node to any other.
- **Recall@10** and **Recall@100** are knn-inspect's over 1,000 held-out queries, each the mean over five seeds of its random entry points: at 10 with the file's own settings (`ef_search` 100, 32 entry points), as k-NN searches it by default, and at 100 with `ef_search` 600 from 32 entry points.
- **Projection s** is the projection's time in the service's `DEBUG` log, in the builds that asked for it; **Latency** and **Host peak** are as above.

| Input | Service | Components | Recall@10 | Recall@100 | Projection s | Latency s | Host peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ip-300000x256-c200-s0.3` | knn-builder | 200 to 1 | 0.147 to 0.950 | 0.150 to 0.999 | 0.27 | 4.54 to 4.92 | 1.36 to 1.32 |
| `ip-300000x256-c200-s0.3` | upstream | 200 to 1 | 0.147 to 0.949 | 0.150 to 0.999 | 0.21 | 5.20 to 5.43 | 2.07 to 2.12 |
| `ip-300000x256-c200-s0.3-fp16` | knn-builder | 200 to 1 | 0.147 to 0.951 | 0.150 to 0.999 | 0.27 | 3.87 to 4.13 | 1.21 to 1.17 |
| `ip-300000x256-c200-s0.3-fp16` | upstream | 200 to 1 | 0.147 to 0.949 | 0.150 to 0.999 | 0.21 | 4.07 to 4.37 | 1.78 to 1.84 |
| `ip-300000x256-c200-s0.3-byte` | knn-builder | 200 to 1 | 0.147 to 0.950 | 0.150 to 0.999 | 0.26 | 3.50 to 3.74 | 1.13 to 1.06 |
| `ip-300000x256-c200-s0.3-byte` | upstream | 200 to 1 | 0.146 to 0.951 | 0.150 to 0.999 | 0.22 | 3.66 to 4.05 | 1.63 to 1.68 |
| `ip-300000x256-c200-s1.0` | knn-builder | 200 to 1 | 0.144 to 0.938 | 0.144 to 1.000 | 0.34 | 4.47 to 4.75 | 1.36 to 1.33 |
| `ip-300000x256-c200-s1.0` | upstream | 200 to 1 | 0.144 to 0.934 | 0.144 to 1.000 | 0.25 | 5.05 to 5.38 | 2.09 to 2.14 |
| `ip-300000x256-c1000-s0.3` | knn-builder | 998 to 1 | 0.033 to 0.936 | 0.033 to 1.000 | 0.34 | 4.61 to 4.97 | 1.37 to 1.32 |
| `ip-300000x256-c1000-s0.3` | upstream | 998 to 1 | 0.033 to 0.935 | 0.033 to 1.000 | 0.26 | 5.12 to 5.46 | 2.07 to 2.11 |
| `ip-300000x256-c200-s2.0` | knn-builder | 1 to 1 | 0.669 to 0.670 | 0.970 to 0.970 | 0.05 | 4.61 to 4.66 | 1.36 to 1.33 |
| `ip-300000x256-c200-s2.0` | upstream | 1 to 1 | 0.677 to 0.671 | 0.969 to 0.969 | 0.26 | 5.03 to 5.33 | 2.09 to 2.13 |

Without the projection both services' graphs fall apart into a component a cluster, and a search finds only the neighbors in the clusters it entered: a recall of 0.14 to 0.15 for 200 clusters and 0.03 for 1,000, at 10 and at 100 alike. With it every such graph is one component on both services, recalling 0.9985 or more at 100 and 0.93 to 0.95 at 10. Both leave the control as built, its graph being connected already: the option goes by what a search can reach, not by what it finds, and the control's recall stays where it was, 0.97 at 100 and 0.67 at 10, below the projected graphs'.

knn-builder's projection is the upstream service's but for the order layer 1's nodes go in (`docs/params.md`). The two promote the same 18,831 nodes, and knn-builder adds 0.2 to 0.7% more links. Its mean recall@10 is within the range of the upstream service's five builds on three of the five split inputs and above it on two, the half_float copy (0.951 against 0.949 at the mean) and 200 clusters at sigma 1.0 (0.938 against 0.934, its builds recalling 0.936 to 0.940 and the upstream's 0.931 to 0.937); at 100 the two are within 0.0003 on every split input, and 0.0008 apart on the control, which neither changes. Its projection takes 0.26 to 0.34 s against 0.21 to 0.26 s, layer 1 being 0.21 to 0.27 s of it against 0.13 to 0.18 s; on the control, where each only measures what a search reaches, it takes 0.05 s against 0.26 s. With the projection knn-builder's builds still end sooner (3.74 to 4.97 s against 4.05 to 5.46 s) and in less host memory (1.06 to 1.33 GB against 1.68 to 2.14 GB). Its host peaks with the projection and without are within 0.07 GB of each other, as far as the same builds' peaks lie apart from one run to the next, so the lower figures with it in the table are no saving; neither service's device peak changes (0.21 to 0.22 GB and 0.26 to 0.27 GB).

The records are in `fixtures/ea7db9f/projection/`, with every build's figures and their ranges in its `summary.md`. `tools/README.md` gives the commands: `make_inputs.py clustered` writes the inputs, `run.py --project` asks for the projection, and `projection.py` scores the files.

## License

Apache-2.0. See [LICENSE](LICENSE).
