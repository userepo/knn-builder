# Tools

Scripts that produce and check the fixtures. The Python ones run in the upstream images, pinned by digest, so their numpy, boto3 and Faiss are the service's own; nothing needs installing on the host but Docker (with GPU access) and Python for `baseline/score.py`.

| Image | Digest | Used for |
| --- | --- | --- |
| `opensearchstaging/remote-vector-index-builder:api-snapshot` | `sha256:78ced9d04c4cbc1106cfceb98e32a772c71565b215c6e68be8fcdcbcdaab64da` | the service at commit 3a02e64 (unreleased 1.3.0), `make_inputs.py`, the baseline runner |
| `opensearchstaging/remote-vector-index-builder:faiss-base-snapshot` | `sha256:b3420e65b89c0d25a58ccc31dd1edfe1019383605ff356df563416b2d35bca9c` | the oracle: Faiss 1.14.3 at `0ca9df4` with the service's patches |
| `localstack/localstack:4.14` | `sha256:3ebc37595918b8accb852f8048fef2aff047d465167edd655528065b07bc364a` | S3 for the baseline, the version upstream's e2e test uses |

Never use the `-latest` tags: they are release 1.2.0, which builds different graphs.

```sh
API=opensearchstaging/remote-vector-index-builder@sha256:78ced9d04c4cbc1106cfceb98e32a772c71565b215c6e68be8fcdcbcdaab64da
BASE=opensearchstaging/remote-vector-index-builder@sha256:b3420e65b89c0d25a58ccc31dd1edfe1019383605ff356df563416b2d35bca9c
RUN="docker run --rm -v $PWD:/work -w /work --entrypoint python"
```

From Git Bash on Windows, Docker needs a Windows path and no path rewriting: `export MSYS_NO_PATHCONV=1` and use `-v "$(pwd -W):/work"` in `RUN`.

## Inputs

`make_inputs.py` writes build inputs in the format k-NN uploads (float32 or float16 vectors, int32 doc ids) with 100 held-out queries each:

```sh
$RUN $API tools/make_inputs.py small fixtures/inputs       # committed
$RUN $API tools/make_inputs.py large target/inputs-large   # for the recall spread
```

## Baseline

`baseline/` runs the service against LocalStack and builds every input five times, one build at a time:

```sh
export INPUTS=fixtures/inputs OUT=target/baseline/small RUNS=5
docker compose -f tools/baseline/docker-compose.yml up --abort-on-container-exit --exit-code-from runner
docker compose -f tools/baseline/docker-compose.yml logs --no-color service > $OUT/service.log
docker compose -f tools/baseline/docker-compose.yml down -v
python tools/baseline/score.py $OUT $INPUTS ../knn-inspect/target/release/knn-inspect
```

`score.py` adds each build's per-step times from the service's DEBUG log and knn-inspect `verify` and `recall` (its seeded search, seed 1, k = 10, over the held-out queries) to `$OUT/summary.json`, with the knn-inspect version: its recall is that version's search. The request's `m`, `ef_construction` and `ef_search` come from each input's `input.json`.

## Oracle

`oracle/oracle.py` writes the file the service writes from the CAGRA graph, the input blobs and the request (`input.json`), replaying `GpuIndexCagra::copyTo` and `write_index`. `check` takes only the graph from each service-built file, checks its degree against the request, rebuilds the file and compares:

```sh
$RUN $BASE tools/oracle/oracle.py check fixtures/service fixtures/inputs
$RUN $BASE tools/oracle/oracle.py check target/baseline/small fixtures/inputs
```

`oracle/levels.py` computes the HNSW level table those files hold without Faiss, as `knn-write` must:

```sh
$RUN $BASE tools/oracle/levels.py "fixtures/service/l2-1000x64*.faiss"
```

`oracle/level_tables.py` writes `knn-write`'s level-table test data from it: the probabilities and neighbor counts for every even M from 2 to 4,096 and level histograms over 2,100,000 nodes. Run it in the image, for glibc's `exp`:

```sh
$RUN $BASE tools/oracle/level_tables.py crates/knn-write/tests/data/level_tables.txt
```

`oracle/build_tables.py` writes `knn-gpu`'s test data from the service's own code: the IVF-PQ list count and lowered m for every vector count up to 3,000, a stride of 97 to 60,000 (where m = 128 stops being lowered) and a spread to 2^31 - 1, the PQ dimension, and numpy's fp32 to fp16 conversion of 7,000 values. Run it in the api image, where the service's package is importable:

```sh
$RUN $API tools/oracle/build_tables.py crates/knn-gpu/tests/data
```

`oracle/requests.py` records how the service answers `POST /_build` bodies, for `knn-builder`'s request parser: for a body its model accepts, the parameters with defaults filled in and the job id; for any other, the status (422, or 400 for a body that is not UTF-8) and the body its own app returns. Accepted bodies are never posted, so no job starts. A body that is not UTF-8 is recorded in hex (`body_hex`). Run it in the api image:

```sh
$RUN $API tools/oracle/requests.py crates/knn-builder/tests/data/requests.json
```

`oracle/capacity.py` records how the service sizes a request, for `knn-builder`'s job store: the GPU and CPU memory estimates of its `calculate_memory_requirements` for some 400 requests, many on either side of a limit, whether each passes the default limits and a smaller GPU's, a few requests under limits equal to their own estimates (the service's comparison at its edge), the messages of its two 507s from its own `JobService`, and its defaults for the store and the limits. Run it in the api image:

```sh
$RUN $API tools/oracle/capacity.py crates/knn-builder/tests/data/capacity.json
```

`oracle/routes.py` records how the service answers HTTP requests, for `knn-builder`'s routes: the status, the headers that matter and the body, for each route and method, the content types a build request may come with, paths no route takes, the status of a job running, completed and failed, and the debug list of jobs (`/_jobs`). Its jobs never start, so it needs no S3. Run it in the api image:

```sh
$RUN $API tools/oracle/routes.py crates/knn-builder/tests/data/routes.json
```

## Parity

`knn-gpu`'s `build_inputs` example builds inputs as the service does, with `knn-gpu` and `knn-write`, and writes a `runs.json` that `baseline/score.py` scores like the service's (it carries its own timings and has no log). In WSL, in the conda environment:

```sh
cargo run --release -p knn-gpu --features gpu --example build_inputs -- target/inputs-large target/rust-baseline/large20 20
```

Then, from Windows or Linux, `python tools/baseline/score.py target/rust-baseline/large20 target/inputs-large ../knn-inspect/target/release/knn-inspect`, and the service's own 20 builds with `RUNS=20` as above.

## End to end

`e2e/docker-compose.yml` runs the upstream service's end-to-end test against knn-builder's image: its client, unchanged, from upstream's own e2e image at 3a02e64, in knn-builder's container's network, where the address the client calls (`localhost:1025`) is knn-builder's, with LocalStack for S3. It builds an index of each of upstream's six test datasets (1M x 128, 384 and 768 float, two 100k half_float and a small one), checks each job through `/_status` and `/_jobs`, and exits 0 when all pass. Build the two images first, the upstream one from a checkout of upstream at 3a02e64 under `target/`, which git and the image's build leave out, on the `faiss-base` image pinned above:

```sh
docker build -t knn-builder:dev .
git clone https://github.com/opensearch-project/remote-vector-index-builder target/upstream && git -C target/upstream checkout 3a02e64
docker tag opensearchstaging/remote-vector-index-builder@sha256:b3420e65b89c0d25a58ccc31dd1edfe1019383605ff356df563416b2d35bca9c opensearchstaging/remote-vector-index-builder:faiss-base-3a02e64
docker build --build-arg BASE_IMAGE_TAG=faiss-base-3a02e64 -f target/upstream/e2e/api/Dockerfile -t remote-vector-index-builder:e2e-3a02e64 target/upstream
docker compose -f tools/e2e/docker-compose.yml up --exit-code-from e2e
```

## Cluster

`cluster/docker-compose.yml` runs a single OpenSearch 3.9.0 node whose k-NN remote index build sends its segments to knn-builder's image, with LocalStack as the S3 repository, as k-NN's own CI runs the upstream service. [knn-inspect](https://github.com/userepo/knn-inspect)'s `tools/extract_opensearch.py --remote`, from a checkout beside this one with its `tools/requirements.txt` installed, registers the repository, turns the feature on sending every segment of 1 KB or more to the builder, indexes seeded vectors, refuses a segment whose native file is not a CAGRA graph (a build that fell back to the data node's), and copies the segment out; knn-inspect's `scan` and `recall`, from the same checkout, then check the files, and `recall` scores the upstream service's segments in knn-inspect's `fixtures/opensearch-3.9.0-remote` alike:

```sh
docker build -t knn-builder:dev .
docker compose -f tools/cluster/docker-compose.yml up -d
EXTRACT="python ../knn-inspect/tools/extract_opensearch.py --remote --dim 64 --container knn-builder-cluster-opensearch --builder-container knn-builder-cluster-builder --localstack http://localhost:4567"
$EXTRACT --out target/cluster/float --spaces l2
$EXTRACT --count 2000 --out target/cluster/float --spaces cosinesimil innerproduct
$EXTRACT --out target/cluster/half-float --encoding half-float --spaces l2
$EXTRACT --count 2000 --out target/cluster/half-float --encoding half-float --spaces cosinesimil
$EXTRACT --out target/cluster/on-disk --encoding on-disk --spaces l2
KNN_INSPECT=../knn-inspect/target/release/knn-inspect
$KNN_INSPECT scan target/cluster
for f in target/cluster/*/*/*.faiss ../knn-inspect/fixtures/opensearch-3.9.0-remote/*/*/*.faiss; do echo "$f"; $KNN_INSPECT recall "$f" --sample 200; done
docker compose -f tools/cluster/docker-compose.yml down -v
```

Where the fixtures hold the same segment built by the upstream service (float `l2` and `cosinesimil`, half_float `l2`, on-disk `l2`), knn-builder's file has its layout and its size. On a run of 2026-10-29 it also had its recall@10 on 200 of its own vectors (1.0, 0.9995, 1.0 and 0.984); the GPU build is not deterministic, so another run's recall can differ in its last digits. cuVS refuses a CAGRA build of a small inner-product or cosine set, 500 random vectors here, as it did for the upstream service when those fixtures were built, and the node then builds the segment itself; 2,000 build. knn-builder does not build `byte` or `binary` vectors yet: their build requests get a 422, and the node builds those segments itself too.

## Burst

`burst/burst.py` sends a burst of build requests to one service as k-NN 3.9.0.0's data nodes would, all at once, and records how each ends for its data node: built in time, refused, failed, or timed out, and how long it waited before building the segment on its own CPU. Each request follows k-NN's client: one retry, a second later, of a 408, 409, 429, 500, 502, 503 or 509; a wait that starts as the request returns, a first status request after three poll intervals and one an interval after, each stretched by 0.8 to 1.2; and the CPU build on a 507, a failed job, a status request that fails, or the timeout. The jobs share one blob per input, differing by tenant id, so 48 jobs need only the four inputs' 2.3 GB in LocalStack. `burst/docker-compose.yml` runs it, in the upstream image for boto3, against knn-builder's image or the upstream one, which knows nothing of the data node's timeout; under WSL2 the upstream service needs `MAX_WORKERS=1`, as two builds at once crash it (as they would knn-gpu's, see `Gpu`):

```sh
docker build -t knn-builder:dev .
$RUN $API tools/make_inputs.py grid target/burst/inputs l2-1000000x128 l2-500000x384 l2-250000x768 l2-100000x768
SERVICE_IMAGE=knn-builder:dev LABEL=knn-builder docker compose -f tools/burst/docker-compose.yml up --abort-on-container-exit --exit-code-from driver
docker compose -f tools/burst/docker-compose.yml down
SERVICE_IMAGE=$API MAX_WORKERS=1 LABEL=upstream-1-worker docker compose -f tools/burst/docker-compose.yml up --abort-on-container-exit --exit-code-from driver
docker compose -f tools/burst/docker-compose.yml down
```

`JOBS` (48), `TIMEOUT_SECONDS` (90), `POLL_SECONDS` (5), `SEED` (1) and `NAMES` set the burst; each run writes `target/burst/<LABEL>.json`.

`burst/knn_check.py` checks the driver's model against k-NN itself, on the cluster above: it sets the node's poll interval and timeout, flushes six small indices at once, and reads the builder's request log. `cadence` (with 200 ms polls, so the builds outlast several) times k-NN's first status request after each build request and the gaps after it; `refused`, with the builder started at `CLIENT_TIMEOUT_SECONDS=10`, below k-NN's first poll, checks that k-NN sends each refused request once, never polls it, and builds the segment itself:

```sh
CLIENT_POLL_INTERVAL_SECONDS=1 docker compose -f tools/cluster/docker-compose.yml up -d
python tools/burst/knn_check.py cadence
CLIENT_TIMEOUT_SECONDS=10 docker compose -f tools/cluster/docker-compose.yml up -d builder
python tools/burst/knn_check.py refused
docker compose -f tools/cluster/docker-compose.yml down -v
```

`fixtures/burst/` holds the results.

## Peak memory

`knn-builder`'s `e2e` example runs whole jobs through its pipeline on the GPU against LocalStack: `put` uploads inputs, `build` runs them and appends each job's stage times and peak resident memory (`VmHWM`, reset before each job: what is resident when the job starts, the CUDA runtime and whatever the allocator kept from earlier jobs, plus the job's own peak; one job per invocation gives a job's figure alone) to `runs.json`, `get` streams the index files to disk for `score.py`. Each runs as a process of its own, so `put` and `get` stay out of the measured peak. In WSL, in the conda environment, with LocalStack on port 4566 and the S3 settings the service reads:

```sh
$RUN $API tools/make_inputs.py peak target/inputs-peak
E2E="cargo run --release -p knn-builder --features gpu --example e2e --"
$E2E put e2e target/inputs-peak l2-1000000x768
$E2E build e2e target/inputs-peak target/e2e/peak l2-1000000x768
$E2E get e2e target/e2e/peak
python tools/baseline/score.py target/e2e/peak target/inputs-peak ../knn-inspect/target/release/knn-inspect
```

For the service, start `localstack` and `service` with `docker compose ... up -d`, run the runner with `docker compose ... run --rm runner` (`INPUTS=target/inputs-peak`, `RUNS=1`), and read the service's peak from `/proc/1/status` in its container (`VmHWM`) before taking it down. `fixtures/peak/` holds the result.

## Stage times and device memory

`grid/run_grid.py` runs `make_inputs.py`'s `grid` set through the same `e2e` example, one input at a time, since the 29 inputs (100k, 250k, 500k and 1M vectors of dimension 128, 384 and 768, fp32 and fp16, with 750k fp32 at each dimension and 2M fp32 at 128 and 384) take 22 GB: it writes an input, uploads it, builds it with m 16 and with m 32, each in a process of its own after a warm-up build of a small input, then deletes it. Each build records its stage times, its index file's size, and the device memory it took at its peak (the free memory before it less the least seen free while it ran, read every 5 ms through `knn_gpu::device_memory`). Samples every 5 ms can miss an allocation shorter than that, so `grid/check_peaks.py` rebuilds five inputs reading the free memory as often as it can (`e2e build --every 0`), whose peaks the 5 ms ones should match, and twice more with device memory held as another process would hold it (`e2e build --leave`, through `knn_gpu::Reservation`): leaving the sampled peak plus 2% and 16 MiB free must complete, and leaving 90% of it must fail out of device memory. Under WSL2 the second completes as well, since the driver pages device memory out to the host rather than refuse it, so there the pair proves nothing and `check.json` says so; on a native Linux driver it checks the peak itself. `grid/cpu_hnsw.py` times the build a data node runs itself, Faiss's HNSW over the same float vectors, on 1 and 4 threads (k-NN's `knn.algo_param.index_thread_qty`: 4 on a node of 32 processors or more), with faiss-cpu 1.15.1. `grid/fit.py` fits a model of each quantity and scores it on the builds it left out. In WSL, in the conda environment, with LocalStack and the S3 settings as above, and faiss-cpu in a venv of its own (`$FAISS_PY`):

```sh
cargo build --release -p knn-builder --features gpu --example e2e
python tools/grid/run_grid.py $CARGO_TARGET_DIR/release/examples/e2e ~/knn-grid/inputs target/grid
python tools/grid/check_peaks.py $CARGO_TARGET_DIR/release/examples/e2e ~/knn-grid/inputs target/grid target/grid/check
CPU="l2-100000x128 l2-100000x384 l2-100000x768 l2-250000x128 l2-250000x384 l2-250000x768"
$FAISS_PY tools/make_inputs.py grid ~/knn-grid/cpu $CPU l2-500000x128 l2-500000x384 l2-500000x768
$FAISS_PY tools/grid/cpu_hnsw.py ~/knn-grid/cpu target/grid 4 $CPU l2-500000x128 l2-500000x384 l2-500000x768
$FAISS_PY tools/grid/cpu_hnsw.py ~/knn-grid/cpu target/grid 1 $CPU l2-500000x128
$FAISS_PY tools/grid/fit.py target/grid target/grid
```

`fixtures/grid/` holds the RTX 4070's results and fits.

See `docs/faiss-layout.md` for the layout and `docs/params.md` for the build parameters.
