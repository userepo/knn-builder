# Tools

Scripts that produce and check the fixtures. The Python ones run in the upstream images, pinned by digest, so their numpy, boto3 and Faiss are the service's own; nothing needs installing on the host but Docker (with GPU access) and Python for `baseline/score.py`.

| Image | Digest | Used for |
| --- | --- | --- |
| `opensearchstaging/remote-vector-index-builder:api-snapshot` | `sha256:e8fecaefc43cfaf4d04f35072f10baedf0ece3a539efac5340e5ab0c095da8bc` | the service at commit ea7db9f (unreleased 1.3.0), `make_inputs.py`, the baseline runner |
| `opensearchstaging/remote-vector-index-builder:faiss-base-snapshot` | `sha256:241eddbe1a2fa81c188aee5edc89a81bc14415a25963f24ecd923f04c4f3d9d1` | the oracle: Faiss 1.14.3 at `0ca9df4` with the service's patches |
| `localstack/localstack:4.14` | `sha256:3ebc37595918b8accb852f8048fef2aff047d465167edd655528065b07bc364a` | S3 for the baseline, the version upstream's e2e test uses |

Never use the `-latest` tags: they are release 1.2.0, which builds different graphs. Upstream built both snapshot images from f11f93c, the last commit before ea7db9f to change the service; ea7db9f changes only `MAINTAINERS.md`.

```sh
API=opensearchstaging/remote-vector-index-builder@sha256:e8fecaefc43cfaf4d04f35072f10baedf0ece3a539efac5340e5ab0c095da8bc
BASE=opensearchstaging/remote-vector-index-builder@sha256:241eddbe1a2fa81c188aee5edc89a81bc14415a25963f24ecd923f04c4f3d9d1
RUN="docker run --rm -v $PWD:/work -w /work --entrypoint python"
```

From Git Bash on Windows, Docker needs a Windows path and no path rewriting: `export MSYS_NO_PATHCONV=1` and use `-v "$(pwd -W):/work"` in `RUN`.

## Inputs

`make_inputs.py` writes build inputs in the format k-NN uploads (float32, float16 or int8 vectors, int32 doc ids) with 100 held-out queries each (1,000 for the `clustered` set):

```sh
$RUN $API tools/make_inputs.py small fixtures/inputs       # committed
$RUN $API tools/make_inputs.py large target/inputs-large   # for the recall spread
$RUN $API tools/make_inputs.py byte fixtures/byte/inputs   # committed
$RUN $API tools/make_inputs.py byte-large target/inputs-byte-large
$RUN $API tools/make_inputs.py binary fixtures/binary/inputs   # committed
$RUN $API tools/make_inputs.py binary-large target/inputs-binary-large
$RUN $API tools/make_inputs.py binary-sizes target/inputs-binary-sizes
$RUN $API tools/make_inputs.py clustered target/inputs-clustered
```

The `clustered` set is for the upper-level projection (`project_link_to_base_layer`): Gaussian mixtures of 300,000 unit vectors of dimension 256 under inner product, each vector and query a random standard normal center plus sigma times standard normal noise, scaled to unit length. Built once each without the projection, at k-NN's default m of 16, knn-builder's CAGRA graphs of 200 clusters at sigma 0.3 or 1.0, or 1,000 at sigma 0.3, fell apart into a component a cluster, while 200 clusters at sigma 2.0 overlap enough to stay one graph. The set also holds the first of them as a half_float field (`-fp16`) and as a byte field (`-byte`: the same unit vectors times 127 over their largest absolute value, rounded, the queries scaled alike), whose graphs fell apart the same way; `make_inputs.py` refuses to run if a copy's seed is not its source's.

`bench/components.py` counts the strongly and weakly connected components of a built file's level 0, in the oracle's image, whose Faiss reads the service's files:

```sh
$RUN $BASE tools/bench/components.py target/e2e/clustered-off/*/run-1.faiss
```

A byte input's vectors are int8 values drawn uniformly over -128 to 127, and its queries the same values as float32, as k-NN takes a byte field's queries. A binary input's vectors are packed bits, as k-NN uploads a binary field's, uniform or, for a name ending in `-clustered`, drawn around a center for every 50 vectors with each bit flipped with probability 0.1; its queries are packed bits too, in `queries.u8`.

## Baseline

`baseline/` runs the service against LocalStack and builds every input five times, one build at a time:

```sh
export INPUTS=fixtures/inputs OUT=target/baseline/small RUNS=5
docker compose -f tools/baseline/docker-compose.yml up --abort-on-container-exit --exit-code-from runner
docker compose -f tools/baseline/docker-compose.yml logs --no-color service > $OUT/service.log
docker compose -f tools/baseline/docker-compose.yml down -v
python tools/baseline/score.py $OUT $INPUTS ../knn-inspect/target/release/knn-inspect
```

`NAMES`, comma separated, builds only those inputs. The byte fixtures come from the same runs, with `INPUTS=fixtures/byte/inputs` and, for the recall spread, `INPUTS=target/inputs-byte-large RUNS=20`, and the binary ones with `fixtures/binary/inputs` and `target/inputs-binary-large`.

`baseline/device_peaks.py` gives each build's peak device memory from the GPU's memory sampled on the host as the run goes, for `binary-sizes`' first look at the NN-descent build's costs: the most memory used in a build's window less what the service's process held before it, the first build of the process, which makes its CUDA context, left out. What the process held is the level the samples last stayed at, two samples or more, before the window, not the least in it: `nvidia-smi` under WSL2 reads 0 MiB used once the GPU has idled for about ten seconds, whatever a process holds. Sample from the host, and keep the service's log with Docker's timestamps:

```sh
export INPUTS=target/inputs-binary-sizes OUT=target/baseline/binary-sizes RUNS=2
mkdir -p $OUT
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 50 > $OUT/gpu.csv & SAMPLER=$!
docker compose -f tools/baseline/docker-compose.yml up --abort-on-container-exit --exit-code-from runner
docker compose -f tools/baseline/docker-compose.yml logs --no-color --timestamps service > $OUT/service.log
docker compose -f tools/baseline/docker-compose.yml down -v
kill $SAMPLER
python tools/baseline/device_peaks.py $OUT
```

`score.py` adds each build's per-step times from the service's DEBUG log and knn-inspect `verify` and `recall` (its seeded search, seed 1, k = 10, over the held-out queries) to `$OUT/summary.json`, with the knn-inspect version: its recall is that version's search. The request's `m`, `ef_construction` and `ef_search` come from each input's `input.json`.

## Oracle

`oracle/oracle.py` writes the file the service writes from the CAGRA graph, the input blobs and the request (`input.json`), replaying `GpuIndexCagra::copyTo` and `write_index`. `check` takes only the graph from each service-built file, checks its degree against the request, rebuilds the file and compares:

```sh
$RUN $BASE tools/oracle/oracle.py check fixtures/service fixtures/inputs
$RUN $BASE tools/oracle/oracle.py check target/baseline/small fixtures/inputs
$RUN $BASE tools/oracle/oracle.py check fixtures/byte/service fixtures/byte/inputs
$RUN $BASE tools/oracle/oracle.py check fixtures/binary/service fixtures/binary/inputs
```

A binary file's graph is read from its bytes, since Faiss reads no binary graph without its vectors.

`oracle/levels.py` computes the HNSW level table those files hold without Faiss, as `knn-write` must:

```sh
$RUN $BASE tools/oracle/levels.py "fixtures/service/l2-1000x64*.faiss"
```

`oracle/level_tables.py` writes `knn-write`'s level-table test data from it: the probabilities and neighbor counts for every even M from 2 to 4,096 and level histograms over 2,100,000 nodes. Run it in the image, for glibc's `exp`:

```sh
$RUN $BASE tools/oracle/level_tables.py crates/knn-write/tests/data/level_tables.txt
```

`oracle/projection.py` writes the test data for `knn-write`'s port of the upstream service's upper-level projection (`project_link_to_base_layer`), from the service's own module and the numpy and Faiss it runs on: the gate's three entry points for 269 graph sizes up to 2^31 - 1, the nodes promoted to layer 1 for 21 pairs of n and m (odd m too, since layer 1's M is the request's m), Faiss's level-0 probability for every m from 2 to 2,048, which alone decides a promotion, the reachable share the gate measures on ten synthetic graphs that the test builds again from the same parameters, and the slots `project_links` changes on ten more under 24 settings in all (shares of a full row from 0 to 1, the last-in-edge protection on and off), with layer 1 synthetic, built from the base graph's own links, or Faiss's `build_layer1` on random vectors; and the layer 1 `build_layer1` builds on 13 sets of vectors the test makes again (L2 and inner product, dimensions that leave 0 to 3 values past a multiple of 4, whole-number vectors whose scores tie, M from 2 to 16, efConstruction from 1 to 100). Faiss runs on one OpenMP thread there, where its HNSW insertion is the same every run (the service's runs on several and varies), so the file comes out the same each time. Run it in the api image:

```sh
$RUN $API tools/oracle/projection.py crates/knn-write/tests/data/projection.json
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

Then, from Windows or Linux, `python tools/baseline/score.py target/rust-baseline/large20 target/inputs-large ../knn-inspect/target/release/knn-inspect`, and the service's own 20 builds with `RUNS=20` as above. The byte parity files come from `target/inputs-byte-large` with 20 runs and `fixtures/byte/inputs` with 5, the binary ones from `target/inputs-binary-large` and `fixtures/binary/inputs` alike, and `oracle.py check` on the example's output directory rebuilds each file it wrote.

## End to end

`e2e/docker-compose.yml` runs the upstream service's end-to-end test against knn-builder's image: its client, unchanged, from upstream's own e2e image at ea7db9f, in knn-builder's container's network, where the address the client calls (`localhost:1025`) is knn-builder's, with LocalStack for S3. It builds an index of each of upstream's six test datasets (1M x 128, 384 and 768 float, two 100k half_float and a small one), checks each job through `/_status` and `/_jobs`, and exits 0 when all pass. Build the two images first, the upstream one from a checkout of upstream at ea7db9f under `target/`, which git and the image's build leave out, on the `faiss-base` image pinned above:

```sh
docker build -t knn-builder:dev .
git clone https://github.com/opensearch-project/remote-vector-index-builder target/upstream && git -C target/upstream checkout ea7db9f
docker tag opensearchstaging/remote-vector-index-builder@sha256:241eddbe1a2fa81c188aee5edc89a81bc14415a25963f24ecd923f04c4f3d9d1 opensearchstaging/remote-vector-index-builder:faiss-base-ea7db9f
docker build --build-arg BASE_IMAGE_TAG=faiss-base-ea7db9f -f target/upstream/e2e/api/Dockerfile -t remote-vector-index-builder:e2e-ea7db9f target/upstream
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
$EXTRACT --out target/cluster/byte --encoding byte --spaces l2
$EXTRACT --count 2000 --out target/cluster/byte --encoding byte --spaces innerproduct
$EXTRACT --out target/cluster/binary --encoding binary --spaces hamming
$EXTRACT --count 2000 --out target/cluster/binary-2000 --encoding binary --spaces hamming
KNN_INSPECT=../knn-inspect/target/release/knn-inspect
$KNN_INSPECT scan target/cluster
for f in target/cluster/*/*/*.faiss ../knn-inspect/fixtures/opensearch-3.9.0-remote/*/*/*.faiss; do echo "$f"; $KNN_INSPECT recall "$f" --sample 200; done
docker compose -f tools/cluster/docker-compose.yml down -v
```

Where the fixtures hold the same segment built by the upstream service (float `l2` and `cosinesimil`, half_float `l2`, on-disk `l2`, byte `l2`, binary `hamming`), knn-builder's file has its layout and its size. On a run of 2026-10-29 it also had its recall@10 on 200 of its own vectors (1.0, 0.9995, 1.0, 0.984, 1.0 and 1.0; 0.9995 for byte `innerproduct`, 1.0 for 2,000 binary vectors), and each byte and binary index found every one of 20 of its own vectors among its 10 nearest when searched; the GPU build is not deterministic, so another run's recall can differ in its last digits. cuVS refuses a CAGRA build of a small inner-product or cosine set, 500 random vectors here, as it did for the upstream service when those fixtures were built, and the node then builds the segment itself; 2,000 build.

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

`JOBS` (48), `TIMEOUT_SECONDS` (90), `POLL_SECONDS` (5), `SEED` (1) and `NAMES` set the burst; each run writes `target/burst/<LABEL>.json`. The burst with binary jobs mixed in, its two binary inputs from `make_inputs.py binary-sizes` copied beside the others:

```sh
SERVICE_IMAGE=knn-builder:dev LABEL=knn-builder-binary-mix JOBS=24 NAMES=hamming-1000000x1024-binary-clustered,hamming-500000x256-binary-clustered,l2-500000x384,l2-100000x768 docker compose -f tools/burst/docker-compose.yml up --abort-on-container-exit --exit-code-from driver
```

`burst/knn_check.py` checks the driver's model against k-NN itself, on the cluster above: it sets the node's poll interval and timeout, flushes six small indices at once, and reads the builder's request log. `cadence` (with 200 ms polls, so the builds outlast several) times k-NN's first status request after each build request and the gaps after it; `refused`, with the builder started at `CLIENT_TIMEOUT_SECONDS=10`, below k-NN's first poll, checks that k-NN sends each refused request once, never polls it, and builds the segment itself:

```sh
CLIENT_POLL_INTERVAL_SECONDS=1 docker compose -f tools/cluster/docker-compose.yml up -d
python tools/burst/knn_check.py cadence
CLIENT_TIMEOUT_SECONDS=10 docker compose -f tools/cluster/docker-compose.yml up -d builder
python tools/burst/knn_check.py refused
docker compose -f tools/cluster/docker-compose.yml down -v
```

`fixtures/burst/` holds the results.

Against knn-builder, the driver also waits once the burst has ended for the service to be idle, no job in flight and every job it took ended by its metrics page's own counts, then saves that page as `<LABEL>-metrics.txt` and asks the status of each job whose end its data node did not see, for `bench/check_metrics.py` (below); a page it cannot have, as of a service that is gone, is noted in the burst's file, which is written all the same. With the `prometheus` profile, a Prometheus (the version CI's `promtool` is, pinned by digest) scrapes the service every second through the burst (`burst/prometheus.yml`), and the driver, given its URL, waits for it to scrape the idle service and keeps what it answers: whether it read the page, the counts by answer and by outcome, which the check holds to the page's, and a few queries an operator would run, recorded as they came. The service's log, for the check's stage times, is the compose log:

```sh
export SERVICE_IMAGE=knn-builder:dev LABEL=knn-builder-prometheus COMPOSE_PROFILES=prometheus PROMETHEUS_URL=http://prometheus:9090
docker compose -f tools/burst/docker-compose.yml up --abort-on-container-exit --exit-code-from driver
docker compose -f tools/burst/docker-compose.yml logs --no-color service > target/burst/knn-builder-prometheus.service.log
docker compose -f tools/burst/docker-compose.yml down
python tools/bench/check_metrics.py target/burst/knn-builder-prometheus.json --log target/burst/knn-builder-prometheus.service.log
```

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

`grid/run_grid.py` runs `make_inputs.py`'s `grid` set through the same `e2e` example, one input at a time, since the 29 inputs (100k, 250k, 500k and 1M vectors of dimension 128, 384 and 768, fp32 and fp16, with 750k fp32 at each dimension and 2M fp32 at 128 and 384) take 22 GB: it writes an input, uploads it, builds it with m 16 and with m 32, each in a process of its own after a warm-up build of a small input, then deletes it. Each build records its stage times, its index file's size, and the device memory it took at its peak (the free memory before it less the least seen free while it ran, read every 5 ms through `knn_gpu::device_memory`). Samples every 5 ms can miss an allocation shorter than that, so `grid/check_peaks.py` rebuilds five inputs reading the free memory as often as it can (`e2e build --every 0`), whose peaks the 5 ms ones should match, and twice more with device memory held as another process would hold it (`e2e build --leave`, through `knn_gpu::Reservation`): leaving the sampled peak plus 2% and 16 MiB free must complete, and leaving 90% of it must fail out of device memory. Under WSL2 the second completes as well, since the driver pages device memory out to the host rather than refuse it, so there the pair proves nothing and `check.json` says so; on a native Linux driver it checks the peak itself. The briefest builds can hold the IVF-PQ build's batch of vectors for less than 5 ms, so `grid/tight_peaks.py` rebuilds every float and byte input of 250,000 vectors or fewer reading the free memory as often as it can, into `tight.json`, and `fit.py` takes each build's peak as the larger of its two readings; the stage times stay those of the 5 ms run, which a busy sampler could slow. Every one of those builds follows a warm-up build, so `grid/first_builds.py` builds six inputs, from 1,000 to 1M vectors, each as its process's first build and again after a warm-up, into `first.json`: what a first build takes besides, and what builds of fewer vectors than the IVF-PQ build's batch take. `grid/cpu_hnsw.py` times the build a data node runs itself, Faiss's HNSW over the same float vectors, on 1 and 4 threads (k-NN's `knn.algo_param.index_thread_qty`: 4 on a node of 32 processors or more), with faiss-cpu 1.15.1. `grid/fit.py` fits a model of each quantity and scores it on the builds it left out. In WSL, in the conda environment, with LocalStack and the S3 settings as above, and faiss-cpu in a venv of its own (`$FAISS_PY`):

```sh
cargo build --release -p knn-builder --features gpu --example e2e
python tools/grid/run_grid.py $CARGO_TARGET_DIR/release/examples/e2e ~/knn-grid/inputs target/grid
python tools/grid/check_peaks.py $CARGO_TARGET_DIR/release/examples/e2e ~/knn-grid/inputs target/grid target/grid/check
python tools/grid/tight_peaks.py $CARGO_TARGET_DIR/release/examples/e2e ~/knn-grid/inputs target/grid
python tools/grid/first_builds.py $CARGO_TARGET_DIR/release/examples/e2e ~/knn-grid/inputs target/grid
CPU="l2-100000x128 l2-100000x384 l2-100000x768 l2-250000x128 l2-250000x384 l2-250000x768"
$FAISS_PY tools/make_inputs.py grid ~/knn-grid/cpu $CPU l2-500000x128 l2-500000x384 l2-500000x768
$FAISS_PY tools/grid/cpu_hnsw.py ~/knn-grid/cpu target/grid 4 $CPU l2-500000x128 l2-500000x384 l2-500000x768
$FAISS_PY tools/grid/cpu_hnsw.py ~/knn-grid/cpu target/grid 1 $CPU l2-500000x128
$FAISS_PY tools/grid/fit.py target/grid target/grid
```

The models are checked, not fitted, on byte builds: `--set grid-byte` runs `make_inputs.py`'s `grid-byte` set (100k, 250k, 500k and 1M byte vectors of dimension 128, 384 and 768), each build after a warm-up on a small byte input, and `cpu_hnsw.py` builds a byte input as k-NN builds a byte field, over `HNSW16,SQ8_direct_signed`:

```sh
python tools/grid/run_grid.py $CARGO_TARGET_DIR/release/examples/e2e ~/knn-grid/byte target/grid-byte --set grid-byte
CPU="l2-100000x128-byte l2-100000x384-byte l2-100000x768-byte l2-250000x128-byte l2-250000x384-byte l2-250000x768-byte"
$FAISS_PY tools/make_inputs.py grid-byte ~/knn-grid/cpu-byte $CPU l2-500000x128-byte l2-500000x384-byte l2-500000x768-byte
$FAISS_PY tools/grid/cpu_hnsw.py ~/knn-grid/cpu-byte target/grid-byte 4 $CPU l2-500000x128-byte l2-500000x384-byte l2-500000x768-byte
$FAISS_PY tools/grid/cpu_hnsw.py ~/knn-grid/cpu-byte target/grid-byte 1 $CPU l2-500000x128-byte
```

The binary build, NN-descent, has models of its own: `--set grid-binary` runs `make_inputs.py`'s `grid-binary` set (clustered binary vectors, 100k to 2M of 256 and 1,024 bits and 512 bits at 100k, 500k and 1M), `cpu_hnsw.py` builds a binary input as k-NN builds a binary field, `BHNSW16,Flat` from Faiss's binary index factory, and `fit.py binary` fits the models to `binary-runs.json` and `binary-cpu.json` in a directory beside the float grid's `fits.json`, whose transfer models it scores on the binary builds:

```sh
python tools/grid/run_grid.py $CARGO_TARGET_DIR/release/examples/e2e ~/knn-grid/binary target/grid-binary --set grid-binary
CPU="hamming-100000x256-binary-clustered hamming-100000x512-binary-clustered hamming-100000x1024-binary-clustered hamming-250000x256-binary-clustered hamming-250000x1024-binary-clustered"
$FAISS_PY tools/make_inputs.py grid-binary ~/knn-grid/cpu-binary $CPU hamming-500000x256-binary-clustered hamming-500000x512-binary-clustered hamming-500000x1024-binary-clustered
$FAISS_PY tools/grid/cpu_hnsw.py ~/knn-grid/cpu-binary target/grid-binary 4 $CPU hamming-500000x256-binary-clustered hamming-500000x512-binary-clustered hamming-500000x1024-binary-clustered
$FAISS_PY tools/grid/cpu_hnsw.py ~/knn-grid/cpu-binary target/grid-binary 1 $CPU hamming-500000x256-binary-clustered
# as binary-runs.json and binary-cpu.json, beside fits.json:
$FAISS_PY tools/grid/fit.py binary fixtures/grid fixtures/grid
```

`fixtures/grid/` holds the RTX 4070's results and fits, the byte ones as `byte-runs.json` and `byte-cpu.json`, which `crates/knn-builder/tests/cost.rs` holds the models to, and the binary ones as `binary-runs.json`, `binary-cpu.json` and `binary-fits.json`, which it ties the binary models to.

The projection a request may ask for runs on the CPU, so it needs no GPU or S3: `crates/knn-write/tests/projection_cost.rs` times it and takes its peak host memory through a counting allocator, on synthetic graphs whose clusters a search cannot leave (100,000 to 1M vectors of 128 and 768 dimensions at m 8 to 32, the vectors' clusters far apart and overlapping, and m 2 and 4 at 1M x 128; efConstruction 50 to 400 and 1 to 32 threads at 1M x m 16) and on connected graphs of 300,000 to 3M nodes, where the gate stops it. It writes `fixtures/grid/projection.json`, to which `cost.rs`'s projection model and `knn-write`'s memory bound (`projection::host_bytes`) are held. In WSL, from the repository's root, in release:

```sh
PROJECTION_COST_OUT=fixtures/grid/projection.json cargo test --release -p knn-write --test projection_cost -- --ignored
```

## Benchmarks

`bench/datasets.py` fetches the datasets the upstream `benchmarking/` tool names and writes them as build inputs in `make_inputs.py`'s layout: sift-128 (ann-benchmarks), ms-marco-384 and cohere-768 (Hugging Face, `navneet1v/datasets`) and gist-960 (ann-benchmarks), 1M vectors each, 9.2 GB of HDF5 files kept in the first directory. Each dataset gives a float input in L2 and in inner product, with the vectors as they are, and the first three a byte input in both spaces, each component scaled over the dataset's range onto -128..127, and a binary one in Hamming space, a bit per component set where it is positive (SIFT's components are counts, so its bits mark the nonzero ones; 77% of them are set). The queries are the first 100 of each file's test set, converted the same way. A dataset is taken whole when the fitted models (`grid/fit.py`, `fixtures/grid/fits.json`) predict its float build's peak device memory within `--gpu-gb` (8, the GPU's), else cut to the most rows, in tens of thousands, that fit; all four fit whole on 8 GB, gist-960's float build predicted at 1.0 GB (4.6 GB in `fixtures/bench/datasets.json`, whose runs still copied the dataset to the device). Every `input.json` records the rows taken, the peak predicted for it and the upstream service's own estimate, and `datasets.json` the sources, their hashes (checked against those the script records) and the sizes chosen. In the conda environment of `bench/environment.yml` (numpy and h5py; apart from the root `environment.yml`, which the container image builds from):

```sh
conda env create -f tools/bench/environment.yml && conda activate knn-bench
python tools/bench/datasets.py ~/knn-bench/hdf5 target/bench/inputs
```

The 17 inputs take 10 GB: the inner product input of each type hard-links its L2 one's blobs, and every input of a dataset its doc ids (copies where the file system has no links, 19 GB then).

`bench/run.py` runs the metrics against one service, from `bench/docker-compose.yml` (the service at `GPU_MEMORY_LIMIT=8` with LocalStack for S3, the upstream one on one worker), driven from the host, where `nvidia-smi` samples the GPU's memory. Its phases: a warm-up build of one input of each data type; `single`, five builds of every input one at a time, each job's latency, the service process's peak resident memory (`VmHWM`, reset before each job through `docker compose exec`), the GPU's memory at its peak during the job less what the service's process held before it (sampled every 50 ms, `baseline/device_peaks.py`'s measure over the request's window; `run.json` keeps when each of the service's processes started, since what an earlier one held does not count) and its per-step times from the service's DEBUG log, with the first run's file kept for the recall; `concurrent`, per data type, five rounds of eight jobs sent at once as k-NN's data nodes send them (`burst/burst.py`'s model at k-NN's default timeout and poll interval), each job's latency as its data node sees it; `sustained`, per data type, four jobs kept in flight for an hour, the next sent as one completes, for jobs per hour (the jobs seen completed within the hour; those still in flight at its end are not counted); and `burst`, per data type, 48 requests at once over its L2 (binary: Hamming) inputs with the 90 s timeout and 5 s polls of `tools/burst`, the service restarted with them, and again before each data type after the first since the upstream service builds on the jobs a burst's data nodes gave up on, and warmed up each time. A run starts from fresh services, and one that fails or loses its service still takes them down and writes what it measured. Each job's file is deleted from the bucket once its job is seen completed, so LocalStack holds the inputs and the jobs in flight, 30 GB at most; under Docker Desktop its disk image on the host still grows with every block first written, by hundreds of GB over a run, until it is trimmed (`fstrim` in the `docker-desktop` distro) and compacted (`diskpart`'s `compact vdisk` on `docker_data.vhdx`, Docker Desktop quit). Each run writes `run.json`, `service.log`, `gpu.csv` and the first files to its directory; `bench/report.py` reduces the runs to the README's tables, knn-inspect `recall` on each first file (seed 1, k 10, cached in `recall.json`), medians over the single builds (the latency as the client saw it beside the sum of the steps from the service's own clock, since on this host the status requests through Docker Desktop's port forwarding paused for 10 to 30 s at the end of some builds of files of 1.5 GB or more), p50 and p99 over the concurrent jobs. `fixtures/bench/` keeps the README's runs (`run.json` and `recall.json` per service, `datasets.json`, and `environment.json` with the versions), from which `report.py` prints the README's tables again; the service logs and GPU samples, 15 MB a run, stay out. `fixtures/ea7db9f/bench-single/` keeps the later single builds of the float and byte inputs, by knn-builder at 3a4eb45 and the upstream service at ea7db9f, whose rows the README shows in place of `fixtures/bench/`'s (the last two commands below; the run is the same for `upstream`). From the repository root, with Docker, `nvidia-smi` and Python on the host:

```sh
docker build -t knn-builder:dev .
python tools/bench/run.py target/bench/knn-builder knn-builder
python tools/bench/run.py target/bench/upstream upstream
python tools/bench/report.py ../knn-inspect/target/release/knn-inspect target/bench/knn-builder target/bench/upstream
python tools/bench/report.py ../knn-inspect/target/release/knn-inspect fixtures/bench/knn-builder fixtures/bench/upstream
python tools/bench/run.py target/bench/single/knn-builder knn-builder --phases single $(ls target/bench/inputs | grep -v 'binary\|datasets.json')
python tools/bench/report.py ../knn-inspect/target/release/knn-inspect fixtures/ea7db9f/bench-single/knn-builder fixtures/ea7db9f/bench-single/upstream
```

`--runs`, `--concurrent`, `--rounds`, `--queue`, `--sustain-seconds`, `--burst-jobs`, `--burst-timeout`, `--burst-poll`, `--phases` and `--seed` set the phases; `--inputs` (repeatable) and names pick the inputs. A dry run on the committed fixtures, every cell filled in a few minutes:

```sh
DRY="--inputs fixtures/inputs --inputs fixtures/byte/inputs --inputs fixtures/binary/inputs --runs 2 --concurrent 4 --rounds 1 --queue 2 --sustain-seconds 30 --burst-jobs 8"
python tools/bench/run.py target/bench/dry/knn-builder knn-builder $DRY
python tools/bench/run.py target/bench/dry/upstream upstream $DRY
python tools/bench/report.py ../knn-inspect/target/release/knn-inspect target/bench/dry/knn-builder target/bench/dry/upstream
```

### On real S3

`--s3-bucket BUCKET` runs against a bucket on real S3 instead of LocalStack, as on a cloud GPU host with the upstream service on its default two workers (`--max-workers 2`). The bucket must exist, in the region `AWS_DEFAULT_REGION` names, and is best kept for the benchmark: the inputs go in at `<input>/vectors.knnvec` and `<input>/doc_ids.knndid` and stay for the next run (an input that has changed since, by its blobs' SHA-256, is sent again), and each job's file is deleted once the job is seen completed. Every container takes `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` from the host's environment, or, with them unset, the SDKs' default chain, which on EC2 means the instance role if the instance metadata service answers containers (IMDSv2 needs a hop limit of 2 for that); `S3_ENDPOINT_URL`, set, names an S3-compatible endpoint. The host fetches and deletes the jobs' files through the `s3` service of `bench/docker-compose.yml` (`bench/s3_proxy.py`, boto3 in the upstream image), so it still needs only Python, and every port the services publish is on the host's loopback. `run.json` records the store and region (not the bucket's name), `GPU_MEMORY_LIMIT` and the worker count; `--s3-bucket` needs `GPU_MEMORY_LIMIT` set, to the GPU's memory in GB, since unset it is 8. On a host with a 24 GB GPU, for instance:

```sh
export AWS_DEFAULT_REGION=us-east-1 GPU_MEMORY_LIMIT=24
python tools/bench/run.py target/bench/cloud/knn-builder knn-builder --s3-bucket my-bench-bucket
python tools/bench/run.py target/bench/cloud/upstream upstream --s3-bucket my-bench-bucket --max-workers 2
python tools/bench/report.py ../knn-inspect/target/release/knn-inspect target/bench/cloud/knn-builder target/bench/cloud/upstream
```

### The projection

`--project` sends every build with `project_link_to_base_layer` true, and `--fetch-runs N` keeps the files of the first N single runs rather than the first alone. Each record of the single phase then carries the projection from the service's DEBUG log: whether it ran (or skipped a connected graph, or promoted too few), the share of the nodes reachable before it, the nodes it promoted, the links it added, and its milliseconds. `bench/projection.py` scores each kept file, its level-0 components (`components.py` in the oracle's image) and knn-inspect recall as the mean over five seeds of random entry points, recall@10 at the file's own search settings and recall@100 at ef_search 600 from 32 entry points, into `projection.json` beside `run.json`, and prints a table by input of the mean and range over the builds of each run; `--delete-files` removes each file once scored, `--summary` writes the table's figures as JSON. On the `clustered` set, with and without the projection on each service:

```sh
P="--inputs target/inputs-clustered --phases single --runs 5 --fetch-runs 5"
python tools/bench/run.py target/bench/projection/knn-builder-off knn-builder $P
python tools/bench/run.py target/bench/projection/knn-builder-on knn-builder $P --project
python tools/bench/run.py target/bench/projection/upstream-off upstream $P
python tools/bench/run.py target/bench/projection/upstream-on upstream $P --project
python tools/bench/projection.py ../knn-inspect/target/release/knn-inspect target/bench/projection/*/ --delete-files --summary target/bench/projection/summary.json
```

### The recall check

`--recall-queries N` starts knn-builder with its recall check on (`RECALL_CHECK_QUERIES`, and `--recall-k` as `RECALL_CHECK_K`), and each warm-up and single record then carries what its job's check found, from the service's log: the recall, the queries and k it is of, and the check's seconds, kept apart from the job's step seconds since the check runs beside the upload. An input whose `input.json` sets `skip_stored_vectors` is requested with it. `bench/recall_parity.py compare` then scores each job's fetched file with `knn-inspect recall --sample`, at the same count and k and seed 1, and the two recalls must be equal to the last digit. A job that leaves its vectors out of the file has no file knn-inspect can search that way, so it is held to its own graph with the vectors put back: `knn-write`'s `with_vectors` example writes that file from the job's file and its input's vector blob, the same as the service writes it with the vectors stored. Another build of the same input would not do, since two builds' graphs differ, at 1,000 vectors as at 1M. A binary job of that kind, which k-NN never sends, is left unscored: knn-format reads no binary graph without its vectors. `recall_parity.py inputs` writes an input again, its blobs hard links, as a `skip_stored_vectors` request or a float input's fp32 blob as a `half_float` one. On the fixtures' inputs of every data type at k 100, where a few recalls fall short of 1, then on seven of the benchmark's 1M inputs:

```sh
cargo build --release -p knn-write --example with_vectors
TOOLS="../knn-inspect/target/release/knn-inspect target/release/examples/with_vectors"
python tools/bench/run.py target/bench/recall-small knn-builder --inputs fixtures/inputs --inputs fixtures/byte/inputs --inputs fixtures/binary/inputs --phases single --runs 1 --recall-queries 100 --recall-k 100
python tools/bench/recall_parity.py compare target/bench/recall-small $TOOLS target/bench/recall-small/parity.json
R=target/bench/recall-inputs
python tools/bench/recall_parity.py inputs $R skip target/bench/inputs/cohere-l2-1000000x768
python tools/bench/recall_parity.py inputs $R fp16from32 target/bench/inputs/cohere-l2-1000000x768
python tools/bench/run.py target/bench/recall-1m knn-builder --inputs target/bench/inputs --inputs $R --phases single --runs 1 --recall-queries 100 cohere-l2-1000000x768 cohere-ip-1000000x768 gist-l2-1000000x960 cohere-l2-1000000x768-byte cohere-hamming-1000000x768-binary cohere-l2-1000000x768-skip cohere-l2-1000000x768-fp16from32
python tools/bench/recall_parity.py compare target/bench/recall-1m $TOOLS target/bench/recall-1m/parity.json
```

`fixtures/ea7db9f/recall-parity/` keeps both runs' `run.json` and `parity.json`.

`bench/recall_cost.py` gives what the check adds a job, from two runs over the same inputs, one with the check off and one with it on: for each input, the medians of the steps' sum, the upload, the check's seconds and the host peak, the host peak's growth from each round's build without the check to that round's with it, and the check's recall, the figures of the README's table, printed as a table and, given a path, written as JSON. Over six of the 1M inputs, the two derived ones from the commands above among them:

```sh
R=target/bench/recall-inputs
N="cohere-l2-1000000x768 gist-l2-1000000x960 cohere-l2-1000000x768-fp16from32 cohere-l2-1000000x768-byte cohere-hamming-1000000x768-binary cohere-l2-1000000x768-skip"
python tools/bench/run.py target/bench/recall-cost/off knn-builder --inputs target/bench/inputs --inputs $R --phases single --runs 5 $N
python tools/bench/run.py target/bench/recall-cost/on knn-builder --inputs target/bench/inputs --inputs $R --phases single --runs 5 --recall-queries 100 $N
python tools/bench/recall_cost.py target/bench/recall-cost/off target/bench/recall-cost/on target/bench/recall-cost/summary.json
```

`fixtures/ea7db9f/recall-cost/` keeps both runs' `run.json` and the summary.

### The metrics

Before each of the service's processes ends, once it is idle, `run.py` saves knn-builder's metrics page to `metrics/process-N.txt` and asks the status of each of that process's jobs whose end it did not see; each record names its process, and `run.json`'s `scrapes` lists the pages, or what kept one from being saved, which does not end the run. A run of every phase has one process up to the bursts and one for each data type's burst. `bench/check_metrics.py` holds each page to the run's own records and the service's log: the build requests by answer, each refusal by the kind its detail names; the jobs ended by outcome, by the error each job's status named; the vectors built by data type; the counts of the completed jobs' histograms; and each stage's seconds summed, against the jobs' `DEBUG` stage-times lines to their nine digits. It prints every row and exits 1 on any difference. A short run of every phase over three of the burst's inputs, two data types:

```sh
python tools/bench/run.py target/bench/metrics knn-builder --inputs target/burst/inputs --runs 1 --concurrent 4 --rounds 1 --queue 2 --sustain-seconds 120 --burst-jobs 24 l2-100000x768 l2-500000x384 hamming-500000x256-binary-clustered
python tools/bench/check_metrics.py target/bench/metrics
```

`fixtures/ea7db9f/metrics/` keeps that run and a burst through Prometheus, each with its pages and its service's log, so the check runs on them as they are. A scrape of a full store, 10,000 jobs held with 1,000 of them in flight and one building, is the `scrapes_a_full_store_in_under_20_ms` test of `crates/knn-builder/tests/service.rs`, on LocalStack as the README runs those tests, held to 20 ms: `cargo test --release -p knn-builder --test service scrapes_a_full_store -- --ignored --nocapture` prints its median.

See `docs/faiss-layout.md` for the layout and `docs/params.md` for the build parameters.
