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

## Parity

`knn-gpu`'s `build_inputs` example builds inputs as the service does, with `knn-gpu` and `knn-write`, and writes a `runs.json` that `baseline/score.py` scores like the service's (it carries its own timings and has no log). In WSL, in the conda environment:

```sh
cargo run --release -p knn-gpu --features gpu --example build_inputs -- target/inputs-large target/rust-baseline/large20 20
```

Then, from Windows or Linux, `python tools/baseline/score.py target/rust-baseline/large20 target/inputs-large ../knn-inspect/target/release/knn-inspect`, and the service's own 20 builds with `RUNS=20` as above.

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

See `docs/faiss-layout.md` for the layout and `docs/params.md` for the build parameters.
