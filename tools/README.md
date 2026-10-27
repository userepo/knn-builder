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

See `docs/faiss-layout.md` for the layout and `docs/params.md` for the build parameters.
