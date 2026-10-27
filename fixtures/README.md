# Fixtures

Synthetic inputs and the files the upstream service built from them, at commit 3a02e64 on an RTX 4070 Laptop GPU. Every value is drawn from a fixed seed; see `tools/README.md` to regenerate them.

| Path | What |
| --- | --- |
| `inputs/<name>/` | one build input: `vectors.knnvec`, `doc_ids.knndid`, `queries.f32` (100 held-out queries), `input.json` (the request's fields and the seed) |
| `service/<name>.faiss` | the first of five service builds of that input |
| `service/baseline-small.json` | all five builds of every input: per-step seconds from the service's DEBUG log, size, sha256, knn-inspect `verify` status and recall@10 |
| `service/baseline-large.json` | the same for two 20,000 x 128 inputs, which are not committed (`make_inputs.py large`) |

| Input | Vectors | Space | Request | Notes |
| --- | --- | --- | --- | --- |
| `l2-1000x64` | 1,000 x 64 | L2 | `float` | |
| `ip-1000x64` | 1,000 x 64, unit length | inner product | `float` | |
| `l2-150x64` | 150 x 64 | L2 | `float` | too small for m = 16: the service lowers m to 7 (degree 28) |
| `l2-1000x64-fp16` | 1,000 x 64 | L2 | `half_float` | an fp16 blob |
| `l2-1000x64-fp16from32` | 1,000 x 64 | L2 | `half_float` | an fp32 blob, converted by the service |
| `l2-1000x64-skip` | 1,000 x 64 | L2 | `float`, `skip_stored_vectors` | graph and ID map only |

`parity/` holds the recall of `knn-gpu` and `knn-write` against the service's, each over its own builds of the same inputs:

| File | What |
| --- | --- |
| `parity/service-20000x128.json` | 20 service builds of each 20,000 x 128 input, scored as `baseline-large.json`: the recall spread `knn-gpu` is held to |
| `parity/knn-gpu-20000x128.json` | 20 builds of the same inputs by `knn-gpu`'s `build_inputs` example |
| `parity/knn-gpu-small.json` | 5 builds of every committed input by the same example |
| `parity/c-api-20000x128.json` | 20 builds of the 20,000 x 128 inputs by an earlier `knn-gpu` that called cuVS's C API, which cannot pass two IVF-PQ search settings (`docs/params.md`) |

Builds are not reproducible: the five files of one input share their headers and level tables but differ in their graphs, so `knn-gpu` is held to the recall spread, and `knn-write` to the oracle, which rebuilds each of these files exactly from its own graph.
