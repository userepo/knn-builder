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

`peak/l2-1000000x768.json` holds one build of the 1M x 768 fp32 input (`make_inputs.py peak`) by each service, on LocalStack: the whole job through `knn-builder`'s pipeline on the GPU (its `e2e` example) and through the service, with stage times, knn-inspect `verify` and `recall`, and the building process's peak resident memory (`VmHWM`). `knn-builder` peaked at 4.24 GB, the service at 8.28 GB; both index files are 3,352,129,561 bytes and verify clean.

`grid/` holds the measurements a forecast of a job starts from, on the RTX 4070 Laptop GPU (8 GB) and its 32-thread CPU, in WSL2 against LocalStack (`tools/grid/`):

| File | What |
| --- | --- |
| `grid/runs.json` | 58 builds by `knn-builder`'s `e2e` example, each of the 29 `grid` inputs with m 16 and 32, and the warm-up builds before them: stage seconds, index file size, peak resident memory and peak device memory |
| `grid/cpu.json` | Faiss HNSW builds (M 16, ef_construction 100) of the fp32 inputs to 500k vectors, on 4 threads and, to 250k and one of 500k, on 1 |
| `grid/check.json` | five of those builds again: reading the free memory as often as it can, and with device memory held to leave the sampled peak plus a margin, then 90% of it, free |
| `grid/fits.json` | each model's terms and its relative error on the builds it was fitted to and on those it left out (the 250k- and 750k-vector builds; the CPU's 500k-vector ones) |

All 58 builds completed. The device peak model, the dataset plus the larger of an IVF-PQ phase (the PQ codes and a fixed batch of rows) and a graph phase (the finished graph and a buffer over at most 500,000 rows), is within 1.6% on the builds left out, and its worst under-estimate on any build is 2.7%; the codes and the graph are floors it only adds to, so it never forecasts less than they take, past the grid included. Reading the free memory in a tight loop found the same peaks as reading it every 5 ms. The held-memory builds prove nothing on this host: WSL2's driver pages device memory out to the host instead of refusing it, so the builds leaving 90% of the peak free completed as well.

The build time is off by 6% at the median and 33% at worst on the builds left out; d 768 at m 32 builds two to five times slower than at m 16 (79.6 s against 14.4 s at 1M vectors), as the IVF-PQ search's lookup table and top-k no longer fit a block's shared memory. Download and upload times carry LocalStack's own spread (one 1M x 384 upload took 9.9 s, its m 32 twin 3.2 s). The index file's size is exact. The CPU model is within 13% on the sizes it was fitted to and under by 14 to 23% at 500k vectors, where HNSW slows faster than n log n. On 4 threads a 500k x 768 build took 62 s. The service's part of the same remote build took 17 s, but the data node's wait is longer: at these transfer rates it uploads the vectors in about 4 s, sees the job complete at its second poll, about 20 s in (the first comes about 15 s after the request, then one every 5 s), and downloads the file in about 5 s, about 29 s in all.

`burst/` holds bursts of 48 build requests, sent at once as k-NN's data nodes would (`tools/burst/`), with a 90 s client timeout and 5 s polls, against each service on the RTX 4070 in WSL2 with LocalStack: twelve each of 1M x 128, 500k x 384, 250k x 768 and 100k x 768 fp32 vectors. Each file lists every job's outcome and when its data node had the index or built the segment itself.

| Run | Built in time | Refused | Failed | Timed out | To the CPU | Data nodes' wait before the CPU |
| --- | --- | --- | --- | --- | --- | --- |
| `knn-builder.json` | 13 | 22 | 13 | 0 | 35 | 361 s |
| `knn-builder-seed2.json` | 7 | 24 | 17 | 0 | 41 | 492 s |
| `upstream-1-worker.json` | 8 | 0 | 0 | 40 | 40 | 3,708 s |
| `upstream-1-worker-seed2.json` | 6 | 0 | 0 | 42 | 42 | 3,881 s |
| `upstream.json`, 2 workers | 0 | 0 | 48 | 0 | 48 | 4,036 s |

knn-builder's failures are the jobs it dropped early; the two-worker upstream run's, status requests that failed once the service had crashed.

The upstream service's default of two workers crashes it under WSL2, at its first two builds at once (exit 139), and every job fails; with one it builds jobs in order, and those that cannot end in time make their data nodes wait out the timeout. knn-builder refuses at once the jobs forecast to miss, and drops early, before their builds, those that the jobs ending meanwhile show will miss after all: its forecast starts from the grid's model, and uploads to LocalStack on this compose network take three to six times as long as the grid's, which it learns only from the first jobs to end. Here the uploads bound both services, which is why they build about as many in time.

`burst/knn-cadence.json` and `burst/knn-refused.json` check the driver against k-NN 3.9.0 on the cluster: with 200 ms polls, k-NN's 37 status requests came 0.49 to 0.71 s after each build request and then 0.16 to 0.24 s apart, as the driver's 0.48 to 0.72 s and 0.16 to 0.24 s; with every build refused, k-NN sent each request once, polled none, counted six build requests failed and no time waited, and built the segments itself.

Builds are not reproducible: the five files of one input share their headers and level tables but differ in their graphs, so `knn-gpu` is held to the recall spread, and `knn-write` to the oracle, which rebuilds each of these files exactly from its own graph.
