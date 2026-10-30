# Fixtures

Synthetic inputs and the files the upstream service built from them, at commit 3a02e64 on an RTX 4070 Laptop GPU. Every value is drawn from a fixed seed; see `tools/README.md` to regenerate them.

| Path | What |
| --- | --- |
| `inputs/<name>/` | one build input: `vectors.knnvec`, `doc_ids.knndid`, `queries.f32` (100 held-out queries), `input.json` (the request's fields and the seed) |
| `service/<name>.faiss` | the first of five service builds of that input |
| `service/baseline-small.json` | all five builds of every input: per-step seconds from the service's DEBUG log, size, sha256, knn-inspect `verify` status and recall@10 |
| `service/baseline-large.json` | the same for two 20,000 x 128 inputs, which are not committed (`make_inputs.py large`) |
| `byte/inputs/<name>/`, `byte/service/` | the same for byte vectors: int8 values drawn uniformly over -128 to 127 (`make_inputs.py byte`), with `byte/service/baseline-small.json` |

| Input | Vectors | Space | Request | Notes |
| --- | --- | --- | --- | --- |
| `l2-1000x64` | 1,000 x 64 | L2 | `float` | |
| `ip-1000x64` | 1,000 x 64, unit length | inner product | `float` | |
| `l2-150x64` | 150 x 64 | L2 | `float` | too small for m = 16: the service lowers m to 7 (degree 28) |
| `l2-1000x64-fp16` | 1,000 x 64 | L2 | `half_float` | an fp16 blob |
| `l2-1000x64-fp16from32` | 1,000 x 64 | L2 | `half_float` | an fp32 blob, converted by the service |
| `l2-1000x64-skip` | 1,000 x 64 | L2 | `float`, `skip_stored_vectors` | graph and ID map only |
| `byte/l2-1000x64-byte` | 1,000 x 64 | L2 | `byte` | |
| `byte/ip-1000x64-byte` | 1,000 x 64 | inner product | `byte` | not unit length: byte vectors cannot be scaled |
| `byte/l2-150x64-byte` | 150 x 64 | L2 | `byte` | the service lowers m to 7, as for float |
| `byte/l2-1000x64-byte-skip` | 1,000 x 64 | L2 | `byte`, `skip_stored_vectors` | graph and ID map only; k-NN never sets it for byte, but the service takes it |

`parity/` holds the recall of `knn-gpu` and `knn-write` against the service's, each over its own builds of the same inputs:

| File | What |
| --- | --- |
| `parity/service-20000x128.json` | 20 service builds of each 20,000 x 128 input, scored as `baseline-large.json`: the recall spread `knn-gpu` is held to |
| `parity/service-byte-20000x128.json` | the same for two 20,000 x 128 byte inputs (`make_inputs.py byte-large`) |
| `parity/knn-gpu-20000x128.json` | 20 builds of the same inputs by `knn-gpu`'s `build_inputs` example |
| `parity/knn-gpu-small.json` | 5 builds of every committed input by the same example |
| `parity/knn-gpu-byte-20000x128.json`, `parity/knn-gpu-byte-small.json` | the same for the byte inputs: 20 builds of each 20,000 x 128 one, 5 of each committed one |
| `parity/c-api-20000x128.json` | 20 builds of the 20,000 x 128 inputs by an earlier `knn-gpu` that called cuVS's C API, which cannot pass two IVF-PQ search settings (`docs/params.md`) |

`peak/l2-1000000x768.json` holds one build of the 1M x 768 fp32 input (`make_inputs.py peak`) by each service, on LocalStack: the whole job through `knn-builder`'s pipeline on the GPU (its `e2e` example) and through the service, with stage times, knn-inspect `verify` and `recall`, and the building process's peak resident memory (`VmHWM`). `knn-builder` peaked at 4.24 GB, the service at 8.28 GB; both index files are 3,352,129,561 bytes and verify clean.

`grid/` holds the measurements a forecast of a job starts from, on the RTX 4070 Laptop GPU (8 GB) and its 32-thread CPU, in WSL2 against LocalStack (`tools/grid/`):

| File | What |
| --- | --- |
| `grid/runs.json` | 58 builds by `knn-builder`'s `e2e` example, each of the 29 `grid` inputs with m 16 and 32, and the warm-up builds before them: stage seconds, index file size, peak resident memory and peak device memory |
| `grid/cpu.json` | Faiss HNSW builds (M 16, ef_construction 100) of the fp32 inputs to 500k vectors, on 4 threads and, to 250k and one of 500k, on 1 |
| `grid/check.json` | five of those builds again: reading the free memory as often as it can, and with device memory held to leave the sampled peak plus a margin, then 90% of it, free |
| `grid/fits.json` | each model's terms and its relative error on the builds it was fitted to and on those it left out (the 250k- and 750k-vector builds; the CPU's 500k-vector ones) |
| `grid/byte-runs.json` | 24 builds of the `grid-byte` inputs (100k to 1M byte vectors of dimension 128, 384 and 768) with m 16 and 32, which the models were not fitted to, recorded as `runs.json`'s |
| `grid/byte-cpu.json` | Faiss HNSW builds of the same inputs to 500k vectors over `SQ8_direct_signed`, as k-NN builds a byte field, on the same thread counts as `cpu.json` |

All 58 builds completed. The device peak model, the dataset plus the larger of an IVF-PQ phase (the PQ codes and a fixed batch of rows) and a graph phase (the finished graph and a buffer over at most 500,000 rows), is within 1.6% on the builds left out, and its worst under-estimate on any build is 2.7%; the codes and the graph are floors it only adds to, so it never forecasts less than they take, past the grid included. Reading the free memory in a tight loop found the same peaks as reading it every 5 ms. The held-memory builds prove nothing on this host: WSL2's driver pages device memory out to the host instead of refusing it, so the builds leaving 90% of the peak free completed as well.

The build time is off by 6% at the median and 33% at worst on the builds left out; d 768 at m 32 builds two to five times slower than at m 16 (79.6 s against 14.4 s at 1M vectors), as the IVF-PQ search's lookup table and top-k no longer fit a block's shared memory. Download and upload times carry LocalStack's own spread (one 1M x 384 upload took 9.9 s, its m 32 twin 3.2 s). The index file's size is exact. The CPU model is within 13% on the sizes it was fitted to and under by 14 to 23% at 500k vectors, where HNSW slows faster than n log n. On 4 threads a 500k x 768 build took 62 s. The service's part of the same remote build took 17 s, but the data node's wait is longer: at these transfer rates it uploads the vectors in about 4 s, sees the job complete at its second poll, about 20 s in (the first comes about 15 s after the request, then one every 5 s), and downloads the file in about 5 s, about 29 s in all.

All 24 byte builds completed, and the models hold for them with one change. The device peak model misses each byte build by what it misses the float build of the same shape, to within 1 MiB where it under-estimates: 0.8% of the peak at the median, and at worst 3.0% under, of a peak smaller by the quarter-size dataset (38 to 76% of the float build's), which the 6% margin still covers. A byte build takes as long as a float one, 0.99 to 1.25 times on this grid, so the build time counts a byte value as 4 bytes, not 1: it is then off by 9% at the median and 29% at worst, against 13% and 44% on the float builds it was fitted to. Download and upload times are within 12% and 38%, the file size within 0.006%. A byte HNSW build on the CPU takes 0.91 to 1.16 times the float one, so the CPU model is the float one, here within 13% (1 thread) and 17% (4 threads) at the median and under by up to 20% and 25% from 250k vectors.

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

The service builds a byte request as it builds a float one, through IVF-PQ with the same parameters, over int8 vectors: the file is the ID map over `IndexHNSWCagra` with numeric type Int8 and an `IndexScalarQuantizer` of `QT_8bit_direct_signed` codes (`IxMp` > `IHc2` > `IxSQ`), or a `null` storage with `skip_stored_vectors`. All 20 small and 40 large byte builds completed and verify clean; recall@10 is 1.0 on every 1,000- and 150-vector build, and 0.770 to 0.779 (L2) and 0.767 to 0.782 (inner product) over the 20,000 x 128 ones, beside 0.758 to 0.778 and 0.758 to 0.796 over 20 builds of each float input (`parity/`).

Builds are not reproducible: the five files of one input share their headers and level tables but differ in their graphs, so `knn-gpu` is held to the recall spread, and `knn-write` to the oracle, which rebuilds each file under `service/` and `byte/service/` exactly from its own graph.

Over the byte inputs, no `knn-gpu` build's recall falls outside the service's range, and the means differ by less than a standard error: 0.7743 against 0.7750 (L2) and 0.7740 against 0.7745 (inner product) over 20 builds each. The oracle rebuilds every byte file `knn-write` wrote from its own graph exactly.
