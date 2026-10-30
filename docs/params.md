# CAGRA build parameters

The parameters the upstream remote-vector-index-builder builds a float, fp16 or byte CAGRA graph with, as `knn-gpu` must reproduce them, and where each comes from. Sources are pinned:

- **rvib**: [remote-vector-index-builder](https://github.com/opensearch-project/remote-vector-index-builder) at `3a02e64`, paths under `remote_vector_index_builder/core/`.
- **Faiss**: Faiss `0ca9df4` with the three patches in rvib's `base_image/patches` applied; paths under `faiss/gpu/`.

Any difference shows up as recall drift, measured against the spread of the service's own builds in `fixtures/parity/service-20000x128.json`.

## Per request

Computed from the request's `doc_count`, `dimension` and `m` (`index_builder/faiss/faiss_index_build_service.py`, `build_index`).

| Parameter | Rule | Source |
| --- | --- | --- |
| `n_lists` | `int(math.sqrt(doc_count))`: a correctly rounded double square root, truncated | `index_builder/index_builder_utils.py:44`, used at `faiss_index_build_service.py:104` |
| effective m | `n_lists' = max(n_lists, 1)`; `candidates = min(n_probes, n_lists') * doc_count // n_lists'`; `max_degree = min(candidates // 2, doc_count - 1)`; `m = max(1, min(m, max_degree // 4))`, with `n_probes` = 5 below | `index_builder_utils.py:47-71`, used at `faiss_index_build_service.py:105` |
| `graph_degree` | effective m x 4 (`GRAPH_DEGREE_PER_M`) | `index_builder_utils.py:18`, `faiss_index_build_service.py:114` |
| `intermediate_graph_degree` | effective m x 4, equal to `graph_degree` | `faiss_index_build_service.py:115` |
| `pq_dim` | `int(dimension / 4)` (`PQ_DIM_COMPRESSION_FACTOR`); 0 below 4 dimensions, which leaves the choice to cuVS | `faiss_index_build_service.py:50, 109-111` |
| metric | `l2` gives cuVS `L2Expanded`, `innerproduct` gives `InnerProduct` | `index_builder_utils.py:84-88`; Faiss `utils/CuvsUtils.h:36-43` (`metricFaissToCuvs(metric, false)`) |
| numeric type | `float` builds over float32, `half_float` over float16 (an fp32 blob is converted to fp16 on download), `byte` over int8, with every other parameter as for float | `common/models/index_builder/faiss/faiss_gpu_index_cagra_builder.py:179-186` |
| `efConstruction`, `efSearch` | the request's `ef_construction` and `ef_search` (defaults 100 and 100), written into the file, not used by the build | `common/models/index_builder/faiss/faiss_index_hnsw_cagra_builder.py:30-33, 72-73` |

Example: 150 vectors with m = 16 give `n_lists` 12, 62 candidates, a maximum degree of 31 and an effective m of 7, so degree 28; the service logs "Using m=7 instead of m=16". 1,000 vectors keep m = 16 (degree 64).

Binary vectors take another path, an NN-descent build with no IVF-PQ settings (`faiss_index_build_service.py:117-124`), which this page does not cover yet.

## Fixed

Set by rvib on every build; where rvib leaves a Faiss field alone, Faiss's default is listed, since it differs from cuVS's in places.

| Parameter | Value | Set by |
| --- | --- | --- |
| graph build algorithm | IVF-PQ | rvib `faiss_gpu_index_cagra_builder.py:40, 124-132` |
| `kmeans_n_iters` | 20 | rvib `faiss/ivf_pq_build_cagra_config.py:23` |
| `kmeans_trainset_fraction` | 0.1 (Faiss's default is 0.5) | rvib `ivf_pq_build_cagra_config.py:25` |
| `pq_bits` | 8 | rvib `ivf_pq_build_cagra_config.py:31` |
| `codebook_kind` | `PER_SUBSPACE` (Faiss default) | Faiss `GpuIndexCagra.h:95` |
| `force_random_rotation` | true (Faiss's default is false) | rvib `ivf_pq_build_cagra_config.py:56` |
| `conservative_memory_allocation` | true (Faiss's default is false) | rvib `ivf_pq_build_cagra_config.py:54` |
| IVF-PQ search `n_probes` | 5 (Faiss's default is 20) | rvib `faiss/ivf_pq_search_cagra_config.py:18` |
| IVF-PQ search `lut_dtype` | `CUDA_R_32F` (Faiss default) | Faiss `GpuIndexCagra.h:142` |
| IVF-PQ search `preferred_shmem_carveout` | 1.0 (Faiss default) | Faiss `GpuIndexCagra.h:169` |
| IVF-PQ search `max_internal_batch_size` | 4096 (Faiss default) | Faiss `GpuIndexCagra.h:173` |
| `refinement_rate` | 1.0 (Faiss's default is 2.0) | rvib `faiss_gpu_index_cagra_builder.py:44` |
| `guarantee_connectivity` | false (Faiss default) | Faiss `GpuIndexCagra.h:192` |
| `store_dataset` | false; the dataset comes from host memory and Faiss copies it to a padded device dataset | rvib `faiss_gpu_index_cagra_builder.py:42`; Faiss `impl/CuvsCagra.cu:214-223` |
| GPU temporary memory | none reserved (`StandardGpuResources.noTempMemory()`) | rvib `faiss_gpu_index_cagra_builder.py:255` |

The intermediate degree is not widened. Unpatched Faiss raises `intermediate_graph_degree` to 1.5 x `graph_degree` when the two are equal; rvib's patch 0002 removes that, so cuVS receives them equal. Calling cuVS directly, as `knn-gpu` does, never widens.

Faiss copies each field above into a default-constructed `cuvs::neighbors::ivf_pq::index_params` and `search_params` (`GpuIndexCagra.cu:63-92`) and into `cagra::index_params` (`impl/CuvsCagra.cu:79-87, 187-195`); every other cuVS field keeps the C++ default of libcuvs 26.08.01. Fields Faiss does not pass include the IVF-PQ search's `internal_distance_dtype` (its own config has one, `GpuIndexCagra.h:151`, but `train_ex` drops it) and `coarse_search_dtype`, so cuVS's defaults apply, `CUDA_R_32F` for both.

## Through cuVS's C++ API

`knn-gpu` reaches cuVS through a C++ shim over the C++ API Faiss uses (`crates/knn-gpu/cpp/knn_cagra.cpp`) and builds the parameters as Faiss does: default-constructed `ivf_pq::index_params`, `ivf_pq::search_params` and `cagra::index_params`, then every field Faiss sets, to the values above. Its dataset path is Faiss's for host vectors: `make_device_padded_dataset` from a host view, then `cagra::build` on that dataset's view (`impl/CuvsCagra.cu:214-223`). Its raft handle is a default-constructed `raft::resources`, as cuVS's C API creates one, where Faiss's comes from `StandardGpuResources`. The shim compiles only against libcuvs 26.08.01 (`cuvs/version_config.h`); another version means checking this page again.

cuVS's C API, which `knn-gpu` called first, cannot pass two of the settings. Its CAGRA build starts from `graph_build_params::ivf_pq_params(extents, metric)` (`cpp/include/cuvs/neighbors/ivf_pq.hpp:3384`) and copies only `n_probes`, `lut_dtype`, `internal_distance_dtype` and `preferred_shmem_carveout` from the C struct (`c/src/neighbors/cagra.cpp:641-647`); the other two keep that constructor's values (`ivf_pq.hpp:3443-3444`). Paths are in cuVS's source at tag v26.08.01 (commit 25b1be4, the commit libcuvs 26.08.01's conda build names).

| Setting | Faiss, so the service and `knn-gpu` | cuVS's C API |
| --- | --- | --- |
| `coarse_search_dtype` | `CUDA_R_32F` (C++ default) | `CUDA_R_16F` |
| `max_internal_batch_size` | 4096 | 131072 |

The first computes the distances to the coarse cluster centers in half precision, which can change which lists a borderline vector probes; the second sets how many queries the search takes at a time. Together they showed in recall, over 20 builds of each 20,000 x 128 input per side (`fixtures/parity/`); the shim changed both at once, so how much each one contributes is not measured. Through the C API, recall fell on levels offset from the service's, and 9 of 20 L2 builds fell outside the service's range, where the service's own second ten builds have none outside its first ten. Through the shim, recall falls on the service's levels, and the means are within 0.001 of the service's:

| Input | Service: mean [range] | Through the C API | Through the shim |
| --- | --- | --- | --- |
| `l2-20000x128` | 0.7695 [0.758, 0.778] | 0.7679 [0.757, 0.780], 9 outside | 0.7701 [0.759, 0.779], 1 outside |
| `ip-20000x128` | 0.7734 [0.758, 0.796] | 0.7701 [0.759, 0.795], none outside | 0.7728 [0.760, 0.796], none outside |
