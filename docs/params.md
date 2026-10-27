# CAGRA build parameters

The parameters the upstream remote-vector-index-builder builds a float or fp16 CAGRA graph with, as `knn-gpu` must reproduce them, and where each comes from. Sources are pinned:

- **rvib**: [remote-vector-index-builder](https://github.com/opensearch-project/remote-vector-index-builder) at `3a02e64`, paths under `remote_vector_index_builder/core/`.
- **Faiss**: Faiss `0ca9df4` with the three patches in rvib's `base_image/patches` applied; paths under `faiss/gpu/`.

Any difference shows up as recall drift, measured against the spread in `fixtures/service/baseline-large.json`.

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
| numeric type | `float` builds over float32, `half_float` over float16 (an fp32 blob is converted to fp16 on download) | `common/models/index_builder/faiss/faiss_gpu_index_cagra_builder.py:179-186` |
| `efConstruction`, `efSearch` | the request's `ef_construction` and `ef_search` (defaults 100 and 100), written into the file, not used by the build | `common/models/index_builder/faiss/faiss_index_hnsw_cagra_builder.py:30-33, 72-73` |

Example: 150 vectors with m = 16 give `n_lists` 12, 62 candidates, a maximum degree of 31 and an effective m of 7, so degree 28; the service logs "Using m=7 instead of m=16". 1,000 vectors keep m = 16 (degree 64).

The binary path (NN-descent, `faiss_index_build_service.py:117-124`) is outside v1.

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

Faiss copies each field above into a default-constructed `cuvs::neighbors::ivf_pq::index_params` and `search_params` (`GpuIndexCagra.cu:63-92`) and into `cagra::index_params` (`impl/CuvsCagra.cu:79-87, 187-195`); every other cuVS field keeps the C++ default of libcuvs 26.08.01. Fields Faiss does not pass include the IVF-PQ search's `internal_distance_dtype` (its own config has one, `GpuIndexCagra.h:151`, but `train_ex` drops it), so cuVS's default applies.

## To check in M3

`knn-gpu` reaches cuVS through its C API (the `cuvs` crate), whose parameter constructors may default differently from the C++ structs Faiss default-constructs. M3 compares every field of the C API's IVF-PQ index and search parameters and CAGRA index parameters, as created, with the C++ defaults, and sets each one that differs.
