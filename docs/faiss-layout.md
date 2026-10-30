# The index file a remote build writes

What `knn-write` must reproduce exactly: the file the upstream remote-vector-index-builder uploads for a float, fp16 or byte field, written by `GpuIndexCagra::copyTo` and `write_index` in its Faiss, and for a binary field by `GpuIndexBinaryCagra::copyTo` and `write_index_binary` (see Binary vectors) (`0ca9df4` with rvib's `base_image/patches`). knn-inspect's `docs/format-notes.md` ("CAGRA graphs") covers how k-NN reads these files; this note covers how they are written.

Every statement below was checked on files the service built (`fixtures/service/`, `fixtures/byte/service/`, `fixtures/binary/service/`): the field values decode as stated and the sizes add up to the file's length. The oracle (`tools/oracle/oracle.py`) rebuilds an identical copy of all 46 files of the M1 baseline, of all 60 service builds of the byte inputs (five of each committed one, 20 of each 20,000 x 128 one), and of all 70 of the binary inputs (five of each committed one, 20 of each 20,000-vector one), from the CAGRA graph, the input blobs and the request alone.

## Layout

All integers are little-endian. A vector is a `u64` element count followed by the elements. A header is the `write_index_header` block (Faiss `impl/index_write.cpp:97-108`).

| Part | Field | Type | Value |
| --- | --- | --- | --- |
| ID map | fourcc | 4 bytes | `IxMp` |
| | `d` | i32 | dimension |
| | `ntotal` | i64 | n |
| | two dummies | i64, i64 | 1048576, 1048576 |
| | `is_trained` | u8 | **0**: the map wrapped the GPU index before training |
| | `metric_type` | i32 | 1 for L2, 0 for inner product (no `metric_arg` follows for either) |
| graph | fourcc | 4 bytes | `IHc2` |
| | header | as above | `d`, n, dummies, `is_trained` 1, metric |
| | `keep_max_size_level0` | u8 | 1 |
| | `base_level_only` | u8 | 1 |
| | `num_base_level_search_entrypoints` | i32 | 32 |
| | `numeric_type_` | i32 | 0 for float32, 1 for fp16, 3 for int8 (Faiss's `NumericType::Int8`) |
| HNSW | `assign_probas` | vector of f64 | see Levels |
| | `cum_nneighbor_per_level` | vector of i32 | 0, 2M, 3M, 4M, ... one more entry than `assign_probas` |
| | `levels` | vector of i32, n entries | each node's level + 1, see Levels |
| | `offsets` | vector of u64, n + 1 entries | 0, then running sums of `cum_nneighbor_per_level[levels[i]]` |
| | `neighbors` | vector of i32, `offsets[n]` entries | level 0 of node i at `offsets[i]`, its 2M CAGRA neighbors in cuVS's order; every other slot -1 |
| | `entry_point`, `max_level` | i32, i32 | -1, -1: `copyTo` resets them and never sets them |
| | `efConstruction`, `efSearch` | i32, i32 | the request's `ef_construction`, `ef_search` |
| | `upper_beam` | i32 | 1 (a constant; the field is deprecated) |
| storage, float32 | fourcc | 4 bytes | `IxF2` for L2, `IxFI` for inner product |
| | header | as above | `d`, n, dummies, `is_trained` 1, metric |
| | vectors | u64 count of floats (n x d), then the floats | the input vectors, row-major, in input order |
| storage, fp16 | fourcc | 4 bytes | `IxSQ` |
| | header | as above | `d`, n, dummies, `is_trained` 1, metric |
| | `qtype`, `rangestat` | i32, i32 | 4 (`QT_fp16`), 0 |
| | `rangestat_arg` | f32 | 0.0 |
| | `d`, `code_size` | u64, u64 | dimension, 2 x dimension |
| | `trained` | vector of f32 | empty |
| | `codes` | vector of u8, n x 2d | the fp16 values, row-major, in input order |
| storage, byte | fourcc | 4 bytes | `IxSQ` |
| | header | as above | `d`, n, dummies, `is_trained` 1, metric |
| | `qtype`, `rangestat` | i32, i32 | 8 (`QT_8bit_direct_signed`), 0 |
| | `rangestat_arg` | f32 | 0.0 |
| | `d`, `code_size` | u64, u64 | dimension, dimension |
| | `trained` | vector of f32 | empty |
| | `codes` | vector of u8, n x d | each int8 value plus 128, row-major, in input order: `copyTo` encodes them so before `add_sa_codes` |
| storage, `skip_stored_vectors` | fourcc | 4 bytes | `null`, and nothing else; the graph header still records n |
| ID map | `id_map` | vector of i64, n entries | the doc ids from the `.knndid` blob, widened from i32, in input order |

M is half the CAGRA graph degree (`copyTo`: `M = graph_degree / 2`), so level 0 holds exactly the graph's row; the rvib degree, m x 4, is always even. The ids come last, after the whole index, so a writer streams the graph and vectors before it needs them.

For an fp16 field built from an fp32 blob (a float field with the fp16 quantizer), the codes are the service's conversion on download: numpy `astype(float16)`, round to nearest even.

For n = 1,000 at degree 64 (M = 32), with 864 empty upper-level slots: 37 + 47 + 56 + 36 + 4,008 + 8,016 + (8 + 4 x 64,864) + 20 + (4 + 33 + 8 + 256,000) + 8,008 = 535,737 bytes, the size of `fixtures/service/l2-1000x64.faiss`.

## Binary vectors

A binary field's file has the same HNSW block between other headers and storage. A binary header (`write_index_binary_header`) is `d` (i32, in bits), `code_size` (i32, d / 8), `ntotal` (i64), `is_trained` (u8) and `metric_type` (i32): 21 bytes, with no dummies and no `metric_arg`. `metric_type` is 1 in all three, `METRIC_L2`, the default a binary index keeps whatever its distance.

| Part | Field | Type | Value |
| --- | --- | --- | --- |
| ID map | fourcc, header | 4 bytes, binary | `IBMp`; `is_trained` **0** |
| graph | fourcc, header | 4 bytes, binary | `IBHc`; `is_trained` 1 |
| | `keep_max_size_level0` | u8 | **0**: `copyTo` sets it false when done |
| | `base_level_only` | u8 | 1 |
| | `num_base_level_search_entrypoints` | i32 | 32; no numeric type follows |
| HNSW | as above | | M is half the graph's degree, truncated |
| storage | fourcc, header | 4 bytes, binary | `IBxF`; `is_trained` 1 |
| | `xb` | vector of u8, n x d / 8 | the packed bits, row-major, in input order |
| storage, `skip_stored_vectors` | fourcc | 4 bytes | `null`: rvib's patch 0001 passes the flag through `write_index_binary` too. k-NN never sets it for a binary field, and its loader takes a binary CAGRA graph only over its vectors |
| ID map | `id_map` | vector of i64, n entries | the doc ids, widened from i32, in input order |

cuVS can return an odd degree for a binary build, lowered to one less than the vector count (39 for 40 vectors at m 16). `copyTo` then takes M = 19 and fills level 0's 38 slots with the first 38 of each node's 39 neighbors.

## Levels

Every node gets an HNSW level, even though nothing but level 0 is used, and the levels decide `levels`, `offsets` and the length of `neighbors`. They come from Faiss's own generator, so a writer must reproduce it exactly:

1. `copyTo` runs on a freshly constructed `IndexHNSWCagra`, whose `HNSW` seeded its generator, `std::mt19937`, with 12345 (`impl/HNSW.cpp:59`) and has drawn nothing.
2. `set_default_probas(M, 1.0 / log(M))` (`impl/HNSW.cpp:77-89`): `levelMult` is computed in double and passed as a **float**. For each level from 0, `float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult))`: both quotients are float divisions, `exp` is C's double `exp`, the product is in double, and it is rounded once to the float `proba`; stop at the first `proba < 1e-9`. Each kept `proba` is widened to double and appended to `assign_probas`, and `cum_nneighbor_per_level` gains 2M for level 0 and M for every level after. Computing the exponentials in float, or keeping the product in double, misses by an ULP on some levels.
3. `prepare_level_tab(n, false)` (`impl/HNSW.cpp:207-230`): for each node in order, `random_level()` (`impl/HNSW.cpp:64-75`) draws `f = mt() / float(mt.max())` in **single precision** (`utils/random.cpp:49-51`), then walks the levels: return the first level with `f < assign_probas[level]`, otherwise subtract it (in double) and go on; past the last, return the last. The node's entry in `levels` is that level + 1.

The same n and M give the same levels on every build, which is why the files' first bytes agree across runs while the graphs differ. For M = 32 the probabilities are 0.96875, 0.0302734..., six levels in all; the 20,000-vector baseline files hold levels 1 to 4 (19,372, 614, 13 and 1 nodes).

`tools/oracle/levels.py` computes the whole table this way, without Faiss, and matches every baseline file. A port must match the C library's double `exp` on these few arguments (the files were written with glibc) and the float32 rounding of the 32-bit draws, bit for bit. `knn-write` computes the probabilities at run time; its tests hold them to glibc's values, from `tools/oracle/level_tables.py`, for every even M from 2 to 4,096 (a field's m up to 2,048) on Linux and Windows, and hold the draws to level histograms over 2,100,000 nodes, far enough to reach nodes where a draw computed in double instead of float lands on another level (the first is node 1,089,903 for M = 2). k-NN accepts any m above 0; a larger M is computed the same way but unverified.

## Order of the graph rows

cuVS returns the graph as an n x degree array of `uint32`, row i for the i-th input vector; `copyTo` copies row i, in cuVS's order, to node i's level-0 slots (`GpuIndexCagra.cu`, end of `copyTo`). Node i is the i-th vector of the input and the i-th entry of `id_map`; the service never reorders.

The ids pass through unchecked: `get_knngraph` widens cuVS's `uint32` ids to 64-bit `idx_t` (`impl/CuvsCagra.cu:349-369`), and `copyTo` stores them in Faiss's 32-bit signed neighbor slots, which keep the low 32 bits. An id from n up to 2^31 - 1 would land in the file as an invalid neighbor, and a larger one as a negative value, `u32::MAX` as -1, Faiss's empty slot. No baseline build held such an id. `knn-write` refuses one instead (`Error::NeighborOutOfRange`): the build fails and k-NN builds the segment on CPU.
