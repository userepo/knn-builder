"""Writes the synthetic build inputs: vector and doc-id blobs in the format
OpenSearch k-NN uploads for a remote build, plus held-out queries.

Each input is a directory holding
  vectors.knnvec  row-major little-endian float32 (or float16 for fp16 blobs)
  doc_ids.knndid  little-endian int32, one per vector
  queries.f32     row-major little-endian float32, for knn-inspect recall
  input.json      the build request's fields (with m, ef_construction and
                  ef_search, which the baseline runner and the oracle read
                  from here) and how the data was drawn

The doc ids are a shuffled, spread-out set rather than 0..n, so a writer
that confuses positions and ids cannot pass by accident.

Usage: python make_inputs.py {small,large,peak} OUT_DIR
  small  the committed fixtures (fixtures/inputs)
  large  the inputs for measuring recall spread (not committed)
  peak   1M x 768 fp32, for measuring peak host memory (not committed;
         3 GB of vectors, drawn in float32 chunks so writing it takes
         little memory)

Runs in the remote-vector-index-builder image, for its numpy.
"""

import json
import pathlib
import sys

import numpy as np

QUERIES = 100

# The request's HNSW parameters for every input: k-NN's defaults.
ALGORITHM_PARAMETERS = {"m": 16, "ef_construction": 100, "ef_search": 100}

# name: (seed, doc_count, dimension, space_type, request data_type,
#        blob dtype, skip_stored_vectors)
# Each input keeps its own seed, so adding an input never changes the
# data of another (and with it the committed service files).
SETS = {
    "small": {
        "ip-1000x64": (1000, 1000, 64, "innerproduct", "float", "<f4", False),
        "l2-1000x64": (1001, 1000, 64, "l2", "float", "<f4", False),
        # A half_float field: fp16 values in the blob.
        "l2-1000x64-fp16": (1002, 1000, 64, "l2", "half_float", "<f2", False),
        # A float field with the fp16 scalar quantizer: fp32 values under
        # data_type half_float, converted by the service.
        "l2-1000x64-fp16from32": (1003, 1000, 64, "l2", "half_float", "<f4", False),
        # Graph and ID map only (k-NN sets this for Faiss multi-bit SQ).
        "l2-1000x64-skip": (1004, 1000, 64, "l2", "float", "<f4", True),
        # Too few vectors for m = 16: the service lowers m (calculate_effective_m).
        "l2-150x64": (1005, 150, 64, "l2", "float", "<f4", False),
    },
    "large": {
        "ip-20000x128": (1000, 20000, 128, "innerproduct", "float", "<f4", False),
        "l2-20000x128": (1001, 20000, 128, "l2", "float", "<f4", False),
    },
    "peak": {
        "l2-1000000x768": (1006, 1_000_000, 768, "l2", "float", "<f4", False),
    },
}

# Above this many values an input's vectors are drawn in float32 chunks and
# written as they are drawn: a float64 draw of 1M x 768 alone takes 6 GB.
# The smaller sets keep their float64 draws, and so their data.
CHUNKED_VALUES = 100_000_000
CHUNK_ROWS = 10_000


def draw(rng, rows, dim, unit):
    x = rng.standard_normal((rows, dim)).astype(np.float32)
    if unit:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def draw32(rng, rows, dim, unit):
    x = rng.standard_normal((rows, dim), dtype=np.float32)
    if unit:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def write_input(out, name, spec):
    seed, doc_count, dim, space, data_type, blob_dtype, skip = spec
    rng = np.random.default_rng(seed)
    unit = space == "innerproduct"
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    if doc_count * dim > CHUNKED_VALUES:
        with (d / "vectors.knnvec").open("wb") as f:
            for start in range(0, doc_count, CHUNK_ROWS):
                rows = min(CHUNK_ROWS, doc_count - start)
                f.write(draw32(rng, rows, dim, unit).astype(blob_dtype).tobytes())
        queries = draw32(rng, QUERIES, dim, unit)
    else:
        vectors = draw(rng, doc_count, dim, unit)
        queries = draw(rng, QUERIES, dim, unit)
        (d / "vectors.knnvec").write_bytes(vectors.astype(blob_dtype).tobytes())
    doc_ids = (rng.permutation(doc_count).astype(np.int64) * 3 + 7).astype("<i4")

    (d / "doc_ids.knndid").write_bytes(doc_ids.tobytes())
    (d / "queries.f32").write_bytes(queries.astype("<f4").tobytes())
    meta = {
        "doc_count": doc_count,
        "dimension": dim,
        "space_type": space,
        "data_type": data_type,
        "blob_dtype": blob_dtype,
        "skip_stored_vectors": skip,
        **ALGORITHM_PARAMETERS,
        "queries": QUERIES,
        "seed": seed,
        "distribution": "standard normal" + (", unit length" if unit else "")
        + (", float32 draws in chunks" if doc_count * dim > CHUNKED_VALUES else ""),
        "doc_ids": "permutation of 0..doc_count, times 3 plus 7",
    }
    (d / "input.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode())


def main():
    which, out = sys.argv[1], pathlib.Path(sys.argv[2])
    for name, spec in SETS[which].items():
        write_input(out, name, spec)
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
