"""Writes the synthetic build inputs: vector and doc-id blobs in the format
OpenSearch k-NN uploads for a remote build, plus held-out queries.

Each input is a directory holding
  vectors.knnvec  row-major little-endian float32 (float16 for fp16 blobs,
                  int8 for byte ones), or for a binary input packed bits,
                  dimension / 8 bytes a vector, as k-NN uploads them
  doc_ids.knndid  little-endian int32, one per vector
  queries.f32     row-major little-endian float32, for knn-inspect recall
                  (whole numbers for a byte input, as k-NN takes its
                  queries); a binary input has queries.u8, packed bits
  input.json      the build request's fields (with m, ef_construction and
                  ef_search, which the baseline runner and the oracle read
                  from here) and how the data was drawn

The doc ids are a shuffled, spread-out set rather than 0..n, so a writer
that confuses positions and ids cannot pass by accident.

Usage: python make_inputs.py SET OUT_DIR [NAME...], SET one of
  small  the committed fixtures (fixtures/inputs)
  large  the inputs for measuring recall spread (not committed)
  peak   1M x 768 fp32, for measuring peak host memory (not committed;
         3 GB of vectors, drawn in float32 chunks so writing it takes
         little memory)
  grid   100k, 250k, 500k and 1M vectors of dimension 128, 384 and 768,
         fp32 and fp16, with 750k fp32 at each dimension and 2M fp32 at
         128 and 384, for timing the stages and measuring device memory
         (not committed; 22 GB in all, so tools/grid/run_grid.py writes
         one at a time), and a small input to warm the GPU up with
  byte        the committed byte fixtures (fixtures/byte/inputs): int8
              values drawn uniformly over -128..127
  byte-large  byte inputs for measuring recall spread (not committed)
  grid-byte   byte inputs of 100k, 250k, 500k and 1M vectors of dimension
              128, 384 and 768, for checking the grid's models on byte
              builds (not committed), and a small one to warm up with
  binary        the committed binary fixtures (fixtures/binary/inputs),
                in the hamming space, of 1,000 vectors and the two small
                counts the service builds without lowering m
  binary-large  binary inputs for measuring recall spread (not committed)
  binary-sizes  binary inputs of 100k, 500k and 1M vectors of 256 and
                1,024 bits, for a first look at the build's costs (not
                committed)
  grid-binary   clustered binary inputs of 100k to 2M vectors of 256 and
                1,024 bits, and 512 at 100k, 500k and 1M, for fitting the
                NN-descent build's models (not committed), and a small one
                to warm up with
With NAMEs, only those inputs of the set.

A binary input's bits are uniform, or, for a name ending in -clustered,
drawn around centers: one center of uniform bits for every 50 vectors,
each vector and query a center's bits with each bit flipped with
probability 0.1, so a graph has neighbors to find.

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
    "grid": {
        # Seeds by name, as everywhere here; the 750k and 2M inputs pin the
        # device peak between and past the others.
        "l2-10000x128": (2000, 10_000, 128, "l2", "float", "<f4", False),
        "l2-100000x128": (2001, 100_000, 128, "l2", "float", "<f4", False),
        "l2-100000x128-fp16": (2002, 100_000, 128, "l2", "half_float", "<f2", False),
        "l2-100000x384": (2003, 100_000, 384, "l2", "float", "<f4", False),
        "l2-100000x384-fp16": (2004, 100_000, 384, "l2", "half_float", "<f2", False),
        "l2-100000x768": (2005, 100_000, 768, "l2", "float", "<f4", False),
        "l2-100000x768-fp16": (2006, 100_000, 768, "l2", "half_float", "<f2", False),
        "l2-250000x128": (2007, 250_000, 128, "l2", "float", "<f4", False),
        "l2-250000x128-fp16": (2008, 250_000, 128, "l2", "half_float", "<f2", False),
        "l2-250000x384": (2009, 250_000, 384, "l2", "float", "<f4", False),
        "l2-250000x384-fp16": (2010, 250_000, 384, "l2", "half_float", "<f2", False),
        "l2-250000x768": (2011, 250_000, 768, "l2", "float", "<f4", False),
        "l2-250000x768-fp16": (2012, 250_000, 768, "l2", "half_float", "<f2", False),
        "l2-500000x128": (2013, 500_000, 128, "l2", "float", "<f4", False),
        "l2-500000x128-fp16": (2014, 500_000, 128, "l2", "half_float", "<f2", False),
        "l2-500000x384": (2015, 500_000, 384, "l2", "float", "<f4", False),
        "l2-500000x384-fp16": (2016, 500_000, 384, "l2", "half_float", "<f2", False),
        "l2-500000x768": (2017, 500_000, 768, "l2", "float", "<f4", False),
        "l2-500000x768-fp16": (2018, 500_000, 768, "l2", "half_float", "<f2", False),
        "l2-1000000x128": (2019, 1_000_000, 128, "l2", "float", "<f4", False),
        "l2-1000000x128-fp16": (2020, 1_000_000, 128, "l2", "half_float", "<f2", False),
        "l2-1000000x384": (2021, 1_000_000, 384, "l2", "float", "<f4", False),
        "l2-1000000x384-fp16": (2022, 1_000_000, 384, "l2", "half_float", "<f2", False),
        "l2-1000000x768": (2023, 1_000_000, 768, "l2", "float", "<f4", False),
        "l2-1000000x768-fp16": (2024, 1_000_000, 768, "l2", "half_float", "<f2", False),
        "l2-750000x128": (2025, 750_000, 128, "l2", "float", "<f4", False),
        "l2-750000x384": (2026, 750_000, 384, "l2", "float", "<f4", False),
        "l2-750000x768": (2027, 750_000, 768, "l2", "float", "<f4", False),
        "l2-2000000x128": (2028, 2_000_000, 128, "l2", "float", "<f4", False),
        "l2-2000000x384": (2029, 2_000_000, 384, "l2", "float", "<f4", False),
    },
    "byte": {
        "ip-1000x64-byte": (3000, 1000, 64, "innerproduct", "byte", "<i1", False),
        "l2-1000x64-byte": (3001, 1000, 64, "l2", "byte", "<i1", False),
        "l2-1000x64-byte-skip": (3002, 1000, 64, "l2", "byte", "<i1", True),
        "l2-150x64-byte": (3003, 150, 64, "l2", "byte", "<i1", False),
    },
    "byte-large": {
        "ip-20000x128-byte": (3004, 20000, 128, "innerproduct", "byte", "<i1", False),
        "l2-20000x128-byte": (3005, 20000, 128, "l2", "byte", "<i1", False),
    },
    "grid-byte": {
        "l2-10000x128-byte": (3010, 10_000, 128, "l2", "byte", "<i1", False),
        "l2-100000x128-byte": (3011, 100_000, 128, "l2", "byte", "<i1", False),
        "l2-100000x384-byte": (3012, 100_000, 384, "l2", "byte", "<i1", False),
        "l2-100000x768-byte": (3013, 100_000, 768, "l2", "byte", "<i1", False),
        "l2-250000x128-byte": (3014, 250_000, 128, "l2", "byte", "<i1", False),
        "l2-250000x384-byte": (3015, 250_000, 384, "l2", "byte", "<i1", False),
        "l2-250000x768-byte": (3016, 250_000, 768, "l2", "byte", "<i1", False),
        "l2-500000x128-byte": (3017, 500_000, 128, "l2", "byte", "<i1", False),
        "l2-500000x384-byte": (3018, 500_000, 384, "l2", "byte", "<i1", False),
        "l2-500000x768-byte": (3019, 500_000, 768, "l2", "byte", "<i1", False),
        "l2-1000000x128-byte": (3020, 1_000_000, 128, "l2", "byte", "<i1", False),
        "l2-1000000x384-byte": (3021, 1_000_000, 384, "l2", "byte", "<i1", False),
        "l2-1000000x768-byte": (3022, 1_000_000, 768, "l2", "byte", "<i1", False),
    },
    "binary": {
        "hamming-1000x256-binary": (4000, 1000, 256, "hamming", "binary", "<u1", False),
        "hamming-1000x256-binary-clustered": (4001, 1000, 256, "hamming", "binary", "<u1", False),
        "hamming-1000x1024-binary-clustered": (4002, 1000, 1024, "hamming", "binary", "<u1", False),
        "hamming-1000x256-binary-skip": (4003, 1000, 256, "hamming", "binary", "<u1", True),
        # Counts the float rule lowers m for, the second below the 64
        # neighbors m = 16 asks for; the service keeps m at 16 for binary.
        "hamming-150x256-binary": (4004, 150, 256, "hamming", "binary", "<u1", False),
        "hamming-40x256-binary": (4005, 40, 256, "hamming", "binary", "<u1", False),
    },
    "binary-large": {
        "hamming-20000x256-binary": (4006, 20000, 256, "hamming", "binary", "<u1", False),
        "hamming-20000x256-binary-clustered": (4007, 20000, 256, "hamming", "binary", "<u1", False),
    },
    "binary-sizes": {
        "hamming-100000x256-binary-clustered": (4008, 100_000, 256, "hamming", "binary", "<u1", False),
        "hamming-100000x1024-binary-clustered": (4009, 100_000, 1024, "hamming", "binary", "<u1", False),
        "hamming-500000x256-binary-clustered": (4010, 500_000, 256, "hamming", "binary", "<u1", False),
        "hamming-500000x1024-binary-clustered": (4011, 500_000, 1024, "hamming", "binary", "<u1", False),
        "hamming-1000000x256-binary-clustered": (4012, 1_000_000, 256, "hamming", "binary", "<u1", False),
        "hamming-1000000x1024-binary-clustered": (4013, 1_000_000, 1024, "hamming", "binary", "<u1", False),
    },
    "grid-binary": {
        "hamming-10000x256-binary-clustered": (4100, 10_000, 256, "hamming", "binary", "<u1", False),
        "hamming-100000x256-binary-clustered": (4101, 100_000, 256, "hamming", "binary", "<u1", False),
        "hamming-100000x512-binary-clustered": (4102, 100_000, 512, "hamming", "binary", "<u1", False),
        "hamming-100000x1024-binary-clustered": (4103, 100_000, 1024, "hamming", "binary", "<u1", False),
        "hamming-250000x256-binary-clustered": (4104, 250_000, 256, "hamming", "binary", "<u1", False),
        "hamming-250000x1024-binary-clustered": (4105, 250_000, 1024, "hamming", "binary", "<u1", False),
        "hamming-500000x256-binary-clustered": (4106, 500_000, 256, "hamming", "binary", "<u1", False),
        "hamming-500000x512-binary-clustered": (4107, 500_000, 512, "hamming", "binary", "<u1", False),
        "hamming-500000x1024-binary-clustered": (4108, 500_000, 1024, "hamming", "binary", "<u1", False),
        "hamming-750000x256-binary-clustered": (4109, 750_000, 256, "hamming", "binary", "<u1", False),
        "hamming-750000x1024-binary-clustered": (4110, 750_000, 1024, "hamming", "binary", "<u1", False),
        "hamming-1000000x256-binary-clustered": (4111, 1_000_000, 256, "hamming", "binary", "<u1", False),
        "hamming-1000000x512-binary-clustered": (4112, 1_000_000, 512, "hamming", "binary", "<u1", False),
        "hamming-1000000x1024-binary-clustered": (4113, 1_000_000, 1024, "hamming", "binary", "<u1", False),
        "hamming-2000000x256-binary-clustered": (4114, 2_000_000, 256, "hamming", "binary", "<u1", False),
        "hamming-2000000x1024-binary-clustered": (4115, 2_000_000, 1024, "hamming", "binary", "<u1", False),
    },
}

# A clustered binary input: vectors for each center, and the chance of each
# bit flipping away from its center's.
VECTORS_PER_CENTER = 50
FLIP = 0.1

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


def draw_bytes(rng, rows, dim):
    return rng.integers(-128, 128, size=(rows, dim), dtype=np.int8)


def draw_bits(rng, rows, dim):
    return rng.integers(0, 256, size=(rows, dim // 8), dtype=np.uint8)


def draw_clustered_bits(rng, rows, dim, centers):
    """Rows of a random center's bits, each flipped with probability FLIP,
    drawn CHUNK_ROWS at a time: a million rows of 1,024 flip draws would
    take 8 GB at once."""
    out = np.empty((rows, dim // 8), dtype=np.uint8)
    for start in range(0, rows, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, rows)
        which = rng.integers(len(centers), size=end - start)
        flips = np.packbits(rng.random((end - start, dim)) < FLIP, axis=1)
        out[start:end] = centers[which] ^ flips
    return out


def distribution(byte, binary, clustered, unit, doc_count, dim):
    if clustered:
        centers = max(doc_count // VECTORS_PER_CENTER, 1)
        return f"{centers} centers of uniform bits, each bit flipped with probability {FLIP}"
    if binary:
        return "uniform bits"
    if byte:
        return "uniform integers -128..127"
    return (
        "standard normal" + (", unit length" if unit else "")
        + (", float32 draws in chunks" if doc_count * dim > CHUNKED_VALUES else "")
    )


def write_input(out, name, spec):
    seed, doc_count, dim, space, data_type, blob_dtype, skip = spec
    rng = np.random.default_rng(seed)
    byte = data_type == "byte"
    binary = data_type == "binary"
    clustered = binary and name.endswith("-clustered")
    # Byte vectors cannot be scaled to unit length; inner product takes them as drawn.
    unit = space == "innerproduct" and not byte
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    if clustered:
        centers = draw_bits(rng, max(doc_count // VECTORS_PER_CENTER, 1), dim)
        vectors = draw_clustered_bits(rng, doc_count, dim, centers)
        queries = draw_clustered_bits(rng, QUERIES, dim, centers)
        (d / "vectors.knnvec").write_bytes(vectors.tobytes())
    elif binary:
        vectors = draw_bits(rng, doc_count, dim)
        queries = draw_bits(rng, QUERIES, dim)
        (d / "vectors.knnvec").write_bytes(vectors.tobytes())
    elif byte:
        vectors = draw_bytes(rng, doc_count, dim)
        queries = draw_bytes(rng, QUERIES, dim)
        (d / "vectors.knnvec").write_bytes(vectors.astype(blob_dtype).tobytes())
    elif doc_count * dim > CHUNKED_VALUES:
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
    if binary:
        (d / "queries.u8").write_bytes(queries.tobytes())
    else:
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
        "distribution": distribution(byte, binary, clustered, unit, doc_count, dim),
        "doc_ids": "permutation of 0..doc_count, times 3 plus 7",
    }
    (d / "input.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode())


def main():
    which, out, names = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3:]
    for name in names:
        if name not in SETS[which]:
            sys.exit(f"no input {name} in {which}")
    for name, spec in SETS[which].items():
        if names and name not in names:
            continue
        write_input(out, name, spec)
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
