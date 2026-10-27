"""The Faiss oracle for knn-write: writes the index file the upstream
remote-vector-index-builder writes, from the same three things knn-write
gets (the CAGRA graph, the vectors, the doc ids, the request), by
replaying GpuIndexCagra::copyTo and write_index on a CPU IndexHNSWCagra.

Runs in the faiss-base-snapshot image pinned in tools/README.md:
its Faiss is the service's (0ca9df4 with base_image/patches), and pip's
faiss-cpu lacks patch 0001, which passes IO_FLAG_SKIP_STORAGE through
the ID map.

Commands:
  extract INDEX OUT_DIR
      From a service-built file: OUT_DIR/graph.u32 (level 0, the CAGRA
      graph, row-major uint32) and OUT_DIR/graph.json (its shape). Nothing
      else is taken from the file.
  write GRAPH_DIR INPUT_DIR OUT
      The oracle's file from GRAPH_DIR (graph.u32, graph.json) and the
      build input in INPUT_DIR: vectors.knnvec and doc_ids.knndid, as the
      service converts them, and input.json for the request (space type,
      data type, ef_construction, ef_search, skip_stored_vectors).
  check SERVICE_DIR INPUTS_DIR
      For every service-built SERVICE_DIR/<name>.faiss, or
      SERVICE_DIR/<name>/run-<k>.faiss: extract, write from
      INPUTS_DIR/<name>, and compare bytes. Exits non-zero on any
      difference.
"""

import json
import math
import pathlib
import sys
import tempfile

import faiss
import numpy as np

# faiss.Float32 and faiss.Float16 (NumericType), as copyTo records them.
NUMERIC = {"float": faiss.Float32, "half_float": faiss.Float16}
METRIC = {"l2": faiss.METRIC_L2, "innerproduct": faiss.METRIC_INNER_PRODUCT}


def load_input(input_dir):
    """The vectors and ids as the service hands them to Faiss
    (core/common/models/vectors_dataset.py, core/tasks.py)."""
    meta = json.loads((input_dir / "input.json").read_text())
    raw = (input_dir / "vectors.knnvec").read_bytes()
    vectors = np.frombuffer(raw, dtype=meta["blob_dtype"]).reshape(meta["doc_count"], meta["dimension"])
    if meta["data_type"] == "half_float":
        # An fp32 blob under half_float is converted to fp16 on download
        # (FP32ToFP16ConvertingBytesIO): numpy's round to nearest even.
        vectors = vectors.astype("<f2")
    ids = np.frombuffer((input_dir / "doc_ids.knndid").read_bytes(), dtype="<i4").astype(np.int64)
    return meta, np.ascontiguousarray(vectors), ids


def graph_degree(meta):
    """The degree the service builds with (docs/params.md): m x 4, with m
    lowered for small inputs (index_builder_utils.py, calculate_ivf_pq_n_lists
    and calculate_effective_m, with n_probes = 5)."""
    n = meta["doc_count"]
    n_lists = max(int(math.sqrt(n)), 1)
    candidates = min(5, n_lists) * n // n_lists
    max_degree = min(candidates // 2, n - 1)
    return max(1, min(meta["m"], max_degree // 4)) * 4


def build(graph, vectors, ids, meta):
    """IndexIDMap over IndexHNSWCagra, field by field as the service leaves
    it: FaissIndexHNSWCagraBuilder, then GpuIndexCagra::copyTo. `meta` is
    the input's input.json."""
    n, degree = graph.shape
    assert n == meta["doc_count"], (n, meta["doc_count"])
    assert degree == graph_degree(meta), f"graph degree {degree}, the request gives {graph_degree(meta)}"
    d = meta["dimension"]
    metric = METRIC[meta["space_type"]]
    numeric = NUMERIC[meta["data_type"]]

    cpu = faiss.IndexHNSWCagra()
    # Faiss only wraps an empty index; the service's ID map wrapped the
    # empty GPU index. Its header fields are set at the end.
    idmap = faiss.IndexIDMap(cpu)
    idmap.own_fields = True
    cpu.this.disown()
    # FaissIndexHNSWCagraBuilder._do_convert_gpu_to_cpu_index
    cpu.hnsw.efConstruction = meta["ef_construction"]
    cpu.hnsw.efSearch = meta["ef_search"]
    cpu.base_level_only = True

    # copyTo: GpuIndex::copyTo, then the CAGRA part.
    cpu.d = d
    cpu.metric_type = metric
    cpu.metric_arg = 0.0
    cpu.is_trained = True
    cpu.hnsw.is_similarity = metric == faiss.METRIC_INNER_PRODUCT
    cpu.ntotal = 0
    cpu.set_numeric_type(numeric)
    m = degree // 2
    if numeric == faiss.Float32:
        storage = faiss.IndexFlatL2(d) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(d)
    else:
        storage = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_fp16, metric)
    storage.this.disown()
    cpu.storage = storage
    cpu.own_fields = True
    cpu.keep_max_size_level0 = True
    cpu.hnsw.reset()
    cpu.hnsw.assign_probas.clear()
    cpu.hnsw.cum_nneighbor_per_level.clear()
    cpu.hnsw.set_default_probas(m, 1.0 / math.log(m))
    cpu.init_level0 = False
    # Draws every vector a level from the HNSW's own generator (seeded
    # 12345 at construction, nothing drawn before), as copyTo does.
    cpu.hnsw.prepare_level_tab(n, False)
    if not meta["skip_stored_vectors"]:
        if numeric == faiss.Float32:
            storage.add(vectors)
        else:
            # The fp16 values are the QT_fp16 codes, as copyTo adds them.
            storage.add_sa_codes(vectors.view(np.uint8).reshape(n, -1))
    cpu.ntotal = n

    # Level 0 of every node from the CAGRA graph; upper levels stay -1.
    offsets = faiss.vector_to_array(cpu.hnsw.offsets).astype(np.int64)
    neighbors = faiss.vector_to_array(cpu.hnsw.neighbors)
    level0 = int(faiss.vector_to_array(cpu.hnsw.cum_nneighbor_per_level)[1])
    assert level0 == degree, (level0, degree)
    for i in range(n):
        neighbors[offsets[i] : offsets[i] + degree] = graph[i].astype(np.int32)
    faiss.copy_array_to_vector(neighbors, cpu.hnsw.neighbors)
    cpu.init_level0 = True

    # The ID map took d and the metric from the GPU index and, wrapping it
    # before training, keeps is_trained false; add_with_ids filled the ids
    # and ntotal.
    idmap.d = d
    idmap.metric_type = metric
    idmap.metric_arg = 0.0
    idmap.is_trained = False
    idmap.ntotal = n
    faiss.copy_array_to_vector(ids, idmap.id_map)
    return idmap


def write(idmap, path, skip):
    """write_index, with IO_FLAG_SKIP_STORAGE for skip_stored_vectors
    (faiss_index_build_service.py, write_cpu_index)."""
    faiss.write_index(idmap, str(path), faiss.IO_FLAG_SKIP_STORAGE if skip else 0)


def extract(index_path, out_dir):
    idmap = faiss.read_index(str(index_path))
    cpu = faiss.downcast_index(idmap.index)
    n = cpu.ntotal
    degree = int(faiss.vector_to_array(cpu.hnsw.cum_nneighbor_per_level)[1])
    offsets = faiss.vector_to_array(cpu.hnsw.offsets).astype(np.int64)
    neighbors = faiss.vector_to_array(cpu.hnsw.neighbors)
    graph = np.stack([neighbors[offsets[i] : offsets[i] + degree] for i in range(n)])
    assert (graph >= 0).all(), "a level-0 slot is empty"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph.u32").write_bytes(graph.astype("<u4").tobytes())
    shape = {"rows": int(n), "degree": int(degree)}
    (out_dir / "graph.json").write_bytes((json.dumps(shape, indent=2) + "\n").encode())
    return shape


def write_from(graph_dir, input_dir, out):
    shape = json.loads((graph_dir / "graph.json").read_text())
    meta, vectors, ids = load_input(input_dir)
    graph = np.frombuffer((graph_dir / "graph.u32").read_bytes(), dtype="<u4").reshape(
        shape["rows"], shape["degree"]
    )
    write(build(graph, vectors, ids, meta), out, meta["skip_stored_vectors"])


def check(service_dir, inputs):
    failed = False
    # <name>.faiss (the committed fixtures) or <name>/run-<k>.faiss (a
    # baseline run's output directory).
    files = [(f.stem, f) for f in service_dir.glob("*.faiss")]
    files += [(f.parent.name, f) for f in service_dir.glob("*/run-*.faiss")]
    for name, service in sorted(files):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            extract(service, tmp)
            write_from(tmp, inputs / name, tmp / "oracle.faiss")
            ours, theirs = (tmp / "oracle.faiss").read_bytes(), service.read_bytes()
        label = service.relative_to(service_dir)
        if ours == theirs:
            print(f"{label}: identical, {len(ours)} bytes")
        else:
            first = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), min(len(ours), len(theirs)))
            print(f"{label}: DIFFERS at byte {first} (oracle {len(ours)} bytes, service {len(theirs)})")
            failed = True
    sys.exit(1 if failed else 0)


def main():
    cmd, *args = sys.argv[1:]
    if cmd == "extract":
        print(json.dumps(extract(pathlib.Path(args[0]), pathlib.Path(args[1]))))
    elif cmd == "write":
        write_from(*map(pathlib.Path, args))
    elif cmd == "check":
        check(*map(pathlib.Path, args))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
