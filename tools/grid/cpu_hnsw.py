"""Times the build a data node runs itself when the service refuses a
segment: Faiss's HNSW over the float vectors (M 16, ef_construction 100,
k-NN's defaults), with doc ids, on 1 and 4 threads (k-NN builds on
knn.algo_param.index_thread_qty threads: 4 on a node of 32 processors or
more, 1 below). Appends a record per build to OUT/cpu.json; fit.py reads
it. The input's float32 vectors stand in for any field's: k-NN builds a
half_float field's graph over fp16 values, which this does not time.

In WSL, with faiss-cpu 1.15.1 (the version knn-inspect pins), after
make_inputs.py grid has written the inputs:

  python tools/grid/cpu_hnsw.py INPUTS OUT THREADS NAME...
"""

import json
import pathlib
import sys
import time

import faiss
import numpy as np

M = 16
EF_CONSTRUCTION = 100


def build(inputs, name, threads):
    meta = json.loads((inputs / name / "input.json").read_text())
    n, d = meta["doc_count"], meta["dimension"]
    vectors = np.fromfile(inputs / name / "vectors.knnvec", dtype=meta["blob_dtype"])
    vectors = vectors.astype(np.float32).reshape(n, d)
    ids = np.fromfile(inputs / name / "doc_ids.knndid", dtype="<i4").astype(np.int64)
    faiss.omp_set_num_threads(threads)
    hnsw = faiss.IndexHNSWFlat(d, M)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    index = faiss.IndexIDMap(hnsw)
    start = time.perf_counter()
    index.add_with_ids(vectors, ids)
    seconds = time.perf_counter() - start
    return {
        "input": name,
        "threads": threads,
        "seconds": round(seconds, 3),
        "shape": {"doc_count": n, "dimension": d, "m": M, "ef_construction": EF_CONSTRUCTION},
        "faiss": faiss.__version__,
        "simd": faiss.get_compile_options().strip(),
    }


def main():
    inputs, out, threads = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
    path = out / "cpu.json"
    records = json.loads(path.read_text()) if path.exists() else []
    for name in sys.argv[4:]:
        record = build(inputs, name, threads)
        print(json.dumps(record), flush=True)
        records = [r for r in records if (r["input"], r["threads"]) != (name, threads)]
        records.append(record)
        out.mkdir(parents=True, exist_ok=True)
        path.write_bytes((json.dumps(records, indent=1) + "\n").encode())


if __name__ == "__main__":
    main()
