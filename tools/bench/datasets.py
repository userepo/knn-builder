"""Fetches the benchmark datasets and writes them as build inputs in
make_inputs.py's layout (vectors.knnvec, doc_ids.knndid, queries.f32 or
queries.u8, input.json), one input per dataset, data type and space:

  sift-128      1M x 128 from ann-benchmarks (SIFT descriptors)
  ms-marco-384  1M x 384 from Hugging Face (navneet1v/datasets)
  cohere-768    1M x 768 from Hugging Face (navneet1v/datasets)
  gist-960      1M x 960 from ann-benchmarks, float only

the sources the upstream service's benchmarking tool names. Each dataset
gives float inputs in L2 and inner product (the vectors as they are, not
scaled to unit length, as that tool takes them), and the first three byte
inputs in both spaces and a binary one in Hamming space:

  byte    each component scaled to int8 over the dataset's range: lo..hi,
          the least and greatest value over the vectors taken, mapped onto
          -128..127 and rounded
  binary  one bit per component, set where the component is positive, so
          128, 384 or 768 bits a vector (SIFT's components are counts, so
          its bits mark the nonzero ones)

The queries come from each file's held-out query set ("test"), the first
100, converted the same way. The doc ids are a shuffled, spread-out set,
as make_inputs.py writes them.

Each dataset is taken whole when its float build fits the GPU, else cut
to the most rows, in tens of thousands, whose build does: the device
memory a build takes at its peak by the models fitted on the RTX 4070
(fixtures/grid/fits.json and binary-fits.json, tools/grid/fit.py) against
GPU_GB, 8 by default, which is also the GPU_MEMORY_LIMIT to run both
services at. Every input.json records the rows taken, the peak predicted
for it and the upstream service's own estimate; OUT_DIR/datasets.json
records the sources and the sizes chosen.

Usage: python tools/bench/datasets.py HDF5_DIR OUT_DIR [--gpu-gb GPU_GB]
           [--rows DATASET=N ...] [DATASET...]

HDF5_DIR keeps the downloaded files (9.2 GB in all; a file already there
is not fetched again); each is checked against its recorded SHA-256.
--rows takes fewer rows of a dataset than fit. With DATASETs, only those.
A dataset's inputs of another row count left under OUT_DIR by an earlier
run are removed, so the directory holds one size of each. Needs numpy and
h5py (tools/bench/environment.yml).
"""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import urllib.request

import h5py
import numpy as np

TOOLS = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = TOOLS.parent / "fixtures"
sys.path.insert(0, str(TOOLS / "grid"))
sys.dont_write_bytecode = True  # no __pycache__ in the checkout
import fit  # noqa: E402

QUERIES = 100
CHUNK_ROWS = 10_000
ROUND_ROWS = 10_000

# The request's HNSW parameters for every input: k-NN's defaults.
ALGORITHM_PARAMETERS = {"m": 16, "ef_construction": 100, "ef_search": 100}

# name: (file, url, sha256 of the file as fetched on 2026-10-30, seed, byte
#        and binary inputs too)
DATASETS = {
    "sift-128": (
        "sift-128-euclidean.hdf5",
        "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984",
        5000,
        True,
    ),
    "ms-marco-384": (
        "ms_marco-384-1m.hdf5",
        "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/ms_marco-384-1m.hdf5?download=true",
        "5f7c584e600e45544e8e9b0d79f9f17da0f1feb8d0f87d4979115d99b823995c",
        5001,
        True,
    ),
    "cohere-768": (
        "cohere-768-l2.hdf5",
        "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/cohere-768-l2.hdf5?download=true",
        "1f324c6928b0513bfa7a9438255f68a62f68ef3435ad28c7aecdd5f577f0c414",
        5002,
        True,
    ),
    "gist-960": (
        "gist-960-euclidean.hdf5",
        "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "8e95831936bfdbfa0a56086942e2cf98cd703517c67f985914183eb4cdbf026a",
        5003,
        False,
    ),
}


def download(url, path, expected):
    """Fetches `path` unless it is there, and checks its SHA-256 either way."""
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        print(f"fetching {url}", flush=True)
        with urllib.request.urlopen(url) as resp, part.open("wb") as out:
            shutil.copyfileobj(resp, out, length=16 << 20)
        part.rename(path)
    digest = sha256(path)
    if digest != expected:
        sys.exit(f"{path}: SHA-256 {digest}, not the recorded {expected}; delete it to fetch again")
    return digest


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class Peaks:
    """The device memory a build takes at its peak, by the fitted models."""

    def __init__(self):
        fits = json.loads((FIXTURES / "grid" / "fits.json").read_bytes())["device_peak_bytes"]
        self.models = [fits["ivf_pq_phase"], fits["graph_phase"]]
        self.cap = fits["rows"]
        binary = json.loads((FIXTURES / "grid" / "binary-fits.json").read_bytes())
        binary = binary["device_peak_bytes"]
        self.binary_models = [binary["nn_descent_phase"], binary["graph_phase"]]
        self.binary_cap = binary["rows"]

    def of(self, n, d, data_type):
        g = 4 * ALGORITHM_PARAMETERS["m"]
        if data_type == "binary":
            shape = (n, d, g)
            return n * d / 8 + max(
                fit.binary_phase_peak(self.binary_models, w, shape, self.binary_cap)
                for w in (0, 1)
            )
        shape = (n, d, 4 if data_type == "float" else 1, g)
        return n * d * shape[2] + max(
            fit.phase_peak(self.models, w, shape, self.cap) for w in (0, 1)
        )


def upstream_estimate(n, d, data_type):
    """The upstream service's GPU memory estimate for the request
    (its calculate_memory_requirements), which GPU_MEMORY_LIMIT bounds."""
    entry = 4 if data_type == "float" else 1
    return (d * entry + ALGORITHM_PARAMETERS["m"] * 8) * 1.1 * n * 0.5


def rows_that_fit(peaks, rows_in_file, d, gpu_bytes):
    """The whole dataset when its float build fits, else the most rows, a
    multiple of ROUND_ROWS, whose float build does."""
    if peaks.of(rows_in_file, d, "float") <= gpu_bytes:
        return rows_in_file
    n = rows_in_file // ROUND_ROWS * ROUND_ROWS
    while n > 0 and peaks.of(n, d, "float") > gpu_bytes:
        n -= ROUND_ROWS
    return n


def to_bytes(x, lo, hi):
    """x over lo..hi onto -128..127, rounded, as int8."""
    scaled = np.rint((x.astype(np.float64) - lo) * (255.0 / (hi - lo))) - 128.0
    return np.clip(scaled, -128, 127).astype("<i1")


def to_bits(x):
    """A bit per component, set where it is positive, packed."""
    return np.packbits(x > 0, axis=1)


def link(src, dst):
    """A hard link where the file system allows, else a copy."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def write_meta(d, meta):
    (d / "input.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode())


def finite(dataset, x, what):
    if not np.isfinite(x).all():
        sys.exit(f"{dataset}: a {what} value is not finite")


def remove_other_sizes(out, short, rows, dim):
    """Removes the dataset's inputs of another row count left by an
    earlier run, so OUT_DIR holds one size of each."""
    for d in sorted(out.glob(f"{short}-*x{dim}*")):
        if d.is_dir() and f"-{rows}x{dim}" not in d.name:
            shutil.rmtree(d)
            print(f"removed {d}, of another row count", flush=True)


def convert(dataset, hdf5, digest, out, rows_wanted, peaks, gpu_bytes):
    """Writes the dataset's inputs and returns the record for datasets.json."""
    file, url, _, seed, all_types = DATASETS[dataset]
    short = dataset.rsplit("-", 1)[0]
    with h5py.File(hdf5, "r") as f:
        train = f["train"]
        rows_in_file, dim = train.shape
        if "test" not in f:
            sys.exit(f"{hdf5}: no held-out queries (no 'test' set)")
        queries = np.asarray(f["test"][:QUERIES], dtype="<f4")
        if queries.shape != (QUERIES, dim):
            sys.exit(f"{hdf5}: {queries.shape[0]} queries of {queries.shape[1]}, not {dim}")
        finite(dataset, queries, "query")
        rows = rows_that_fit(peaks, rows_in_file, dim, gpu_bytes)
        if rows <= 0:
            sys.exit(f"{dataset}: no row count fits {gpu_bytes} bytes")
        if rows_wanted is not None:
            if rows_wanted > rows:
                print(f"{dataset}: {rows_wanted} rows asked, {rows} fit", flush=True)
            rows = min(rows, rows_wanted)
        print(f"{dataset}: {rows} of {rows_in_file} rows x {dim}", flush=True)
        remove_other_sizes(out, short, rows, dim)

        rng = np.random.default_rng(seed)
        doc_ids = (rng.permutation(rows).astype(np.int64) * 3 + 7).astype("<i4")
        names = {
            ("float", "l2"): f"{short}-l2-{rows}x{dim}",
            ("float", "innerproduct"): f"{short}-ip-{rows}x{dim}",
        }
        if all_types:
            names |= {
                ("byte", "l2"): f"{short}-l2-{rows}x{dim}-byte",
                ("byte", "innerproduct"): f"{short}-ip-{rows}x{dim}-byte",
                ("binary", "hamming"): f"{short}-hamming-{rows}x{dim}-binary",
            }
        dirs = {key: out / name for key, name in names.items()}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # The float vectors as they are, and the dataset's range for the
        # byte ones.
        lo, hi = np.inf, -np.inf
        with (dirs["float", "l2"] / "vectors.knnvec").open("wb") as blob:
            for start in range(0, rows, CHUNK_ROWS):
                chunk = np.asarray(train[start : min(start + CHUNK_ROWS, rows)], dtype="<f4")
                finite(dataset, chunk, "vector")
                lo, hi = min(lo, float(chunk.min())), max(hi, float(chunk.max()))
                blob.write(chunk.tobytes())
        if not hi > lo:
            sys.exit(f"{dataset}: every value is {lo}")
        one_bits = 0
        if all_types:
            with (
                (dirs["byte", "l2"] / "vectors.knnvec").open("wb") as bytes_blob,
                (dirs["binary", "hamming"] / "vectors.knnvec").open("wb") as bits_blob,
            ):
                for start in range(0, rows, CHUNK_ROWS):
                    chunk = np.asarray(train[start : min(start + CHUNK_ROWS, rows)], dtype="<f4")
                    bytes_blob.write(to_bytes(chunk, lo, hi).tobytes())
                    bits = to_bits(chunk)
                    one_bits += int(np.unpackbits(bits).sum())
                    bits_blob.write(bits.tobytes())

    (dirs["float", "l2"] / "doc_ids.knndid").write_bytes(doc_ids.tobytes())
    (dirs["float", "l2"] / "queries.f32").write_bytes(queries.tobytes())
    if all_types:
        (dirs["byte", "l2"] / "queries.f32").write_bytes(
            to_bytes(queries, lo, hi).astype("<f4").tobytes()
        )
        (dirs["binary", "hamming"] / "queries.u8").write_bytes(to_bits(queries).tobytes())
    # The inner product input of each type holds the same vectors, ids and
    # queries as its L2 one.
    for data_type, l2, ip in (("float", "l2", "innerproduct"), ("byte", "l2", "innerproduct")):
        if (data_type, ip) in dirs:
            for blob in ("vectors.knnvec", "queries.f32"):
                link(dirs[data_type, l2] / blob, dirs[data_type, ip] / blob)
    for key, d in dirs.items():
        if key != ("float", "l2"):
            link(dirs["float", "l2"] / "doc_ids.knndid", d / "doc_ids.knndid")

    source = {
        "dataset": dataset,
        "file": file,
        "url": url,
        "sha256": digest,
        "bytes": hdf5.stat().st_size,
        "rows_in_file": rows_in_file,
        "rows_taken": rows,
        "gpu_bytes": gpu_bytes,
        "queries": f"the first {QUERIES} of the file's test set",
    }
    inputs = {}
    for (data_type, space), d in dirs.items():
        blob_dtype = {"float": "<f4", "byte": "<i1", "binary": "<u1"}[data_type]
        distribution = {
            "float": "the dataset's vectors as they are",
            "byte": f"each component over the dataset's range {lo}..{hi} onto -128..127, rounded",
            "binary": "a bit per component, set where it is positive"
            + f"; {one_bits / (rows * dim):.4f} of the bits set",
        }[data_type]
        predicted = peaks.of(rows, dim, data_type)
        estimate = upstream_estimate(rows, dim, data_type)
        meta = {
            "doc_count": rows,
            "dimension": dim,
            "space_type": space,
            "data_type": data_type,
            "blob_dtype": blob_dtype,
            "skip_stored_vectors": False,
            **ALGORITHM_PARAMETERS,
            "queries": QUERIES,
            "seed": seed,
            "distribution": distribution,
            "doc_ids": "permutation of 0..doc_count, times 3 plus 7",
            "source": source,
            "device_peak_bytes_predicted": round(predicted),
            "upstream_gpu_estimate_bytes": round(estimate),
        }
        write_meta(d, meta)
        inputs[d.name] = {
            "data_type": data_type,
            "space_type": space,
            "device_peak_bytes_predicted": round(predicted),
            "upstream_gpu_estimate_bytes": round(estimate),
        }
        print(
            f"  {d.name}: predicted peak {predicted / 1e9:.2f} GB,"
            f" upstream estimate {estimate / 1e9:.2f} GB",
            flush=True,
        )
    return {**source, "dimension": dim, "inputs": inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("hdf5_dir", type=pathlib.Path)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--gpu-gb", type=float, default=8.0)
    parser.add_argument("--rows", action="append", default=[], metavar="DATASET=N")
    parser.add_argument("datasets", nargs="*")
    args = parser.parse_args()
    for dataset in args.datasets:
        if dataset not in DATASETS:
            sys.exit(f"no dataset {dataset}; one of {', '.join(DATASETS)}")
    selected = args.datasets or list(DATASETS)
    rows_wanted = {}
    for setting in args.rows:
        dataset, _, n = setting.partition("=")
        if dataset not in selected or not n.isdigit() or int(n) == 0:
            sys.exit(f"--rows takes DATASET=N, a selected dataset and N above 0, not {setting}")
        rows_wanted[dataset] = int(n)
    gpu_bytes = round(args.gpu_gb * 1e9)
    peaks = Peaks()
    args.hdf5_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for dataset in selected:
        file, url, expected, _, _ = DATASETS[dataset]
        hdf5 = args.hdf5_dir / file
        digest = download(url, hdf5, expected)
        records[dataset] = convert(
            dataset, hdf5, digest, args.out_dir, rows_wanted.get(dataset), peaks, gpu_bytes
        )
    summary = args.out_dir / "datasets.json"
    if summary.exists():
        records = {**json.loads(summary.read_bytes()).get("datasets", {}), **records}
    summary.write_bytes((json.dumps({"datasets": records}, indent=1) + "\n").encode())


if __name__ == "__main__":
    main()
