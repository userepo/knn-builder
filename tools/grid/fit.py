"""Fits the models a forecast of a job starts from to the grid's builds
(run_grid.py's runs.json) and the CPU reference (cpu_hnsw.py's cpu.json):
each stage's seconds, the index file's size, the device memory a build
takes at its peak, and the CPU build's seconds per thread count. The fit
leaves out the grid's 250,000- and 750,000-vector builds and the CPU's
500,000-vector ones, and reports its error on them: they test it between
the points it saw. Writes OUT/fits.json.

  python tools/grid/fit.py RUNS_DIR OUT

The terms, from docs/params.md: n vectors of dimension d, s bytes a value
(4, 2 for fp16, 1 for byte), graph degree g = 4m. The IVF-PQ build's
k-means runs on n/10 vectors over sqrt(n) lists, and its search probes 5
lists of about sqrt(n) vectors for each of n queries, so both cost about
n^1.5 d; the graph's optimization costs about n g^2; the dataset copy
n d s. A byte build takes as long as a float one, so the build's time
counts a byte value as 4 bytes (build_size), as knn-builder's cost model
does.

That search keeps its fast kernel only while the lookup table (pq_dim x
256 fp16 values, pq_dim = d/4) and the fused top-k of g candidates fit in
a block's shared memory (cuVS's compute_similarity_select). On the RTX
4070 (101,376 bytes a block) only d 768 at m 32 overflows it, and its
search then costs about four times as much: the `n^1.5 d, slow search`
term, gated on that measured case.

The device peak is the dataset plus the larger of two phases: the IVF-PQ
build and search (a buffer of a fixed number of rows of d floats, and the
PQ codes, n d / 4 bytes) and the graph's optimization (the graph, n g 4
bytes, and a buffer of g ids for at most a fixed number of rows). What
must be on the device is a floor the fit adds to: the codes in the first
phase, the finished graph in the second, so no job is forecast to take
less than its dataset, codes and graph, however far past the grid it is.
The phase that a build peaks in is found by fitting each phase to the
builds it bounds, in turn, until no build changes phase; the row cap of
the optimization buffer is the candidate that fits best, and fits.json
gives the range of caps that fit within 1% of it, which says how well the
grid pins the cap down.

Each linear model is a sum of non-negative terms, chosen among its terms
by least squares on relative error, so a small job counts as much as a
large one.
"""

import itertools
import json
import math
import pathlib
import sys

import numpy as np

GPU_HELD_OUT = (250_000, 750_000)
CPU_HELD_OUT = 500_000
MIB = 1 << 20


def shape_of(record):
    shape = record["shape"]
    s = {"<f4": 4, "<f2": 2, "<i1": 1}[shape["blob_dtype"]]
    return shape["doc_count"], shape["dimension"], s, 4 * shape["m"]


def build_size(s):
    """Bytes a value as the build's time counts it: a byte one as a float
    one, which it builds as fast as."""
    return 4 if s == 1 else s


def slow_search(d, g):
    return d == 768 and g == 128


TERMS = {
    "download_seconds": lambda n, d, s, g: {"1": 1.0, "n d s": n * d * s, "n": n},
    "build_seconds": lambda n, d, s, g: {
        "1": 1.0,
        "n d s": n * d * build_size(s),
        "n^1.5 d": n**1.5 * d,
        "n g^2": n * g * g,
        "n^1.5 d, slow search": n**1.5 * d if slow_search(d, g) else 0.0,
    },
    "file_bytes": lambda n, d, s, g: {"1": 1.0, "n d s": n * d * s, "n g": n * g, "n": n},
    "upload_seconds": lambda n, d, s, g: {"1": 1.0, "n d s": n * d * s, "n g": n * g, "n": n},
}

OBSERVED = {
    "download_seconds": lambda r: r["seconds"]["download"],
    "build_seconds": lambda r: r["seconds"]["build"],
    "file_bytes": lambda r: r["file_size"],
    "upload_seconds": lambda r: r["seconds"]["upload"],
}


def ivf_phase(n, d, s, g, cap):
    return {"1": 1.0, "d": float(d), "n d": n * d}


def ivf_floor(n, d, s, g):
    # The PQ codes: pq_dim = d / 4 bytes a vector at 8 bits.
    return n * d / 4


def graph_phase(n, d, s, g, cap):
    return {"1": 1.0, "n g": n * g, "min(n, rows) g": min(n, cap) * g}


def graph_floor(n, d, s, g):
    # The finished graph: g uint32 ids a vector.
    return 4 * n * g


PHASES = ((ivf_phase, ivf_floor), (graph_phase, graph_floor))


def cpu_terms(n, d):
    # An HNSW insertion visits about log n nodes, each a distance over d
    # values and the graph bookkeeping around it, which d does not change.
    return {
        "1": 1.0,
        "n log2(n)": n * math.log2(n),
        "n log2(n) d": n * math.log2(n) * d,
    }


def fit(rows, ys, scales=None):
    """The non-negative least-squares fit on error relative to `scales`
    (the values themselves when absent): the subset of terms whose
    least-squares coefficients are all non-negative with the least squared
    relative error (for this few terms, trying every subset finds the
    non-negative optimum)."""
    names = list(rows[0])
    x = np.array([[row[t] for t in names] for row in rows])
    y = np.array(ys, dtype=float)
    w = 1.0 / np.array(scales if scales is not None else ys, dtype=float)
    best = None
    for k in range(1, len(names) + 1):
        for subset in itertools.combinations(range(len(names)), k):
            a = x[:, subset] * w[:, None]
            if not a.any(axis=0).all():
                continue
            coef, *_ = np.linalg.lstsq(a, y * w, rcond=None)
            if (coef < 0).any():
                continue
            error = float(((a @ coef - y * w) ** 2).sum())
            if best is None or error < best[0]:
                best = (error, {names[i]: float(c) for i, c in zip(subset, coef)})
    # No term fits with a non-negative coefficient: nothing to add.
    return best[1] if best else {}


def predict(model, row):
    return sum(c * row.get(t, 0.0) for t, c in model.items())


def errors(predicted, ys):
    relative = [p / y - 1.0 for p, y in zip(predicted, ys)]
    if not relative:
        return None
    return {
        "points": len(relative),
        "median_abs": round(float(np.median(np.abs(relative))), 4),
        "max_abs": round(float(np.max(np.abs(relative))), 4),
        "most_under": round(float(min(relative)), 4),
    }


def linear(terms, observe, train, held):
    model = fit([terms(*shape_of(r)) for r in train], [observe(r) for r in train])

    def score(records):
        return errors(
            [predict(model, terms(*shape_of(r))) for r in records], [observe(r) for r in records]
        )

    return {"terms": model, "train": score(train), "held_out": score(held)}


def phase_peak(models, which, shape, cap):
    terms, floor = PHASES[which]
    return floor(*shape) + predict(models[which], terms(*shape, cap))


def device_fit(train, cap):
    """The two-phase device model for one row cap: the phases' models and
    their squared relative error on `train`."""
    over = [r["device"]["peak_bytes"] - dataset(r) for r in train]
    # Start from the phases' rough sizes (the IVF-PQ buffer about 128k
    # rows of d floats, the graph's buffer as large as the graph), then
    # alternate.
    phase = [
        int(2 * graph_floor(*shape) > ivf_floor(*shape) + 4 * shape[1] * 131_072)
        for shape in map(shape_of, train)
    ]
    models = [{}, {}]
    for _ in range(20):
        for which, (terms, floor) in enumerate(PHASES):
            mine = [(r, o) for r, o, p in zip(train, over, phase) if p == which]
            if mine:
                # What the floor leaves, relative to the build's whole peak.
                models[which] = fit(
                    [terms(*shape_of(r), cap) for r, _ in mine],
                    [o - floor(*shape_of(r)) for r, o in mine],
                    [r["device"]["peak_bytes"] for r, _ in mine],
                )
        new = [
            int(
                phase_peak(models, 1, shape_of(r), cap) > phase_peak(models, 0, shape_of(r), cap)
            )
            for r in train
        ]
        if new == phase:
            break
        phase = new
    predicted = [device_peak(models, r, cap) for r in train]
    error = sum((p / r["device"]["peak_bytes"] - 1.0) ** 2 for p, r in zip(predicted, train))
    return error, models


def dataset(record):
    n, d, s, _ = shape_of(record)
    return n * d * s


def device_peak(models, record, cap):
    shape = shape_of(record)
    return dataset(record) + max(phase_peak(models, w, shape, cap) for w in (0, 1))


def device(train, held):
    caps = range(100_000, 2_000_001, 20_000)
    fitted = [device_fit(train, cap) + (cap,) for cap in caps]
    error, models, cap = min(fitted, key=lambda f: f[0])
    near = [c for e, _, c in fitted if e <= error * 1.01]
    ys = lambda rs: [r["device"]["peak_bytes"] for r in rs]  # noqa: E731
    return {
        "model": "n d s + max(n d / 4 + IVF-PQ phase, 4 n g + graph phase)",
        "ivf_pq_phase": models[0],
        "graph_phase": models[1],
        "rows": cap,
        "rows_within_1_percent": [min(near), max(near)],
        "train": errors([device_peak(models, r, cap) for r in train], ys(train)),
        "held_out": errors([device_peak(models, r, cap) for r in held], ys(held)),
    }


def main():
    runs_dir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    runs = json.loads((runs_dir / "runs.json").read_text())
    builds = [
        r
        for r in runs
        if r["task_status"] == "COMPLETED_INDEX_BUILD" and r["shape"]["doc_count"] >= 100_000
    ]
    train = [r for r in builds if r["shape"]["doc_count"] not in GPU_HELD_OUT]
    held = [r for r in builds if r["shape"]["doc_count"] in GPU_HELD_OUT]
    fits = {
        "failed_builds": [
            {"input": r["input"], "m": r["shape"]["m"], "error": r["error_message"]}
            for r in runs
            if r["task_status"] != "COMPLETED_INDEX_BUILD"
        ]
    }
    for target, terms in TERMS.items():
        fits[target] = linear(terms, OBSERVED[target], train, held)
    fits["device_peak_bytes"] = device(train, held)
    cpu_path = runs_dir / "cpu.json"
    if cpu_path.exists():
        cpu = json.loads(cpu_path.read_text())
        for threads in sorted({r["threads"] for r in cpu}):
            mine = [r for r in cpu if r["threads"] == threads]
            cpu_train = [r for r in mine if r["shape"]["doc_count"] != CPU_HELD_OUT]
            cpu_held = [r for r in mine if r["shape"]["doc_count"] == CPU_HELD_OUT]
            rows = lambda rs: [  # noqa: E731
                cpu_terms(r["shape"]["doc_count"], r["shape"]["dimension"]) for r in rs
            ]
            model = fit(rows(cpu_train), [r["seconds"] for r in cpu_train])

            def score(records, model=model, rows=rows):
                return errors(
                    [predict(model, row) for row in rows(records)], [r["seconds"] for r in records]
                )

            fits[f"cpu_seconds_{threads}_threads"] = {
                "terms": model,
                "train": score(cpu_train),
                "held_out": score(cpu_held),
            }
    out.mkdir(parents=True, exist_ok=True)
    (out / "fits.json").write_bytes((json.dumps(fits, indent=1) + "\n").encode())
    print(json.dumps(fits, indent=1))


if __name__ == "__main__":
    main()
