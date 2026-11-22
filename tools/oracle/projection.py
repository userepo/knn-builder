"""Writes knn-write's test data for the upper-level projection, from the
upstream service's own module (core/index_builder/faiss/
cagra_upper_level_projection.py) and the numpy it runs on.

Usage: python projection.py OUT
  OUT  crates/knn-write/tests/data/projection.json

Run it in the api-snapshot image (tools/README.md), whose numpy draws the
gate's entry points and whose Faiss draws the promoted nodes. The file
holds:
  probes    n -> the gate's entry points, default_rng(12345).choice(n,
            min(3, n), replace=False)
  promoted  the ids draw_promoted(n, m) promotes: whole for small n, else
            their count, sum, and first and last ten
  gate      max_reachable_share on graphs built by graph() below, which
            knn-write's test builds again from the same parameters
  level0    m -> faiss.HNSW(m).assign_probas[0] as f64 bits in hex, for
            every m from 2 to 2048: a node is promoted where its draw is
            at least this, so it decides the draw alone
  layer1    build_layer1 on vectors() below, which knn-write's test
            builds again: the graph, as a string of ids
  layer1_small  the same for 100 vectors or fewer, built on SMALL_THREADS
            OpenMP threads: Faiss adds a level that small on one thread
            whatever its thread count, so each is the graph one thread
            builds, every run
  links     project_links on a graph() with the promoted ids and layer-1
            graph each case lists, for a few settings: the links it wrote
            and every slot it changed, with its new id; layer 1 is
            synthetic (lcg draws), built from the base graph's own links
            between promoted nodes, or Faiss's build_layer1 on random
            vectors; its long lists of integers are strings of them, space
            separated, the changes as index and id in turn

Faiss runs on one OpenMP thread here, where its HNSW insertion, and so
build_layer1, is the same every run (on several threads, as the service's
are on a machine of 8 processors or more, a level of more than 100 nodes
varies). Only layer1_small is built on more, and comes out the same too, so
a run writes the same file again.
"""

import json
import pathlib
import struct
import sys

import faiss
import numpy as np

from remote_vector_index_builder.core.index_builder.faiss.cagra_upper_level_projection import (
    build_layer1,
    draw_promoted,
    max_reachable_share,
    project_links,
)

PROBE_N = list(range(0, 260)) + [
    1_000,
    9_999,
    10_000,
    10_001,
    65_535,
    65_536,
    99_991,
    1_000_000,
    2**31 - 1,
]
# m from the smallest that promotes to past any practical setting, odd and
# even, since layer 1's M is the request's m itself.
PROMOTED_M = [1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 24, 31, 32, 48, 64, 100, 512, 2048]
PROMOTED_SMALL_N = 5_000
PROMOTED_LARGE = [(1_000_000, 5), (1_000_000, 16), (2_100_000, 32)]
# n, degree, clusters, bridge_every, empty_every, seed
GRAPHS = [
    (1_000, 8, 1, 0, 0, 1),  # one random cluster: everything reachable
    (1_000, 8, 10, 0, 0, 2),  # ten closed clusters: a tenth
    (1_003, 8, 7, 97, 0, 3),  # seven clusters, a few links across
    (1_003, 8, 7, 13, 0, 4),  # more links across
    (1_003, 8, 7, 2_003, 0, 10),  # four links across
    (500, 6, 3, 0, 3, 5),  # empty slots
    (1_000, 1, 1, 0, 0, 6),  # one link a node: cycles and tails
    (2, 4, 1, 0, 0, 7),
    (1, 3, 1, 0, 0, 8),
    (300, 4, 300, 0, 0, 9),  # every node alone, linked to itself
]
STOP_AT = [0.99, 1.0, 0.05]
# (n, d, clusters, kind, metric, M, ef, seed) for build_layer1 on vectors().
LAYER1_CASES = [
    (300, 16, 4, "float", "l2", 4, 40, 31),
    (300, 16, 4, "float", "ip", 4, 40, 32),
    (500, 13, 3, "float", "l2", 8, 20, 33),  # a tail of 1
    (500, 30, 5, "float", "ip", 8, 100, 34),  # a tail of 2
    (400, 7, 2, "byte", "l2", 5, 30, 35),  # a tail of 3; equal scores
    (400, 32, 6, "byte", "ip", 16, 64, 36),
    (1_000, 64, 10, "coarse", "l2", 16, 100, 37),
    (2_000, 24, 8, "float", "l2", 12, 50, 38),
    (1_500, 11, 4, "byte", "ip", 6, 40, 39),
    (200, 5, 1, "byte", "l2", 2, 8, 40),
    (300, 9, 3, "float", "l2", 3, 1, 41),
    (3, 4, 1, "float", "l2", 2, 10, 42),
    (1, 4, 1, "float", "ip", 2, 10, 43),
]
# The same of 100 vectors or fewer, which Faiss adds on one thread whatever
# its thread count, and the count they are built on.
LAYER1_SMALL_CASES = [
    (80, 8, 3, "float", "l2", 4, 40, 44),
    (100, 16, 4, "float", "ip", 8, 40, 45),  # the most it adds so
    (64, 6, 2, "byte", "l2", 6, 20, 46),  # equal scores
]
SMALL_THREADS = 8
# (name, graph() parameters, promoted, layer 1, [(max_share, protect)]):
# promoted ("draw", m) or ("ids", [...]); layer 1 ("synthetic", M, seed,
# empty_every, self_every), ("present", M) or ("faiss", M, ef, d, seed).
DEFAULT = (0.5, True)
LINK_CASES = [
    (
        "clusters",
        (600, 8, 6, 0, 0, 11),
        ("draw", 4),
        ("synthetic", 4, 21, 7, 11),
        [DEFAULT, (0.5, False), (0.25, True), (1.0, True), (0.0, True), (0.3, True)],
    ),
    (
        "empty slots",
        (500, 10, 5, 0, 3, 12),
        ("draw", 5),
        ("synthetic", 5, 22, 9, 0),
        [DEFAULT, (0.1, True), (0.7, False)],
    ),
    ("links present", (400, 8, 4, 0, 0, 13), ("draw", 3), ("present", 6), [DEFAULT, (0.5, False)]),
    (
        "protection",
        (300, 4, 1, 0, 0, 14),
        ("draw", 2),
        ("synthetic", 2, 24, 0, 0),
        [DEFAULT, (0.5, False), (1.0, True)],
    ),
    ("faiss layer 1", (2_000, 16, 10, 0, 0, 15), ("draw", 8), ("faiss", 8, 40, 16, 25), [DEFAULT]),
    ("one promoted", (50, 4, 1, 0, 0, 16), ("ids", [7]), ("synthetic", 3, 26, 0, 0), [DEFAULT]),
    (
        "degree 32",
        (1_500, 32, 3, 101, 0, 17),
        ("draw", 16),
        ("synthetic", 16, 27, 5, 0),
        [DEFAULT, (0.5, False)],
    ),
    (
        "repeated ids",
        (600, 8, 150, 0, 0, 18),
        ("draw", 2),
        ("present", 4),
        [DEFAULT, (0.5, False)],
    ),
    (
        # Full rows of 10 at a share of 0.7: numpy 1.26 multiplies the
        # float32 share by the int in float64 (6.99..., so 6 slots), where
        # numpy 2's float32 product would round to 7.
        "cap rounding",
        (500, 10, 5, 0, 0, 19),
        ("draw", 3),
        ("synthetic", 8, 29, 0, 0),
        [(0.7, True), (0.7, False), DEFAULT],
    ),
    (
        "repeated ids, synthetic",
        (600, 8, 150, 0, 0, 18),
        ("draw", 2),
        ("synthetic", 4, 28, 0, 13),
        [DEFAULT],
    ),
]


def lcg(seed):
    """A 64-bit LCG (Knuth's MMIX constants), its top 31 bits a draw."""
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        yield state >> 33


def graph(n, degree, clusters, bridge_every, empty_every, seed):
    """n rows of `degree` ids: node i links within its cluster (n split in
    `clusters` runs of ceil(n / clusters) ids), except every bridge_every-th
    slot, which links anywhere, and every empty_every-th, which is empty
    (-1). Slots count from 0 over the whole table."""
    draws = lcg(seed)
    size = -(-n // clusters)
    out = np.empty(n * degree, dtype=np.int32)
    for i in range(n):
        low = (i // size) * size
        high = min(n, low + size)
        for s in range(degree):
            slot = i * degree + s
            draw = next(draws)
            if empty_every and slot % empty_every == 0:
                out[slot] = -1
            elif bridge_every and slot % bridge_every == 0:
                out[slot] = draw % n
            else:
                out[slot] = low + draw % (high - low)
    return out


def vectors(n, d, clusters, kind, seed):
    """n rows of d float32 values, row after row, each from an lcg draw r:
    float, cluster * 7 + (r % 20001 - 10000) / 997; coarse, cluster * 7 +
    (r % 4097 - 2048) / 64; byte, cluster * 20 + r % 256 - 128 (whole
    numbers, so many equal scores). Node i is in cluster i % clusters."""
    draws = lcg(seed)
    out = np.empty((n, d), dtype=np.float32)
    for i in range(n):
        cluster = i % clusters
        for j in range(d):
            r = next(draws)
            if kind == "float":
                out[i, j] = np.float32(cluster * 7) + np.float32(r % 20001 - 10000) / np.float32(997)
            elif kind == "coarse":
                out[i, j] = np.float32(cluster * 7) + np.float32(r % 4097 - 2048) / np.float32(64)
            else:
                out[i, j] = np.float32(cluster * 20 + r % 256 - 128)
    return out


def layer1(case):
    n, d, clusters, kind, metric, M, ef, seed = case
    x = vectors(n, d, clusters, kind, seed)
    faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
    first = build_layer1(x, faiss_metric, M, ef)
    again = build_layer1(x, faiss_metric, M, ef)
    assert (first == again).all(), "one thread should build the same layer 1 every run"
    return {
        "n": n,
        "dimension": d,
        "clusters": clusters,
        "kind": kind,
        "metric": metric,
        "m": M,
        "ef_construction": ef,
        "seed": seed,
        "first_row": [float(v) for v in x[0]],
        "graph": numbers(first.ravel()),
    }


def on_several_threads(case):
    """build_layer1 on the case's vectors, on SMALL_THREADS OpenMP threads."""
    n, d, clusters, kind, metric, M, ef, seed = case
    x = vectors(n, d, clusters, kind, seed)
    faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
    faiss.omp_set_num_threads(SMALL_THREADS)
    try:
        assert faiss.omp_get_max_threads() == SMALL_THREADS
        return numbers(build_layer1(x, faiss_metric, M, ef).ravel())
    finally:
        faiss.omp_set_num_threads(1)


def layer1_small(case):
    """layer1(case), held to what SMALL_THREADS threads build, run after
    run: the same graph."""
    record = layer1(case)
    assert record["n"] <= 100
    for _ in range(5):
        assert on_several_threads(case) == record["graph"], (
            f"{SMALL_THREADS} threads should build one thread's layer 1 of {record['n']} nodes"
        )
    return {**record, "threads": SMALL_THREADS}


def layer1_of(spec, neighbors, n, degree, promoted):
    """The layer-1 graph [len(promoted), M], positions into promoted."""
    kind, M = spec[0], spec[1]
    npro = len(promoted)
    out = np.full((npro, M), -1, dtype=np.int32)
    if kind == "synthetic":
        _, _, seed, empty_every, self_every = spec
        draws = lcg(seed)
        for i in range(npro):
            for s in range(M):
                slot = i * M + s
                draw = next(draws)
                if empty_every and slot % empty_every == 0:
                    continue
                out[i, s] = i if self_every and slot % self_every == 0 else draw % npro
    elif kind == "present":
        position = {int(p): i for i, p in enumerate(promoted)}
        for i, p in enumerate(promoted):
            row = [position[int(v)] for v in neighbors[p * degree : (p + 1) * degree] if int(v) in position]
            row = [v for v in dict.fromkeys(row) if v != i][:M]
            out[i, : len(row)] = row
    else:
        _, _, ef, d, seed = spec
        rng = np.random.RandomState(seed)
        centers = rng.normal(size=(8, d)).astype(np.float32) * 10
        vectors = centers[rng.randint(0, 8, size=n)] + rng.normal(size=(n, d)).astype(np.float32)
        out = build_layer1(np.ascontiguousarray(vectors[promoted]), faiss.METRIC_L2, M, ef)
    return out


def numbers(values):
    """Integers as one string, space separated, to keep the file small."""
    return " ".join(str(int(v)) for v in values)


def links(case):
    name, params, promoted_spec, layer1_spec, settings = case
    n, degree = params[0], params[1]
    base = graph(*params)
    if promoted_spec[0] == "draw":
        promoted = draw_promoted(n, promoted_spec[1]).astype(np.int64)
    else:
        promoted = np.array(promoted_spec[1], dtype=np.int64)
    layer1 = layer1_of(layer1_spec, base, n, degree, promoted)
    offsets = np.arange(n + 1, dtype=np.int64) * degree
    results = []
    for max_share, protect in settings:
        neighbors = base.copy()
        result = {"max_share": max_share, "protect_last_in_edge": protect}
        try:
            result["written"] = project_links(neighbors, offsets, degree, promoted, layer1, max_share, protect)
            changed = np.flatnonzero(neighbors != base)
            result["changes"] = numbers(x for i in changed for x in (i, neighbors[i]))
        except RuntimeError as e:
            result["error"] = str(e)
        results.append(result)
    return {
        "name": name,
        "n": n,
        "degree": degree,
        "clusters": params[2],
        "bridge_every": params[3],
        "empty_every": params[4],
        "seed": params[5],
        "promoted": numbers(promoted),
        "layer1_degree": int(layer1.shape[1]),
        "layer1": numbers(layer1.ravel()),
        "results": results,
    }


def promoted_summary(ids):
    ids = [int(i) for i in ids]
    return {
        "count": len(ids),
        "sum": sum(ids),
        "first": ids[:10],
        "last": ids[-10:],
    }


def main():
    # One OpenMP thread: Faiss's HNSW insertion is then the same every run,
    # so build_layer1 is, and so is this file.
    faiss.omp_set_num_threads(1)
    out = pathlib.Path(sys.argv[1])
    probes = {
        str(n): [int(i) for i in np.random.default_rng(12345).choice(n, min(3, n), replace=False)]
        for n in PROBE_N
    }
    promoted = [
        {"n": PROMOTED_SMALL_N, "m": m, "ids": [int(i) for i in draw_promoted(PROMOTED_SMALL_N, m)]}
        for m in PROMOTED_M
    ] + [
        {"n": n, "m": m, **promoted_summary(draw_promoted(n, m))} for n, m in PROMOTED_LARGE
    ]
    gate = []
    for n, degree, clusters, bridge_every, empty_every, seed in GRAPHS:
        neighbors = graph(n, degree, clusters, bridge_every, empty_every, seed)
        offsets = np.arange(n + 1, dtype=np.int64) * degree
        gate.append(
            {
                "n": n,
                "degree": degree,
                "clusters": clusters,
                "bridge_every": bridge_every,
                "empty_every": empty_every,
                "seed": seed,
                "first_row": [int(i) for i in neighbors[:degree]],
                "shares": {
                    repr(stop_at): max_reachable_share(neighbors, offsets, degree, stop_at=stop_at)
                    for stop_at in STOP_AT
                },
            }
        )
    level0 = {}
    for m in range(2, 2049):
        # Held in a name: a temporary HNSW is freed under its vector.
        hnsw = faiss.HNSW(m)
        level0[str(m)] = struct.pack(">d", hnsw.assign_probas.at(0)).hex()
    record = {
        "generated_by": "tools/oracle/projection.py",
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "probes": probes,
        "promoted": promoted,
        "gate": gate,
        "level0": level0,
        "layer1": [layer1(case) for case in LAYER1_CASES],
        "layer1_small": [layer1_small(case) for case in LAYER1_SMALL_CASES],
        "links": [links(case) for case in LINK_CASES],
    }
    # The threads are Faiss's to use: on them, a level of more than 100
    # nodes comes out another graph than one thread's.
    large = max(LAYER1_CASES)
    one_thread = next(r["graph"] for r in record["layer1"] if r["n"] == large[0])
    assert any(on_several_threads(large) != one_thread for _ in range(5)), (
        f"{SMALL_THREADS} threads should build another layer 1 of {large[0]} nodes"
    )
    out.write_bytes((json.dumps(record, indent=1) + "\n").encode())


if __name__ == "__main__":
    main()
