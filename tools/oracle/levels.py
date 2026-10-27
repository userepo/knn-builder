"""The HNSW level table of a service-built index file, computed without
Faiss: an executable statement of docs/faiss-layout.md ("Levels") for
knn-write to port, checked against the files.

Usage: python levels.py FILE_GLOB...
  For each glob of service-built files sharing n and M (say every run of
  one input), checks that the files agree with each other and with the
  computed assign_probas, cum_nneighbor_per_level, levels and offsets.
  Exits non-zero on any difference.

Needs numpy (any version: it uses only numpy's MT19937 stream and IEEE
float32 conversion and division, which the standard requires to be
correctly rounded, so every platform gives the same results; the exp goes
through Python's math.exp, the C library's).
"""

import glob
import math
import struct
import sys

import numpy as np

# The ID map's fourcc and header, then the graph's fourcc, header and four
# CAGRA fields: the HNSW block starts at this offset (docs/faiss-layout.md).
HNSW_OFFSET = 37 + 47


def read_table(path):
    data = open(path, "rb").read()
    pos = HNSW_OFFSET
    table = []
    for fmt in ("d", "i", "i", "Q"):
        count = struct.unpack_from("<Q", data, pos)[0]
        table.append(np.frombuffer(data, dtype="<" + fmt, count=count, offset=pos + 8))
        pos += 8 + count * struct.calcsize(fmt)
    return table  # assign_probas, cum_nneighbor_per_level, levels, offsets


def default_probas(m):
    """HNSW::set_default_probas(int M, float levelMult) with levelMult =
    1.0 / log(M): the quotients are float, exp is C's double exp, and the
    product is rounded once to the float `proba`, then widened to double."""
    level_mult = np.float32(1.0 / math.log(m))
    probas, cum, total, level = [], [0], 0, 0
    while True:
        a = float(np.float32(-level) / level_mult)
        b = float(np.float32(-1) / level_mult)
        proba = np.float32(math.exp(a) * (1 - math.exp(b)))
        if proba < 1e-9:
            break
        probas.append(float(proba))
        total += 2 * m if level == 0 else m
        cum.append(total)
        level += 1
    return probas, cum


def level_table(n, m):
    """HNSW::prepare_level_tab(n, false) on a fresh HNSW: std::mt19937
    seeded with 12345, each draw mt() / float(mt.max()) in float, walked
    down the probabilities in double; stores level + 1."""
    probas, cum = default_probas(m)
    # numpy's legacy seeding is std::mt19937's; drawing the full 32-bit
    # range returns the raw outputs.
    raw = np.random.RandomState(12345).randint(0, 2**32, size=n, dtype=np.uint64).astype(np.uint32)
    # One float32 rounding of the draw and one of the quotient, then double.
    rem = (raw.astype(np.float32) / np.float32(4294967295)).astype(np.float64)
    # random_level, for every node at once with the same double arithmetic:
    # the first level whose probability exceeds what is left, subtracting
    # each one passed; past them all, the last.
    level = np.full(n, len(probas) - 1, dtype=np.int32)
    undecided = np.ones(n, dtype=bool)
    for j, p in enumerate(probas):
        hit = undecided & (rem < p)
        level[hit] = j
        undecided &= ~hit
        rem = np.where(undecided, rem - p, rem)
    levels = level + 1
    offsets = np.concatenate([[0], np.cumsum(np.array(cum)[levels])]).astype(np.uint64)
    return np.array(probas), np.array(cum, dtype=np.int32), levels, offsets


def main():
    ok = True
    for pattern in sys.argv[1:]:
        files = sorted(glob.glob(pattern))
        tables = [read_table(f) for f in files]
        first = tables[0]
        n, m = len(first[2]), int(first[1][1]) // 2
        agree = all(all(np.array_equal(a, b) for a, b in zip(t, first)) for t in tables)
        computed = level_table(n, m)
        match = all(np.array_equal(a, b) for a, b in zip(computed, first))
        print(f"{pattern}: n={n} M={m}, {len(files)} files agree={agree}, computed table matches={match}")
        ok &= agree and match
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
