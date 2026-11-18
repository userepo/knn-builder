"""Counts the connected components of each index file's level 0, the
graph a CAGRA build leaves: strongly connected (a search can get from any
node of one to any other) and weakly connected (linked ignoring direction),
with the largest strong component's share of the nodes.

Usage: python components.py [--json] FILE...

With --json, one JSON object a line: file, nodes, strong, largest, weak.

Runs in the faiss-base image, for its Faiss, which reads the service's
files, and its scipy.
"""

import json
import sys

import faiss
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def level0(path):
    """The node count and each node's level-0 links, as a sparse matrix."""
    # The ID map owns the graph index: keep it alive while reading.
    outer = faiss.read_index(path)
    index = faiss.downcast_index(outer.index)
    hnsw = index.hnsw
    n = index.ntotal
    neighbors = faiss.vector_to_array(hnsw.neighbors)
    offsets = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
    width = hnsw.nb_neighbors(0)
    cols = neighbors[offsets[:n, None] + np.arange(width)[None, :]].reshape(-1).astype(np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), width)
    keep = cols >= 0
    links = csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.int8), (rows[keep], cols[keep])), shape=(n, n)
    )
    return n, links


def main():
    as_json = sys.argv[1:2] == ["--json"]
    for path in sys.argv[2:] if as_json else sys.argv[1:]:
        n, links = level0(path)
        strong, labels = connected_components(links, directed=True, connection="strong")
        weak, _ = connected_components(links, directed=True, connection="weak")
        largest = np.bincount(labels).max() / n
        if as_json:
            print(json.dumps({"file": path, "nodes": n, "strong": int(strong),
                              "largest": float(largest), "weak": int(weak)}), flush=True)
        else:
            print(f"{path}: {n} nodes, components: {strong} strong (largest {largest:.1%}), {weak} weak")


if __name__ == "__main__":
    main()
