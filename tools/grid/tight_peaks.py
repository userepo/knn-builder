"""Measures again the device peaks of the grid's builds of 250,000 vectors
or fewer, float and byte, reading the free memory as often as it can (e2e
build --every 0) instead of every 5 ms. Those builds take about a second
or less, and the IVF-PQ build holds its batch of rows on the device only
briefly, so a 5 ms sample can miss it: the peak of 100,000 vectors of
dimension 128 read 68 MiB sampled and 100 MiB here. fit.py and knn-builder's
cost tests take a build's peak as the larger of its two readings; the
grid's stage times stay those of its 5 ms run, which a busy sampler could
slow.

Writes OUT/tight.json: each build's input, m and peak. In WSL, in the conda
environment, with LocalStack and the S3 settings as for run_grid.py:

  python tools/grid/tight_peaks.py E2E INPUTS OUT
"""

import json
import pathlib
import shutil
import sys

sys.dont_write_bytecode = True  # no __pycache__ in the checkout

import run_grid  # noqa: E402

LARGEST = 250_000


def cases():
    """Every float and byte grid build of LARGEST vectors or fewer."""
    out = []
    for which in ("grid", "grid-byte"):
        for name in run_grid.grid_names(which):
            if int(name.split("-")[1].split("x")[0]) <= LARGEST:
                out.append((which, name))
    return out


def main():
    e2e, inputs, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    results = []
    for which, name in cases():
        warm_up = run_grid.WARM_UP[which]
        run_grid.write(inputs, which, warm_up)
        run_grid.run(e2e, "put", run_grid.BUCKET, inputs, warm_up)
        run_grid.write(inputs, which, name)
        run_grid.run(e2e, "put", run_grid.BUCKET, inputs, name)
        for m in run_grid.MS:
            target = out / f"{name}-m{m}"
            run_grid.run(
                e2e, "build", run_grid.BUCKET, inputs, target, "--m", m, "--every", 0, warm_up, name
            )
            record = next(
                r
                for r in json.loads((target / "runs.json").read_text())
                if r["input"] == name and r["shape"]["m"] == m
            )
            results.append({"input": name, "m": m, "peak_bytes": record["device"]["peak_bytes"]})
            print(json.dumps(results[-1]), flush=True)
        run_grid.run(e2e, "drop", run_grid.BUCKET, name)
        shutil.rmtree(inputs / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tight.json").write_bytes((json.dumps(results, indent=1) + "\n").encode())


if __name__ == "__main__":
    main()
