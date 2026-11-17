"""Measures what a process's first build takes of the device besides a
later one, and builds of fewer vectors than the IVF-PQ build's batch: each
of a few inputs built alone in a process of its own, then again after a
warm-up build, reading the free memory as often as it can (e2e build
--every 0). cuVS loads its kernels and creates its library handles in the
first build and keeps them, so that build's peak is higher by what it
keeps. knn-builder's cost tests hold the memory a build is forecast to need
against both peaks.

Writes OUT/first.json: each build's input, m, its peak as a process's first
build and its peak after a warm-up. In WSL, in the conda environment, with
LocalStack and the S3 settings as for run_grid.py:

  python tools/grid/first_builds.py E2E INPUTS OUT
"""

import json
import pathlib
import sys

sys.dont_write_bytecode = True  # no __pycache__ in the checkout

import run_grid  # noqa: E402

M = 16
# make_inputs.py's set and input: two below the batch's 65,536 rows in both
# float types, the grid's warm-up, and three sizes up to the largest.
CASES = (
    ("small", "l2-1000x64"),
    ("small", "l2-1000x64-fp16"),
    ("grid", "l2-10000x128"),
    ("large", "l2-20000x128"),
    ("grid", "l2-100000x768"),
    ("grid", "l2-1000000x768"),
)


def peak(e2e, inputs, target, names):
    """The device peak of the last of `names`, built in one process."""
    run_grid.run(e2e, "build", run_grid.BUCKET, inputs, target, "--m", M, "--every", 0, *names)
    record = next(
        r
        for r in json.loads((target / "runs.json").read_text())
        if r["input"] == names[-1] and r["shape"]["m"] == M
    )
    return record["device"]["peak_bytes"]


def main():
    e2e, inputs, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    warm_up = run_grid.WARM_UP["grid"]
    run_grid.write(inputs, "grid", warm_up)
    run_grid.run(e2e, "put", run_grid.BUCKET, inputs, warm_up)
    results = []
    for which, name in CASES:
        run_grid.write(inputs, which, name)
        run_grid.run(e2e, "put", run_grid.BUCKET, inputs, name)
        results.append(
            {
                "input": name,
                "m": M,
                "first_peak_bytes": peak(e2e, inputs, out / f"{name}-first", [name]),
                "peak_bytes": peak(e2e, inputs, out / f"{name}-later", [warm_up, name]),
            }
        )
        print(json.dumps(results[-1]), flush=True)
        if name != warm_up:
            run_grid.run(e2e, "drop", run_grid.BUCKET, name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "first.json").write_bytes((json.dumps(results, indent=1) + "\n").encode())


if __name__ == "__main__":
    main()
