"""Runs make_inputs.py's grid through knn-builder's e2e example, one input
at a time: writes it, uploads it, builds it with m 16 and 32, each build in
a process of its own after a small warm-up build, then deletes it from S3
and from disk. 22 GB of inputs never sit on disk or in LocalStack at once.
Each build's record (stage times, file size, peak resident and device
memory) lands in OUT/runs.json; fit.py reads it.

In WSL, in the conda environment, with LocalStack on port 4566 and the S3
settings the service reads (tools/README.md):

  python tools/grid/run_grid.py E2E INPUTS OUT [NAME...]

E2E is the built example binary, INPUTS a scratch directory, NAMEs a subset
of the grid.
"""

import pathlib
import shutil
import subprocess
import sys

BUCKET = "grid"
WARM_UP = "l2-10000x128"
MS = (16, 32)
TOOLS = pathlib.Path(__file__).resolve().parent.parent


def grid_names():
    sys.path.insert(0, str(TOOLS))
    sys.dont_write_bytecode = True  # no __pycache__ in the checkout
    import make_inputs  # noqa: PLC0415

    return [name for name in make_inputs.SETS["grid"] if name != WARM_UP]


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def write(inputs, name):
    if not (inputs / name / "input.json").exists():
        run(sys.executable, TOOLS / "make_inputs.py", "grid", inputs, name)


def main():
    e2e, inputs, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    names = sys.argv[4:] or grid_names()
    write(inputs, WARM_UP)
    run(e2e, "put", BUCKET, inputs, WARM_UP)
    for name in names:
        write(inputs, name)
        run(e2e, "put", BUCKET, inputs, name)
        for m in MS:
            run(e2e, "build", BUCKET, inputs, out, "--m", m, WARM_UP, name)
        run(e2e, "drop", BUCKET, name)
        shutil.rmtree(inputs / name)


if __name__ == "__main__":
    main()
