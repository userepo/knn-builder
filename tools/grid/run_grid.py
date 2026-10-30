"""Runs make_inputs.py's grid through knn-builder's e2e example, one input
at a time: writes it, uploads it, builds it with m 16 and 32, each build in
a process of its own after a small warm-up build, then deletes it from S3
and from disk. 22 GB of inputs never sit on disk or in LocalStack at once.
Each build's record (stage times, file size, peak resident and device
memory) lands in OUT/runs.json; fit.py reads it.

In WSL, in the conda environment, with LocalStack on port 4566 and the S3
settings the service reads (tools/README.md):

  python tools/grid/run_grid.py E2E INPUTS OUT [--set SET] [NAME...]

E2E is the built example binary, INPUTS a scratch directory, SET the
make_inputs.py set to run (grid, the default, or grid-byte, whose builds
warm up on a byte input of their own), NAMEs a subset of it.
"""

import pathlib
import shutil
import subprocess
import sys

BUCKET = "grid"
# Each set's small input, built before every build to warm its kernels up.
WARM_UP = {"grid": "l2-10000x128", "grid-byte": "l2-10000x128-byte"}
MS = (16, 32)
TOOLS = pathlib.Path(__file__).resolve().parent.parent


def grid_names(which):
    sys.path.insert(0, str(TOOLS))
    sys.dont_write_bytecode = True  # no __pycache__ in the checkout
    import make_inputs  # noqa: PLC0415

    return [name for name in make_inputs.SETS[which] if name != WARM_UP[which]]


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def write(inputs, which, name):
    if not (inputs / name / "input.json").exists():
        run(sys.executable, TOOLS / "make_inputs.py", which, inputs, name)


def main():
    e2e, inputs, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    rest = sys.argv[4:]
    which = "grid"
    if rest[:1] == ["--set"]:
        which, rest = rest[1], rest[2:]
    names = rest or grid_names(which)
    warm_up = WARM_UP[which]
    write(inputs, which, warm_up)
    run(e2e, "put", BUCKET, inputs, warm_up)
    for name in names:
        write(inputs, which, name)
        run(e2e, "put", BUCKET, inputs, name)
        for m in MS:
            run(e2e, "build", BUCKET, inputs, out, "--m", m, warm_up, name)
        run(e2e, "drop", BUCKET, name)
        shutil.rmtree(inputs / name)


if __name__ == "__main__":
    main()
