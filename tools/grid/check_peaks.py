"""Checks the device peaks run_grid.py sampled every 5 ms, which can miss
an allocation shorter than that, so the true peak can only be at or above
the sampled one. Each case rebuilds an input through the e2e example:

- once reading the free memory as often as it can (--every 0: about every
  35 us under WSL2, the time one cudaMemGetInfo call takes), whose peak the
  5 ms one should match if the samples miss nothing that lasts longer than
  that;
- once with device memory held as another process would hold it, leaving
  the sampled peak plus a margin free, which must complete;
- once leaving 90% of the sampled peak free, which must fail out of device
  memory. It shows whether holding memory takes it from the build at all:
  under WSL2 the driver pages device memory out instead of refusing it, so
  this build completes too, and the pair proves nothing there; on a native
  Linux driver the pair checks the peak itself.

Writes OUT/check.json. In WSL, in the conda environment, with LocalStack
and the S3 settings as for run_grid.py, after it has recorded the cases in
RUNS_DIR/runs.json:

  python tools/grid/check_peaks.py E2E INPUTS RUNS_DIR OUT
"""

import json
import pathlib
import shutil
import sys

sys.dont_write_bytecode = True  # no __pycache__ in the checkout

import run_grid  # noqa: E402

WARM_UP = run_grid.WARM_UP["grid"]

# Each phase's largest builds, both data types, and the smallest build
# with the largest overhead for its size.
CASES = [
    ("l2-100000x768", 16),
    ("l2-500000x384-fp16", 16),
    ("l2-1000000x128", 32),
    ("l2-1000000x768", 16),
    ("l2-2000000x128", 32),
]
MARGIN = 0.02
MARGIN_BYTES = 16 << 20
SHORT = 0.9


def build(e2e, inputs, target, name, m, *flags):
    run_grid.run(
        e2e, "build", run_grid.BUCKET, inputs, target, "--m", m, *flags, WARM_UP, name
    )
    return next(
        r
        for r in json.loads((target / "runs.json").read_text())
        if r["input"] == name and r["shape"]["m"] == m
    )


def main():
    e2e, inputs, runs_dir, out = sys.argv[1], *map(pathlib.Path, sys.argv[2:5])
    runs = json.loads((runs_dir / "runs.json").read_text())
    sampled = {
        (r["input"], r["shape"]["m"]): r["device"]["peak_bytes"]
        for r in runs
        if r["task_status"] == "COMPLETED_INDEX_BUILD"
    }
    run_grid.write(inputs, "grid", WARM_UP)
    run_grid.run(e2e, "put", run_grid.BUCKET, inputs, WARM_UP)
    results = []
    for name, m in CASES:
        peak = sampled[(name, m)]
        run_grid.write(inputs, "grid", name)
        run_grid.run(e2e, "put", run_grid.BUCKET, inputs, name)
        tight = build(e2e, inputs, out / "tight", name, m, "--every", 0)
        held = {}
        for label, leave in (("margin", int(peak * (1 + MARGIN)) + MARGIN_BYTES),
                             ("short", int(peak * SHORT))):
            record = build(e2e, inputs, out / label, name, m, "--leave", leave)
            held[label] = {
                "left_free_bytes": record["device"]["free_before_bytes"],
                "task_status": record["task_status"],
                "error_message": record.get("error_message"),
            }
        results.append({
            "input": name,
            "m": m,
            "sampled_peak_bytes": peak,
            "tight_peak_bytes": tight["device"]["peak_bytes"],
            "tight_over_sampled": round(tight["device"]["peak_bytes"] / peak - 1.0, 4),
            "held": held,
            # The short build failing is what makes the margin build's
            # completion evidence.
            "held_conclusive": held["short"]["task_status"] == "FAILED_INDEX_BUILD",
        })
        print(json.dumps(results[-1]), flush=True)
        run_grid.run(e2e, "drop", run_grid.BUCKET, name)
        shutil.rmtree(inputs / name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_bytes((json.dumps(results, indent=1) + "\n").encode())
    worst = max(r["tight_over_sampled"] for r in results)
    conclusive = sum(r["held_conclusive"] for r in results)
    print(f"tight peaks at most {worst:+.2%} over the sampled ones; "
          f"held-memory checks conclusive in {conclusive} of {len(results)}")


if __name__ == "__main__":
    main()
