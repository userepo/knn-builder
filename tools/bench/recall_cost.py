"""What knn-builder's recall check adds a job, from two runs of run.py over
the same inputs: one with the check off, one with it on (--recall-queries).

For each input, the medians over each run's single builds: the steps' sum
and the upload from the service's own clock, the check's seconds, and the
service's peak host memory. The check runs beside the upload, which waits
for it before it completes, so what it adds a job is the upload's growth;
the steps' sum carries the build's own variation besides. The host peaks
are also compared round by round, each run's first build with the other's
first and so on, since a peak can drift over a run's rounds alike in both.

Usage: python tools/bench/recall_cost.py OFF_DIR ON_DIR [OUT_JSON]

Prints a Markdown table, and writes its figures to OUT_JSON when given.
Standard library only.
"""

import json
import pathlib
import statistics
import sys


def singles(run_dir):
    """Each input's completed single builds, and the run's record."""
    record = json.loads((run_dir / "run.json").read_text())
    jobs = {}
    for r in record.get("single", []):
        if r.get("task_status") == "COMPLETED_INDEX_BUILD" and r.get("seconds"):
            jobs.setdefault(r["input"], []).append(r)
    return record, jobs


def median(values):
    return statistics.median(values) if values else None


def figures(jobs):
    # The check's seconds are kept apart from the steps', in recall_check.
    steps = [sum(r["seconds"].values()) for r in jobs]
    out = {
        "builds": len(jobs),
        "steps_s": median(steps),
        "upload_s": median([r["seconds"]["upload"] for r in jobs]),
        "host_peak_bytes": median([r["host_peak_bytes"] for r in jobs if "host_peak_bytes" in r]),
    }
    checks = [r["recall_check"] for r in jobs if "recall_check" in r]
    if checks:
        out["check_s"] = median([c["seconds"] for c in checks])
        out["recall"] = median([c["recall"] for c in checks])
        out["queries"] = checks[0]["queries"]
        out["k"] = checks[0]["k"]
    return out


def added_host(off_jobs, on_jobs):
    """The median of the host peak's growth from each round's build with the
    check off to that round's with it on."""
    off_by_run = {r["run"]: r["host_peak_bytes"] for r in off_jobs if "host_peak_bytes" in r}
    return median([r["host_peak_bytes"] - off_by_run[r["run"]]
                   for r in on_jobs if "host_peak_bytes" in r and r["run"] in off_by_run])


def main():
    args = sys.argv[1:]
    if len(args) not in (2, 3):
        sys.exit(__doc__.split("\n\n")[2])
    off_record, off = singles(pathlib.Path(args[0]))
    on_record, on = singles(pathlib.Path(args[1]))
    if off_record["settings"].get("recall_check_queries"):
        sys.exit(f"{args[0]} ran with the check on")
    if not on_record["settings"].get("recall_check_queries"):
        sys.exit(f"{args[1]} ran with the check off")
    if off_record["image_id"] != on_record["image_id"]:
        sys.exit("the two runs are of different images")
    report = {
        "image_id": on_record["image_id"],
        "recall_check_queries": on_record["settings"]["recall_check_queries"],
        "recall_check_k": on_record["settings"]["recall_check_k"],
        "inputs": {},
    }
    for name in sorted(set(off) & set(on)):
        if not any("recall_check" in r for r in on[name]):
            sys.exit(f"{name}: no build of {args[1]} reported its check")
    print("| Input | Data type | Builds | Steps s | with the check | Upload s | with the check "
          "| Check s | Host peak GB | with the check | Added GB | Recall |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in sorted(set(off) & set(on)):
        meta = on_record["inputs"][name]
        a, b = figures(off[name]), figures(on[name])
        b["host_added_bytes"] = added_host(off[name], on[name])
        report["inputs"][name] = {"data_type": meta["data_type"],
                                  "skip_stored_vectors": meta.get("skip_stored_vectors", False),
                                  "off": a, "on": b}
        kind = meta["data_type"] + (", skip_stored_vectors" if meta.get("skip_stored_vectors") else "")
        print(f"| `{name}` | {kind} | {a['builds']} and {b['builds']} | {a['steps_s']:.1f} | "
              f"{b['steps_s']:.1f} | {a['upload_s']:.1f} | {b['upload_s']:.1f} | "
              f"{b['check_s']:.1f} | {a['host_peak_bytes'] / 1e9:.2f} | "
              f"{b['host_peak_bytes'] / 1e9:.2f} | {b['host_added_bytes'] / 1e9:.2f} | "
              f"{b['recall']:.3f} |")
    missing = set(off) ^ set(on)
    if missing:
        print(f"\nin one run only: {', '.join(sorted(missing))}")
    if len(args) == 3:
        pathlib.Path(args[2]).write_bytes((json.dumps(report, indent=1) + "\n").encode())
        print(f"\nwrote {args[2]}")


if __name__ == "__main__":
    main()
