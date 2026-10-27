"""Scores a baseline run: the service's per-step DEBUG timings for every
build, and knn-inspect `verify` and `recall` on every index file.

Usage: python score.py OUT_DIR INPUTS_DIR KNN_INSPECT
  OUT_DIR      what run_baseline.py wrote (runs.json, <input>/run-<k>.faiss)
               plus service.log, the service container's log; or what
               knn-gpu's build_inputs example wrote, whose runs.json carries
               each build's seconds itself and which has no log
  INPUTS_DIR   the inputs the run built (for queries.f32 and input.json)
  KNN_INSPECT  path to the knn-inspect binary

Writes OUT_DIR/summary.json, recording the knn-inspect version. Recall is
knn-inspect's own seeded search of each graph (seed 1, the file's efSearch
and entry-point count, k = 10) over the input's held-out queries, so it
depends on that version; a file without its vectors gets `verify` only.
Standard library only.
"""

import json
import pathlib
import re
import statistics
import subprocess
import sys

RECALL = {"seed": 1, "k": 10, "queries": "the input's queries.f32"}

START = r"Starting task execution for vector path: (\S+)"
STEPS = {
    "download": r"Vector download time for vector path (\S+): ([\d.]+) seconds",
    "build": r"Index build time for vector path (\S+): ([\d.]+) seconds",
    "conversion": r"Index conversion time for vector path (\S+): ([\d.]+) seconds",
    "write": r"Index write time for vector path (\S+): ([\d.]+) seconds",
    "upload": r"Total upload time for path (\S+): ([\d.]+) seconds",
}


def timings(log):
    """Per vector path, one dict of step seconds per build, in log order.
    Every build the service runs logs "Starting task execution" first, so a
    build that fails partway still takes its own slot; builds ran one at a
    time, so the k-th slot of a path belongs to its k-th build."""
    found = {}
    for line in log.splitlines():
        m = re.search(START, line)
        if m:
            found.setdefault(m[1], []).append({})
            continue
        for step, pattern in STEPS.items():
            m = re.search(pattern, line)
            if m and found.get(m[1]):
                found[m[1]][-1][step] = float(m[2])
    return found


def knn_inspect(binary, *args):
    out = subprocess.run([binary, *args, "--json"], capture_output=True, text=True, check=False)
    return json.loads(out.stdout)


def knn_inspect_version(binary):
    out = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def spread(values):
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 5),
    }


def main():
    out, inputs, binary = map(pathlib.Path, sys.argv[1:4])
    runs = json.loads((out / "runs.json").read_text())
    log = out / "service.log"
    steps = timings(log.read_text(encoding="utf-8", errors="replace")) if log.exists() else {}

    summary = {}
    builds_seen = {}
    for record in runs:
        name, run = record["input"], record["run"]
        meta = json.loads((inputs / name / "input.json").read_text())
        entry = summary.setdefault(name, {"input": meta, "runs": []})
        path = f"{name}/vectors.knnvec"
        nth = builds_seen.get(path, 0)
        builds_seen[path] = nth + 1
        row = {k: record[k] for k in ("run", "task_status", "wall_seconds", "size", "sha256") if k in record}
        slots = steps.get(path, [])
        row["seconds"] = slots[nth] if nth < len(slots) else record.get("seconds", {})
        if "out_of_range" in record:
            row["out_of_range"] = record["out_of_range"]
        if record["task_status"] == "COMPLETED_INDEX_BUILD":
            index = out / name / f"run-{run}.faiss"
            row["verify"] = knn_inspect(binary, "verify", str(index))["status"]
            if not meta["skip_stored_vectors"]:
                report = knn_inspect(
                    binary, "recall", "--queries", str(inputs / name / "queries.f32"),
                    "--seed", str(RECALL["seed"]), "--k", str(RECALL["k"]), str(index),
                )
                row["recall_at_10"] = report["recall"]
        entry["runs"].append(row)

    for entry in summary.values():
        recalls = [r["recall_at_10"] for r in entry["runs"] if "recall_at_10" in r]
        if recalls:
            entry["recall_at_10"] = spread(recalls)
        builds = [r["seconds"]["build"] for r in entry["runs"] if "build" in r["seconds"]]
        if builds:
            entry["build_seconds"] = spread(builds)
        hashes = {r["sha256"] for r in entry["runs"] if "sha256" in r}
        if hashes:
            entry["distinct_files"] = len(hashes)

    report = {"knn_inspect": knn_inspect_version(binary), "recall": RECALL, "inputs": summary}
    # Bytes, not text: write_text would give CRLF line endings on Windows.
    (out / "summary.json").write_bytes((json.dumps(report, indent=2) + "\n").encode())
    for name, entry in summary.items():
        print(
            name,
            "recall", entry.get("recall_at_10"),
            "build s", entry.get("build_seconds"),
            "verify", sorted({r.get("verify") for r in entry["runs"]}),
            "distinct files", entry.get("distinct_files", "-"),
        )


if __name__ == "__main__":
    main()
