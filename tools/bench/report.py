"""Reduces run.py's runs to the benchmark's tables, in Markdown on standard
output: for each data type, one table of the single builds (per input and
service: the median job latency as the client saw it, the steps' sum from
the service's own clock, per-step seconds, peak host and device memory,
and recall@10 of the first run's file) and one of the concurrency
metrics (per service: the concurrent jobs' latency p50 and p99, the burst's
outcomes, and the sustained queue's jobs per hour).

The steps are aligned across the services: the upstream service logs
download, build, conversion, write and upload; knn-builder download, build
and upload with the write streamed inside the upload; so the table's build
is the upstream build, conversion and write together.

Recall is knn-inspect's seeded search (seed 1, k 10) over each input's
held-out queries, as tools/baseline/score.py runs it; each run directory
keeps the results in recall.json, so a second report reads them.

Usage: python tools/bench/report.py KNN_INSPECT RUN_DIR...

Standard library only.
"""

import json
import math
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RECALL = {"seed": 1, "k": 10}


def knn_inspect(binary, *args):
    out = subprocess.run([binary, *args, "--json"], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        sys.exit(f"knn-inspect {' '.join(args)} failed:\n{out.stderr.strip()}")
    return json.loads(out.stdout)


def knn_inspect_version(binary):
    out = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def recall(binary, run_dir, run):
    """recall@10 of each input's first file, from run_dir/recall.json or
    knn-inspect."""
    cache = run_dir / "recall.json"
    found = json.loads(cache.read_bytes()) if cache.exists() else {}
    found.setdefault("recall", RECALL)
    found.setdefault("inputs", {})
    indexes = {name: run_dir / name / "run-1.faiss" for name in run["inputs"]}
    indexes = {name: path for name, path in indexes.items() if path.exists()}
    if not indexes:
        # Nothing to score (the committed runs keep no index files): the
        # cache stands as it is, whatever knn-inspect is at hand.
        return found["inputs"]
    version = knn_inspect_version(binary)
    if found.get("knn_inspect") != version:
        # Another knn-inspect: the files at hand are scored again, the
        # recalls of files no longer at hand kept with their version.
        found["inputs"] = {n: r for n, r in found["inputs"].items() if n not in indexes}
    found["knn_inspect"] = version
    changed = False
    for name, info in run["inputs"].items():
        index = indexes.get(name)
        if name in found["inputs"] or index is None:
            continue
        changed = True
        directory = ROOT / info["dir"]
        meta = json.loads((directory / "input.json").read_bytes())
        if meta.get("skip_stored_vectors"):
            found["inputs"][name] = None  # a graph without its vectors
            continue
        queries = directory / "queries.u8"
        if not queries.exists():
            queries = directory / "queries.f32"
        report = knn_inspect(
            binary, "recall", "--queries", str(queries),
            "--seed", str(RECALL["seed"]), "--k", str(RECALL["k"]), str(index),
        )
        found["inputs"][name] = report["recall"]
        print(f"recall {run['service']} {name}: {report['recall']}", file=sys.stderr, flush=True)
    if changed:
        cache.write_bytes((json.dumps(found, indent=1) + "\n").encode())
    return found["inputs"]


def median(values):
    return statistics.median(values) if values else None


def percentile(values, p):
    """The nearest-rank percentile."""
    if not values:
        return None
    values = sorted(values)
    return values[max(math.ceil(p / 100 * len(values)) - 1, 0)]


def build_seconds(seconds):
    """The build step across the services: build, conversion and write."""
    return sum(seconds.get(k, 0.0) for k in ("build", "conversion", "write"))


def fmt(value, digits=1):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def gb(value):
    return "-" if value is None else f"{value / 1e9:.2f}"


def single_rows(run, recalls):
    """Per input: the medians over its completed single builds."""
    rows = {}
    by_input = {}
    for r in run.get("single", []):
        if r["task_status"] == "COMPLETED_INDEX_BUILD":
            by_input.setdefault(r["input"], []).append(r)
    for name, records in by_input.items():
        # A record whose log block lacked a step line has no step medians.
        with_steps = [
            r for r in records
            if r.get("seconds") and {"download", "build", "upload"} <= r["seconds"].keys()
        ]
        rows[name] = {
            "n": len(records),
            "latency": median([r["wall_seconds"] for r in records]),
            "steps": median([sum(r["seconds"].values()) for r in with_steps]),
            "download": median([r["seconds"]["download"] for r in with_steps]),
            "build": median([build_seconds(r["seconds"]) for r in with_steps]),
            "upload": median([r["seconds"]["upload"] for r in with_steps]),
            "host_peak": median([r["host_peak_bytes"] for r in records]),
            "device_peak": median(
                [r["device_peak_bytes"] for r in records if r.get("device_peak_bytes") is not None]
            ),
            "recall": recalls.get(name),
        }
    return rows


def concurrency_row(run, data_type):
    row = {}
    if group := run.get("concurrent", {}).get(data_type):
        built = [r["seen_after"] for r in group["records"] if r["outcome"] == "built"]
        row["concurrent"] = f"{group['jobs']} x {group['rounds']}"
        row["built"] = f"{len(built)} of {len(group['records'])}"
        row["p50"] = percentile(built, 50)
        row["p99"] = percentile(built, 99)
    if group := run.get("burst", {}).get(data_type):
        o = group["outcomes"]
        other = sum(n for k, n in o.items() if k not in ("built", "refused", "failed", "timed_out"))
        row["burst"] = (
            f"{group['jobs']}: {o.get('built', 0)} built, {o.get('refused', 0)} refused, "
            f"{o.get('failed', 0)} failed, {o.get('timed_out', 0)} timed out"
            + (f", {other} other" if other else "")
        )
        row["cpu_wait"] = group["cpu_wait_seconds"]
    if group := run.get("sustained", {}).get(data_type):
        row["jobs_per_hour"] = group["jobs_per_hour"]
        row["sustained"] = f"{group['queue']} in flight for {group['seconds']} s"
    return row


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: report.py KNN_INSPECT RUN_DIR...")
    binary, *dirs = sys.argv[1:]
    runs = []
    for d in map(pathlib.Path, dirs):
        run = json.loads((d / "run.json").read_bytes())
        run["_recalls"] = recall(binary, d, run)
        run["_rows"] = single_rows(run, run["_recalls"])
        runs.append(run)
    types = {}
    for run in runs:
        for name, info in run["inputs"].items():
            types.setdefault(info["data_type"], set()).add(name)

    for data_type in sorted(types):
        print(f"### {data_type}\n")
        print("| Input | Service | Builds | Latency s | Steps s | Download s | Build s "
              "| Upload s | Host peak GB | Device peak GB | Recall@10 |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name in sorted(types[data_type]):
            for run in runs:
                row = run["_rows"].get(name)
                if row is None:
                    continue
                print(
                    f"| `{name}` | {run['service']} | {row['n']} | {fmt(row['latency'])} "
                    f"| {fmt(row['steps'])} "
                    f"| {fmt(row['download'])} | {fmt(row['build'])} | {fmt(row['upload'])} "
                    f"| {gb(row['host_peak'])} | {gb(row['device_peak'])} "
                    f"| {fmt(row['recall'], 3)} |"
                )
        print()
        print("| Service | Concurrent jobs | Built | Latency p50 s | p99 s | Burst "
              "| CPU wait s | Sustained | Jobs per hour |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for run in runs:
            row = concurrency_row(run, data_type)
            print(
                f"| {run['service']} | {row.get('concurrent', '-')} | {row.get('built', '-')} "
                f"| {fmt(row.get('p50'))} | {fmt(row.get('p99'))} | {row.get('burst', '-')} "
                f"| {fmt(row.get('cpu_wait'))} | {row.get('sustained', '-')} "
                f"| {fmt(row.get('jobs_per_hour'))} |"
            )
        print()
    for run in runs:
        gpu = run["gpu"]
        print(f"{run['service']}: image `{run['image']}` ({run['image_id']}), "
              f"{gpu['name']} ({gpu['memory_mib']} MiB, driver {gpu['driver']}), "
              f"MAX_WORKERS {run['max_workers']}, started {run['started']}  ")


if __name__ == "__main__":
    main()
