"""Scores the builds of run.py runs for the upper-level projection: each
fetched index file's level-0 components (components.py, in the oracle's
faiss-base image) and its recall over several seeds of knn-inspect's
random entry points, kept in RUN_DIR/projection.json; then, over every
RUN_DIR given, a table by input of the mean and range over each run's
builds, with the projection's outcome and time and the builds' peaks.

Recall is measured twice a file, each the mean over --seeds seeds:
  recall@10   k 10, at the file's ef_search and entry points, as k-NN
              searches a graph built this way
  recall@100  k 100, ef_search 600 from 32 random entry points

--jobs files (8 by default) score at once, each running its knn-inspect
calls in turn. A file already scored keeps its scores, so --delete-files
(each file removed once scored) leaves a run's directory small and its
scores kept.

Usage: python tools/bench/projection.py KNN_INSPECT RUN_DIR... [--seeds N]
           [--jobs N] [--delete-files] [--summary PATH]

--summary writes the table's figures as JSON too. Needs Docker and the
knn-inspect binary; the runs' inputs where run.json names them.
"""

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = "opensearchstaging/remote-vector-index-builder@sha256:241eddbe1a2fa81c188aee5edc89a81bc14415a25963f24ecd923f04c4f3d9d1"
# name: (k, ef_search, entry points); None takes the file's.
RECALLS = {"recall@10": (10, None, None), "recall@100": (100, 600, 32)}


class ToolError(Exception):
    """Docker or knn-inspect failed: the message carries what it said."""


def tool(what, args):
    """The command's standard output, or a ToolError with its own words."""
    try:
        out = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as error:
        raise ToolError(f"{what} did not start: {error}") from error
    if out.returncode != 0:
        raise ToolError(f"{what} failed ({out.returncode}):\n{out.stderr.strip()}")
    return out.stdout


def components(paths):
    """components.py's figures for each file, by path relative to ROOT."""
    rel = [p.resolve().relative_to(ROOT).as_posix() for p in paths]
    out = tool(
        "components.py in Docker",
        ["docker", "run", "--rm", "-v", f"{ROOT}:/work", "-w", "/work",
         "--entrypoint", "python", BASE, "tools/bench/components.py", "--json", *rel],
    )
    found = {}
    for line in out.splitlines():
        row = json.loads(line)
        found[row.pop("file")] = row
    return {p: found[r] for p, r in zip(paths, rel)}


def recall(knn_inspect, path, queries, k, ef, entries, seed):
    args = [knn_inspect, "recall", "--queries", str(queries), "--k", str(k),
            "--seed", str(seed), "--json"]
    if ef is not None:
        args += ["--ef-search", str(ef)]
    if entries is not None:
        args += ["--entry-points", str(entries)]
    return json.loads(tool(f"knn-inspect recall of {path}", [*args, str(path)]))["recall"]


def score(knn_inspect, run_dir, seeds, delete, jobs):
    """Scores each fetched file of the run not yet scored: projection.json."""
    run = json.loads((run_dir / "run.json").read_text())
    cache_path = run_dir / "projection.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get("seeds") not in (None, seeds):
        sys.exit(f"{cache_path} was scored over {cache['seeds']} seeds, not {seeds}")
    cache["seeds"] = seeds
    files = cache.setdefault("files", {})
    todo = [
        p for name in run["inputs"] for p in sorted((run_dir / name).glob("run-*.faiss"))
        if f"{name}/{p.name}" not in files
    ]
    if not todo:
        return run, files
    counted = components(todo)

    def recalls(path):
        queries = ROOT / run["inputs"][path.parent.name]["dir"] / "queries.f32"
        return {
            label: statistics.mean(
                recall(knn_inspect, path, queries, k, ef, entries, seed)
                for seed in range(1, seeds + 1)
            )
            for label, (k, ef, entries) in RECALLS.items()
        }

    # Files score on --jobs threads, each a knn-inspect at a time; the
    # cache is written from this thread alone, as each file finishes.
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        pending = {pool.submit(recalls, path): path for path in todo}
        try:
            for done in concurrent.futures.as_completed(pending):
                path = pending[done]
                key = f"{path.parent.name}/{path.name}"
                files[key] = {**counted[path], **done.result()}
                print(f"scored {run_dir.name} {key}: {files[key]}", file=sys.stderr, flush=True)
                cache_path.write_bytes((json.dumps(cache, indent=1) + "\n").encode())
                if delete:
                    path.unlink()
        except BaseException:
            # One file's failure ends the scoring: no file waiting starts.
            pool.shutdown(cancel_futures=True)
            raise
    return run, files


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values), "n": len(values)}


def projection_seconds(record):
    """The projection's seconds from its log line: gate, layer 1 and links."""
    p = record.get("projection") or {}
    ms = [p[k] for k in ("gate_ms", "layer1_build_ms", "projection_ms") if k in p]
    # A projection that logged no times (failed, stopped, too few promoted)
    # has none, rather than zero seconds.
    return sum(ms) / 1000 if ms else None


def summarize(run, files):
    """By input: the figures over the run's builds."""
    builds = {}
    for record in run.get("single", []):
        builds.setdefault(record["input"], []).append(record)
    out = {}
    for name in sorted(run["inputs"]):
        records = builds.get(name, [])
        scored = [s for key, s in files.items() if key.split("/")[0] == name]
        outcomes = {}
        for r in records:
            outcome = (r.get("projection") or {}).get("outcome", "none")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        out[name] = {
            "builds": len(records),
            "completed": sum(r["task_status"] == "COMPLETED_INDEX_BUILD" for r in records),
            "outcomes": outcomes,
            "strong": spread([s["strong"] for s in scored]),
            "largest": spread([s["largest"] for s in scored]),
            "recall@10": spread([s["recall@10"] for s in scored]),
            "recall@100": spread([s["recall@100"] for s in scored]),
            "reachable_before": spread([(r.get("projection") or {}).get("reachable") for r in records]),
            "promoted": spread([(r.get("projection") or {}).get("promoted") for r in records]),
            "links_added": spread([(r.get("projection") or {}).get("links_added") for r in records]),
            "projection_seconds": spread([projection_seconds(r) for r in records]),
            "wall_seconds": spread([r.get("wall_seconds") for r in records]),
            "host_peak_bytes": spread([r.get("host_peak_bytes") for r in records]),
            "device_peak_bytes": spread([r.get("device_peak_bytes") for r in records]),
        }
    return out


def fmt(s, digits=3, scale=1.0):
    if s is None:
        return "-"
    mean, lo, hi = (s[k] / scale for k in ("mean", "min", "max"))
    if lo == hi:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("knn_inspect")
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--delete-files", action="store_true")
    parser.add_argument("--summary", type=pathlib.Path)
    args = parser.parse_args()
    for value, what in ((args.seeds, "--seeds"), (args.jobs, "--jobs")):
        if value <= 0:
            sys.exit(f"{what} takes a number above 0")

    summary = {}
    for run_dir in args.run_dirs:
        try:
            run, files = score(args.knn_inspect, run_dir, args.seeds, args.delete_files, args.jobs)
        except ToolError as error:
            sys.exit(str(error))
        flag = run["settings"].get("project_link_to_base_layer", False)
        label = f"{run['service']} {'with' if flag else 'without'} the projection"
        summary[run_dir.name] = {"service": run["service"], "image_id": run["image_id"],
                                 "project_link_to_base_layer": flag, "label": label,
                                 "inputs": summarize(run, files)}

    print("| Input | Run | Builds | Outcomes | Strong components | Largest | recall@10 | recall@100"
          " | Projection s | Wall s | Host GB | Device GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    names = sorted({n for run in summary.values() for n in run["inputs"]})
    for name in names:
        for run in summary.values():
            row = run["inputs"].get(name)
            if row is None:
                continue
            outcomes = ", ".join(f"{k} {v}" for k, v in sorted(row["outcomes"].items()))
            print(f"| {name} | {run['label']} | {row['completed']}/{row['builds']} | {outcomes}"
                  f" | {fmt(row['strong'], 0)} | {fmt(row['largest'])} | {fmt(row['recall@10'])}"
                  f" | {fmt(row['recall@100'])} | {fmt(row['projection_seconds'], 2)}"
                  f" | {fmt(row['wall_seconds'], 2)} | {fmt(row['host_peak_bytes'], 2, 1e9)}"
                  f" | {fmt(row['device_peak_bytes'], 2, 1e9)} |")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_bytes((json.dumps({"seeds": args.seeds, "recalls": RECALLS, "runs": summary},
                                             indent=1) + "\n").encode())


if __name__ == "__main__":
    main()
