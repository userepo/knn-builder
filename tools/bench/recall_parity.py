"""Holds knn-builder's recall check to knn-inspect on the files it stored.

A run of tools/bench/run.py with --recall-queries records what each job's
check found. `compare` scores each job's fetched file with
`knn-inspect recall --sample`, with the same count, k and seed, and the two
recalls must be equal: the check searches the job's vectors and graph in
memory as knn-inspect searches the file written from them.

A job that leaves its vectors out of the file (skip_stored_vectors) has no
file knn-inspect can search that way: k-NN searches such a graph over the
quantized vectors Lucene keeps on the data node. Its check gives the
graph's recall over the full vectors, so it is held to its own graph with
the vectors put back: knn-write's `with_vectors` example writes that file
from the job's file and its input's vector blob. (The file of another
build of the same input would not do: two builds' graphs differ.) A binary
job of that kind, which k-NN never sends, is left unscored: knn-format
reads no binary graph without its vectors.

  recall_parity.py inputs OUT KIND INPUT_DIR...
      For each input directory, OUT/<name>-<KIND>: its input.json changed
      and its other files hard links of the input's (copies, where the
      file system links none). KIND is
        skip        requested with skip_stored_vectors
        fp16from32  a float input's fp32 blob requested as half_float,
                    which the service converts to fp16
  recall_parity.py compare RUN_DIR KNN_INSPECT WITH_VECTORS [OUT_JSON]
      Compares every job of the run's first single round, prints each,
      writes OUT_JSON when given, and exits 1 unless every job was checked
      and every one scored was equal.

Needs Python 3, a knn-inspect binary and the example's
(cargo build --release -p knn-write --example with_vectors), nothing else.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# knn-builder's seed for its sample and its searches' entry draws.
SEED = 1
KINDS = {
    "skip": {"skip_stored_vectors": True},
    "fp16from32": {"data_type": "half_float"},
}


def make_inputs(out, kind, directories):
    if kind not in KINDS:
        sys.exit(f"no kind {kind}; one of {', '.join(KINDS)}")
    for directory in directories:
        meta = json.loads((directory / "input.json").read_text())
        if kind == "fp16from32" and (meta["data_type"], meta.get("blob_dtype")) != ("float", "<f4"):
            sys.exit(f"{directory.name} is not a float input of fp32 values")
        meta.update(KINDS[kind])
        meta["derived_from"] = directory.name
        target = out / f"{directory.name}-{kind}"
        target.mkdir(parents=True, exist_ok=True)
        for source in directory.iterdir():
            link = target / source.name
            if source.name == "input.json":
                link.write_bytes((json.dumps(meta, indent=2) + "\n").encode())
                continue
            link.unlink(missing_ok=True)
            try:
                os.link(source, link)
            except OSError:
                shutil.copyfile(source, link)
        print(f"wrote {target}")


def run(binary, *args):
    # A path resolved, since Windows finds no program by a relative path
    # with slashes; a bare name left for the search of PATH.
    if "/" in binary or os.sep in binary:
        binary = str(pathlib.Path(binary).resolve())
    try:
        out = subprocess.run([binary, *args], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        sys.exit(f"no program {binary}")
    if out.returncode != 0:
        sys.exit(f"{binary} {' '.join(args)} failed:\n{out.stderr.strip()}")
    return out.stdout


def compare(run_dir, knn_inspect, with_vectors, out):
    record = json.loads((run_dir / "run.json").read_text())
    inputs = record["inputs"]
    jobs = {r["input"]: r for r in record.get("single", []) if r.get("run") == 1}
    report = {
        "service": record["service"],
        "image_id": record["image_id"],
        "knn_inspect": run(knn_inspect, "--version").strip(),
        "seed": SEED,
        "inputs": {},
    }
    failed, unscored = [], []
    for name in sorted(inputs):
        meta = inputs[name]
        row = {k: meta[k] for k in ("data_type", "space_type", "doc_count", "dimension",
                                    "skip_stored_vectors")}
        report["inputs"][name] = row
        job = jobs.get(name, {})
        found = job.get("recall_check")
        if job.get("task_status") != "COMPLETED_INDEX_BUILD" or found is None:
            row["error"] = job.get("error_message") or "the job reported no recall"
            failed.append(name)
            print(f"FAILED  {name}: {row['error']}", flush=True)
            continue
        row["job"] = found
        path = run_dir / name / "run-1.faiss"
        scored = "its file"
        if meta["skip_stored_vectors"]:
            if meta["data_type"] == "binary":
                row["unscored"] = "knn-format reads no binary graph without its vectors"
                unscored.append(name)
                print(f"unscored {name}: the job's {found['recall']}; {row['unscored']}",
                      flush=True)
                continue
            graph_only, path = path, run_dir / name / "with-vectors.faiss"
            run(with_vectors, str(graph_only), str(ROOT / meta["dir"] / "vectors.knnvec"),
                str(path))
            scored = "its graph with the vectors put back"
        theirs = json.loads(run(
            knn_inspect, "recall", "--sample", str(found["queries"]), "--k", str(found["k"]),
            "--seed", str(SEED), "--json", str(path),
        ))["recall"]
        if meta["skip_stored_vectors"]:
            path.unlink()
        row["knn_inspect"] = {"recall": theirs, "of": scored}
        row["equal"] = theirs == found["recall"]
        if not row["equal"]:
            failed.append(name)
        print(f"{'equal  ' if row['equal'] else 'DIFFERS'} {name}: the job's {found['recall']}, "
              f"knn-inspect's {theirs} on {scored} (recall@{found['k']}, "
              f"{found['queries']} queries)", flush=True)
    if out is not None:
        out.write_bytes((json.dumps(report, indent=1) + "\n").encode())
        print(f"wrote {out}")
    equal = len(inputs) - len(failed) - len(unscored)
    print(f"{equal} of {len(inputs)} equal, {len(unscored)} unscored, {len(failed)} failed")
    return 1 if failed else 0


def main():
    args = sys.argv[1:]
    if len(args) >= 4 and args[0] == "inputs":
        make_inputs(pathlib.Path(args[1]), args[2], [pathlib.Path(a) for a in args[3:]])
    elif len(args) in (4, 5) and args[0] == "compare":
        out = pathlib.Path(args[4]) if len(args) == 5 else None
        sys.exit(compare(pathlib.Path(args[1]), args[2], args[3], out))
    else:
        sys.exit(__doc__.split("\n\n")[3])


if __name__ == "__main__":
    main()
