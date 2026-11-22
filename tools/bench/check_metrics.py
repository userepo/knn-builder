"""Holds knn-builder's metrics pages to the counts of the run that saved
them: tools/bench/run.py's, a page for each of the service's processes,
or a burst's (tools/burst/burst.py).

For each page, from the run's own records and the service's log:

- build requests by answer: each job taken, and each refusal by the kind
  its detail names (the 409, twice, as the data node retries it once, and
  each 507's three kinds and the full store);
- jobs ended by outcome: each job's end as the runner saw it, by the
  error its status named, or, for a job whose end it did not see (a data
  node that timed out), the status asked once the service was idle;
- vectors built, by data type: the vectors of the jobs completed;
- the counts of the completed jobs' histograms: job, queue and forecast
  ratio;
- with the service's log, each stage's count and seconds summed, against
  the stage times of the `DEBUG` lines of the jobs completed, and the
  recall's count;
- no job in flight.

A process or a burst whose page the run could not have fails the check,
a process the run ended before it could scrape among them, as do a run
that ended with an error and a Prometheus that was asked and did not
answer.

A retry of a job held (`held`) is the client's, and the run does not know
of one: it is listed, not checked.

Usage: python tools/bench/check_metrics.py RUN_DIR
       python tools/bench/check_metrics.py BURST_JSON [--log SERVICE_LOG]

With a burst, --log (the service's `docker compose logs`) adds the stage
times; without it they are not checked. Exits 1 on any difference.
Standard library only.
"""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "burst"))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True  # no __pycache__ in the checkout
from burst import page_values  # noqa: E402
from run import KNN_BUILDER_STEPS  # noqa: E402

# A refusal's answer by its detail, as the service words each 507.
REFUSALS = [
    ("would be seen ended in about", "late"),
    ("the data node builds it in about", "cpu_sooner"),
    ("Insufficient available resources", "too_large"),
    ("request store is at capacity", "store_full"),
    ("is already held with other parameters", "conflict"),
]
# A failed job's outcome by its error, most particular first.
FAILURES = [
    ("the job was dropped: forecast again", "forecast_late"),
    ("the data node's timeout of", "timed_out"),
    ("the recall check failed", "low_recall"),
    ("the job was canceled", "canceled"),
    ("the pipeline stopped", "canceled"),
    ("stage panicked", "panicked"),
    ("building the graph: it needs about", "device_memory"),
    ("deriving the build parameters", "bad_parameters"),
    (" is outside ", "bad_parameters"),
    ("building the graph: ", "build_failed"),
    ("writing the index", "write_failed"),
    ("uploading the index", "upload_failed"),
    ("downloading the inputs", "download_failed"),
    ("half_float vector blob", "download_failed"),
    ("do not fit in host memory", "host_memory"),
]
# The stages of the DEBUG line, as the page names them.
STAGES = {"download": "download", "build": "build", "projection": "projection",
          "recall_check": "recall_check", "upload": "upload"}
# Seconds the page and the log may differ by: the log writes nine digits.
SECONDS_TOLERANCE = 1e-6


def first(table, text):
    for words, name in table:
        if words in text:
            return name
    return None


def answer(record):
    """The service's answer to the record's request, as the page counts it."""
    if record.get("job_id"):
        return "accepted"
    status = record.get("status")
    detail = record.get("detail")
    text = json.dumps(detail) if detail is not None else ""
    # 422 for a body that is no request, 400 for one that is not UTF-8.
    if status in (400, 422):
        return "invalid"
    if status in (409, 507):
        return first(REFUSALS, text) or f"status {status}"
    return f"status {status}"


def end(record):
    """How the record's job ended, as the page counts it; None when no
    one knows."""
    status = record.get("task_status")
    outcome = record.get("outcome")
    if status == "COMPLETED_INDEX_BUILD" or outcome == "built":
        return "completed"
    if status == "FAILED_INDEX_BUILD":
        return first(FAILURES, record.get("error_message") or "")
    if outcome == "failed":
        return first(FAILURES, str(record.get("detail") or ""))
    final = record.get("final") or {}
    if final.get("task_status") == "COMPLETED_INDEX_BUILD":
        return "completed"
    if final.get("task_status") == "FAILED_INDEX_BUILD":
        return first(FAILURES, final.get("error_message") or "")
    return None


def family(values, name, label):
    """The values of the metric `name`, by its one label's value."""
    out = {}
    prefix = f"{name}{{{label}=\""
    for series, value in values.items():
        if series.startswith(prefix) and series.endswith('"}'):
            out[series[len(prefix):-2]] = value
    return out


def check_page(title, values, records, metas, steps):
    """The page's rows against the records': (what, page, run, same)."""
    rows = []

    def compare(what, page, run, same=None):
        ok = page == run if same is None else same
        rows.append((what, page, run, ok))

    answers = {}
    for r in records:
        a = answer(r)
        # The data node sends a 409 again, and gets it again.
        answers[a] = answers.get(a, 0) + (2 if a == "conflict" else 1)
    requests = family(values, "knn_builder_build_requests_total", "result")
    for result in sorted(set(requests) | set(answers)):
        page = requests.get(result)
        if result == "held":
            rows.append((f"build_requests {result}", page, "-", True))
            continue
        compare(f"build_requests {result}", page, float(answers.get(result, 0)))

    accepted = [r for r in records if r.get("job_id")]
    ends = {}
    unknown = 0
    for r in accepted:
        e = end(r)
        if e is None:
            unknown += 1
        else:
            ends[e] = ends.get(e, 0) + 1
    ended = family(values, "knn_builder_jobs_ended_total", "outcome")
    for outcome in sorted(set(ended) | set(ends)):
        compare(f"jobs_ended {outcome}", ended.get(outcome), float(ends.get(outcome, 0)))
    compare("jobs whose end no one knows", None, unknown, unknown == 0)

    completed = [r for r in accepted if end(r) == "completed"]
    vectors = {}
    for r in completed:
        meta = metas[r["input"]]
        vectors[meta["data_type"]] = vectors.get(meta["data_type"], 0) + meta["doc_count"]
    built = family(values, "knn_builder_vectors_built_total", "data_type")
    for data_type in sorted(set(built) | set(vectors)):
        compare(f"vectors_built {data_type}", built.get(data_type), float(vectors.get(data_type, 0)))

    # A page without the stages is not one with none in flight.
    in_flight = family(values, "knn_builder_jobs_in_flight", "stage")
    compare("jobs_in_flight", sum(in_flight.values()) if in_flight else None, 0.0)
    for histogram in ("job_seconds", "queue_seconds", "forecast_ratio"):
        compare(f"{histogram} count", values.get(f"knn_builder_{histogram}_count"),
                float(len(completed)))

    if steps is not None:
        logged = [steps.get(r["job_id"]) for r in completed]
        missing = sum(1 for s in logged if s is None)
        compare("completed jobs without a stage-times line", None, missing, missing == 0)
        logged = [s for s in logged if s is not None]
        for key, stage in STAGES.items():
            times = [s[key] for s in logged if key in s]
            label = f'{{stage="{stage}"}}'
            count = values.get(f"knn_builder_stage_seconds_count{label}")
            total = values.get(f"knn_builder_stage_seconds_sum{label}")
            compare(f"stage_seconds count {stage}", count, float(len(times)))
            same = total is not None and abs(total - sum(times)) <= SECONDS_TOLERANCE * max(1, len(times))
            compare(f"stage_seconds sum {stage}", total, round(sum(times), 9), same)
        checked = sum(1 for s in logged if "recall_check" in s) + ends.get("low_recall", 0)
        compare("recall count", values.get("knn_builder_recall_count"), float(checked))

    print(f"== {title}")
    width = max(len(row[0]) for row in rows)
    for what, page, run, ok in rows:
        print(f"{'ok ' if ok else 'BAD'} {what:<{width}}  page {page}  run {run}")
    return all(ok for *_, ok in rows)


def bench_steps(log):
    """Each completed job's stage seconds by job id, from knn-builder's
    `DEBUG` lines: run.py's steps, with the recall check's, which run.py
    keeps apart."""
    steps = {}
    for m in KNN_BUILDER_STEPS.finditer(log):
        steps[m[1]] = {"download": float(m[2]), "build": float(m[3]), "upload": float(m[6])}
        if m[4] is not None:
            steps[m[1]]["projection"] = float(m[4])
        if m[5] is not None:
            steps[m[1]]["recall_check"] = float(m[5])
    return steps


def bench(run_dir):
    record = json.loads((run_dir / "run.json").read_text())
    metas = record["inputs"]
    log = (run_dir / "service.log").read_text(encoding="utf-8", errors="replace")
    steps = bench_steps(log)
    records = record["warm_up"] + record.get("single", [])
    for phase in ("concurrent", "sustained", "burst"):
        for group in record.get(phase, {}).values():
            records += group["records"]
    if not record.get("scrapes"):
        sys.exit(f"{run_dir / 'run.json'} has no scrapes")
    ok = True
    for scrape in record["scrapes"]:
        if "error" in scrape:
            print(f"== process {scrape['process']}\nBAD no page: {scrape['error']}")
            ok = False
            continue
        values = page_values((run_dir / scrape["file"]).read_text())
        mine = [r for r in records if r.get("process") == scrape["process"]]
        ok &= check_page(f"process {scrape['process']}: {scrape['file']}", values, mine, metas, steps)
    # A run that stopped early scraped no page of its last process.
    scraped = {scrape["process"] for scrape in record["scrapes"]}
    for process in sorted({r["process"] for r in records if "process" in r} - scraped):
        unchecked = sum(1 for r in records if r.get("process") == process)
        print(f"== process {process}\nBAD no page: {unchecked} records not checked")
        ok = False
    if "error" in record:
        print(f"== the run\nBAD ended with an error: {record['error']}")
        ok = False
    return ok


def burst_run(path, log):
    summary = json.loads(path.read_text())
    if "metrics_error" in summary:
        sys.exit(f"{path} has no metrics page: {summary['metrics_error']}")
    if not summary.get("metrics"):
        sys.exit(f"{path} has no metrics page")
    values = page_values((path.parent / summary["metrics"]).read_text())
    steps = None
    if log is not None:
        steps = bench_steps(log.read_text(encoding="utf-8", errors="replace"))
    ok = check_page(str(path), values, summary["records"], summary["input_meta"], steps)
    if summary.get("prometheus"):
        ok &= check_prometheus(summary["prometheus"], values)
    if "prometheus_error" in summary:
        print(f"== Prometheus\nBAD no answer: {summary['prometheus_error']}")
        ok = False
    return ok


def check_prometheus(answers, values):
    """What Prometheus read of the idle service against the page."""
    rows = []
    up = [value for _, value in answers["up"]]
    rows.append(("up", up, [1.0], up == [1.0]))
    for name, metric, label in (
        ("build_requests", "knn_builder_build_requests_total", "result"),
        ("jobs_ended", "knn_builder_jobs_ended_total", "outcome"),
    ):
        seen = {labels[label]: value for labels, value in answers[name]}
        page = family(values, metric, label)
        rows.append((name, seen, page, seen == page))
    print("== Prometheus")
    for what, seen, page, ok in rows:
        print(f"{'ok ' if ok else 'BAD'} {what}  Prometheus {seen}  page {page}")
    for name in ("samples_scraped", "refusal_share", "drain_seconds_max", "forecast_ratio_p50",
                 "build_seconds_p90", "store_fullness"):
        print(f"    {name}: {[value for _, value in answers[name]]}")
    return all(ok for *_, ok in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--log", type=pathlib.Path)
    args = parser.parse_args()
    if args.path.is_dir():
        ok = bench(args.path)
    else:
        ok = burst_run(args.path, args.log)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
