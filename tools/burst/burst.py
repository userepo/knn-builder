"""Sends a burst of build requests to a remote index build service as
OpenSearch k-NN's data nodes would, all at once, and records how each ends
for the data node: built in time, refused, failed, or timed out, and how
long the data node waited before it built the segment on its own CPU.

Each request is one data node's, as k-NN 3.9.0.0's client makes it
(`RemoteIndexHTTPClient`, `RemoteIndexHTTPClientRetryStrategy`,
`RemoteIndexPoller`):

- `POST /_build`; a 408, 409, 429, 500, 502, 503 or 509, or a connection
  that fails, is retried once after 1 s, and any other answer but 200 is a
  failure, the data node building the segment itself at once.
- Its wait starts as the request returns: it sleeps three poll intervals,
  then asks `GET /_status` once an interval, each sleep stretched by a
  factor drawn from 0.8 to 1.2, while less than its timeout has passed.
  A completed job ends the wait; a failed one, a status request that
  fails, or the timeout passing sends the segment to the CPU.

The inputs are `make_inputs.py`'s, one blob per input, uploaded once: the
jobs differ by tenant id, which with the vector path makes the job id, so
a burst of many jobs needs only a few blobs. Jobs take the inputs in turn.

Once the burst has ended and the service has no job in flight, it saves
the service's metrics page (knn-builder's; the upstream service has none)
and asks the status of each job whose end its data node did not see, for
tools/bench/check_metrics.py to hold the page to the burst's own counts.
With PROMETHEUS_URL set, it waits for that Prometheus to scrape the idle
service, then saves what PROMETHEUS_QUERIES answer. A page or an answer
it cannot have is noted in the burst's file, which is written all the
same.

Runs as the driver of docker-compose.yml, in the upstream image (for
boto3). Settings from the environment: SERVICE_URL, S3_ENDPOINT_URL,
INPUTS (a directory), NAMES (comma-separated inputs under it), JOBS,
TIMEOUT_SECONDS and POLL_SECONDS (k-NN's client timeout and poll
interval), SEED, OUT (a directory), LABEL (the run's name, also the
tenant ids' prefix) and PROMETHEUS_URL (optional). Writes OUT/LABEL.json,
and for knn-builder OUT/LABEL-metrics.txt.

`DataNode` and `at_once` are the data node's model and the burst, and
`scrape_when_idle` and `ask_ends` what follows it, for tools/bench/run.py
to drive the same way; they need the standard library only.
"""

import hashlib
import http.client
import json
import math
import os
import pathlib
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# RemoteIndexHTTPClientRetryStrategy: one retry, a second apart.
RETRIED = {408, 409, 429, 500, 502, 503, 509}
RETRY_DELAY = 1.0
# RemoteIndexPoller.
FIRST_POLL = 3
JITTER = (0.8, 1.2)
# The client's own socket timeout is not k-NN's to set here; a request this
# slow counts as one that failed.
REQUEST_TIMEOUT = 60.0
# How often the metrics page is asked whether any job is still in flight,
# and for how long at most.
IDLE_POLL = 1.0
IDLE_TIMEOUT = 900.0
# What a scrape can fail with: the connection or a wait that gave up, a
# response cut short, a page or an answer that does not parse.
SCRAPE_ERRORS = (OSError, http.client.HTTPException, ValueError)
# What a Prometheus scraping the burst's service is asked once it has
# scraped the idle service: whether it read the page, the counts the check
# holds to the page, and the queries an operator would run. A window of
# 15 minutes takes in a burst from the service's start.
PROMETHEUS_QUERIES = {
    "up": "up",
    "samples_scraped": "scrape_samples_scraped",
    "build_requests": "sum by (result) (knn_builder_build_requests_total)",
    "jobs_ended": "sum by (outcome) (knn_builder_jobs_ended_total)",
    "refusal_share": (
        'sum(increase(knn_builder_build_requests_total{result=~"store_full|too_large|late|cpu_sooner"}[15m]))'
        " / sum(increase(knn_builder_build_requests_total[15m]))"
    ),
    "drain_seconds_max": "max_over_time(knn_builder_drain_seconds[15m])",
    "forecast_ratio_p50": (
        "histogram_quantile(0.5, sum by (le) (increase(knn_builder_forecast_ratio_bucket[15m])))"
    ),
    "build_seconds_p90": (
        'histogram_quantile(0.9, sum by (le) (increase(knn_builder_stage_seconds_bucket{stage="build"}[15m])))'
    ),
    "store_fullness": "knn_builder_jobs_held / knn_builder_jobs_held_limit",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def put_inputs(endpoint, bucket, inputs, create=True):
    """Uploads each input's two blobs to `bucket` unless the same blobs
    are there; `inputs` maps each name to its directory. Each blob goes up
    with its SHA-256 as metadata, and one there under the name with
    another, or none, is replaced: a bucket that outlives a run would
    otherwise serve an input's old vectors after the input changed. The
    bucket is created first when `create` says so, as LocalStack's must
    be; a bucket on real S3 is the user's and must exist."""
    import boto3  # noqa: PLC0415  (the upstream image's; the driver only)
    import botocore.exceptions  # noqa: PLC0415

    s3 = boto3.client("s3", endpoint_url=endpoint)
    if create:
        try:
            s3.create_bucket(Bucket=bucket)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise
    for name, directory in inputs.items():
        for blob in ("vectors.knnvec", "doc_ids.knndid"):
            key = f"{name}/{blob}"
            digest = sha256(directory / blob)
            try:
                there = s3.head_object(Bucket=bucket, Key=key)["Metadata"].get("sha256")
            except botocore.exceptions.ClientError:
                there = None
            if there != digest:
                s3.upload_file(
                    str(directory / blob), bucket, key,
                    ExtraArgs={"Metadata": {"sha256": digest}},
                )
                print(f"put {key}", flush=True)


class DataNode:
    """k-NN's data nodes, each remote build of a segment as its client
    makes it: `service` the service's URL, `inputs` each input's directory
    by name, `bucket` where their blobs are, `label` the tenant ids'
    prefix, `timeout` and `poll` the client's seconds, `seed` the jitter's."""

    def __init__(self, service, inputs, bucket, label, timeout, poll, seed=1):
        self.service = service
        self.inputs = inputs
        self.bucket = bucket
        self.label = label
        self.timeout = float(timeout)
        self.poll = float(poll)
        self.seed = int(seed)

    def request(self, method, path, body=None):
        """One request, retried as k-NN retries it: (status, parsed body)."""
        data = None if body is None else json.dumps(body).encode()
        for attempt in (0, 1):
            req = urllib.request.Request(
                self.service + path, data=data, method=method,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in RETRIED and attempt == 0:
                    time.sleep(RETRY_DELAY)
                    continue
                try:
                    detail = json.loads(e.read())
                except ValueError:
                    detail = None
                return e.code, detail
            except OSError as e:
                if attempt == 0:
                    time.sleep(RETRY_DELAY)
                    continue
                return None, str(e)
        raise AssertionError("unreachable")

    def body(self, index, name):
        meta = json.loads((self.inputs[name] / "input.json").read_text())
        return {
            "container_name": self.bucket,
            "vector_path": f"{name}/vectors.knnvec",
            "doc_id_path": f"{name}/doc_ids.knndid",
            "tenant_id": f"{self.label}-{index}",
            "dimension": meta["dimension"],
            "doc_count": meta["doc_count"],
            "data_type": meta["data_type"],
            "index_parameters": {"space_type": meta["space_type"]},
        }

    def build(self, index, name, start, record):
        """One data node's remote build of one segment, its outcome and
        the seconds from `start` (a monotonic time) into `record`."""
        rng = random.Random(self.seed * 100_003 + index)
        jitter = lambda: rng.uniform(*JITTER)  # noqa: E731
        sent = time.monotonic()
        status, answer = self.request("POST", "/_build", self.body(index, name))
        if status != 200:
            record.update(outcome="refused", status=status, detail=answer,
                          cpu_after=time.monotonic() - start)
            return
        job_id = answer["job_id"]
        waiting = time.monotonic()
        record["job_id"] = job_id
        record["post_seconds"] = waiting - sent
        time.sleep(self.poll * FIRST_POLL * jitter())
        while time.monotonic() - waiting < self.timeout:
            status, answer = self.request("GET", f"/_status/{job_id}")
            if status != 200 or not answer or "task_status" not in answer:
                record.update(outcome="status_failed", status=status, detail=answer,
                              cpu_after=time.monotonic() - start)
                return
            task = answer["task_status"]
            if task == "COMPLETED_INDEX_BUILD":
                record.update(outcome="built", seen_after=time.monotonic() - start,
                              file_name=answer.get("file_name"))
                return
            if task == "FAILED_INDEX_BUILD":
                record.update(outcome="failed", detail=answer.get("error_message"),
                              cpu_after=time.monotonic() - start)
                return
            time.sleep(self.poll * jitter())
        record.update(outcome="timed_out", cpu_after=time.monotonic() - start)

    def wait_for_heart_beat(self):
        for _ in range(120):
            try:
                if self.request("GET", "/_heart_beat")[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(1)
        raise SystemExit("the service never answered its heart beat")


def at_once(node, jobs):
    """`jobs`, (index, input name) pairs, sent all at once; the records
    in that order, each with its outcome."""
    records = [{"index": index, "input": name} for index, name in jobs]
    gate = threading.Barrier(len(records) + 1)
    start = None

    def run(record):
        try:
            gate.wait()
        except threading.BrokenBarrierError:  # the burst was interrupted before it was sent
            record.update(outcome="error", detail="not sent", cpu_after=0.0)
            return
        try:
            node.build(record["index"], record["input"], start, record)
        except Exception as e:  # noqa: BLE001  (one job's, not the burst's)
            record.update(outcome="error", detail=repr(e), cpu_after=time.monotonic() - start)

    threads = [threading.Thread(target=run, args=(r,), daemon=True) for r in records]
    for t in threads:
        t.start()
    start = time.monotonic()
    gate.wait()
    for t in threads:
        t.join()
    return records


def page_values(text):
    """The samples of a Prometheus text page, each series as the page
    writes its name and labels, with its value."""
    values = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            series, _, value = line.rpartition(" ")
            values[series] = float(value)
    return values


def scrape_when_idle(service, timeout=IDLE_TIMEOUT):
    """`service`'s metrics page once it is idle, or None for a service
    with no page (the upstream one answers 404). Idle is no job in flight
    and every job taken ended, by the page's own counts: the pipeline lets
    go of a job a moment before the service counts its end. A job a data
    node gave up on runs until the service ends it, at its deadline at the
    latest."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(f"{service}/metrics", timeout=REQUEST_TIMEOUT) as resp:
                page = resp.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        values = page_values(page)
        in_flight = sum(v for s, v in values.items() if s.startswith("knn_builder_jobs_in_flight{"))
        taken = values.get('knn_builder_build_requests_total{result="accepted"}', 0.0)
        ended = sum(v for s, v in values.items() if s.startswith("knn_builder_jobs_ended_total{"))
        if in_flight == 0 and ended == taken:
            return page
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"after {timeout:g} s, {in_flight:g} jobs in flight and {taken - ended:g} not ended"
            )
        time.sleep(IDLE_POLL)


def seen_end(record):
    """Whether the runner saw its job end: run.py's own builds by their
    status, a data node's by its outcome."""
    return (
        record.get("task_status") in ("COMPLETED_INDEX_BUILD", "FAILED_INDEX_BUILD")
        or record.get("outcome") in ("built", "failed")
    )


def ask_ends(node, records):
    """The status of each job of `records` whose end the runner did not
    see (a data node that timed out, or a status request that failed),
    asked once the service is idle, into its record as `final`."""
    for record in records:
        if record.get("job_id") and not seen_end(record):
            try:
                status, answer = node.request("GET", f"/_status/{record['job_id']}")
            except SCRAPE_ERRORS as e:  # an answer cut short or no JSON
                status, answer = None, repr(e)
            record["final"] = answer if status == 200 else {"status": status, "detail": answer}


def query_prometheus(url, after, timeout=120.0):
    """What PROMETHEUS_QUERIES answer at `url`, once it has scraped the
    service at or after `after` (Unix seconds): each query's samples, as
    (labels, value) pairs."""
    def query(expr):
        q = urllib.parse.urlencode({"query": expr})
        with urllib.request.urlopen(f"{url}/api/v1/query?{q}", timeout=REQUEST_TIMEOUT) as resp:
            answer = json.loads(resp.read())
        if answer.get("status") != "success":
            raise ValueError(f"Prometheus refused {expr}: {answer}")
        return [(r["metric"], float(r["value"][1])) for r in answer["data"]["result"]]

    deadline = time.monotonic() + timeout
    while True:
        try:
            scraped = query("timestamp(knn_builder_build_info)")
        except SCRAPE_ERRORS:
            scraped = []
        if scraped and scraped[0][1] >= after:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Prometheus had not scraped the idle service after {timeout:g} s")
        time.sleep(IDLE_POLL)
    # NaN, as a quantile of no observations is, and the infinity of one
    # past the last bucket, are no JSON numbers.
    return {
        name: [[labels, value if math.isfinite(value) else None] for labels, value in query(expr)]
        for name, expr in PROMETHEUS_QUERIES.items()
    }


def after_burst(node, records, prometheus_url):
    """What follows a burst: the service's page once idle (None for a
    service with none), each unseen end asked into `records`, and what the
    Prometheus at `prometheus_url`, when given, answers; with what kept
    the page or the answers from being had, by the summary's key. The
    burst's own results never wait on any of it."""
    page = prometheus = None
    errors = {}
    try:
        page = scrape_when_idle(node.service)
    except SCRAPE_ERRORS as e:
        errors["metrics_error"] = repr(e)
    if page is not None:
        idle = time.time()
        ask_ends(node, records)
        if prometheus_url:
            try:
                prometheus = query_prometheus(prometheus_url, idle)
            except SCRAPE_ERRORS as e:
                errors["prometheus_error"] = repr(e)
    return page, prometheus, errors


def summarize(records):
    """The burst's outcome counts, the jobs sent to the CPU (every outcome
    but built, a job whose request raised among them) and their wait, and
    when the last job built was seen ended."""
    outcomes = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    to_cpu = [r for r in records if r["outcome"] != "built"]
    return {
        "outcomes": outcomes,
        "to_cpu": len(to_cpu),
        "cpu_wait_seconds": round(sum(r["cpu_after"] for r in to_cpu), 1),
        "built_seen_after_max": round(
            max((r["seen_after"] for r in records if r["outcome"] == "built"), default=0.0), 1
        ),
    }


def main():
    bucket = "burst"
    inputs_dir = pathlib.Path(os.environ["INPUTS"])
    names = os.environ["NAMES"].split(",")
    jobs = int(os.environ["JOBS"])
    out = pathlib.Path(os.environ["OUT"])
    label = os.environ["LABEL"]
    node = DataNode(
        os.environ["SERVICE_URL"],
        {name: inputs_dir / name for name in names},
        bucket,
        label,
        os.environ["TIMEOUT_SECONDS"],
        os.environ["POLL_SECONDS"],
        os.environ.get("SEED", "1"),
    )
    put_inputs(os.environ["S3_ENDPOINT_URL"], bucket, node.inputs)
    node.wait_for_heart_beat()
    records = at_once(node, [(i, names[i % len(names)]) for i in range(jobs)])
    out.mkdir(parents=True, exist_ok=True)
    page, prometheus, errors = after_burst(node, records, os.environ.get("PROMETHEUS_URL"))
    if page is not None:
        (out / f"{label}-metrics.txt").write_bytes(page.encode())
    metas = {name: json.loads((node.inputs[name] / "input.json").read_text()) for name in names}
    summary = {
        "label": label,
        "service": node.service,
        "jobs": jobs,
        "inputs": names,
        # What the check needs of each input: the vectors it counts built.
        "input_meta": {
            name: {k: meta[k] for k in ("data_type", "doc_count")} for name, meta in metas.items()
        },
        "timeout_seconds": node.timeout,
        "poll_seconds": node.poll,
        "seed": node.seed,
        **summarize(records),
        "metrics": None if page is None else f"{label}-metrics.txt",
        "prometheus": prometheus,
        **errors,
        "records": records,
    }
    (out / f"{label}.json").write_bytes((json.dumps(summary, indent=1) + "\n").encode())
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
