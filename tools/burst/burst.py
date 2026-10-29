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

Runs as the driver of docker-compose.yml, in the upstream image (for
boto3). Settings from the environment: SERVICE_URL, S3_ENDPOINT_URL,
INPUTS (a directory), NAMES (comma-separated inputs under it), JOBS,
TIMEOUT_SECONDS and POLL_SECONDS (k-NN's client timeout and poll
interval), SEED, OUT (a directory) and LABEL (the run's name, also the
tenant ids' prefix). Writes OUT/LABEL.json.
"""

import json
import os
import pathlib
import random
import threading
import time
import urllib.error
import urllib.request

import boto3
import botocore.exceptions

BUCKET = "burst"
SERVICE = os.environ["SERVICE_URL"]
INPUTS = pathlib.Path(os.environ["INPUTS"])
NAMES = os.environ["NAMES"].split(",")
JOBS = int(os.environ["JOBS"])
TIMEOUT = float(os.environ["TIMEOUT_SECONDS"])
POLL = float(os.environ["POLL_SECONDS"])
SEED = int(os.environ.get("SEED", "1"))
OUT = pathlib.Path(os.environ["OUT"])
LABEL = os.environ["LABEL"]

# RemoteIndexHTTPClientRetryStrategy: one retry, a second apart.
RETRIED = {408, 409, 429, 500, 502, 503, 509}
RETRY_DELAY = 1.0
# RemoteIndexPoller.
FIRST_POLL = 3
JITTER = (0.8, 1.2)
# The client's own socket timeout is not k-NN's to set here; a request this
# slow counts as one that failed.
REQUEST_TIMEOUT = 60.0


def put_inputs():
    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    try:
        s3.create_bucket(Bucket=BUCKET)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise
    for name in NAMES:
        for blob in ("vectors.knnvec", "doc_ids.knndid"):
            key = f"{name}/{blob}"
            try:
                s3.head_object(Bucket=BUCKET, Key=key)
            except botocore.exceptions.ClientError:
                s3.upload_file(str(INPUTS / name / blob), BUCKET, key)
                print(f"put {key}", flush=True)


def request(method, path, body=None):
    """One request, retried as k-NN retries it: (status, parsed body)."""
    data = None if body is None else json.dumps(body).encode()
    for attempt in (0, 1):
        req = urllib.request.Request(
            SERVICE + path, data=data, method=method,
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


def body(index, name):
    meta = json.loads((INPUTS / name / "input.json").read_text())
    return {
        "container_name": BUCKET,
        "vector_path": f"{name}/vectors.knnvec",
        "doc_id_path": f"{name}/doc_ids.knndid",
        "tenant_id": f"{LABEL}-{index}",
        "dimension": meta["dimension"],
        "doc_count": meta["doc_count"],
        "data_type": meta["data_type"],
        "index_parameters": {"space_type": meta["space_type"]},
    }


def client(index, name, start, record):
    """One data node's remote build of one segment."""
    rng = random.Random(SEED * 100_003 + index)
    jitter = lambda: rng.uniform(*JITTER)  # noqa: E731
    sent = time.monotonic()
    status, answer = request("POST", "/_build", body(index, name))
    if status != 200:
        record.update(outcome="refused", status=status, detail=answer,
                      cpu_after=time.monotonic() - start)
        return
    job_id = answer["job_id"]
    waiting = time.monotonic()
    record["post_seconds"] = waiting - sent
    time.sleep(POLL * FIRST_POLL * jitter())
    while time.monotonic() - waiting < TIMEOUT:
        status, answer = request("GET", f"/_status/{job_id}")
        if status != 200 or not answer or "task_status" not in answer:
            record.update(outcome="status_failed", status=status, detail=answer,
                          cpu_after=time.monotonic() - start)
            return
        task = answer["task_status"]
        if task == "COMPLETED_INDEX_BUILD":
            record.update(outcome="built", seen_after=time.monotonic() - start)
            return
        if task == "FAILED_INDEX_BUILD":
            record.update(outcome="failed", detail=answer.get("error_message"),
                          cpu_after=time.monotonic() - start)
            return
        time.sleep(POLL * jitter())
    record.update(outcome="timed_out", cpu_after=time.monotonic() - start)


def main():
    put_inputs()
    for _ in range(120):
        try:
            if request("GET", "/_heart_beat")[0] == 200:
                break
        except OSError:
            pass
        time.sleep(1)
    else:
        raise SystemExit("the service never answered its heart beat")
    records = [{"index": i, "input": NAMES[i % len(NAMES)]} for i in range(JOBS)]
    gate = threading.Barrier(JOBS + 1)
    start = None

    def run(record):
        gate.wait()
        client(record["index"], record["input"], start, record)

    threads = [threading.Thread(target=run, args=(r,)) for r in records]
    for t in threads:
        t.start()
    start = time.monotonic()
    gate.wait()
    for t in threads:
        t.join()
    outcomes = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    to_cpu = [r for r in records if r["outcome"] != "built"]
    summary = {
        "label": LABEL,
        "service": SERVICE,
        "jobs": JOBS,
        "inputs": NAMES,
        "timeout_seconds": TIMEOUT,
        "poll_seconds": POLL,
        "seed": SEED,
        "outcomes": outcomes,
        "to_cpu": len(to_cpu),
        "cpu_wait_seconds": round(sum(r["cpu_after"] for r in to_cpu), 1),
        "built_seen_after_max": round(max((r["seen_after"] for r in records if r["outcome"] == "built"), default=0.0), 1),
        "records": records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{LABEL}.json").write_bytes((json.dumps(summary, indent=1) + "\n").encode())
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
