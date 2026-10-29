"""Checks burst.py's model of k-NN's client against k-NN itself, on the
single-node cluster of tools/cluster/docker-compose.yml: it sets the node's
poll interval and timeout, makes several indices whose segments go to the
builder, flushes them all at once, and reads the builder's request log for
how k-NN called it.

- `cadence`: when k-NN first asked each job's status after its request
  returned, and how far apart its later polls came. burst.py waits three
  intervals for the first and one for each later, each stretched by 0.8 to
  1.2, so with 200 ms intervals the first should come 0.48 to 0.72 s in
  and the rest 0.16 to 0.24 s apart. (At k-NN's 5 s these small builds
  would end before the first poll, and show no later ones.)

Each index's vectors stay small, 12k by default, so that all of them wait
in the node's indexing buffer (10% of its 1 GB heap) until the flush: a
fuller buffer writes, and so builds, the earlier indices' segments before.
- `refused`: with the builder's own timeout (CLIENT_TIMEOUT_SECONDS) below
  k-NN's first poll, it refuses every job with 507; k-NN should send each
  build request once, never ask its status, and build the segment itself,
  as burst.py counts a refusal.

Runs on the host, with Python 3 and nothing else:

  python tools/burst/knn_check.py cadence|refused [--indices N] [--count N]
"""

import argparse
import datetime
import json
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

OPENSEARCH = "http://localhost:9200"
LOCALSTACK = "http://localhost:4567"
BUILDER = "knn-builder-cluster-builder"
REPOSITORY = "knn-builder-check"
BUCKET = "knn-builder-check"
DIMENSION = 64


def call(method, url, body=None, ndjson=False):
    data = None
    headers = {}
    if body is not None:
        data = body.encode() if ndjson else json.dumps(body).encode()
        headers["Content-Type"] = "application/x-ndjson" if ndjson else "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            text = resp.read()
            return resp.status, json.loads(text) if text else None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def setup(poll_ms, timeout):
    call("PUT", f"{LOCALSTACK}/{BUCKET}")
    status, answer = call("PUT", f"{OPENSEARCH}/_snapshot/{REPOSITORY}", {
        "type": "s3",
        "settings": {
            "bucket": BUCKET, "base_path": "vectors", "region": "us-east-1",
            "endpoint": "http://localstack:4566", "s3_upload_retry_enabled": False,
        },
    })
    assert status == 200, answer
    status, answer = call("PUT", f"{OPENSEARCH}/_cluster/settings", {"persistent": {
        "knn.remote_index_build.enabled": True,
        "knn.remote_index_build.repository": REPOSITORY,
        "knn.remote_index_build.service.endpoint": "http://builder:1025",
        "knn.remote_index_build.poll.interval": f"{poll_ms}ms",
        "knn.remote_index_build.client.timeout": f"{timeout}s",
    }})
    assert status == 200, answer


def make_index(name, count, rng):
    call("DELETE", f"{OPENSEARCH}/{name}")
    status, answer = call("PUT", f"{OPENSEARCH}/{name}", {
        "settings": {
            "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1,
            "index.knn": True,
            "index.knn.remote_index_build.enabled": True,
            "index.knn.remote_index_build.size.min": "1kb",
        },
        "mappings": {"properties": {"vec": {
            "type": "knn_vector", "dimension": DIMENSION,
            "method": {"name": "hnsw", "engine": "faiss", "space_type": "l2"},
        }}},
    })
    assert status == 200, answer
    lines = []
    for i in range(count):
        lines.append(json.dumps({"index": {"_id": str(i)}}))
        lines.append(json.dumps({"vec": [round(rng.uniform(-1, 1), 4) for _ in range(DIMENSION)]}))
    status, answer = call("POST", f"{OPENSEARCH}/{name}/_bulk", "\n".join(lines) + "\n", ndjson=True)
    assert status == 200 and not answer["errors"], str(answer)[:500]


def flush_all(names):
    results = {}

    def flush(name):
        results[name] = call("POST", f"{OPENSEARCH}/{name}/_flush?force=true&wait_if_ongoing=true")[0]

    threads = [threading.Thread(target=flush, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


LINE = re.compile(r"^(\S+Z)\s+\w+\s+(\S+): (.*)$")


def builder_log(since):
    out = subprocess.run(
        ["docker", "logs", "--since", since, BUILDER], capture_output=True, text=True, check=True
    )
    return (out.stdout + out.stderr).splitlines()


def stamp(text):
    """A log line's time, as seconds since the epoch: 2026-10-29T21:13:42.587414Z."""
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def cadence(lines):
    """Per job: seconds from its request's return to each status request."""
    queued, polls = {}, {}
    for line in lines:
        m = LINE.match(line)
        if not m:
            continue
        t, _, message = stamp(m.group(1)), m.group(2), m.group(3)
        q = re.search(r"job queued job_id=(\w+)", message)
        if q:
            queued[q.group(1)] = t
        p = re.search(r"request method=GET path=\"/_status/(\w+)\"", message)
        if p:
            polls.setdefault(p.group(1), []).append(t)
    jobs = {}
    for job_id, start in queued.items():
        times = polls.get(job_id, [])
        jobs[job_id] = {
            "first_poll_s": round(times[0] - start, 2) if times else None,
            "gaps_s": [round(b - a, 2) for a, b in zip(times, times[1:])],
        }
    return jobs


def stats():
    _, answer = call("GET", f"{OPENSEARCH}/_plugins/_knn/stats")
    node = next(iter(answer["nodes"].values()))
    return node["remote_vector_index_build_stats"]["client_stats"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cadence", "refused"])
    parser.add_argument("--indices", type=int, default=6)
    parser.add_argument("--count", type=int, default=12_000)
    parser.add_argument("--poll-ms", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    setup(args.poll_ms, args.timeout)
    rng = random.Random(7)
    names = [f"check-{args.mode}-{i}" for i in range(args.indices)]
    for name in names:
        make_index(name, args.count, rng)
    before = stats()
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 1))
    flushed = flush_all(names)
    after = stats()
    lines = builder_log(since)
    posts = [line for line in lines if 'request method=POST path="/_build"' in line]
    statuses = {}
    for line in posts:
        code = re.search(r"status=(\d+)", line).group(1)
        statuses[code] = statuses.get(code, 0) + 1
    report = {
        "mode": args.mode,
        "flush_status": flushed,
        "build_requests": statuses,
        "status_requests": sum('path="/_status/' in line for line in lines),
        "knn_client_stats_delta": {k: after[k] - before.get(k, 0) for k in after},
        "jobs": cadence(lines),
    }
    print(json.dumps(report, indent=1))
    if args.mode == "cadence":
        firsts = [j["first_poll_s"] for j in report["jobs"].values() if j["first_poll_s"] is not None]
        gaps = [g for j in report["jobs"].values() for g in j["gaps_s"]]
        print(f"first polls {min(firsts)} to {max(firsts)} s; later gaps "
              f"{min(gaps) if gaps else None} to {max(gaps) if gaps else None} s", file=sys.stderr)


if __name__ == "__main__":
    main()
