"""Builds every input under $INPUTS $RUNS times through the remote vector
index build service at $SERVICE_URL and keeps each index file it writes.

Runs as the runner of docker-compose.yml, in the service's own image (for
boto3). Builds run one at a time, so the service's per-step DEBUG timings
can be matched to runs by order. Each run uses its own tenant_id: the
service derives a job id from vector_path and tenant_id, and would hand
back the earlier job for a repeat.

Writes $OUT/<input>/run-<k>.faiss and $OUT/runs.json.
"""

import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

import boto3
import botocore.exceptions

BUCKET = "baseline"
SERVICE = os.environ["SERVICE_URL"]
# A build that runs longer than this is recorded as timed out; upstream's
# e2e test allows 20 minutes too.
JOB_TIMEOUT_SECONDS = 1200


def http(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        SERVICE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def wait_for_service():
    for _ in range(120):
        try:
            http("GET", "/_heart_beat")
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)
    sys.exit("the service did not come up")


def request(name, meta, run):
    return {
        "repository_type": "s3",
        "container_name": BUCKET,
        "vector_path": f"{name}/vectors.knnvec",
        "doc_id_path": f"{name}/doc_ids.knndid",
        "tenant_id": f"run-{run}",
        "dimension": meta["dimension"],
        "doc_count": meta["doc_count"],
        "data_type": meta["data_type"],
        "engine": "faiss",
        "index_parameters": {
            "space_type": meta["space_type"],
            "algorithm": "hnsw",
            "algorithm_parameters": {
                "ef_construction": meta["ef_construction"],
                "ef_search": meta["ef_search"],
                "m": meta["m"],
            },
        },
        "skip_stored_vectors": meta["skip_stored_vectors"],
    }


def main():
    inputs = pathlib.Path(os.environ["INPUTS"])
    out = pathlib.Path(os.environ["OUT"])
    runs = int(os.environ["RUNS"])
    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])

    wait_for_service()
    for _ in range(60):
        try:
            s3.create_bucket(Bucket=BUCKET)
            break
        except s3.exceptions.BucketAlreadyOwnedByYou:
            break
        except botocore.exceptions.EndpointConnectionError:  # LocalStack still starting
            time.sleep(1)
    else:
        sys.exit("LocalStack did not come up")

    names = sorted(p.name for p in inputs.iterdir() if (p / "input.json").exists())
    metas = {}
    for name in names:
        metas[name] = json.loads((inputs / name / "input.json").read_text())
        for blob in ("vectors.knnvec", "doc_ids.knndid"):
            s3.upload_file(str(inputs / name / blob), BUCKET, f"{name}/{blob}")

    records = []
    failed = False
    for run in range(1, runs + 1):
        for name in names:
            body = request(name, metas[name], run)
            start = time.monotonic()
            job = http("POST", "/_build", body)["job_id"]
            while True:
                status = http("GET", f"/_status/{job}")
                if status["task_status"] != "RUNNING_INDEX_BUILD":
                    break
                if time.monotonic() - start > JOB_TIMEOUT_SECONDS:
                    status = {"task_status": "TIMED_OUT", "error_message": "still running"}
                    break
                time.sleep(0.2)
            # Polled every 0.2 s, so this overstates a short build by up to
            # that much; the per-step seconds come from the service's log.
            wall = time.monotonic() - start
            record = {
                "input": name,
                "run": run,
                "job_id": job,
                "task_status": status["task_status"],
                "wall_seconds": round(wall, 3),
            }
            if status["task_status"] == "COMPLETED_INDEX_BUILD":
                record["file_name"] = status["file_name"]
                local = out / name / f"run-{run}.faiss"
                local.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, f"{name}/{status['file_name']}", str(local))
                data = local.read_bytes()
                record["size"] = len(data)
                record["sha256"] = hashlib.sha256(data).hexdigest()
            else:
                record["error_message"] = status.get("error_message")
                failed = True
            print(json.dumps(record), flush=True)
            records.append(record)

    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_bytes((json.dumps(records, indent=2) + "\n").encode())
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
