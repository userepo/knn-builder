"""Runs the benchmark's metrics against one remote index build service on
the GPU, over the inputs tools/bench/datasets.py wrote (or make_inputs.py's),
and records every measurement in OUT_DIR/run.json for report.py.

The service runs from docker-compose.yml beside this file, with LocalStack
for S3 or, given --s3-bucket, a bucket on real S3, and is driven from the
host, where nvidia-smi samples the GPU's memory. The phases, in order:

  single      a warm-up build of one input of each data type, then RUNS
              builds of every input one at a time: each job's latency (from
              its request to the status that shows it completed, asked
              every 0.2 s), the service process's peak resident memory
              (VmHWM, reset before each job), the GPU's memory at its peak
              during the job less what the service process held before it
              (nvidia-smi sampled every 50 ms, the measure of
              tools/baseline/device_peaks.py over the request's window,
              from the POST to the status that shows the job ended), and
              its per-step times and any upper-level projection it ran
              from the service's DEBUG log; the files of the first
              --fetch-runs runs (1 by default) are fetched to
              OUT_DIR/<input>/run-N.faiss for the recall
  concurrent  per data type, ROUNDS rounds of CONCURRENT jobs sent at once
              as k-NN's data nodes send them (tools/burst/burst.py's model,
              k-NN's default timeout and poll interval), over the data
              type's inputs in turn: each job's latency as its data node
              sees it, for the p50 and p99
  sustained   per data type, QUEUE jobs kept in flight for SUSTAIN seconds,
              the next sent as one completes, over the data type's inputs
              in turn: the jobs seen completed within the window, for jobs
              per hour (the jobs still in flight at its end are not
              counted, so an hour-long window understates the rate by at
              most QUEUE jobs)
  burst       per data type, BURST_JOBS requests at once over the data
              type's L2 (binary: Hamming) inputs, with the client timeout
              and poll interval tools/burst ran in (90 s, 5 s), the
              service restarted with them, and again before each data
              type after the first, since the upstream service builds on
              the jobs a burst's data nodes gave up on, and warmed up each
              time: how many were built in time, refused, failed or timed
              out, as the burst fixtures count them

Before each of the service's processes ends (a phase that restarts it,
or the run's end), once no job is in flight, the run saves knn-builder's
metrics page to OUT_DIR/metrics/process-N.txt and asks the status of each
job whose end it did not see; each record names its process, for
tools/bench/check_metrics.py to hold each page to the run's own counts.

A run starts from fresh services and an empty OUT_DIR record: whatever a
run before left in Docker is taken down first, and its files and recall
are replaced. A run that fails or is interrupted still takes the services
down and writes run.json with the phases that finished and the error; a
service that stops answering its heart beat ends the run the same way.

Each job's file is deleted once its job is seen completed (a run's
1,500 jobs would else leave hundreds of GB), so the bucket holds about
30 GB at most: the inputs and one concurrent round's files. LocalStack
keeps them in Docker's disk, which under Docker Desktop still grows on the
host with every block first written, by hundreds of GB over a run, until
the image is trimmed and compacted; see that the host has the room.

With --s3-bucket BUCKET the inputs and the jobs' files are on real S3, in
that bucket, which must exist and is best given to the benchmark alone:
the inputs go in at <input>/vectors.knnvec and <input>/doc_ids.knndid,
are left there for the next run (an input already there is not sent
again, unless its blobs have changed since, by their SHA-256), and the
jobs' files are deleted as above. Every container takes
AWS_DEFAULT_REGION, which must be set, and AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY and AWS_SESSION_TOKEN from this environment, or,
those unset, finds credentials by the SDKs' default chain (an instance
role, if the instance metadata service answers containers);
S3_ENDPOINT_URL, set, names an S3-compatible endpoint instead of AWS's.
S3_PAYLOAD_SIGNING_ENABLED in this environment reaches knn-builder as it
is, on either S3, and run.json records it: true has it sign every upload's
body, as it did before it left them unsigned over HTTPS.
The host reads and deletes the jobs' files through the s3 service of
docker-compose.yml, which signs with boto3, so it still needs no AWS
library.

Usage: python tools/bench/run.py OUT_DIR SERVICE [--inputs DIR ...]
           [--runs N] [--concurrent N] [--rounds N] [--queue N]
           [--sustain-seconds S] [--burst-jobs N] [--burst-timeout S]
           [--burst-poll S] [--phases LIST] [--seed N] [--project]
           [--fetch-runs N] [--s3-bucket BUCKET] [--max-workers N]
           [--recall-queries N] [--recall-k K] [NAME...]

--project asks every build for the upper-level projection
(project_link_to_base_layer), for tools/bench/projection.py.
--recall-queries turns on knn-builder's recall check (RECALL_CHECK_QUERIES,
with --recall-k as RECALL_CHECK_K, 10 by default), and each warm-up and
single job records what its check found, from the service's log, for
tools/bench/recall_parity.py; the upstream service has no such check. An
input whose input.json sets skip_stored_vectors is requested with it.

SERVICE is knn-builder (the knn-builder:dev image) or upstream (the pinned
api-snapshot image on one worker, since two builds at once crash it under
WSL2; --max-workers sets another count, such as its default 2 on native
Linux); SERVICE_IMAGE in the environment names another image. --inputs
(default target/bench/inputs) may repeat; NAMEs pick inputs under them.
GPU_MEMORY_LIMIT (8) is the GPU's memory in GB as the services are told
it, printed at the start beside the GPU's own; --s3-bucket needs it set.
SERVICE_PORT (1026) and S3_PORT (4568) are the host ports the services
answer on, on the loopback only. Needs Docker with GPU access and Compose
2.20 or later, nvidia-smi and Python 3 on the host, nothing else. Writes
OUT_DIR/run.json, service.log and gpu.csv.
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "burst"))
sys.path.insert(0, str(ROOT / "tools" / "baseline"))
sys.dont_write_bytecode = True  # no __pycache__ in the checkout
import burst  # noqa: E402
import device_peaks  # noqa: E402

UPSTREAM = "opensearchstaging/remote-vector-index-builder@sha256:e8fecaefc43cfaf4d04f35072f10baedf0ece3a539efac5340e5ab0c095da8bc"
# name: (image, MAX_WORKERS)
SERVICES = {"knn-builder": ("knn-builder:dev", "2"), "upstream": (UPSTREAM, "1")}
# LocalStack's bucket; a run on real S3 names its own.
LOCAL_BUCKET = "bench"
PHASES = ("single", "concurrent", "sustained", "burst")
# k-NN's defaults: knn.remote_index_build.client.timeout and poll_interval.
KNN_TIMEOUT = 3600
KNN_POLL = 5
# A single build's status is asked this often.
STATUS_POLL = 0.2
# A single build that runs longer than this is recorded as timed out, as
# tools/baseline/run_baseline.py does.
JOB_TIMEOUT = 1200
GPU_SAMPLE_MS = 50
COMPOSE = ["docker", "compose", "-f", str(HERE / "docker-compose.yml")]
# A docker command that runs longer than this has lost its engine (one
# hung for hours collecting the log of a service whose engine had died
# with the disk full). `up`, which pulls the images the first time, is
# not held to it.
DOCKER_TIMEOUT = 1800

# The upstream service's log: a job's block runs from its start line to
# its end line, the step lines between them for its vector path its. On
# more than one worker the blocks of different paths interleave; jobs of
# one path in flight at once cannot be told apart, and none of them gets
# step times, nor any job of the path that starts before the last of
# them has ended.
UPSTREAM_START = re.compile(r"Starting task execution for vector path: (\S+)")
UPSTREAM_END = re.compile(r"Job (\w+) with vector path (\S+) completed with status")
# A job whose record expired while it ran ends with this line instead.
UPSTREAM_EXPIRED = re.compile(r"Job \w+ with vector path (\S+) was deleted during execution")
# A build that raised, before its start line or after it: the line does
# not say which, so how many jobs of its path are in flight is no longer
# known, and the path gets no step times for the rest of the process.
UPSTREAM_RAISED = re.compile(r"Build process failed for job \w+ with vector path (\S+?): ")
# The process starting: whatever was in flight in the one before is gone.
UPSTREAM_BOOT = re.compile(r"Started server process")
UPSTREAM_STEPS = {
    "download": re.compile(r"Vector download time for vector path (\S+): ([\d.]+) seconds"),
    "build": re.compile(r"Index build time for vector path (\S+): ([\d.]+) seconds"),
    "conversion": re.compile(r"Index conversion time for vector path (\S+): ([\d.]+) seconds"),
    "write": re.compile(r"Index write time for vector path (\S+): ([\d.]+) seconds"),
    "upload": re.compile(r"Total upload time for path (\S+): ([\d.]+) seconds"),
}
# A job that projected links has its projection's time between the build's
# and the upload's, and one whose recall check ran, the check's after it.
KNN_BUILDER_STEPS = re.compile(
    r"job stage times job_id=(\w+) download_s=([\d.eE+-]+) build_s=([\d.eE+-]+)"
    r"(?: projection_s=([\d.eE+-]+))?(?: recall_check_s=([\d.eE+-]+))? upload_s=([\d.eE+-]+)"
)
# How long a job's writer waited, of the upload's seconds: for a part to
# be taken, for the recall check, and for the upload to complete.
KNN_BUILDER_WAITS = re.compile(
    r"job stage times job_id=(\w+) .*?\bupload_part_wait_s=([\d.eE+-]+)"
    r" upload_check_wait_s=([\d.eE+-]+) upload_finish_wait_s=([\d.eE+-]+)"
)
# The upper-level projection's lines in each service's log: whether it
# ran, the share of the nodes reachable before it, its counts and times.
UPSTREAM_PROJECTION = re.compile(
    r"Upper-level projection(?: (skipped))?: ([\d.]+)% of \d+ nodes reachable(.*)"
)
UPSTREAM_NOTHING = re.compile(r"Upper-level projection: (\d+) promoted, nothing to do")
UPSTREAM_FAILED = re.compile(r"Upper-level projection failed")
KNN_BUILDER_PROJECTION = re.compile(
    r"upper-level projection( skipped| stopped|: nothing to do| failed)?(?:[^\"]*?) vector_path=\"([^\"]+)\"(.*)"
)
KNN_BUILDER_COMPLETED = re.compile(r"job completed job_id=(\w+) vector_path=\"([^\"]+)\"")
# What a job's recall check found, on its completion line.
KNN_BUILDER_RECALL = re.compile(
    r"job completed job_id=(\w+) .*?\brecall=([\d.eE+-]+) recall_queries=(\d+) recall_k=(\d+)"
)
FIELD = re.compile(r"(\w+)=([\d.eE+-]+)(?=[\s),]|$)")


def now():
    return datetime.datetime.now().astimezone()


def run(*args, env=None, capture=False, timeout=DOCKER_TIMEOUT):
    """A docker command: its output, when wanted, through a file rather
    than a pipe, so that the timeout's kill of the client ends the call
    even while the compose plugin it spawned keeps the pipe open. A None
    in `env` unsets that variable."""
    merged = {k: v for k, v in {**os.environ, **(env or {})}.items() if v is not None}
    args = list(map(str, args))
    if not capture:
        subprocess.run(args, env=merged, check=True, timeout=timeout)
        return b""
    with tempfile.TemporaryFile() as out:
        subprocess.run(args, env=merged, check=True, timeout=timeout, stdout=out)
        out.seek(0)
        return out.read()


class Service:
    """The compose services: up with one set of client settings at a time,
    their logs collected before each change, down at the end. `bucket` is
    a bucket on real S3, or None for LocalStack's."""

    def __init__(self, image, max_workers, out, inputs, bucket, recall_check):
        self.bucket = bucket or LOCAL_BUCKET
        queries, k = recall_check
        self.env = {
            "SERVICE_IMAGE": image,
            "MAX_WORKERS": max_workers,
            "RECALL_CHECK_QUERIES": str(queries),
            "RECALL_CHECK_K": str(k),
            "GPU_MEMORY_LIMIT": gpu_memory_limit(),
            "SERVICE_PORT": os.environ.get("SERVICE_PORT", "1026"),
            "S3_PORT": os.environ.get("S3_PORT", "4568"),
            "BENCH_BUCKET": self.bucket,
            "INPUT_DIRS": ",".join(
                str(d.resolve().relative_to(ROOT)).replace("\\", "/") for d in inputs.values()
            ),
        }
        if bucket is None:
            # LocalStack's endpoint inside the compose network and its test
            # credentials, whatever this environment holds for real S3.
            self.s3_service = "localstack"
            self.env.update(
                COMPOSE_PROFILES="localstack",
                S3_ENDPOINT_URL="http://localstack:4566",
                AWS_ACCESS_KEY_ID="test",
                AWS_SECRET_ACCESS_KEY="test",
                AWS_SESSION_TOKEN=None,
                AWS_DEFAULT_REGION="us-east-1",
                BENCH_CREATE_BUCKET="1",
            )
        else:
            # The credentials, region and any endpoint are this
            # environment's, passed through as they are.
            self.s3_service = "s3"
            self.env.update(COMPOSE_PROFILES="s3", BENCH_CREATE_BUCKET=None)
        # `down` with both profiles, so a run takes down what a run before
        # left, whichever S3 it had.
        self.down_env = {**self.env, "COMPOSE_PROFILES": "localstack,s3"}
        # 127.0.0.1, where the compose file publishes the ports: Windows
        # tries localhost's ::1 first, and a refused connection there costs
        # each request 2 s.
        self.url = f"http://127.0.0.1:{self.env['SERVICE_PORT']}"
        self.s3 = f"http://127.0.0.1:{self.env['S3_PORT']}"
        self.log = out / "service.log"
        self.log.write_bytes(b"")
        self.settings = None
        # When each of the service's processes started, for what each held
        # on the device before a job: an earlier one's is not counted.
        self.starts = []
        # The process up, numbered from 1 as each starts.
        self.process = 0

    def start(self, timeout, poll):
        """Starts the service with these client settings; a change restarts
        it, its log kept first. The first start takes down what a run
        before left, so no job store or bucket carries over."""
        if self.settings == (timeout, poll):
            return
        if self.settings is not None:
            # Cleared first: a collection that fails or a restart that
            # fails must not have the log collected again on the way down.
            self.settings = None
            self.collect_log()
        else:
            run(*COMPOSE, "down", "-v", env=self.down_env)
        env = {
            **self.env,
            "CLIENT_TIMEOUT_SECONDS": str(timeout),
            "CLIENT_POLL_INTERVAL_SECONDS": str(poll),
        }
        # No timeout: the first `up` pulls the images, 14 GB for the
        # upstream one.
        run(*COMPOSE, "up", "-d", self.s3_service, "service", env=env, timeout=None)
        # Before the wait: what the process holds from its start shows
        # only until the GPU idles, which a slow wait for S3 can outlast.
        self.starts.append(now())
        self.process += 1
        self.settings = (timeout, poll)
        self.wait()

    def restart(self):
        """Restarts the service alone, with the settings it has: a fresh
        job store, its log kept (the container's log survives a restart,
        so it is collected whole at the end)."""
        run(*COMPOSE, "restart", "service", env=self.env)
        self.starts.append(now())
        self.process += 1
        self.wait()

    def wait(self):
        timeout, poll = self.settings
        burst.DataNode(self.url, {}, self.bucket, "", timeout, poll).wait_for_heart_beat()
        if self.s3_service == "localstack":
            self.wait_for_localstack()
        else:
            self.wait_for_bucket()

    def wait_for_localstack(self):
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"{self.s3}/_localstack/health", timeout=5) as resp:
                    if json.loads(resp.read()).get("services", {}).get("s3") in ("available", "running"):
                        return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise SystemExit("LocalStack never reported S3 up")

    def wait_for_bucket(self):
        """Waits up to two minutes for the s3 service to reach the bucket;
        ends the run at once where it will not (s3_proxy.py's 502: no
        credentials, access denied, no such bucket)."""
        reason = "the s3 service never answered"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                # s3_proxy.py answers within its own shorter timeouts.
                with urllib.request.urlopen(f"{self.s3}/_health", timeout=30):
                    return
            except urllib.error.HTTPError as e:
                reason = e.read().decode(errors="replace").strip() or f"HTTP {e.code}"
                if e.code == 502:
                    break
            except OSError as e:
                reason = str(e)
            time.sleep(1)
        raise SystemExit(f"S3 bucket {self.bucket} not reachable: {reason}")

    def upload(self):
        run(*COMPOSE, "run", "--rm", "uploader", env=self.env)

    def collect_log(self):
        out = run(*COMPOSE, "logs", "--no-color", "service", env=self.env, capture=True)
        with self.log.open("ab") as f:
            f.write(out)

    def exec(self, command):
        out = run(*COMPOSE, "exec", "-T", "service", "sh", "-c", command,
                  env=self.env, capture=True)
        return out.decode(errors="replace")

    def reset_peak_rss(self):
        self.exec("echo 5 > /proc/1/clear_refs")

    def peak_rss(self):
        """The service process's VmHWM, in bytes."""
        status = self.exec("grep VmHWM /proc/1/status")
        return int(status.split()[1]) * 1024

    def fetch(self, key, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.s3}/{self.bucket}/{urllib.parse.quote(key)}"
        with urllib.request.urlopen(url) as resp, path.open("wb") as f:
            shutil.copyfileobj(resp, f, length=8 << 20)
            expected = resp.headers.get("Content-Length")
        size = path.stat().st_size
        # A transfer cut short ends without an error from urllib.
        if expected is not None and size != int(expected):
            raise OSError(f"fetched {size} of {expected} bytes of {key}")
        return size

    def delete(self, key):
        url = f"{self.s3}/{self.bucket}/{urllib.parse.quote(key)}"
        request = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(request, timeout=60):
            pass

    def down(self):
        try:
            if self.settings is not None:
                self.collect_log()
        finally:
            run(*COMPOSE, "down", "-v", env=self.down_env)


class Sampler:
    """nvidia-smi on the host, the GPU's memory used every GPU_SAMPLE_MS."""

    def __init__(self, path):
        self.path = path
        self.file = path.open("wb")
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,memory.used",
             "--format=csv,noheader,nounits", f"-lms={GPU_SAMPLE_MS}"],
            stdout=self.file,
        )

    def stop(self):
        self.process.terminate()
        self.process.wait()
        self.file.close()

    def samples(self):
        out = []
        for line in self.path.read_text().splitlines():
            stamp, _, used = line.partition(", ")
            # A line cut short as the sampler stopped, or a value the
            # driver does not report ([N/A]), is no sample.
            try:
                local = datetime.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f")
                at = local.astimezone()
                used = int(used)
            except ValueError:
                continue
            # nvidia-smi stamps local time, which repeats an hour when
            # daylight-saving time ends: a stamp before the one before it
            # is that hour's second pass, else the builds of that hour
            # have no samples in their windows.
            if out and out[-1][0] - at > datetime.timedelta(minutes=30):
                at = local.replace(fold=1).astimezone()
            out.append((at, used))
        return out


def gpu_memory_limit():
    """GPU_MEMORY_LIMIT as docker-compose.yml passes it: this
    environment's, or 8."""
    return os.environ.get("GPU_MEMORY_LIMIT") or "8"


def gpu_info():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    name, driver, total = (x.strip() for x in out.stdout.strip().split(","))
    return {"name": name, "driver": driver, "memory_mib": int(total)}


def image_id(image):
    """The image's id, once compose has pulled it."""
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def attach_device_peaks(records, samples, starts):
    """Each build's device peak, by device_peaks.py's measure over its
    request's window, and what the service's process held before it, of
    the processes up at `starts` the last up before the build."""
    for record in records:
        if "start" not in record:
            continue
        start = datetime.datetime.fromisoformat(record["start"])
        peak, count, held = device_peaks.peak(
            samples,
            start,
            datetime.datetime.fromisoformat(record["end"]),
            max((s for s in starts if s <= start), default=None),
        )
        record["device_samples"] = count
        record["device_held_bytes"] = held << 20
        record["device_peak_bytes"] = None if peak is None else peak << 20


def steps_from_log(log):
    """Each job's step seconds by job id, from either service's log."""
    steps = {}
    for m in KNN_BUILDER_STEPS.finditer(log):
        steps[m[1]] = {"download": float(m[2]), "build": float(m[3]), "upload": float(m[6])}
        if m[4] is not None:
            steps[m[1]]["projection"] = float(m[4])
    # By vector path: the open block and the jobs of the path in flight.
    # The block is None from the second job's start until every one has
    # ended, so a job that starts as another of its path is still
    # building never takes that one's lines.
    blocks = {}

    def close(path):
        """One job of `path` ended: its block, when it ran alone."""
        block, in_flight = blocks.pop(path, (None, 0))
        if in_flight > 1:
            blocks[path] = (None, in_flight - 1)
            return None
        return block

    for line in log.splitlines():
        if UPSTREAM_BOOT.search(line):
            blocks.clear()
        elif m := UPSTREAM_START.search(line):
            block, in_flight = blocks.get(m[1], ({}, 0))
            blocks[m[1]] = (block if in_flight == 0 else None, in_flight + 1)
        elif m := UPSTREAM_END.search(line):
            if (block := close(m[2])) is not None:
                steps[m[1]] = block
        elif m := UPSTREAM_EXPIRED.search(line):
            close(m[1])
        elif m := UPSTREAM_RAISED.search(line):
            # An endless count: no start or end opens a block again.
            blocks[m[1]] = (None, float("inf"))
        else:
            for step, pattern in UPSTREAM_STEPS.items():
                if m := pattern.search(line):
                    block = blocks.get(m[1], (None, 0))[0]
                    if block is not None:
                        block[step] = float(m[2])
    return steps


def recalls_from_log(log):
    """What each job's recall check found, by job id, from knn-builder's
    log: the recall, as the service printed it, the queries and k it is
    of, and the check's seconds, which ran beside the upload's and so are
    kept out of the job's step seconds."""
    out = {
        m[1]: {"recall": float(m[2]), "queries": int(m[3]), "k": int(m[4])}
        for m in KNN_BUILDER_RECALL.finditer(log)
    }
    for m in KNN_BUILDER_STEPS.finditer(log):
        if m[5] is not None and m[1] in out:
            out[m[1]]["seconds"] = float(m[5])
    return out


def fields(text):
    return {k: float(v) for k, v in FIELD.findall(text)}


def upload_waits_from_log(log):
    """Each job's waits within its upload step by job id, from
    knn-builder's log: the seconds its writer waited for the upload to
    take a part, those the written file waited for the recall check, and
    those the upload took to complete once the last part was handed over.
    The rest of the upload's step seconds is the write. A service that
    does not log them gives none."""
    return {
        m[1]: {"part": float(m[2]), "check": float(m[3]), "finish": float(m[4])}
        for m in KNN_BUILDER_WAITS.finditer(log)
    }


def projections_from_log(log):
    """Each job's upper-level projection by job id, from either service's
    log: its outcome (projected; skipped, the graph connected; nothing, too
    few promoted; stopped; failed), the share of the nodes reachable before
    it, and its counts and milliseconds. knn-builder logs a projection by
    vector path ahead of the job's completion line, which names both, and
    the upstream service's projection lines name no path: one build at a
    time, as the single phase runs them, pairs them up."""
    out = {}
    pending = {}
    block = None
    for line in log.splitlines():
        if m := KNN_BUILDER_PROJECTION.search(line):
            outcome = {" skipped": "skipped", " stopped": "stopped", " failed": "failed",
                       ": nothing to do": "nothing"}.get(m[1], "projected")
            pending[m[2]] = {"outcome": outcome, **fields(m[3])}
        elif m := KNN_BUILDER_COMPLETED.search(line):
            if m[2] in pending:
                out[m[1]] = pending.pop(m[2])
        elif UPSTREAM_START.search(line):
            block = None
        elif m := UPSTREAM_PROJECTION.search(line):
            block = {"outcome": "skipped" if m[1] else "projected",
                     "reachable": float(m[2]) / 100, **fields(m[3])}
        elif m := UPSTREAM_NOTHING.search(line):
            block = {"outcome": "nothing", "promoted": float(m[1])}
        elif UPSTREAM_FAILED.search(line):
            block = {"outcome": "failed"}
        elif (m := UPSTREAM_END.search(line)) and block is not None:
            out[m[1]] = block
            block = None
    return out


class BenchNode(burst.DataNode):
    """A data node whose request leaves the vectors out of the file when
    its input says so (skip_stored_vectors), as k-NN's does for a field
    whose vectors Lucene keeps."""

    def body(self, index, name):
        body = super().body(index, name)
        meta = json.loads((self.inputs[name] / "input.json").read_text())
        if meta.get("skip_stored_vectors"):
            body["skip_stored_vectors"] = True
        return body


class ProjectingNode(BenchNode):
    """A data node whose requests ask for the upper-level projection, as
    k-NN sends a field's project_link_to_base_layer."""

    def body(self, index, name):
        body = super().body(index, name)
        body["index_parameters"]["algorithm_parameters"] = {"project_link_to_base_layer": True}
        return body


class Bench:
    def __init__(self, args, inputs, metas):
        self.args = args
        self.inputs = inputs
        self.metas = metas
        self.out = args.out_dir
        self.label = args.service
        image, max_workers = SERVICES.get(args.service, (None, "1"))
        if args.max_workers is not None:
            max_workers = str(args.max_workers)
        image = os.environ.get("SERVICE_IMAGE", image)
        if image is None:
            sys.exit(f"SERVICE is {', '.join(SERVICES)}, or set SERVICE_IMAGE; not {args.service}")
        gpu = gpu_info()
        print(f"GPU_MEMORY_LIMIT {gpu_memory_limit()} GB for {gpu['name']}, "
              f"which has {gpu['memory_mib'] * 2**20 / 1e9:.1f} GB", flush=True)
        # A run's directory never holds an input: OUT_DIR given as an
        # inputs directory would otherwise lose them below.
        for name in inputs:
            if (self.out / name / "input.json").exists():
                sys.exit(f"{self.out} holds the input {name}: give a run directory")
        self.out.mkdir(parents=True, exist_ok=True)
        self.service = Service(image, max_workers, self.out, inputs, args.s3_bucket,
                               (args.recall_queries, args.recall_k))
        # What a run before left here: its files, and what report.py and
        # projection.py cached of them.
        for cache in ("recall.json", "projection.json"):
            (self.out / cache).unlink(missing_ok=True)
        for name in inputs:
            shutil.rmtree(self.out / name, ignore_errors=True)
        self.stop = threading.Event()
        self.record = {
            "service": self.label,
            "image": image,
            "image_id": None,
            "max_workers": int(max_workers),
            "gpu": gpu,
            "started": now().isoformat(),
            "settings": {
                "runs": args.runs,
                "concurrent": args.concurrent,
                "rounds": args.rounds,
                "queue": args.queue,
                "sustain_seconds": args.sustain_seconds,
                "burst_jobs": args.burst_jobs,
                "burst_timeout_seconds": args.burst_timeout,
                "burst_poll_seconds": args.burst_poll,
                "knn_timeout_seconds": KNN_TIMEOUT,
                "knn_poll_seconds": KNN_POLL,
                "seed": args.seed,
                "gpu_sample_ms": GPU_SAMPLE_MS,
                "project_link_to_base_layer": args.project,
                "recall_check_queries": args.recall_queries,
                "recall_check_k": args.recall_k,
                "s3_payload_signing_enabled": os.environ.get("S3_PAYLOAD_SIGNING_ENABLED"),
                "fetch_runs": args.fetch_runs,
                "gpu_memory_limit_gb": float(gpu_memory_limit()),
                # Where the bucket was, not its name.
                "s3": s3_setting(args.s3_bucket),
            },
            "inputs": {
                name: {
                    "dir": str(d.resolve().relative_to(ROOT)).replace("\\", "/"),
                    **{k: metas[name][k] for k in ("data_type", "space_type", "doc_count", "dimension")},
                    "skip_stored_vectors": bool(metas[name].get("skip_stored_vectors")),
                }
                for name, d in inputs.items()
            },
            "warm_up": [],
            "scrapes": [],
        }

    def alive(self):
        """Raises when the service no longer answers its heart beat, so a
        run against a dead service ends with what it measured rather than
        recording refusals for hours."""
        try:
            with urllib.request.urlopen(f"{self.service.url}/_heart_beat", timeout=10) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        self.stop.set()
        raise RuntimeError("the service stopped answering its heart beat")

    def by_type(self):
        """The inputs of each data type, in name order."""
        types = {}
        for name in sorted(self.inputs):
            types.setdefault(self.metas[name]["data_type"], []).append(name)
        return types

    def stamp(self, records):
        """Marks `records` as the service's process up now."""
        for record in records:
            record["process"] = self.service.process

    def records_of(self, process):
        """Every record of the jobs sent to `process`."""
        records = self.record["warm_up"] + self.record.get("single", [])
        for phase in ("concurrent", "sustained", "burst"):
            for group in self.record.get(phase, {}).values():
                records += group["records"]
        return [r for r in records if r.get("process") == process]

    def end_process(self):
        """Before the service's process ends: once no job is in flight,
        its metrics page, and the status of each of its jobs whose end the
        run did not see. A process sent no job, or a service with no page,
        has nothing to check. A page the run cannot have, a job still in
        flight when the wait gives up among them, is noted for the check to
        report: the run's measurements come first."""
        process = self.service.process
        records = self.records_of(process)
        if not records:
            return
        try:
            page = burst.scrape_when_idle(self.service.url)
        except burst.SCRAPE_ERRORS as e:
            self.record["scrapes"].append({
                "process": process,
                "error": repr(e),
                "at": now().isoformat(),
            })
            print(f"no metrics of process {process}: {e!r}", file=sys.stderr, flush=True)
            return
        if page is None:
            return
        path = self.out / "metrics" / f"process-{process}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(page.encode())
        timeout, poll = self.service.settings
        burst.ask_ends(burst.DataNode(self.service.url, {}, self.service.bucket, "", timeout, poll),
                       records)
        self.record["scrapes"].append({
            "process": process,
            "file": path.relative_to(self.out).as_posix(),
            "at": now().isoformat(),
        })
        print(f"metrics of process {process}: {path}", flush=True)

    def node(self, label, timeout=KNN_TIMEOUT, poll=KNN_POLL):
        node_type = ProjectingNode if self.args.project else BenchNode
        return node_type(
            self.service.url, self.inputs, self.service.bucket, f"{self.label}-{label}", timeout,
            poll, self.args.seed,
        )

    def one_build(self, node, index, name):
        """One job, its status asked every STATUS_POLL: the record."""
        record = {"input": name, "index": index}
        self.alive()
        self.service.reset_peak_rss()
        started = now()
        start = time.monotonic()
        status, answer = node.request("POST", "/_build", node.body(index, name))
        if status != 200:
            record.update(task_status="REFUSED", status=status, detail=answer)
            return record
        job_id = answer["job_id"]
        while True:
            status, answer = node.request("GET", f"/_status/{job_id}")
            if status != 200 or not answer or answer.get("task_status") != "RUNNING_INDEX_BUILD":
                break
            if time.monotonic() - start > JOB_TIMEOUT:
                answer = {"task_status": "TIMED_OUT", "error_message": "still running"}
                break
            time.sleep(STATUS_POLL)
        wall = time.monotonic() - start
        ended = now()
        # A status request that failed answers with its status, or the
        # error's text when the connection did.
        detail = answer if isinstance(answer, dict) else None
        record.update(
            job_id=job_id,
            task_status=detail.get("task_status") if detail else f"status {status}",
            wall_seconds=round(wall, 3),
            start=started.isoformat(),
            end=ended.isoformat(),
            host_peak_bytes=self.service.peak_rss(),
        )
        if record["task_status"] == "COMPLETED_INDEX_BUILD":
            record["file_name"] = detail["file_name"]
        else:
            record["error_message"] = detail.get("error_message") if detail else answer
        return record

    def discard(self, record):
        """Deletes the file of a job seen completed, so the bucket holds
        only the inputs and the jobs in flight. A delete that fails is
        noted on the record, not the run's end: it is housekeeping."""
        if not record.get("file_name"):
            return
        try:
            self.service.delete(f"{record['input']}/{record['file_name']}")
        except OSError as e:
            record["discard_error"] = repr(e)
            print(f"delete of {record['input']}/{record['file_name']} failed: {e!r}",
                  file=sys.stderr, flush=True)

    def warm_up(self, phase, data_types=None):
        """A build of one input of each data type (of those named, when
        named), so the service has loaded its GPU code before the timed
        jobs."""
        node = self.node(f"warm-up-{phase}")
        for index, (data_type, names) in enumerate(self.by_type().items()):
            if data_types is not None and data_type not in data_types:
                continue
            record = self.one_build(node, index, names[0])
            self.discard(record)
            record["phase"] = phase
            self.stamp([record])
            self.record["warm_up"].append(record)
            print(f"warm-up {data_type}: {record['task_status']} in {record.get('wall_seconds')} s", flush=True)

    def single(self):
        node = self.node("single")
        # Each phase's records are the run's from the start, so a run that
        # fails midway keeps what the phase measured.
        records = self.record["single"] = []
        for run_number in range(1, self.args.runs + 1):
            for i, name in enumerate(sorted(self.inputs)):
                record = self.one_build(node, run_number * len(self.inputs) + i, name)
                record["run"] = run_number
                self.stamp([record])
                if run_number <= self.args.fetch_runs and record["task_status"] == "COMPLETED_INDEX_BUILD":
                    path = self.out / name / f"run-{run_number}.faiss"
                    record["size"] = self.service.fetch(f"{name}/{record['file_name']}", path)
                self.discard(record)
                records.append(record)
                print(f"single {name} run {run_number}: {record['task_status']} in {record.get('wall_seconds')} s", flush=True)

    def concurrent(self):
        node = self.node("concurrent")
        phase = self.record["concurrent"] = {}
        for data_type, names in self.by_type().items():
            rounds = []
            phase[data_type] = {"jobs": self.args.concurrent, "rounds": self.args.rounds, "records": rounds}
            for round_number in range(1, self.args.rounds + 1):
                self.alive()
                base = round_number * self.args.concurrent
                jobs = [(base + i, names[i % len(names)]) for i in range(self.args.concurrent)]
                records = burst.at_once(node, jobs)
                self.stamp(records)
                for r in records:
                    r["round"] = round_number
                    self.discard(r)
                rounds.extend(records)
                built = [r["seen_after"] for r in records if r["outcome"] == "built"]
                print(f"concurrent {data_type} round {round_number}: {len(built)} of {len(records)} built, "
                      f"last seen after {max(built, default=0):.1f} s", flush=True)

    def sustained(self):
        node = self.node("sustained")
        phase = self.record["sustained"] = {}
        for data_type, names in self.by_type().items():
            records = []
            lock = threading.Lock()
            counter = {"next": 0}
            seconds = self.args.sustain_seconds
            start = time.monotonic()

            failures = []

            def worker():
                while True:
                    with lock:
                        if time.monotonic() - start >= seconds or self.stop.is_set():
                            return
                        index = counter["next"]
                        counter["next"] += 1
                    record = {"index": index, "input": names[index % len(names)]}
                    try:
                        self.alive()
                        node.build(index, record["input"], start, record)
                    except RuntimeError as e:  # the service is down: the queue ends
                        failures.append(e)
                        return
                    except Exception as e:  # noqa: BLE001  (the job's, not the queue's)
                        record.update(outcome="error", detail=repr(e),
                                      cpu_after=time.monotonic() - start)
                    self.stamp([record])
                    with lock:
                        records.append(record)
                    self.discard(record)

            # Daemon threads, and `stop` set on an error or interrupt: the
            # queue never outlives the run.
            threads = [threading.Thread(target=worker, daemon=True) for _ in range(self.args.queue)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if failures:
                raise failures[0]
            records.sort(key=lambda r: r["index"])
            completed = sum(
                1 for r in records if r["outcome"] == "built" and r["seen_after"] <= seconds
            )
            phase[data_type] = {
                "queue": self.args.queue,
                "seconds": seconds,
                "completed_within": completed,
                "jobs_per_hour": round(completed * 3600 / seconds, 2),
                "records": records,
            }
            print(f"sustained {data_type}: {completed} completed in {seconds} s, "
                  f"{phase[data_type]['jobs_per_hour']} jobs per hour", flush=True)

    def burst(self):
        node = self.node("burst", self.args.burst_timeout, self.args.burst_poll)
        phase = self.record["burst"] = {}
        for i, (data_type, names) in enumerate(self.by_type().items()):
            if i:
                # The upstream service builds the jobs a burst's data
                # nodes gave up on all the same; restarted, it meets the
                # next burst with an empty queue, as the first one.
                self.end_process()
                self.service.restart()
            self.warm_up("burst", [data_type])
            self.alive()
            l2 = [n for n in names if self.metas[n]["space_type"] in ("l2", "hamming")] or names
            jobs = [(i, l2[i % len(l2)]) for i in range(self.args.burst_jobs)]
            records = burst.at_once(node, jobs)
            self.stamp(records)
            for r in records:
                self.discard(r)
            phase[data_type] = {
                "jobs": self.args.burst_jobs,
                "inputs": l2,
                "timeout_seconds": self.args.burst_timeout,
                "poll_seconds": self.args.burst_poll,
                **burst.summarize(records),
                "records": records,
            }
            print(f"burst {data_type}: {phase[data_type]['outcomes']}", flush=True)

    def attach_steps(self):
        log = self.service.log.read_text(encoding="utf-8", errors="replace")
        steps = steps_from_log(log)
        projections = projections_from_log(log)
        recalls = recalls_from_log(log)
        waits = upload_waits_from_log(log)
        for record in self.record["warm_up"] + self.record.get("single", []):
            if "job_id" in record:
                record["seconds"] = steps.get(record["job_id"])
                if record["job_id"] in waits:
                    record["upload_waits"] = waits[record["job_id"]]
                if record["job_id"] in projections:
                    record["projection"] = projections[record["job_id"]]
                if record["job_id"] in recalls:
                    record["recall_check"] = recalls[record["job_id"]]
        for phase in ("concurrent", "sustained", "burst"):
            for group in self.record.get(phase, {}).values():
                for record in group["records"]:
                    if "job_id" in record:
                        record["seconds"] = steps.get(record["job_id"])
                        if record["job_id"] in waits:
                            record["upload_waits"] = waits[record["job_id"]]

    def attach_device_peaks(self, samples):
        starts = self.service.starts
        self.record["service_starts"] = [s.isoformat() for s in starts]
        attach_device_peaks(self.record["warm_up"] + self.record.get("single", []), samples, starts)

    def save(self):
        self.record["ended"] = now().isoformat()
        path = self.out / "run.json"
        path.write_bytes((json.dumps(self.record, indent=1) + "\n").encode())

    def main(self):
        phases = self.args.phases
        sampler = Sampler(self.out / "gpu.csv")
        try:
            self.service.start(KNN_TIMEOUT, KNN_POLL)
            self.record["image_id"] = image_id(self.record["image"])
            self.service.upload()
            defaults = [p for p in phases if p != "burst"]
            if defaults:
                self.warm_up("default")
            for phase in defaults:
                getattr(self, phase)()
            if "burst" in phases:
                if self.service.settings != (self.args.burst_timeout, self.args.burst_poll):
                    self.end_process()
                self.service.start(self.args.burst_timeout, self.args.burst_poll)
                self.burst()
            self.end_process()
        except BaseException as e:
            self.record["error"] = repr(e)
            self.stop.set()
            raise
        finally:
            # What was measured before an error is still worth keeping:
            # every step here runs whatever the one before did.
            try:
                self.service.down()
            finally:
                try:
                    sampler.stop()
                finally:
                    try:
                        self.attach_steps()
                        self.attach_device_peaks(sampler.samples())
                    finally:
                        self.save()
        print(f"wrote {self.out / 'run.json'}", flush=True)


def s3_setting(bucket):
    """Where a run's bucket was, for run.json: LocalStack, or real S3 in
    a region, at AWS's endpoint or another."""
    if bucket is None:
        return {"store": "localstack"}
    return {
        "store": "s3",
        "region": os.environ["AWS_DEFAULT_REGION"],
        "endpoint": "custom" if os.environ.get("S3_ENDPOINT_URL") else "aws",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("service")
    parser.add_argument("--inputs", action="append", type=pathlib.Path, default=[])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--queue", type=int, default=4)
    parser.add_argument("--sustain-seconds", type=int, default=3600)
    parser.add_argument("--burst-jobs", type=int, default=48)
    parser.add_argument("--burst-timeout", type=int, default=90)
    parser.add_argument("--burst-poll", type=int, default=5)
    parser.add_argument("--phases", default=",".join(PHASES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--fetch-runs", type=int, default=1)
    parser.add_argument("--s3-bucket")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--recall-queries", type=int, default=0)
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("names", nargs="*")
    # Intermixed: the usage puts the names after the options, where
    # parse_args on older Pythons (3.12.3 among them) has given them an
    # empty list already and refuses them as unrecognized.
    args = parser.parse_intermixed_args()
    if args.s3_bucket is not None:
        if not args.s3_bucket:
            sys.exit("--s3-bucket takes a bucket's name")
        if not os.environ.get("AWS_DEFAULT_REGION"):
            sys.exit("--s3-bucket needs AWS_DEFAULT_REGION, the bucket's region")
        # Unset it means 8, which on a larger GPU would run the whole
        # benchmark with the services told the wrong size.
        if not os.environ.get("GPU_MEMORY_LIMIT"):
            sys.exit("--s3-bucket needs GPU_MEMORY_LIMIT, the GPU's memory in GB")
    if args.max_workers is not None and args.max_workers <= 0:
        sys.exit("--max-workers takes a number above 0")
    if args.recall_queries < 0:
        sys.exit("--recall-queries takes a number, 0 or more")
    if args.recall_queries > 0 and args.service == "upstream":
        sys.exit("--recall-queries is knn-builder's: the upstream service has no recall check")
    if not 1 <= args.recall_k <= 10_000:
        sys.exit("--recall-k takes a number from 1 to 10000")
    try:
        if float(gpu_memory_limit()) <= 0:
            raise ValueError
    except ValueError:
        sys.exit(f"GPU_MEMORY_LIMIT takes a number of GB above 0, not {gpu_memory_limit()}")
    args.phases = [p for p in args.phases.split(",") if p]
    for phase in args.phases:
        if phase not in PHASES:
            sys.exit(f"no phase {phase}; one of {', '.join(PHASES)}")
    for value, what in ((args.runs, "--runs"), (args.concurrent, "--concurrent"),
                        (args.rounds, "--rounds"), (args.queue, "--queue"),
                        (args.sustain_seconds, "--sustain-seconds"),
                        (args.burst_jobs, "--burst-jobs"), (args.burst_timeout, "--burst-timeout"),
                        (args.burst_poll, "--burst-poll"), (args.fetch_runs, "--fetch-runs")):
        if value <= 0:
            sys.exit(f"{what} takes a number above 0")

    inputs = {}
    for directory in args.inputs or [ROOT / "target" / "bench" / "inputs"]:
        if not directory.resolve().is_relative_to(ROOT):
            sys.exit(f"--inputs {directory} is outside {ROOT}, which is all the uploader mounts")
        for d in sorted(directory.iterdir()):
            if (d / "input.json").exists():
                if d.name in inputs:
                    sys.exit(f"input {d.name} is under two --inputs directories")
                inputs[d.name] = d
    if args.names:
        if missing := [n for n in args.names if n not in inputs]:
            sys.exit(f"no input {', '.join(missing)}")
        inputs = {n: inputs[n] for n in args.names}
    if not inputs:
        sys.exit("no inputs")
    metas = {name: json.loads((d / "input.json").read_text()) for name, d in inputs.items()}
    Bench(args, inputs, metas).main()


if __name__ == "__main__":
    main()
