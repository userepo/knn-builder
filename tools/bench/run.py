"""Runs the benchmark's metrics against one remote index build service on
the GPU, over the inputs tools/bench/datasets.py wrote (or make_inputs.py's),
and records every measurement in OUT_DIR/run.json for report.py.

The service runs from docker-compose.yml beside this file, with LocalStack
for S3, and is driven from the host, where nvidia-smi samples the GPU's
memory. The phases, in order:

  single      a warm-up build of one input of each data type, then RUNS
              builds of every input one at a time: each job's latency (from
              its request to the status that shows it completed, asked
              every 0.2 s), the service process's peak resident memory
              (VmHWM, reset before each job), the GPU's memory at its peak
              during the job less the least seen (nvidia-smi sampled every
              50 ms, the measure of tools/baseline/device_peaks.py over the
              request's window, from the POST to the status that shows the
              job ended), and its per-step times from the service's DEBUG
              log; the first run's file is fetched to
              OUT_DIR/<input>/run-1.faiss for the recall
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

A run starts from fresh services and an empty OUT_DIR record: whatever a
run before left in Docker is taken down first, and its files and recall
are replaced. A run that fails or is interrupted still takes the services
down and writes run.json with the phases that finished and the error; a
service that stops answering its heart beat ends the run the same way.

LocalStack keeps the inputs and the files of the jobs in flight in
Docker's disk. Each job's file is deleted once its job is seen completed
(a run's 1,500 jobs would else leave hundreds of GB), so the bucket holds
about 30 GB at most: the inputs and one concurrent round's files. Under
Docker Desktop the disk image on the host still grows with every block
first written, by hundreds of GB over a run, until the image is trimmed
and compacted; see that the host has the room.

Usage: python tools/bench/run.py OUT_DIR SERVICE [--inputs DIR ...]
           [--runs N] [--concurrent N] [--rounds N] [--queue N]
           [--sustain-seconds S] [--burst-jobs N] [--burst-timeout S]
           [--burst-poll S] [--phases LIST] [--seed N] [NAME...]

SERVICE is knn-builder (the knn-builder:dev image) or upstream (the pinned
api-snapshot image on one worker, since two builds at once crash it under
WSL2); SERVICE_IMAGE in the environment names another image. --inputs
(default target/bench/inputs) may repeat; NAMEs pick inputs under them.
SERVICE_PORT (1026) and S3_PORT (4568) are the host ports the services
answer on. Needs Docker with GPU access, nvidia-smi and Python 3 on the
host, nothing else. Writes OUT_DIR/run.json, service.log and gpu.csv.
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
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "burst"))
sys.dont_write_bytecode = True  # no __pycache__ in the checkout
import burst  # noqa: E402

UPSTREAM = "opensearchstaging/remote-vector-index-builder@sha256:78ced9d04c4cbc1106cfceb98e32a772c71565b215c6e68be8fcdcbcdaab64da"
# name: (image, MAX_WORKERS)
SERVICES = {"knn-builder": ("knn-builder:dev", "2"), "upstream": (UPSTREAM, "1")}
BUCKET = "bench"
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
# its end line (one worker, so blocks never interleave), the step lines
# between them its.
UPSTREAM_START = re.compile(r"Starting task execution for vector path: (\S+)")
UPSTREAM_END = re.compile(r"Job (\w+) with vector path (\S+) completed with status")
UPSTREAM_STEPS = {
    "download": re.compile(r"Vector download time for vector path \S+: ([\d.]+) seconds"),
    "build": re.compile(r"Index build time for vector path \S+: ([\d.]+) seconds"),
    "conversion": re.compile(r"Index conversion time for vector path \S+: ([\d.]+) seconds"),
    "write": re.compile(r"Index write time for vector path \S+: ([\d.]+) seconds"),
    "upload": re.compile(r"Total upload time for path \S+: ([\d.]+) seconds"),
}
KNN_BUILDER_STEPS = re.compile(
    r"job stage times job_id=(\w+) download_s=([\d.eE+-]+) build_s=([\d.eE+-]+) upload_s=([\d.eE+-]+)"
)


def now():
    return datetime.datetime.now().astimezone()


def run(*args, env=None, capture=False, timeout=DOCKER_TIMEOUT):
    """A docker command: its output, when wanted, through a file rather
    than a pipe, so that the timeout's kill of the client ends the call
    even while the compose plugin it spawned keeps the pipe open."""
    merged = {**os.environ, **(env or {})}
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
    their logs collected before each change, down at the end."""

    def __init__(self, image, max_workers, out, inputs):
        self.env = {
            "SERVICE_IMAGE": image,
            "MAX_WORKERS": max_workers,
            "SERVICE_PORT": os.environ.get("SERVICE_PORT", "1026"),
            "S3_PORT": os.environ.get("S3_PORT", "4568"),
            "INPUT_DIRS": ",".join(
                str(d.resolve().relative_to(ROOT)).replace("\\", "/") for d in inputs.values()
            ),
        }
        self.url = f"http://localhost:{self.env['SERVICE_PORT']}"
        self.s3 = f"http://localhost:{self.env['S3_PORT']}"
        self.log = out / "service.log"
        self.log.write_bytes(b"")
        self.settings = None

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
            run(*COMPOSE, "down", "-v", env=self.env)
        env = {
            **self.env,
            "CLIENT_TIMEOUT_SECONDS": str(timeout),
            "CLIENT_POLL_INTERVAL_SECONDS": str(poll),
        }
        # No timeout: the first `up` pulls the images, 14 GB for the
        # upstream one.
        run(*COMPOSE, "up", "-d", "localstack", "service", env=env, timeout=None)
        self.settings = (timeout, poll)
        self.wait()

    def restart(self):
        """Restarts the service alone, with the settings it has: a fresh
        job store, its log kept (the container's log survives a restart,
        so it is collected whole at the end)."""
        run(*COMPOSE, "restart", "service", env=self.env)
        self.wait()

    def wait(self):
        timeout, poll = self.settings
        burst.DataNode(self.url, {}, BUCKET, "", timeout, poll).wait_for_heart_beat()
        self.wait_for_s3()

    def wait_for_s3(self):
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"{self.s3}/_localstack/health", timeout=5) as resp:
                    if json.loads(resp.read()).get("services", {}).get("s3") in ("available", "running"):
                        return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise SystemExit("LocalStack never reported S3 up")

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
        with urllib.request.urlopen(f"{self.s3}/{BUCKET}/{key}") as resp, path.open("wb") as f:
            shutil.copyfileobj(resp, f, length=8 << 20)
        return path.stat().st_size

    def delete(self, key):
        request = urllib.request.Request(f"{self.s3}/{BUCKET}/{key}", method="DELETE")
        with urllib.request.urlopen(request, timeout=60):
            pass

    def down(self):
        try:
            if self.settings is not None:
                self.collect_log()
        finally:
            run(*COMPOSE, "down", "-v", env=self.env)


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


def steps_from_log(log):
    """Each job's step seconds by job id, from either service's log."""
    steps = {}
    for m in KNN_BUILDER_STEPS.finditer(log):
        steps[m[1]] = {"download": float(m[2]), "build": float(m[3]), "upload": float(m[4])}
    block = None
    for line in log.splitlines():
        if UPSTREAM_START.search(line):
            block = {}
        elif (m := UPSTREAM_END.search(line)) and block is not None:
            steps[m[1]] = block
            block = None
        elif block is not None:
            for step, pattern in UPSTREAM_STEPS.items():
                if m := pattern.search(line):
                    block[step] = float(m[1])
    return steps


class Bench:
    def __init__(self, args, inputs, metas):
        self.args = args
        self.inputs = inputs
        self.metas = metas
        self.out = args.out_dir
        self.label = args.service
        image, max_workers = SERVICES.get(args.service, (None, "1"))
        image = os.environ.get("SERVICE_IMAGE", image)
        if image is None:
            sys.exit(f"SERVICE is {', '.join(SERVICES)}, or set SERVICE_IMAGE; not {args.service}")
        gpu = gpu_info()
        # A run's directory never holds an input: OUT_DIR given as an
        # inputs directory would otherwise lose them below.
        for name in inputs:
            if (self.out / name / "input.json").exists():
                sys.exit(f"{self.out} holds the input {name}: give a run directory")
        self.out.mkdir(parents=True, exist_ok=True)
        self.service = Service(image, max_workers, self.out, inputs)
        # What a run before left here: its files, and the recall report.py
        # cached of them.
        (self.out / "recall.json").unlink(missing_ok=True)
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
            },
            "inputs": {
                name: {
                    "dir": str(d.resolve().relative_to(ROOT)).replace("\\", "/"),
                    **{k: metas[name][k] for k in ("data_type", "space_type", "doc_count", "dimension")},
                }
                for name, d in inputs.items()
            },
            "warm_up": [],
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

    def node(self, label, timeout=KNN_TIMEOUT, poll=KNN_POLL):
        return burst.DataNode(
            self.service.url, self.inputs, BUCKET, f"{self.label}-{label}", timeout, poll,
            self.args.seed,
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
                if run_number == 1 and record["task_status"] == "COMPLETED_INDEX_BUILD":
                    path = self.out / name / "run-1.faiss"
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
                self.service.restart()
            self.warm_up("burst", [data_type])
            self.alive()
            l2 = [n for n in names if self.metas[n]["space_type"] in ("l2", "hamming")] or names
            jobs = [(i, l2[i % len(l2)]) for i in range(self.args.burst_jobs)]
            records = burst.at_once(node, jobs)
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
        steps = steps_from_log(self.service.log.read_text(encoding="utf-8", errors="replace"))
        for record in self.record["warm_up"] + self.record.get("single", []):
            if "job_id" in record:
                record["seconds"] = steps.get(record["job_id"])
        for phase in ("concurrent", "sustained", "burst"):
            for group in self.record.get(phase, {}).values():
                for record in group["records"]:
                    if "job_id" in record:
                        record["seconds"] = steps.get(record["job_id"])

    def attach_device_peaks(self, samples):
        for record in self.record["warm_up"] + self.record.get("single", []):
            if "start" not in record:
                continue
            start = datetime.datetime.fromisoformat(record["start"])
            end = datetime.datetime.fromisoformat(record["end"])
            used = [u for t, u in samples if start <= t <= end]
            record["device_samples"] = len(used)
            record["device_peak_bytes"] = (max(used) - min(used)) << 20 if used else None

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
                self.service.start(self.args.burst_timeout, self.args.burst_poll)
                self.burst()
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
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    args.phases = [p for p in args.phases.split(",") if p]
    for phase in args.phases:
        if phase not in PHASES:
            sys.exit(f"no phase {phase}; one of {', '.join(PHASES)}")
    for value, what in ((args.runs, "--runs"), (args.concurrent, "--concurrent"),
                        (args.rounds, "--rounds"), (args.queue, "--queue"),
                        (args.sustain_seconds, "--sustain-seconds"),
                        (args.burst_jobs, "--burst-jobs"), (args.burst_timeout, "--burst-timeout"),
                        (args.burst_poll, "--burst-poll")):
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
