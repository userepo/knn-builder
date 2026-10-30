"""Each build's peak device memory in a baseline run, from the GPU's memory
sampled beside it: run on the host, as the run starts,

  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 50 > OUT_DIR/gpu.csv

and keep the service's log with Docker's timestamps (`docker compose logs
--timestamps service > OUT_DIR/service.log`). A build's window runs from
the service's "Starting task execution" to its job's "completed" line, and
its peak is the most memory used in the window less the least, so the
service's own CUDA context and resources, once a first build has made them,
are not counted. The first build in the service's process makes them
inside its window, so it is left out. The GPU must run nothing else
meanwhile, and an allocation shorter than the sampling interval can be
missed: a first look, not a measurement to fit to. nvidia-smi stamps its
samples in the host's local time.

  python tools/baseline/device_peaks.py OUT_DIR

Writes OUT_DIR/device.json: per build in log order, its vector path, its
build seconds from the service's DEBUG log, and its peak in MiB.
Standard library only.
"""

import datetime
import json
import pathlib
import re
import sys

START = re.compile(r"(\S+Z) .*Starting task execution for vector path: (\S+)")
BUILD = re.compile(r"Index build time for vector path (\S+): ([\d.]+) seconds")
END = re.compile(r"(\S+Z) .*Job \w+ with vector path (\S+) completed with status")


def utc(stamp):
    """Docker's RFC 3339 stamp, nanoseconds cut to microseconds."""
    whole, _, fraction = stamp.rstrip("Z").partition(".")
    return datetime.datetime.fromisoformat(f"{whole}.{fraction[:6] or '0'}+00:00")


def samples(path):
    out = []
    for line in path.read_text().splitlines():
        stamp, _, used = line.partition(", ")
        try:
            local = datetime.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f")
        except ValueError:
            continue
        out.append((local.astimezone(datetime.timezone.utc), int(used)))
    return out


def windows(log):
    """(path, start, end, build seconds) per build, in log order."""
    open_, seconds, done = {}, {}, []
    for line in log.splitlines():
        if m := START.search(line):
            open_[m[2]] = utc(m[1])
        elif m := BUILD.search(line):
            seconds[m[1]] = float(m[2])
        elif (m := END.search(line)) and m[2] in open_:
            done.append((m[2], open_.pop(m[2]), utc(m[1]), seconds.pop(m[2], None)))
    return done


def main():
    out = pathlib.Path(sys.argv[1])
    gpu = samples(out / "gpu.csv")
    builds = windows((out / "service.log").read_text(encoding="utf-8", errors="replace"))
    records = []
    for path, start, end, seconds in builds[1:]:
        used = [u for t, u in gpu if start <= t <= end]
        records.append({
            "vector_path": path,
            "build_seconds": seconds,
            "samples": len(used),
            "peak_mib": max(used) - min(used) if used else None,
        })
    (out / "device.json").write_bytes((json.dumps(records, indent=1) + "\n").encode())
    for r in records:
        print(r)


if __name__ == "__main__":
    main()
