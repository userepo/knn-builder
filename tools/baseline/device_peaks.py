"""Each build's peak device memory in a baseline run, from the GPU's memory
sampled beside it: run on the host, as the run starts,

  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 50 > OUT_DIR/gpu.csv

and keep the service's log with Docker's timestamps (`docker compose logs
--timestamps service > OUT_DIR/service.log`). A build's window runs from
the service's "Starting task execution" to its job's "completed" line, and
its peak is the most memory used in the window less what the service's
process held before it, so the process's own CUDA context and resources,
once a first build has made them, are not counted. The first build in the
service's process makes them inside its window, so it is left out. The GPU
must run nothing else meanwhile, and an allocation shorter than the
sampling interval can be missed: a first look, not a measurement to fit
to. nvidia-smi stamps its samples in the host's local time.

What a process holds is the level the samples last stayed at before the
window, not the least in the window: nvidia-smi under WSL2 reads 0 MiB
used once the GPU has idled for about ten seconds, whatever a process
holds, until the GPU next runs work, so a window that starts idle sees
nothing held, and a window of a few samples can miss the level the
process holds. The readings pass through one to three values on the way
down to 0 and back up, a sample each, so the level held is the last
stretch of HELD_RUN equal samples or more.

  python tools/baseline/device_peaks.py OUT_DIR

Writes OUT_DIR/device.json: per build in log order, its vector path, its
build seconds from the service's DEBUG log, and its peak in MiB.
Standard library only.
"""

import bisect
import datetime
import json
import pathlib
import re
import sys

# Equal samples in a row that make a level held. The values on the way to
# and from an idle GPU last a sample each, and builds a second apart leave
# the level between them only a few samples when nvidia-smi falls behind
# its interval (100 ms and more against 50 asked).
HELD_RUN = 2

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


def held(gpu, start, since=None):
    """The MiB the process held before `start`: of the samples, (time, MiB
    used) in time order, the value of the last stretch of HELD_RUN or more
    equal ones, not zero, that begins before it (and may run on past it),
    or 0 when there is none. A stretch that ends before `since`, when the
    process started, was another process's."""
    j = bisect.bisect_left(gpu, start, key=lambda sample: sample[0]) - 1
    while j >= 0:
        value = gpu[j][1]
        first = j
        while first > 0 and gpu[first - 1][1] == value:
            first -= 1
        last = j
        while last + 1 < len(gpu) and gpu[last + 1][1] == value:
            last += 1
        if since is not None and gpu[last][0] < since:
            break
        if value and last - first + 1 >= HELD_RUN:
            return value
        j = first - 1
    return 0


def peak(gpu, start, end, since=None):
    """A build's (peak MiB, samples in its window, MiB held before it):
    the most used from `start` to `end` less what the process, started at
    `since`, held before; the peak is None without a sample in the
    window, or when none reaches what the process held (a window that saw
    only the readings of a GPU idle or waking up)."""
    used = [u for t, u in gpu if start <= t <= end]
    before = held(gpu, start, since)
    most = max(used, default=None)
    return (most - before if most is not None and most >= before else None), len(used), before


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
        peak_mib, samples_in_window, _ = peak(gpu, start, end)
        records.append({
            "vector_path": path,
            "build_seconds": seconds,
            "samples": samples_in_window,
            "peak_mib": peak_mib,
        })
    (out / "device.json").write_bytes((json.dumps(records, indent=1) + "\n").encode())
    for r in records:
        print(r)


if __name__ == "__main__":
    main()
