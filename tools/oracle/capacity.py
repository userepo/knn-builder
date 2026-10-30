"""Records how the service sizes a build request, for knn-builder's job
store (crates/knn-builder/tests/data/capacity.json): the GPU and CPU memory
estimates of `calculate_memory_requirements` for a spread of requests,
whether each passes the limits, a few requests under limits equal to
their own estimates (the service's comparison at its edge), the
messages of the service's two 507s, from its own JobService, and its
defaults for the store and the limits.

The limits are the defaults, where the CPU limit always binds first, and
a smaller GPU's, where the GPU limit can. The spread takes in requests on
either side of each limit, where the estimate's rounding decides, and
sizes far past them.

Runs in the api image (tools/README.md):
  $RUN $API tools/oracle/capacity.py crates/knn-builder/tests/data/capacity.json
"""

import json
import math
import random
import sys

sys.path[:0] = ["/app/remote_vector_index_builder"]

from app.base.config import Settings  # noqa: E402
from app.base.exceptions import CapacityError  # noqa: E402
from app.services.job_service import JobService  # noqa: E402
from app.storage.memory import InMemoryRequestStore  # noqa: E402
from app.utils.memory import calculate_memory_requirements  # noqa: E402
from core.common.models import IndexBuildParameters  # noqa: E402

# As main.py computes them, from GB.
GPU_LIMIT = Settings().gpu_memory_limit * 10**9
CPU_LIMIT = Settings().cpu_memory_limit * 10**9
LIMITS = [(GPU_LIMIT, CPU_LIMIT), (8.0 * 10**9, 64.0 * 10**9)]


def request(dimension, doc_count, data_type="float", m=16, path="a/_0_f.knnvec"):
    return IndexBuildParameters.model_validate(
        {
            "container_name": "bucket",
            "vector_path": path,
            "doc_id_path": "a/_0_f.knndid",
            "dimension": dimension,
            "doc_count": doc_count,
            "data_type": data_type,
            "index_parameters": {"algorithm_parameters": {"m": m}},
        }
    )


def estimate(dimension, doc_count, data_type, m):
    gpu, cpu = calculate_memory_requirements(request(dimension, doc_count, data_type, m))
    return {
        "dimension": dimension,
        "doc_count": doc_count,
        "data_type": data_type,
        "m": m,
        "gpu": gpu,
        "cpu": cpu,
        # JobService._create_workflow's test, for each pair of limits.
        "fits": [not (gpu > g or cpu > c) for g, c in LIMITS],
    }


def around_limits():
    """Counts on either side of the count where each estimate crosses its
    limit, for a few shapes."""
    for dimension, data_type, m in [
        (768, "float", 16),
        (768, "half_float", 16),
        (128, "float", 32),
        (1536, "half_float", 8),
        (3, "float", 0),
        (768, "byte", 16),
        (4096, "byte", 32),
    ]:
        size = {"float": 4, "half_float": 2, "byte": 1}[data_type]
        index = (dimension * size + m * 8) * 1.1
        per_cpu = index + dimension * size
        per_gpu = index * 0.5
        bounds = [(c, per_cpu) for _, c in LIMITS] + [(g, per_gpu) for g, _ in LIMITS]
        for limit, per in bounds:
            middle = int(limit / per)
            for doc_count in range(middle - 2, middle + 3):
                yield dimension, doc_count, data_type, m
    # CPU estimates of exactly 32e9 bytes, the default limit, found by a
    # search: at the limit, not over it. No GPU estimate found lands on
    # one; at_limits tests that edge.
    yield 6, 50, "float", 72727267
    yield 12, 25, "float", 145454534


def spread():
    rng = random.Random(20261028)
    for _ in range(300):
        yield (
            rng.choice([1, 2, 3, 64, 100, 128, 384, 768, 1024, 1536, 4096, 16000]),
            rng.choice([5, 1000, 10**6, rng.randrange(5, 2**32), rng.randrange(5, 2**40)]),
            rng.choice(["float", "half_float"]),
            rng.choice([16, 32, 0, -1, 512, rng.randrange(-(2**31), 2**31)]),
        )
    # Byte requests, drawn apart so the draws above stay as they were.
    rng = random.Random(20261030)
    for _ in range(60):
        yield (
            rng.choice([1, 64, 128, 384, 768, 1536, 4096]),
            rng.choice([5, 1000, 10**6, rng.randrange(5, 2**32)]),
            "byte",
            rng.choice([16, 32, 0, -1, 512]),
        )
    # Sizes far past every limit, up to 64-bit counts.
    for dimension, doc_count in [(2**32, 2**32), (2**63, 2**63), (2**64 - 1, 2**64 - 1)]:
        for m in [16, 2**63 - 1, -(2**63)]:
            yield dimension, doc_count, "float", m


def at_limits():
    """The test at its edge: requests under limits equal to their own
    estimates, and one float below either."""
    for dimension, doc_count, data_type, m in [
        (768, 10**6, "float", 16),
        (100, 12345, "half_float", 7),
        (3, 5, "float", -1),
        (384, 54321, "byte", 16),
    ]:
        gpu, cpu = calculate_memory_requirements(
            request(dimension, doc_count, data_type, m)
        )
        below = [
            (gpu, cpu),
            (math.nextafter(gpu, 0), cpu),
            (gpu, math.nextafter(cpu, 0)),
        ]
        for g, c in below:
            yield {
                "dimension": dimension,
                "doc_count": doc_count,
                "data_type": data_type,
                "m": m,
                "limits": {"gpu": g, "cpu": c},
                "fits": not (gpu > g or cpu > c),
            }


class Executor:
    def submit_workflow(self, workflow):
        pass


def refusals():
    """The messages of the two 507s: a store at capacity, and an estimate
    over a limit."""
    store = InMemoryRequestStore(Settings(request_store_max_size=1))
    service = JobService(store, Executor(), GPU_LIMIT, CPU_LIMIT)
    service.create_job(request(8, 10, path="a/_0_held.knnvec"))
    try:
        service.create_job(request(8, 10, path="a/_0_full.knnvec"))
    except CapacityError as e:
        full = str(e)
    service = JobService(
        InMemoryRequestStore(Settings()), Executor(), GPU_LIMIT, CPU_LIMIT
    )
    try:
        service.create_job(request(768, 10**8, path="a/_0_large.knnvec"))
    except CapacityError as e:
        large = str(e)
    return {
        "full": {"vector_path": "a/_0_full.knnvec", "message": full},
        "too_large": {"vector_path": "a/_0_large.knnvec", "message": large},
    }


def main():
    estimates = [estimate(*case) for case in [*around_limits(), *spread()]]
    settings = Settings()
    table = {
        "defaults": {
            "max_jobs": settings.request_store_max_size,
            "ttl_seconds": settings.request_store_ttl_seconds,
            "gpu_memory_limit": GPU_LIMIT,
            "cpu_memory_limit": CPU_LIMIT,
        },
        "limits": [{"gpu": g, "cpu": c} for g, c in LIMITS],
        "estimates": estimates,
        "at_limits": list(at_limits()),
        "refusals": refusals(),
    }
    with open(sys.argv[1], "wb") as out:
        out.write((json.dumps(table, indent=1, allow_nan=False) + "\n").encode())
    fits = [sum(e["fits"][i] for e in estimates) for i in range(len(LIMITS))]
    print(f"{len(estimates)} estimates; fit under each pair of limits: {fits}")


if __name__ == "__main__":
    main()
