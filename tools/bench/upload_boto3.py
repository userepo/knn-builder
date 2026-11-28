"""Times boto3 uploads of a file as the upstream service makes them, over
a grid of chunk sizes and concurrencies, beside knn-store's
`upload_bench` example.

    upload_boto3.py BUCKET KEY SIZE_MIB CHUNK_MIB[,...] CONCURRENCY[,...] RUNS [--dir DIR]

The upload is the upstream service's at ea7db9f: `upload_file` of a local
file with a `TransferConfig` of the chunk size (its threshold too) and
the concurrency, and `ChecksumAlgorithm` CRC32. The service's own
settings are a chunk of 50 MiB and a concurrency of 8 on 8 CPUs. The file
is written once to DIR, by default /dev/shm, so that reading it costs
what a file just written costs the service: memory, not a disk.

Each upload prints one line of JSON: the chunk size and concurrency, the
run, the seconds and the rate in MiB/s. A first line gives the versions.
Every upload goes to KEY, which is deleted at the end, and the file is
removed. The region and credentials are boto3's own (AWS_DEFAULT_REGION,
the instance's role); S3_ENDPOINT_URL, when set, names another endpoint.
"""

import argparse
import json
import os
import pathlib
import time

import boto3
import botocore
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

MIB = 1 << 20
BLOCK = 16 * MIB


def numbers(text):
    return [int(n) for n in text.split(",")]


def crt_version():
    """The AWS CRT's version when it is installed: boto3 may then send an
    upload through it instead of through its own threads."""
    try:
        import awscrt  # noqa: PLC0415
    except ImportError:
        return None
    return awscrt.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bucket")
    parser.add_argument("key")
    parser.add_argument("size_mib", type=int)
    parser.add_argument("chunk_mib", type=numbers)
    parser.add_argument("concurrency", type=numbers)
    parser.add_argument("runs", type=int)
    parser.add_argument("--dir", type=pathlib.Path, default=pathlib.Path("/dev/shm"))
    args = parser.parse_args()

    print(json.dumps({
        "client": "boto3",
        "boto3": boto3.__version__,
        "botocore": botocore.__version__,
        "awscrt": crt_version(),
        "cpus": os.cpu_count(),
    }), flush=True)

    path = args.dir / f"upload-boto3-{os.getpid()}.bin"
    block = bytes(i % 251 for i in range(BLOCK))
    left = args.size_mib * MIB
    with path.open("wb") as out:
        while left > 0:
            left -= out.write(block[:min(left, BLOCK)])
    try:
        for chunk in args.chunk_mib:
            for concurrency in args.concurrency:
                # A client of its own, so each setting starts on new
                # connections, with a pool that holds them all.
                client = boto3.client(
                    "s3",
                    endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
                    config=Config(max_pool_connections=max(10, concurrency)),
                )
                transfer = TransferConfig(
                    multipart_chunksize=chunk * MIB,
                    multipart_threshold=chunk * MIB,
                    max_concurrency=concurrency,
                )
                for run in range(1, args.runs + 1):
                    start = time.perf_counter()
                    client.upload_file(
                        str(path),
                        args.bucket,
                        args.key,
                        Config=transfer,
                        ExtraArgs={"ChecksumAlgorithm": "CRC32"},
                    )
                    seconds = time.perf_counter() - start
                    print(json.dumps({
                        "client": "boto3",
                        "part_mib": chunk,
                        "in_flight": concurrency,
                        "run": run,
                        "seconds": round(seconds, 3),
                        "mib_per_s": round(args.size_mib / seconds),
                    }), flush=True)
        boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None).delete_object(
            Bucket=args.bucket, Key=args.key
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
