"""Uploads build inputs to the benchmark's bucket, for run.py: each
directory in $INPUT_DIRS (comma separated, under /work) is one input, its
name the directory's, its blobs vectors.knnvec and doc_ids.knndid; a blob
already there, by its SHA-256, is not sent again. The bucket is $BENCH_BUCKET, on
LocalStack or real S3 as $S3_ENDPOINT_URL says (unset: real S3), and is
created only when $BENCH_CREATE_BUCKET is 1. Runs as the uploader of
docker-compose.yml, in the upstream image for its boto3.
"""

import os
import pathlib
import sys

sys.path.insert(0, "/work/tools/burst")
sys.dont_write_bytecode = True  # no __pycache__ in the checkout
import burst  # noqa: E402


def main():
    directories = [pathlib.Path("/work", d) for d in os.environ["INPUT_DIRS"].split(",") if d]
    burst.put_inputs(
        os.environ.get("S3_ENDPOINT_URL") or None,
        os.environ["BENCH_BUCKET"],
        {d.name: d for d in directories},
        create=os.environ.get("BENCH_CREATE_BUCKET") == "1",
    )


if __name__ == "__main__":
    main()
