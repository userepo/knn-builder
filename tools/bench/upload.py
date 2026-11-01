"""Uploads build inputs to the benchmark's bucket in LocalStack, for
run.py: each directory in $INPUT_DIRS (comma separated, under /work) is
one input, its name the directory's, its blobs vectors.knnvec and
doc_ids.knndid; a blob already there is not sent again. Runs as the
uploader of docker-compose.yml, in the upstream image for its boto3.
"""

import os
import pathlib
import sys

sys.path.insert(0, "/work/tools/burst")
sys.dont_write_bytecode = True  # no __pycache__ in the checkout
import burst  # noqa: E402

BUCKET = "bench"


def main():
    directories = [pathlib.Path("/work", d) for d in os.environ["INPUT_DIRS"].split(",") if d]
    burst.put_inputs(os.environ["S3_ENDPOINT_URL"], BUCKET, {d.name: d for d in directories})


if __name__ == "__main__":
    main()
