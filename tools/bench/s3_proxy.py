"""Serves the benchmark's bucket on real S3 to run.py on the host as
LocalStack serves its bucket: GET /BUCKET/KEY streams an object, DELETE
/BUCKET/KEY deletes one, and GET /_health answers 200 once the bucket can
be reached. Runs as the s3 service of docker-compose.yml, in the upstream
image for its boto3, so the host needs no AWS library: boto3 signs every
request with the credentials and region of the container's environment,
or of the SDK's default chain where they are unset.

/_health answers 502 with S3's reason where waiting will not help (no
credentials, access denied, no such bucket) and 503 where it might (S3
not reached, or failing on its side), so run.py stops at once on the
first and waits on the second. The service listens on port 4566, as
LocalStack does, and serves no other bucket than BENCH_BUCKET.
"""

import http.server
import os
import urllib.parse

import boto3
import botocore.config
import botocore.exceptions

BUCKET = os.environ["BENCH_BUCKET"]
CHUNK = 8 << 20
ENDPOINT = os.environ.get("S3_ENDPOINT_URL") or None
# boto3's clients are safe to share between threads.
S3 = boto3.client("s3", endpoint_url=ENDPOINT)
# /_health's client: one attempt on short timeouts, so where S3 cannot be
# reached it says so in seconds, not after the minutes the default
# timeouts and retries take.
PROBE = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    config=botocore.config.Config(
        connect_timeout=5, read_timeout=10, retries={"total_max_attempts": 1}
    ),
)


class Handler(http.server.BaseHTTPRequestHandler):
    def reply(self, status, text):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def key(self):
        """The object's key, or None after answering 404 for a path
        outside the bucket."""
        prefix = f"/{BUCKET}/"
        if not self.path.startswith(prefix) or len(self.path) == len(prefix):
            self.reply(404, f"only keys of the bucket {BUCKET} are served")
            return None
        return urllib.parse.unquote(self.path[len(prefix):])

    def failed(self, error):
        """Answers with S3's status and reason for a request it refused,
        502 for any other failure."""
        if isinstance(error, botocore.exceptions.ClientError):
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 502
            self.reply(status, str(error))
        else:
            self.reply(502, str(error))

    def do_GET(self):  # noqa: N802  (http.server's name)
        if self.path == "/_health":
            try:
                PROBE.head_bucket(Bucket=BUCKET)
            except botocore.exceptions.ClientError as e:
                # S3 refusing the request will not pass; S3 failing may.
                status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0
                self.reply(503 if status >= 500 else 502, str(e))
            except botocore.exceptions.NoCredentialsError as e:
                self.reply(502, str(e))
            except botocore.exceptions.BotoCoreError as e:
                self.reply(503, str(e))
            else:
                self.reply(200, "ok")
            return
        key = self.key()
        if key is None:
            return
        try:
            obj = S3.get_object(Bucket=BUCKET, Key=key)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.failed(e)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(obj["ContentLength"]))
        self.end_headers()
        # A transfer that breaks midway leaves the client short of
        # Content-Length, which it reads as a failed fetch.
        for chunk in obj["Body"].iter_chunks(CHUNK):
            self.wfile.write(chunk)

    def do_DELETE(self):  # noqa: N802  (http.server's name)
        key = self.key()
        if key is None:
            return
        try:
            S3.delete_object(Bucket=BUCKET, Key=key)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.failed(e)
            return
        self.send_response(204)
        self.end_headers()


def main():
    http.server.ThreadingHTTPServer(("", 4566), Handler).serve_forever()


if __name__ == "__main__":
    main()
