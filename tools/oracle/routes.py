"""Records how the service answers HTTP requests, for knn-builder's routes
(crates/knn-builder/tests/data/routes.json): the status, the headers that
matter and the body, for each route and method, the content types a build
request may come with, paths no route takes, the status of a job
running, completed and failed, and the debug list of jobs.

The service runs as its own app does, in a test client, but with jobs
that never start: the workflow executor takes them and does nothing, so
a job stays running until the oracle sets its state.

Runs in the api image (tools/README.md):
  $RUN $API tools/oracle/routes.py crates/knn-builder/tests/data/routes.json
"""

import json
import sys

sys.path[:0] = ["/app/remote_vector_index_builder"]

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.models.job import JobStatus  # noqa: E402

BUILD = {
    "container_name": "bucket",
    "vector_path": "segments/_0_target_field.knnvec",
    "doc_id_path": "segments/_0_target_field.knndid",
    "dimension": 768,
    "doc_count": 1000000,
}
BODY = json.dumps(BUILD)
JSON = {"content-type": "application/json"}
HEADERS = ["content-type", "allow", "location", "www-authenticate"]


def case(name, method, path, headers=None, body=None):
    return {"case": name, "method": method, "path": path, "headers": headers or {}, "body": body}


def other(path, **changes):
    return json.dumps({**BUILD, "vector_path": path, **changes})


CASES = [
    case("heart beat", "GET", "/_heart_beat"),
    case("heart beat, HEAD", "HEAD", "/_heart_beat"),
    case("heart beat, POST", "POST", "/_heart_beat"),
    case("heart beat, query", "GET", "/_heart_beat?x=1"),
    case("heart beat, trailing slash", "GET", "/_heart_beat/"),
    case("heart beat, trailing slash and query", "GET", "/_heart_beat/?x=1&y=2"),
    case("status, trailing slash", "GET", "/_status/0123abc/"),
    case("heart beat, two trailing slashes", "GET", "/_heart_beat//"),
    case("heart beat, an Authorization header", "GET", "/_heart_beat",
         {"authorization": "Basic dXNlcjpwYXNz"}),
    case("root", "GET", "/"),
    case("no such path", "GET", "/nope"),
    case("no such path, POST", "POST", "/nope", JSON, BODY),
    case("build", "POST", "/_build", JSON, BODY),
    case("build again", "POST", "/_build", JSON, BODY),
    case("build, GET", "GET", "/_build"),
    case("build, PUT", "PUT", "/_build", JSON, BODY),
    case("build, OPTIONS", "OPTIONS", "/_build"),
    case("build, HEAD", "HEAD", "/_build"),
    case("build, no content type", "POST", "/_build", {}, other("a/_0_none.knnvec")),
    case("build, charset", "POST", "/_build",
         {"content-type": "application/json; charset=utf-8"}, other("a/_0_charset.knnvec")),
    case("build, upper case type", "POST", "/_build",
         {"content-type": "Application/JSON"}, other("a/_0_upper.knnvec")),
    case("build, +json type", "POST", "/_build",
         {"content-type": "application/vnd.api+json"}, other("a/_0_plus.knnvec")),
    case("build, text/plain", "POST", "/_build",
         {"content-type": "text/plain"}, other("a/_0_text.knnvec")),
    case("build, form type", "POST", "/_build",
         {"content-type": "application/x-www-form-urlencoded"}, other("a/_0_form.knnvec")),
    case("build, type without subtype", "POST", "/_build",
         {"content-type": "application"}, other("a/_0_bare.knnvec")),
    case("build, text/plain and empty", "POST", "/_build", {"content-type": "text/plain"}, ""),
    case("build, invalid", "POST", "/_build", JSON, json.dumps({**BUILD, "doc_count": 4})),
    case("build, too large", "POST", "/_build", JSON,
         other("a/_0_large.knnvec", doc_count=100_000_000)),
    case("build, same path, other parameters", "POST", "/_build", JSON,
         json.dumps({**BUILD, "dimension": 16})),
    case("build, trailing slash", "POST", "/_build/", JSON, other("a/_0_slash.knnvec")),
    case("status, no such job", "GET", "/_status/0123abc"),
    case("status, empty id", "GET", "/_status/"),
    case("status, past the id", "GET", "/_status/0123abc/x"),
    case("status, POST", "POST", "/_status/0123abc"),
    case("status, HEAD, no such job", "HEAD", "/_status/0123abc"),
    # Percent-encoded paths: Starlette routes on the decoded path.
    case("status, an id not UTF-8", "GET", "/_status/%FF"),
    case("status, an encoded id", "GET", "/_status/%C3%A9"),
    case("status, an encoded percent", "GET", "/_status/%2541"),
    case("status, an encoded slash", "GET", "/_status/a%2Fb"),
    case("status, an encoded trailing slash", "GET", "/_status/a%2F"),
    case("heart beat, encoded", "GET", "/_heart%5Fbeat"),
    case("jobs", "GET", "/_jobs"),
    case("docs", "GET", "/docs"),
    case("openapi", "GET", "/openapi.json"),
]


def answer(client, c):
    response = client.request(
        c["method"], c["path"], headers=c["headers"],
        content=c["body"].encode() if c["body"] is not None else None,
        follow_redirects=False,
    )
    headers = {h: response.headers[h] for h in HEADERS if h in response.headers}
    return {"status": response.status_code, "response_headers": headers, "response_body": response.text}


def statuses(client):
    """The status of a job in each state, set straight in the store."""
    service = app.state.job_service
    body = other("a/_0_states.knnvec")
    job_id = client.post("/_build", content=body, headers=JSON).json()["job_id"]
    out = {}
    for name, data in [
        ("running", {}),
        ("completed", {"status": JobStatus.COMPLETED, "file_name": "_0_states.faiss"}),
        ("failed", {"status": JobStatus.FAILED, "error_message": "it failed",
                    "file_name": None}),
    ]:
        if data:
            service.request_store.update(job_id, data)
        response = client.get(f"/_status/{job_id}")
        out[name] = {
            "status": response.status_code,
            "response_headers": {h: response.headers[h] for h in HEADERS if h in response.headers},
            "response_body": response.text,
        }
    return out


def listing(client):
    """`/_jobs` over three jobs, in order: running, completed and failed,
    one with a path not ASCII and one with a tenant id holding DEL, the
    store emptied first."""
    service = app.state.job_service
    service.request_store._store.clear()
    bodies = [
        other("a/_0_listed.knnvec"),
        other("a/_0_été.knnvec"),
        # A tenant id with DEL, which json.dumps escapes as it does past ASCII.
        other("a/_0_tenant.knnvec", tenant_id="t\x7f"),
    ]
    ids = [client.post("/_build", content=b, headers=JSON).json()["job_id"] for b in bodies]
    service.request_store.update(
        ids[1], {"status": JobStatus.COMPLETED, "file_name": "_0_été.faiss"}
    )
    service.request_store.update(ids[2], {"status": JobStatus.FAILED, "error_message": "it failed"})
    response = client.get("/_jobs")
    return {
        "bodies": bodies,
        "status": response.status_code,
        "response_headers": {h: response.headers[h] for h in HEADERS if h in response.headers},
        "response_body": response.text,
    }


def main():
    # Jobs taken, never run.
    app.state.job_service.workflow_executor.submit_workflow = lambda workflow: None
    client = TestClient(app)
    table = {
        "cases": [{**c, **answer(client, c)} for c in CASES],
        "statuses": statuses(client),
        "listing": listing(client),
    }
    with open(sys.argv[1], "wb") as out:
        out.write((json.dumps(table, indent=1) + "\n").encode())
    print(f"{len(table['cases'])} cases")


if __name__ == "__main__":
    main()
