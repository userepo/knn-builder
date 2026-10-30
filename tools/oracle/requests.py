"""Records how the service answers build requests: knn-builder's request
parser is tested against this table (crates/knn-builder/tests/data/
requests.json).

Each case is a raw request body. A body the service's model accepts is
recorded with the parameters it parses to, defaults filled in, and the job
id the service derives; it is never posted, so no job starts. Any other
body is posted to the service's own app and its answer recorded: the
status and the body, of field, message and type per error for a 422. A
body that is not UTF-8 is recorded in hex (`body_hex`).

Runs in the api image (tools/README.md):
  $RUN $API tools/oracle/requests.py crates/knn-builder/tests/data/requests.json
"""

import json
import sys

sys.path[:0] = ["/app/remote_vector_index_builder"]

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.main import app  # noqa: E402
from app.utils.hash import generate_job_id  # noqa: E402
from app.utils.request import create_request_parameters  # noqa: E402
from core.common.models import IndexBuildParameters  # noqa: E402

REQUIRED = {
    "container_name": "bucket",
    "vector_path": "segments/_0_target_field.knnvec",
    "doc_id_path": "segments/_0_target_field.knndid",
    "dimension": 768,
    "doc_count": 1000000,
}

FULL = {
    "repository_type": "s3",
    **REQUIRED,
    "tenant_id": "cluster-a",
    "data_type": "float",
    "engine": "faiss",
    "index_parameters": {
        "space_type": "l2",
        "algorithm": "hnsw",
        "algorithm_parameters": {"ef_construction": 100, "ef_search": 100, "m": 16},
    },
    "skip_stored_vectors": False,
}


def with_(**changes):
    body = dict(REQUIRED)
    for key, value in changes.items():
        if value is ...:
            body.pop(key, None)
        else:
            body[key] = value
    return body


def params(**changes):
    return with_(index_parameters={"algorithm_parameters": dict(changes)})


CASES = {
    # Accepted.
    "required only": REQUIRED,
    "every field": FULL,
    "half_float, inner product": with_(
        data_type="half_float", index_parameters={"space_type": "innerproduct"}
    ),
    "skip stored vectors": with_(skip_stored_vectors=True),
    "extra algorithm parameter": params(m=32, encoder={"name": "sq", "bits": 4}),
    "unknown index parameter": with_(index_parameters={"space_type": "l2", "foo": 1}),
    "dimension as a numeric string": with_(dimension="768"),
    "dimension as an integral float": with_(dimension=768.0),
    "doc_count 5": with_(doc_count=5),
    "bool as a string": with_(skip_stored_vectors="true"),
    "bool as 0": with_(skip_stored_vectors=0),
    "byte data": with_(data_type="byte"),
    "binary data, hamming": with_(data_type="binary", index_parameters={"space_type": "hamming"}),
    # Pairs k-NN never sends, which the service takes: binary builds in
    # Hamming distance whatever the space, other vectors under hamming in L2.
    "binary data, l2": with_(data_type="binary", index_parameters={"space_type": "l2"}),
    "float data, hamming": with_(index_parameters={"space_type": "hamming"}),
    "m 0": params(m=0),
    "negative ef_search": params(ef_search=-1),
    "dimension padded string": with_(dimension=" 768 "),
    "dimension signed string": with_(dimension="+768"),
    "doc_count past 32 bits": with_(doc_count=5_000_000_000),
    "bool as Yes": with_(skip_stored_vectors="Yes"),
    "bool as off": with_(skip_stored_vectors="off"),
    "bool as 1.0": with_(skip_stored_vectors=1.0),
    "empty tenant_id": with_(tenant_id=""),
    "dimension true": with_(dimension=True),
    "dimension decimal string": with_(dimension="768.0"),
    "dimension underscored string": with_(dimension="7_68"),
    "bool as 2": with_(skip_stored_vectors=2),
    "bool as null": with_(skip_stored_vectors=None),
    "tenant_id null": with_(tenant_id=None),
    "index_parameters null": with_(index_parameters=None),
    # Edges, either way.
    "path ending in a newline": with_(vector_path="segments/_0.knnvec\n"),
    "path with a newline inside": with_(vector_path="segments\n_0.knnvec"),
    "bool as TRUE": with_(skip_stored_vectors="TRUE"),
    "bool as a decimal string": with_(skip_stored_vectors="1.0"),
    "bool as 0.5": with_(skip_stored_vectors=0.5),
    "bool as -0.0": with_(skip_stored_vectors=-0.0),
    "bool as 2.0": with_(skip_stored_vectors=2.0),
    "dimension false": with_(dimension=False),
    "dimension exponent string": with_(dimension="1e3"),
    "dimension past 64 bits": with_(dimension=1e20),
    "dimension 2**64": with_(dimension=2**64),
    "dimension 2**64 as a string": with_(dimension=str(2**64)),
    "m below -2**63": params(m=-1e30),
    "dimension padded decimal underscored": with_(dimension="  12_3.000 "),
    "two unknown fields": {**REQUIRED, "zeta": 1, "alpha": 2},
    "unknown field before a bad one": {"zeta": 1, **with_(doc_count=0)},
    # Refused.
    "empty object": {},
    "unknown field": with_(color="red"),
    "path without .knnvec": with_(vector_path="segments/_0.vec"),
    "path .knnvec alone": with_(vector_path=".knnvec"),
    "dimension 0": with_(dimension=0),
    "dimension negative": with_(dimension=-3),
    "dimension text": with_(dimension="abc"),
    "dimension fractional": with_(dimension=1.5),
    "dimension null": with_(dimension=None),
    "dimension list": with_(dimension=[768]),
    "doc_count 4": with_(doc_count=4),
    "unknown data_type": with_(data_type="int8"),
    "unknown space_type": with_(index_parameters={"space_type": "cosinesimil"}),
    "unknown algorithm": with_(index_parameters={"algorithm": "ivf"}),
    "unknown engine": with_(engine="lucene"),
    "unknown repository_type": with_(repository_type="gcs"),
    "container_name a number": with_(container_name=123),
    "index_parameters a string": with_(index_parameters="hnsw"),
    "algorithm_parameters a list": with_(index_parameters={"algorithm_parameters": [16]}),
    "m text": params(m="sixteen"),
    "bool text": with_(skip_stored_vectors="maybe"),
    "several errors": with_(container_name=..., doc_count=1, data_type="int8"),
    "every index parameter wrong": with_(
        index_parameters={
            "algorithm_parameters": {"m": "x", "ef_search": "y", "ef_construction": "z"},
            "space_type": "q",
            "algorithm": "w",
        }
    ),
}

REQUIRED_JSON = json.dumps(REQUIRED)


def ignored(raw):
    """The required fields and `raw`, verbatim, as an unknown index
    parameter: the service ignores it, so it reaches neither the parsed
    parameters nor this table, whose JSON could not always hold it."""
    return REQUIRED_JSON[:-1] + ', "index_parameters": {"ignored": ' + raw + "}}"


def kept(raw):
    """The required fields and `raw`, verbatim, as an extra algorithm
    parameter, which the service keeps as it parsed it."""
    return (
        REQUIRED_JSON[:-1]
        + ', "index_parameters": {"algorithm_parameters": {"x": '
        + raw
        + "}}}"
    )


RAW = {
    "not JSON": "{",
    "a JSON list": "[]",
    "JSON null": "null",
    "trailing comma": '{"container_name": "bucket",}',
    "not JSON after non-ASCII": '{"container_name": "bécket" "x"}',
    "duplicate key": '{"container_name": "a", "container_name": "b", "vector_path": "p.knnvec", '
    '"doc_id_path": "p.knndid", "dimension": 8, "doc_count": 10}',
    # Bodies at the edges of JSON and of UTF-8, either way.
    "empty body": "",
    "whitespace body": "   ",
    "BOM": b"\xef\xbb\xbf" + REQUIRED_JSON.encode(),
    "BOM alone": b"\xef\xbb\xbf",
    "invalid UTF-8": REQUIRED_JSON.encode().replace(b'"bucket"', b'"b\xffcket"'),
    "invalid UTF-8 after a JSON error": b'{"container_name": , "x": "\xff"}',
    "UTF-16": REQUIRED_JSON.encode("utf-16"),
    "UTF-8 of a surrogate": ignored('"@"').encode().replace(b"@", b"\xed\xa0\x80"),
    "escaped lone surrogate": ignored('"\\ud800"'),
    "NaN": ignored("NaN"),
    "Infinity": ignored("-Infinity"),
    "NaN dimension": REQUIRED_JSON.replace('"dimension": 768', '"dimension": NaN'),
    "nesting past 128 levels": ignored("[" * 200 + "]" * 200),
    "number past f64": ignored("1e400"),
    "dimension past f64": REQUIRED_JSON.replace('"dimension": 768', '"dimension": 1e400'),
    "trailing data": REQUIRED_JSON + " x",
    # Decimals a fast float parser can round one step off.
    "decimal rounded to even": kept("9007199254740993.0"),
    "decimal near the least normal": kept("2.2250738585072011e-308"),
}


def record(client, body):
    # As Starlette reads it: json.loads of the bytes, which drops a BOM
    # and reads UTF-16 and UTF-32 too.
    try:
        value = json.loads(body)
        parsed = IndexBuildParameters.model_validate(value)
    except (ValueError, ValidationError):
        response = client.post(
            "/_build", content=body, headers={"Content-Type": "application/json"}
        )
        return {"status": response.status_code, "response": response.json()}
    return {
        "status": 200,
        "parsed": parsed.model_dump(mode="json"),
        "job_id": generate_job_id(create_request_parameters(parsed)),
    }


def entry(name, body):
    """A case: its body as text, or in hex when it is not UTF-8."""
    raw = body if isinstance(body, bytes) else body.encode()
    try:
        shown = {"body": raw.decode()}
    except UnicodeDecodeError:
        shown = {"body_hex": raw.hex()}
    return {"case": name, **shown, **record(TestClient(app), raw)}


def main():
    table = [entry(name, json.dumps(value)) for name, value in CASES.items()]
    table += [entry(name, body) for name, body in RAW.items()]
    with open(sys.argv[1], "wb") as out:
        out.write((json.dumps(table, indent=2, allow_nan=False) + "\n").encode())
    print(f"{len(table)} cases")


if __name__ == "__main__":
    main()
