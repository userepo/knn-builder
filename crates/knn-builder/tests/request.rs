//! The request parser against the service's own answers:
//! `tests/data/requests.json`, recorded by `tools/oracle/requests.py`. An
//! accepted body parses to the same parameters and job id; a refused one
//! to the same status and body. The cases knn-builder answers differently
//! are listed in `DIVERGENT`, each with why.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use knn_builder::request::{self, FieldError, Rejection};
use knn_builder::{DataType, JobError, SpaceType};
use serde_json::{Value, json};

const TABLE: &str = include_str!("data/requests.json");

/// A case's name, knn-builder's status, and each error's field and type.
type Divergent = (&'static str, u16, &'static [(&'static str, &'static str)]);

/// Cases knn-builder answers otherwise than the service, with its status
/// and the field and type of each error:
///
/// - Vectors it cannot build yet, which the service takes. Refused, k-NN
///   builds the segment on the data node, rather than sending a job that
///   would only fail.
/// - Integers past 64 bits, which the service takes, as Python takes any
///   integer, and then refuses with 507: no build holds such a size.
/// - Bodies Python's `json.loads` reads and knn-builder does not: forms
///   outside RFC 8259 JSON in UTF-8, and JSON past `serde_json`'s limits,
///   none of which k-NN's client sends (the module's docs).
const DIVERGENT: [Divergent; 13] = [
    ("byte data", 422, &[("body.data_type", "unsupported")]),
    (
        "binary data, hamming",
        422,
        &[
            ("body.data_type", "unsupported"),
            ("body.index_parameters.space_type", "unsupported"),
        ],
    ),
    // serde_json reads an integer this large as a float.
    (
        "dimension 2**64",
        422,
        &[("body.dimension", "int_parsing_size")],
    ),
    (
        "dimension 2**64 as a string",
        422,
        &[("body.dimension", "less_than_equal")],
    ),
    ("UTF-16", 400, &[]),
    ("UTF-8 of a surrogate", 400, &[]),
    (
        "escaped lone surrogate",
        422,
        &[("body[208]", "json_invalid")],
    ),
    ("NaN", 422, &[("body[201]", "json_invalid")]),
    ("Infinity", 422, &[("body[202]", "json_invalid")]),
    ("NaN dimension", 422, &[("body[142]", "json_invalid")]),
    ("number past f64", 422, &[("body[205]", "json_invalid")]),
    ("dimension past f64", 422, &[("body[146]", "json_invalid")]),
    (
        "nesting past 128 levels",
        422,
        &[("body[326]", "json_invalid")],
    ),
];

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(TABLE)
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

/// A case's body: its text, or its bytes in hex when they are not UTF-8.
fn body(case: &Value) -> Vec<u8> {
    if let Some(text) = case["body"].as_str() {
        return text.as_bytes().to_vec();
    }
    let hex = case["body_hex"].as_str().unwrap().as_bytes();
    hex.chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn answers_as_the_service_does() {
    let cases = cases();
    assert!(cases.len() >= 80, "{} cases", cases.len());
    let mut divergent = 0;
    for case in &cases {
        let name = case["case"].as_str().unwrap();
        let parsed = request::parse(&body(case));
        let service = case["status"].as_u64().unwrap();
        if let Some((_, status, expected)) = DIVERGENT.iter().find(|(n, ..)| *n == name) {
            divergent += 1;
            let rejection = parsed.unwrap_err();
            let got: Vec<(&str, &str)> = rejection
                .errors()
                .iter()
                .map(|e| (e.field.as_str(), e.kind))
                .collect();
            assert_eq!(
                (rejection.status(), got),
                (*status, expected.to_vec()),
                "{name}"
            );
            assert!(
                u64::from(*status) != service || rejection.body() != case["response"],
                "{name}: answered as the service does"
            );
            continue;
        }
        match service {
            200 => {
                let parsed = parsed.unwrap_or_else(|r| panic!("{name}: {r:?}"));
                assert_eq!(parsed.to_json(), case["parsed"], "{name}");
                assert_eq!(parsed.job_id().as_str(), case["job_id"], "{name}");
            }
            400 | 422 => {
                let rejection = parsed.err().unwrap_or_else(|| panic!("{name}: accepted"));
                assert_eq!(u64::from(rejection.status()), service, "{name}");
                assert_eq!(rejection.body(), case["response"], "{name}");
            }
            status => panic!("{name}: status {status}"),
        }
    }
    assert_eq!(
        divergent,
        DIVERGENT.len(),
        "every divergent case is in the table"
    );
}

/// A 400 has no field errors (its body is checked against the table).
#[test]
fn refuses_a_body_that_is_not_utf8_with_400() {
    let rejection = request::parse(b"{\"container_name\": \"\xff\"}").unwrap_err();
    assert_eq!(rejection, Rejection::Unreadable);
    assert_eq!(rejection.status(), 400);
    assert_eq!(rejection.errors(), []);
}

#[test]
fn maps_an_accepted_request_to_its_job() {
    let body = json!({
        "container_name": "bucket",
        "vector_path": "a/_0_f.knnvec",
        "doc_id_path": "a/_0_f.knndid",
        "dimension": "128",
        "doc_count": 1000,
        "data_type": "half_float",
        "index_parameters": {
            "space_type": "innerproduct",
            "algorithm_parameters": {"m": 32, "ef_construction": 256, "ef_search": -1},
        },
        "skip_stored_vectors": "yes",
    });
    let job = request::parse(body.to_string().as_bytes())
        .unwrap()
        .job()
        .unwrap();
    assert_eq!(job.bucket, "bucket");
    assert_eq!(job.vector_path, "a/_0_f.knnvec");
    assert_eq!(job.doc_id_path, "a/_0_f.knndid");
    assert_eq!((job.dimension, job.doc_count), (128, 1000));
    assert_eq!(job.data_type, DataType::HalfFloat);
    assert_eq!(job.space_type, SpaceType::InnerProduct);
    assert_eq!((job.m, job.ef_construction, job.ef_search), (32, 256, -1));
    assert!(job.skip_stored_vectors);
}

/// What the service accepts and then fails in its build, knn-builder
/// fails before starting: past 32 bits, or an m below 0.
#[test]
fn fails_a_job_no_build_can_take() {
    let with = |field: &str, value: Value| {
        let mut body = json!({
            "container_name": "b",
            "vector_path": "p.knnvec",
            "doc_id_path": "p.knndid",
            "dimension": 8,
            "doc_count": 10,
        });
        match field {
            "m" | "ef_search" => {
                body["index_parameters"] = json!({"algorithm_parameters": {field: value}});
            }
            _ => body[field] = value,
        }
        request::parse(body.to_string().as_bytes()).unwrap().job()
    };
    let out_of_range = |name: &'static str, value: i128, min: i128, max: i128| {
        Err(JobError::OutOfRange {
            name,
            value,
            min,
            max,
        })
    };
    let u32_max = i128::from(u32::MAX);
    assert_eq!(
        with("doc_count", json!(5_000_000_000_u64)),
        out_of_range("doc_count", 5_000_000_000, 0, u32_max)
    );
    assert_eq!(
        with("dimension", json!(u32_max + 1)),
        out_of_range("dimension", u32_max + 1, 0, u32_max)
    );
    assert_eq!(with("m", json!(-1)), out_of_range("m", -1, 0, u32_max));
    assert_eq!(
        with("ef_search", json!(1_i64 << 31)),
        out_of_range("ef_search", 1 << 31, i32::MIN.into(), i32::MAX.into())
    );
    assert!(with("doc_count", json!(u32::MAX)).is_ok());
    assert!(with("ef_search", json!(i32::MIN)).is_ok());
}

/// Past what knn-builder holds, an integer is refused as pydantic refuses
/// one past a bound; the service's model has none there.
#[test]
fn refuses_integers_past_64_bits() {
    let body = |dimension: &str, m: &str| {
        format!(
            r#"{{"container_name":"b","vector_path":"p.knnvec","doc_id_path":"p.knndid","dimension":"{dimension}","doc_count":10,"index_parameters":{{"algorithm_parameters":{{"m":"{m}"}}}}}}"#
        )
    };
    let rejection = request::parse(body("18446744073709551616", "-9223372036854775809").as_bytes())
        .unwrap_err();
    assert_eq!(
        rejection.errors(),
        [
            FieldError {
                field: "body.dimension".into(),
                message: "Input should be less than or equal to 18446744073709551615".into(),
                kind: "less_than_equal",
            },
            FieldError {
                field: "body.index_parameters.algorithm_parameters.m".into(),
                message: "Input should be greater than or equal to -9223372036854775808".into(),
                kind: "greater_than_equal",
            },
        ]
    );
    let at_the_bounds = body("18446744073709551615", "-9223372036854775808");
    assert!(request::parse(at_the_bounds.as_bytes()).is_ok());
}

/// The job id is the service's: SHA-256 of `vector_path-tenant_id`.
#[test]
fn derives_the_job_id_from_path_and_tenant() {
    let parse = |tenant: &str| {
        let body = json!({
            "container_name": "b",
            "vector_path": "p.knnvec",
            "doc_id_path": "p.knndid",
            "dimension": 8,
            "doc_count": 10,
            "tenant_id": tenant,
        });
        request::parse(body.to_string().as_bytes()).unwrap()
    };
    // printf 'p.knnvec-t' | sha256sum
    assert_eq!(
        parse("t").job_id().as_str(),
        "91f2e75221ddc094120b8234ee5b4af7a9b43caf860990ac9ec1c13f91812261"
    );
    assert_ne!(parse("t").job_id(), parse("u").job_id());
}
