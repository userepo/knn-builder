//! The HTTP API against the service's own answers,
//! `tests/data/routes.json`, recorded by `tools/oracle/routes.py`; its
//! Basic Auth; and, against LocalStack (ignored by default, as the
//! pipeline's tests are), a job's life through it. The cases knn-builder
//! answers differently are listed in `DIVERGENT`, each with why.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::{S3, ids, le_ids};
use knn_builder::FakeBuilder;
use knn_builder::api;
use knn_builder::config::Credentials;
use knn_builder::service::{Service, Settings};
use knn_store::{Store, Transfers};
use serde_json::{Value, json};
use tower::ServiceExt as _;

const ROUTES: &str = include_str!("data/routes.json");

/// The headers the oracle records.
const HEADERS: [&str; 4] = ["content-type", "allow", "location", "www-authenticate"];

/// Cases knn-builder answers otherwise than the service, with its status
/// and body:
///
/// - The same job id with other parameters is a conflict, 409; the
///   service compares only the path and tenant the id is made of, and
///   answers with the job held.
/// - FastAPI's documentation and the service's debug list of jobs are not
///   served.
const DIVERGENT: [(&str, u16, &str); 4] = [
    (
        "build, same path, other parameters",
        409,
        r#"{"detail":"Job 96c423eb32079f67c20f0ec969f6eef553e3cef4a79c92979e1f7a542218c58f with vector path segments/_0_target_field.knnvec is already held with other parameters"}"#,
    ),
    ("jobs", 404, r#"{"detail":"Not Found"}"#),
    ("docs", 404, r#"{"detail":"Not Found"}"#),
    ("openapi", 404, r#"{"detail":"Not Found"}"#),
];

/// An answer: its status, the recorded headers it has, and its body.
#[derive(Debug, PartialEq)]
struct Answer {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

/// Sends a request to `router`, as the oracle's client sends it, with the
/// `Host` it names.
async fn send(
    router: &api::App,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> Answer {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header("host", "testserver");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let request = request.body(Body::from(body.to_owned())).unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let headers = HEADERS
        .iter()
        .filter_map(|name| {
            let value = response.headers().get(*name)?;
            Some(((*name).to_owned(), value.to_str().unwrap().to_owned()))
        })
        .collect();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    Answer {
        status,
        headers,
        body: String::from_utf8(body.to_vec()).unwrap(),
    }
}

/// A store on an S3 that is not there: jobs taken through it fail in the
/// background, which these tests do not look at.
async fn nowhere() -> Store {
    let mut settings = knn_store::Settings::from_env();
    settings.endpoint_url = Some("http://127.0.0.1:9".into());
    settings.region = Some("us-east-1".into());
    Store::new(&settings, Transfers::default()).await
}

fn service(store: Store, builder: FakeBuilder, settings: Settings) -> Arc<Service> {
    Arc::new(Service::start(store, builder, settings))
}

#[tokio::test(flavor = "multi_thread")]
async fn answers_as_the_service_does() {
    let table: Value = serde_json::from_str(ROUTES).unwrap();
    let cases = table["cases"].as_array().unwrap();
    assert!(cases.len() >= 30, "{} cases", cases.len());
    let service = service(
        nowhere().await,
        FakeBuilder::new(Duration::ZERO),
        Settings::default(),
    );
    let router = api::app(service, None);
    let mut divergent = 0;
    for case in cases {
        let name = case["case"].as_str().unwrap();
        let headers: Vec<(&str, &str)> = case["headers"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str().unwrap()))
            .collect();
        let answer = send(
            &router,
            case["method"].as_str().unwrap(),
            case["path"].as_str().unwrap(),
            &headers,
            case["body"].as_str().unwrap_or_default(),
        )
        .await;
        if let Some((_, status, body)) = DIVERGENT.iter().find(|(n, ..)| *n == name) {
            divergent += 1;
            assert_eq!(
                (answer.status, answer.body.as_str()),
                (*status, *body),
                "{name}"
            );
            assert_ne!(
                u64::from(answer.status),
                case["status"],
                "{name}: answered as the service does"
            );
            continue;
        }
        let headers: Vec<(String, String)> = case["response_headers"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
            .collect();
        let want = Answer {
            status: u16::try_from(case["status"].as_u64().unwrap()).unwrap(),
            headers: HEADERS
                .iter()
                .filter_map(|h| headers.iter().find(|(k, _)| k == h).cloned())
                .collect(),
            body: case["response_body"].as_str().unwrap().to_owned(),
        };
        assert_eq!(answer, want, "{name}");
    }
    assert_eq!(
        divergent,
        DIVERGENT.len(),
        "every divergent case is in the table"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn asks_for_basic_auth_when_it_is_set() {
    let credentials = Credentials {
        username: "admin".into(),
        password: "s3cret:x".into(),
    };
    let service = service(
        nowhere().await,
        FakeBuilder::new(Duration::ZERO),
        Settings::default(),
    );
    let router = api::app(service, Some(credentials));
    let challenge = ("www-authenticate".to_owned(), "Basic".to_owned());
    let refused = |message: &str| Answer {
        status: 401,
        headers: vec![
            ("content-type".to_owned(), "application/json".to_owned()),
            challenge.clone(),
        ],
        body: format!(r#"{{"detail":"{message}"}}"#),
    };
    // admin:s3cret:x, admin:wrong, and admin with no colon.
    let right = "Basic YWRtaW46czNjcmV0Ong=";
    for (authorization, answer) in [
        (None, refused("Not authenticated")),
        (
            Some("Bearer YWRtaW46czNjcmV0Ong="),
            refused("Not authenticated"),
        ),
        (
            Some("Basic !!!"),
            refused("Invalid authentication credentials"),
        ),
        (
            Some("Basic YWRtaW4="),
            refused("Invalid authentication credentials"),
        ),
        // The password with more after it, and the username.
        (
            Some("Basic YWRtaW46czNjcmV0Onh0cmE="),
            refused("Incorrect username or password"),
        ),
        (
            Some("Basic YWRtaW5YOnMzY3JldDp4"),
            refused("Incorrect username or password"),
        ),
        (
            Some("Basic YWRtaW46d3Jvbmc="),
            refused("Incorrect username or password"),
        ),
    ] {
        let headers: Vec<(&str, &str)> = authorization
            .map(|a| ("authorization", a))
            .into_iter()
            .collect();
        for (method, path) in [
            ("GET", "/_heart_beat"),
            ("GET", "/_status/abc"),
            ("POST", "/_build"),
        ] {
            assert_eq!(
                send(&router, method, path, &headers, "").await,
                answer,
                "{authorization:?} {path}"
            );
        }
    }
    let ok = send(
        &router,
        "GET",
        "/_heart_beat",
        &[("authorization", right)],
        "",
    )
    .await;
    assert_eq!(ok.status, 200);
    // Any case of the scheme.
    let ok = send(
        &router,
        "GET",
        "/_heart_beat",
        &[("authorization", "basic YWRtaW46czNjcmV0Ong=")],
        "",
    )
    .await;
    assert_eq!(ok.status, 200);
    let missing = send(
        &router,
        "GET",
        "/_status/abc",
        &[("authorization", right)],
        "",
    )
    .await;
    assert_eq!(
        (missing.status, missing.body.as_str()),
        (404, r#"{"detail":"Job not found"}"#)
    );
}

/// The request k-NN would send for the segment `name`.
fn build_body(s3: &S3, name: &str) -> Value {
    json!({
        "container_name": s3.bucket,
        "vector_path": format!("{name}/_0_vectors.knnvec"),
        "doc_id_path": format!("{name}/_0_vectors.knndid"),
        "dimension": 16,
        "doc_count": 600,
    })
}

async fn put_inputs(s3: &S3, body: &Value) {
    let values =
        (0..600 * 16_u32).flat_map(|i| f32::from(u16::try_from(i % 997).unwrap()).to_le_bytes());
    s3.put(
        body["vector_path"].as_str().unwrap(),
        values.collect(),
        None,
    )
    .await;
    s3.put(
        body["doc_id_path"].as_str().unwrap(),
        le_ids(&ids(600)),
        None,
    )
    .await;
}

async fn post(router: &api::App, body: &Value) -> Answer {
    send(
        router,
        "POST",
        "/_build",
        &[("content-type", "application/json")],
        &body.to_string(),
    )
    .await
}

/// The status of `job_id` once it is no longer running.
async fn ended(router: &api::App, job_id: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let answer = send(router, "GET", &format!("/_status/{job_id}"), &[], "").await;
        assert_eq!(answer.status, 200, "{}", answer.body);
        let status: Value = serde_json::from_str(&answer.body).unwrap();
        if status["task_status"] != "RUNNING_INDEX_BUILD" {
            return status;
        }
        assert!(Instant::now() < deadline, "job {job_id} still running");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn builds_a_job_through_the_api() {
    let s3 = S3::new("api-build").await;
    let body = build_body(&s3, "api");
    put_inputs(&s3, &body).await;
    let builder = FakeBuilder::new(Duration::from_millis(300));
    let router = api::app(
        service(s3.store.clone(), builder.clone(), Settings::default()),
        None,
    );

    let first = post(&router, &body).await;
    assert_eq!(first.status, 200, "{}", first.body);
    let job_id = serde_json::from_str::<Value>(&first.body).unwrap()["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    // A retry gets the same job; the same path with other parameters, 409.
    assert_eq!(post(&router, &body).await, first);
    let mut other = body.clone();
    other["dimension"] = json!(8);
    let conflict = post(&router, &other).await;
    assert_eq!(conflict.status, 409, "{}", conflict.body);
    // Invalid, 422.
    let mut invalid = body.clone();
    invalid["doc_count"] = json!(4);
    assert_eq!(post(&router, &invalid).await.status, 422);

    let running = send(&router, "GET", &format!("/_status/{job_id}"), &[], "").await;
    assert_eq!(
        running.body,
        r#"{"task_status":"RUNNING_INDEX_BUILD","file_name":null,"error_message":null}"#
    );
    assert_eq!(
        ended(&router, &job_id).await,
        json!({
            "task_status": "COMPLETED_INDEX_BUILD",
            "file_name": "_0_vectors.faiss",
            "error_message": null,
        })
    );
    assert!(s3.get("api/_0_vectors.faiss").await.is_some());
    assert_eq!(builder.builds(), 1);
    let missing = send(&router, "GET", "/_status/0123abc", &[], "").await;
    assert_eq!(
        (missing.status, missing.body.as_str()),
        (404, r#"{"detail":"Job not found"}"#)
    );
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn answers_for_a_running_job_past_its_ttl() {
    let s3 = S3::new("api-ttl").await;
    let body = build_body(&s3, "ttl");
    put_inputs(&s3, &body).await;
    let ttl = Duration::from_millis(100);
    let settings = Settings {
        ttl,
        ..Settings::default()
    };
    let router = api::app(
        service(
            s3.store.clone(),
            FakeBuilder::new(Duration::from_millis(600)),
            settings,
        ),
        None,
    );
    let job_id = serde_json::from_str::<Value>(&post(&router, &body).await.body).unwrap()["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    tokio::time::sleep(ttl * 3).await;
    let status = send(&router, "GET", &format!("/_status/{job_id}"), &[], "").await;
    assert_eq!(status.status, 200);
    assert!(
        status.body.contains("RUNNING_INDEX_BUILD"),
        "{}",
        status.body
    );
    assert_eq!(
        ended(&router, &job_id).await["task_status"],
        "COMPLETED_INDEX_BUILD"
    );
    tokio::time::sleep(ttl * 2).await;
    let expired = send(&router, "GET", &format!("/_status/{job_id}"), &[], "").await;
    assert_eq!(expired.status, StatusCode::NOT_FOUND.as_u16());
    s3.remove().await;
}
