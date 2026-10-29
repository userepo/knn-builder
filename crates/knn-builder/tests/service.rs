//! The service core against a real S3 API, with the fake builder:
//! LocalStack in CI. Ignored by default, as the pipeline's tests are; the
//! job store's own rules are unit tests in `src/service.rs`.
//!
//! Each test makes a bucket of its own and removes it when it passes.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{S3, ids, le_ids};
use knn_builder::FakeBuilder;
use knn_builder::request::{self, BuildRequest, JobId};
use knn_builder::service::{Service, Settings, Status};
use serde_json::{Value, json};

const DOC_COUNT: u32 = 600;
const DIMENSION: u32 = 16;

/// The request k-NN would send for the segment `name`, with `changes`
/// to its body.
fn request(s3: &S3, name: &str, changes: &Value) -> BuildRequest {
    let mut body = json!({
        "container_name": s3.bucket,
        "vector_path": format!("{name}/_0_vectors.knnvec"),
        "doc_id_path": format!("{name}/_0_vectors.knndid"),
        "dimension": DIMENSION,
        "doc_count": DOC_COUNT,
    });
    for (field, value) in changes.as_object().unwrap() {
        body[field] = value.clone();
    }
    request::parse(body.to_string().as_bytes()).unwrap()
}

/// Puts the blobs of `request`.
async fn put_inputs(s3: &S3, request: &BuildRequest) {
    let values = (0..DOC_COUNT * DIMENSION)
        .flat_map(|i| f32::from(u16::try_from(i % 997).unwrap()).to_le_bytes());
    s3.put(&request.vector_path, values.collect(), None).await;
    s3.put(&request.doc_id_path, le_ids(&ids(DOC_COUNT)), None)
        .await;
}

/// The status of `job_id` once it has ended.
async fn ended(service: &Service, job_id: &JobId) -> Status {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match service.status(job_id) {
            Some(Status::Running) => {}
            Some(status) => return status,
            None => panic!("job {job_id} is not held"),
        }
        assert!(Instant::now() < deadline, "job {job_id} still running");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn builds_a_job_once_however_often_it_is_requested() {
    let s3 = S3::new("service-once").await;
    let request = request(&s3, "once", &json!({}));
    put_inputs(&s3, &request).await;
    let builder = FakeBuilder::new(Duration::from_millis(300));
    let service = Arc::new(Service::start(
        s3.store.clone(),
        builder.clone(),
        Settings::default(),
    ));
    // Requests at once, as a retry can race the first.
    let submits: Vec<_> = (0..16)
        .map(|_| {
            let (service, request) = (Arc::clone(&service), request.clone());
            tokio::spawn(async move { service.submit(request) })
        })
        .collect();
    for submit in submits {
        assert_eq!(submit.await.unwrap(), Ok(request.job_id()));
    }
    assert_eq!(service.status(&request.job_id()), Some(Status::Running));
    let file_name = "_0_vectors.faiss".to_owned();
    assert_eq!(
        ended(&service, &request.job_id()).await,
        Status::Completed {
            file_name: file_name.clone()
        }
    );
    assert!(s3.get(&format!("once/{file_name}")).await.is_some());
    // A retry after the job ended gets it too.
    assert_eq!(service.submit(request.clone()), Ok(request.job_id()));
    assert_eq!(builder.builds(), 1);
    Arc::into_inner(service).unwrap().close().await;
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn holds_a_failure_and_stores_nothing() {
    let s3 = S3::new("service-fails").await;
    let request = request(&s3, "fails", &json!({}));
    put_inputs(&s3, &request).await;
    let builder = FakeBuilder::failing(Duration::from_millis(50), "no GPU here");
    let service = Service::start(s3.store.clone(), builder, Settings::default());
    let job_id = service.submit(request.clone()).unwrap();
    let Status::Failed { error_message } = ended(&service, &job_id).await else {
        panic!("not failed");
    };
    assert!(error_message.contains("no GPU here"), "{error_message}");
    let mut keys = s3.keys().await;
    keys.sort();
    assert_eq!(
        keys,
        [request.doc_id_path.clone(), request.vector_path.clone()]
    );
    assert_eq!(s3.pending_uploads().await, 0);
    service.close().await;
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn fails_at_once_a_job_no_build_can_take() {
    let s3 = S3::new("service-range").await;
    let builder = FakeBuilder::new(Duration::ZERO);
    let service = Service::start(s3.store.clone(), builder.clone(), Settings::default());
    let changes = json!({"index_parameters": {"algorithm_parameters": {"m": -1}}});
    let job_id = service.submit(request(&s3, "range", &changes)).unwrap();
    // Failed when submitted: there are no inputs to download.
    assert_eq!(
        service.status(&job_id),
        Some(Status::Failed {
            error_message: "m -1 is outside 0 to 4294967295".into()
        })
    );
    assert_eq!(builder.builds(), 0);
    service.close().await;
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn expires_a_job_only_after_it_ends() {
    let s3 = S3::new("service-ttl").await;
    let request = request(&s3, "ttl", &json!({}));
    put_inputs(&s3, &request).await;
    let ttl = Duration::from_millis(100);
    let settings = Settings {
        ttl,
        ..Settings::default()
    };
    let builder = FakeBuilder::new(Duration::from_millis(600));
    let service = Service::start(s3.store.clone(), builder, settings);
    let job_id = service.submit(request).unwrap();
    tokio::time::sleep(ttl * 3).await;
    assert_eq!(service.status(&job_id), Some(Status::Running));
    assert!(matches!(
        ended(&service, &job_id).await,
        Status::Completed { .. }
    ));
    tokio::time::sleep(ttl * 2).await;
    assert_eq!(service.status(&job_id), None);
    service.close().await;
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn cancels_its_jobs_when_closed() {
    let s3 = S3::new("service-close").await;
    let builder = FakeBuilder::new(Duration::from_secs(1));
    let service = Service::start(s3.store.clone(), builder, Settings::default());
    for name in ["close-a", "close-b", "close-c"] {
        let request = request(&s3, name, &json!({}));
        put_inputs(&s3, &request).await;
        service.submit(request).unwrap();
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    let start = Instant::now();
    service.close().await;
    // Not three builds: the one under way ends, the rest never start.
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "{:?}",
        start.elapsed()
    );
    let keys = s3.keys().await;
    let index = |key: &String| Path::new(key).extension().is_some_and(|e| e == "faiss");
    assert!(!keys.iter().any(index), "{keys:?}");
    assert_eq!(s3.pending_uploads().await, 0);
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn drops_the_jobs_the_data_node_stopped_waiting_for() {
    let s3 = S3::new("service-deadline").await;
    let first = request(&s3, "first", &json!({}));
    let second = request(&s3, "second", &json!({}));
    for request in [&first, &second] {
        put_inputs(&s3, request).await;
    }
    // Each forecast to end within a second, so both are taken; but the
    // first's build takes 3 s, past the data node's 2 s.
    let settings = Settings {
        client_timeout: Duration::from_secs(2),
        poll_interval: Duration::ZERO,
        ..Settings::default()
    };
    let builder = FakeBuilder::new(Duration::from_secs(3));
    let service = Service::start(s3.store.clone(), builder.clone(), settings);
    let ids = [first.clone(), second.clone()].map(|request| service.submit(request).unwrap());
    let dropped = Status::Failed {
        error_message: "the data node's timeout of 2 s passed, so the job was dropped".to_owned(),
    };
    // The first ran its build to the end, then stored nothing; the second,
    // waiting for the builder, never built.
    for (job_id, name) in ids.iter().zip(["first", "second"]) {
        assert_eq!(ended(&service, job_id).await, dropped, "{name}");
        assert!(s3.get(&format!("{name}/_0_vectors.faiss")).await.is_none());
    }
    assert_eq!(builder.builds(), 1);
    service.close().await;
    s3.remove().await;
}
