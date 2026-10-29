//! The pipeline against a real S3 API, with the fake builder: LocalStack in
//! CI. Ignored by default; run them with the service's own settings
//! pointing at LocalStack, as knn-store's are (see the README).
//!
//! Each test makes a bucket of its own and removes it when it passes.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

mod common;

use std::time::{Duration, Instant};

use common::{S3, ids, le_ids};
use knn_builder::cost::{Fitted, Stages};
use knn_builder::forecast::Forecaster;
use knn_builder::{
    Builder, Built, DataType, FakeBuilder, Job, JobError, Pipeline, SpaceType, Span, Stage,
};
use knn_gpu::{BuildParams, fp16};
use knn_write::{IndexFile, IndexSpec, Vectors};

/// What the pipeline tests do with their bucket.
trait Inputs {
    async fn put_inputs(&self, job: &Job, values: &[f32], blob: Blob, kms_key: Option<&str>);
    fn pipeline(&self, builder: impl Builder) -> Pipeline;
}

impl Inputs for S3 {
    /// Puts a job's blobs: `values` encoded as `blob` says, and doc ids
    /// counting down from 9,999.
    async fn put_inputs(&self, job: &Job, values: &[f32], blob: Blob, kms_key: Option<&str>) {
        let vectors = match blob {
            Blob::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Blob::F16 => halves(values)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        };
        self.put(&job.vector_path, vectors, kms_key).await;
        self.put(&job.doc_id_path, le_ids(&ids(job.doc_count)), None)
            .await;
    }

    /// A pipeline on this store, with no host-memory budget to speak of.
    fn pipeline(&self, builder: impl Builder) -> Pipeline {
        Pipeline::start(self.store.clone(), builder, u64::MAX)
    }
}

/// How a test encodes a vector blob.
#[derive(Debug, Clone, Copy)]
enum Blob {
    F32,
    F16,
}

fn job(bucket: &str, name: &str, doc_count: u32, dimension: u32) -> Job {
    Job {
        bucket: bucket.to_owned(),
        vector_path: format!("{name}/_0_vectors.knnvec"),
        doc_id_path: format!("{name}/_0_vectors.knndid"),
        dimension,
        doc_count,
        data_type: DataType::Float,
        space_type: SpaceType::L2,
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        skip_stored_vectors: false,
    }
}

fn values(job: &Job) -> Vec<f32> {
    let len = job.doc_count * job.dimension;
    (0..len)
        .map(|i| f32::from(u16::try_from(i % 9_973).unwrap()) * 0.31 - 1_500.0)
        .collect()
}

fn halves(values: &[f32]) -> Vec<u16> {
    let mut out = vec![0; values.len()];
    fp16::convert(values, &mut out);
    out
}

/// The file the pipeline must store for `job`: knn-write's, from the
/// inputs as stored (`halves` when they are fp16) and the fake's graph.
fn expected(job: &Job, values: &[f32], half: bool) -> Vec<u8> {
    let metric = job.space_type.build_metric();
    let params = BuildParams::new(job.doc_count, job.dimension, job.m, metric).unwrap();
    let degree = params.graph_degree as usize;
    let graph = FakeBuilder::graph(job.doc_count as usize, degree);
    let halves = halves(values);
    let ids = ids(job.doc_count);
    let file = IndexFile::new(IndexSpec {
        dimension: job.dimension as usize,
        metric: job.space_type.file_metric(),
        graph: &graph,
        graph_degree: degree,
        vectors: if half {
            Vectors::F16(&halves)
        } else {
            Vectors::F32(values)
        },
        ids: &ids,
        ef_construction: job.ef_construction,
        ef_search: job.ef_search,
        skip_stored_vectors: job.skip_stored_vectors,
    })
    .unwrap();
    let mut out = Vec::new();
    file.write_to(&mut out).unwrap();
    out
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn stores_the_index_of_each_kind_of_job() {
    let s3 = S3::new("kinds").await;
    let float = job(&s3.bucket, "float", 600, 16);
    let mut fp16_blob = job(&s3.bucket, "fp16", 600, 16);
    fp16_blob.data_type = DataType::HalfFloat;
    fp16_blob.space_type = SpaceType::InnerProduct;
    // A float field with the fp16 quantizer: fp32 values, stored as fp16.
    let mut fp32_blob = job(&s3.bucket, "fp32-as-half", 600, 16);
    fp32_blob.data_type = DataType::HalfFloat;
    let mut skipped = job(&s3.bucket, "skipped", 600, 16);
    skipped.skip_stored_vectors = true;
    // A small count, whose m the service lowers.
    let small = job(&s3.bucket, "small", 40, 8);

    let cases = [
        (&float, Blob::F32, false),
        (&fp16_blob, Blob::F16, true),
        (&fp32_blob, Blob::F32, true),
        (&skipped, Blob::F32, false),
        (&small, Blob::F32, false),
    ];
    for &(job, blob, _) in &cases {
        s3.put_inputs(job, &values(job), blob, None).await;
    }
    let pipeline = s3.pipeline(FakeBuilder::new(Duration::ZERO));
    let handles: Vec<_> = cases
        .iter()
        .map(|&(job, ..)| pipeline.submit(job.clone()))
        .collect();
    for (handle, &(job, _, half)) in handles.into_iter().zip(&cases) {
        let built = handle.outcome().await.unwrap();
        assert_eq!(built.file_name, "_0_vectors.faiss");
        let stored = s3.get(&job.index_key()).await.unwrap();
        assert!(
            stored == expected(job, &values(job), half),
            "{} differs",
            job.vector_path
        );
    }
    pipeline.close().await;
    assert_eq!(s3.pending_uploads().await, 0);
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn overlaps_the_stages() {
    let s3 = S3::new("overlap").await;
    let jobs: Vec<Job> = (0..3)
        .map(|i| job(&s3.bucket, &format!("job{i}"), 2_000, 32))
        .collect();
    for job in &jobs {
        s3.put_inputs(job, &values(job), Blob::F32, None).await;
    }
    let pipeline = s3.pipeline(FakeBuilder::new(Duration::from_millis(400)));
    let handles: Vec<_> = jobs
        .iter()
        .map(|job| pipeline.submit(job.clone()))
        .collect();
    let mut built: Vec<Built> = Vec::new();
    for handle in handles {
        built.push(handle.outcome().await.unwrap());
    }
    pipeline.close().await;
    let [first, second, third] = [0, 1, 2].map(|i| built[i].timings);

    // The second job downloads while the first builds, and the first
    // uploads while the second builds.
    assert!(second.download.start < first.build.end);
    assert!(first.upload.start < second.build.end);
    // The builder takes one job at a time.
    assert!(second.build.start >= first.build.end);
    assert!(third.build.start >= second.build.end);
    // The third waits for the second's slot: it cannot download before
    // the builder, done with the first, has taken the second.
    assert!(third.download.start >= first.build.end);
    s3.remove().await;
}

/// 2 MiB: two small jobs (1 MiB each, rounded up) fit at once, and a
/// large one (about 7.7 MB) takes all of it.
const BUDGET: u64 = 2 << 20;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn keeps_the_jobs_in_memory_within_the_budget() {
    let s3 = S3::new("budget").await;
    let jobs = [
        job(&s3.bucket, "small1", 600, 16),
        job(&s3.bucket, "small2", 600, 16),
        job(&s3.bucket, "large", 20_000, 32),
        job(&s3.bucket, "small3", 600, 16),
    ];
    for job in &jobs {
        s3.put_inputs(job, &values(job), Blob::F32, None).await;
    }
    let builder = FakeBuilder::new(Duration::from_millis(400));
    let pipeline = Pipeline::start(s3.store.clone(), builder, BUDGET);
    let handles: Vec<_> = jobs
        .iter()
        .map(|job| pipeline.submit(job.clone()))
        .collect();
    let mut timings = Vec::new();
    for handle in handles {
        timings.push(handle.outcome().await.unwrap().timings);
    }
    pipeline.close().await;
    let [small1, small2, large, small3] = [0, 1, 2, 3].map(|i| timings[i]);

    // Two small jobs fit: the second downloads while the first builds.
    assert!(small2.download.start < small1.build.end);
    // The large one waits until both have ended, and the next small one
    // until it has: neither downloads while the builder works.
    assert!(large.download.start >= small2.upload.end);
    assert!(small3.download.start >= large.upload.end);
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn fails_a_build_the_free_device_memory_cannot_hold() {
    let s3 = S3::new("device-memory").await;
    let small = job(&s3.bucket, "small", 600, 16);
    let large = job(&s3.bucket, "large", 20_000, 32);
    for job in [&small, &large] {
        s3.put_inputs(job, &values(job), Blob::F32, None).await;
    }
    // Free memory for the small build, not the large one, as if another
    // process had taken the rest.
    let free = u64::midpoint(
        Fitted::device_bytes_needed(&small),
        Fitted::device_bytes_needed(&large),
    );
    assert!(Fitted::device_bytes_needed(&small) < free);
    let builder = FakeBuilder::new(Duration::ZERO).with_free_device_memory(free);
    let pipeline = s3.pipeline(builder.clone());
    let large_outcome = pipeline.submit(large.clone()).outcome();
    let small_outcome = pipeline.submit(small.clone()).outcome();
    let failed = large_outcome.await;
    assert!(
        matches!(failed, Err(JobError::DeviceMemory { .. })),
        "{failed:?}"
    );
    small_outcome.await.unwrap();
    pipeline.close().await;
    // The large job never reached the builder.
    assert_eq!(builder.builds(), 1);
    s3.remove().await;
}

/// How far a forecast may miss when it knows each stage's time, in
/// seconds: what the stages' hand-offs take.
const FORECAST_SLACK: f64 = 0.05;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn ends_the_jobs_when_the_forecast_says() {
    let s3 = S3::new("forecast").await;
    // The budget test's jobs: the large one waits for both small ones
    // before it, and the last small one for the large one.
    let jobs = [
        job(&s3.bucket, "small1", 600, 16),
        job(&s3.bucket, "small2", 600, 16),
        job(&s3.bucket, "large", 20_000, 32),
        job(&s3.bucket, "small3", 600, 16),
    ];
    for job in &jobs {
        s3.put_inputs(job, &values(job), Blob::F32, None).await;
    }
    let builder = FakeBuilder::new(Duration::from_millis(400));
    let pipeline = Pipeline::start(s3.store.clone(), builder, BUDGET);
    let handles: Vec<_> = jobs
        .iter()
        .map(|job| pipeline.submit(job.clone()))
        .collect();
    let submitted = (Instant::now(), pipeline.in_flight());
    let order: Vec<_> = submitted.1.iter().map(|f| f.job.clone()).collect();
    assert_eq!(order, jobs, "in the order submitted");
    // Partway: the first jobs are on, or past, the builder.
    tokio::time::sleep(Duration::from_millis(700)).await;
    let partway = (Instant::now(), pipeline.in_flight());
    assert!(partway.1.len() < jobs.len(), "{:?}", partway.1);
    assert!(
        partway.1.iter().any(|f| f.stage != Stage::Queued),
        "{:?}",
        partway.1
    );
    let mut timings = Vec::new();
    for handle in handles {
        timings.push(handle.outcome().await.unwrap().timings);
    }
    assert!(pipeline.in_flight().is_empty(), "every job has ended");
    pipeline.close().await;

    // Told what each stage of each job took, the forecast plays the
    // pipeline's rules: each job ends when it did.
    let took = |job: &Job| {
        let t = timings[jobs.iter().position(|j| j == job).unwrap()];
        let seconds = |span: Span| span.duration().as_secs_f64();
        Stages {
            download: seconds(t.download),
            build: seconds(t.build),
            upload: seconds(t.upload),
        }
    };
    let forecaster = Forecaster::new(took, BUDGET);
    for (label, (now, in_flight)) in [("at submission", &submitted), ("partway", &partway)] {
        let forecast = forecaster.forecast(*now, in_flight, None);
        for (flight, forecast) in in_flight.iter().zip(&forecast.ends) {
            let i = jobs.iter().position(|j| *j == flight.job).unwrap();
            let ended = timings[i].upload.end.saturating_duration_since(*now);
            let (forecast, ended) = (forecast.as_secs_f64(), ended.as_secs_f64());
            println!("job {i} {label}: forecast {forecast:.3} s, ended after {ended:.3} s");
            assert!(
                (forecast - ended).abs() <= FORECAST_SLACK,
                "job {i} {label}: forecast {forecast:.3} s, ended after {ended:.3} s"
            );
        }
    }
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn cancels_a_job_waiting_for_memory() {
    let s3 = S3::new("budget-cancel").await;
    let first = job(&s3.bucket, "first", 600, 16);
    let waiting = job(&s3.bucket, "waiting", 20_000, 32);
    let after = job(&s3.bucket, "after", 600, 16);
    for job in [&first, &waiting, &after] {
        s3.put_inputs(job, &values(job), Blob::F32, None).await;
    }
    let builder = FakeBuilder::new(Duration::from_secs(1));
    let pipeline = Pipeline::start(s3.store.clone(), builder.clone(), BUDGET);
    let first = pipeline.submit(first);
    // The large one waits for the first's memory, through its build, and
    // the small one behind it waits in the queue.
    let second = pipeline.submit(waiting.clone());
    let third = pipeline.submit(after);
    tokio::time::sleep(Duration::from_millis(300)).await;
    second.cancel();
    let ended = tokio::time::timeout(Duration::from_millis(300), second.outcome()).await;
    assert_eq!(ended, Ok(Err(JobError::Canceled)));
    let first = first.outcome().await.unwrap().timings;
    let third = third.outcome().await.unwrap().timings;
    // Once the large one is gone, the small one fits beside the first,
    // and downloads while the first still builds.
    assert!(third.download.start < first.build.end);
    // The large one was never built.
    assert_eq!(builder.builds(), 2);
    pipeline.close().await;
    assert_eq!(s3.get(&waiting.index_key()).await, None);
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn holds_three_jobs_at_most() {
    let s3 = S3::new("bound").await;
    // A large graph over few values: each index file is about 17 times
    // its vector blob, so uploads are the slow stage and every graph waits
    // for upload room.
    let jobs: Vec<Job> = (0..5)
        .map(|i| job(&s3.bucket, &format!("job{i}"), 200_000, 4))
        .collect();
    for job in &jobs {
        s3.put_inputs(job, &values(job), Blob::F32, None).await;
    }
    let pipeline = s3.pipeline(FakeBuilder::new(Duration::ZERO));
    let handles: Vec<_> = jobs
        .iter()
        .map(|job| pipeline.submit(job.clone()))
        .collect();
    let mut timings = Vec::new();
    for handle in handles {
        timings.push(handle.outcome().await.unwrap().timings);
    }
    pipeline.close().await;
    // One uploading, one built, one downloaded: a fourth starts
    // downloading only once the upload three jobs back has ended.
    for k in 0..2 {
        assert!(
            timings[k + 3].download.start >= timings[k].upload.end,
            "job {} started downloading while job {k} still uploaded",
            k + 3
        );
    }
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn fails_jobs_without_storing_anything() {
    let s3 = S3::new("fails").await;
    let missing = job(&s3.bucket, "missing", 600, 16);
    let mut short = job(&s3.bucket, "short", 600, 16);
    short.doc_count = 700;
    // A half_float blob of neither size, and no doc-id blob: the size is
    // the error, as the service reports it.
    let mut wrong_half = job(&s3.bucket, "half", 600, 16);
    wrong_half.data_type = DataType::HalfFloat;
    wrong_half.dimension = 12;
    s3.put(&wrong_half.vector_path, vec![0; 600 * 16 * 4], None)
        .await;
    let mut zero_m = job(&s3.bucket, "zero-m", 600, 16);
    zero_m.m = 0;
    let good = job(&s3.bucket, "good", 600, 16);
    // Blobs of 600 x 16 values for the rest.
    for job in [&short, &zero_m, &good] {
        let mut inputs = job.clone();
        inputs.doc_count = 600;
        inputs.dimension = 16;
        s3.put_inputs(&inputs, &values(&inputs), Blob::F32, None)
            .await;
    }

    let pipeline = s3.pipeline(FakeBuilder::new(Duration::ZERO));
    let outcome = |job: &Job| pipeline.submit(job.clone()).outcome();
    let error = outcome(&missing).await.unwrap_err();
    assert!(
        matches!(
            error,
            JobError::Download(knn_store::Error::S3 {
                call: "HeadObject",
                ..
            })
        ),
        "{error}"
    );
    let error = outcome(&short).await.unwrap_err();
    assert!(
        matches!(error, JobError::Download(knn_store::Error::Size { .. })),
        "{error}"
    );
    let error = outcome(&wrong_half).await.unwrap_err();
    assert!(matches!(error, JobError::HalfFloatSize { .. }), "{error}");
    let error = outcome(&zero_m).await.unwrap_err();
    assert!(matches!(error, JobError::Params(_)), "{error}");
    // The pipeline carries on.
    outcome(&good).await.unwrap();
    pipeline.close().await;

    // A failing build, and a graph the writer refuses.
    let failing = s3.pipeline(FakeBuilder::failing(Duration::ZERO, "out of device memory"));
    let error = failing.submit(good.clone()).outcome().await.unwrap_err();
    assert_eq!(error, JobError::Build("out of device memory".to_owned()));
    failing.close().await;
    let refused = s3.pipeline(FakeBuilder::naming_missing_vectors(Duration::ZERO));
    let mut bad_graph = good.clone();
    bad_graph.vector_path = "good/_1_vectors.knnvec".to_owned();
    s3.put_inputs(&bad_graph, &values(&bad_graph), Blob::F32, None)
        .await;
    let error = refused
        .submit(bad_graph.clone())
        .outcome()
        .await
        .unwrap_err();
    assert!(matches!(error, JobError::Write(_)), "{error}");
    refused.close().await;

    for job in [&missing, &short, &wrong_half, &zero_m, &bad_graph] {
        assert_eq!(s3.get(&job.index_key()).await, None, "{}", job.vector_path);
    }
    assert_eq!(s3.pending_uploads().await, 0);
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn cancels_jobs() {
    let s3 = S3::new("cancel").await;
    let building = job(&s3.bucket, "building", 600, 16);
    let waiting = job(&s3.bucket, "waiting", 600, 16);
    let queued = job(&s3.bucket, "queued", 600, 16);
    let after = job(&s3.bucket, "after", 600, 16);
    for job in [&building, &waiting, &queued, &after] {
        s3.put_inputs(job, &values(job), Blob::F32, None).await;
    }
    let builder = FakeBuilder::new(Duration::from_secs(1));
    let pipeline = s3.pipeline(builder.clone());
    let first = pipeline.submit(building.clone());
    let second = pipeline.submit(waiting.clone());
    let third = pipeline.submit(queued.clone());
    // Still queued, behind the second: withdrawn, it ends at once, long
    // before the first one's build frees a slot.
    third.cancel();
    let ended = tokio::time::timeout(Duration::from_millis(300), third.outcome()).await;
    assert_eq!(ended, Ok(Err(JobError::Canceled)));
    // Midway through the first one's build, which cannot stop, with the
    // second downloaded and waiting for the builder. The first is canceled
    // while another task awaits it.
    let canceler = first.canceler();
    let waiting_for_first = tokio::spawn(first.outcome());
    tokio::time::sleep(Duration::from_millis(300)).await;
    canceler.cancel();
    second.cancel();
    assert_eq!(waiting_for_first.await.unwrap(), Err(JobError::Canceled));
    assert_eq!(second.outcome().await, Err(JobError::Canceled));
    // Only the first was built, and thrown away.
    assert_eq!(builder.builds(), 1);
    // The pipeline carries on.
    pipeline.submit(after.clone()).outcome().await.unwrap();
    assert_eq!(builder.builds(), 2);
    pipeline.close().await;
    for job in [&building, &waiting, &queued] {
        assert_eq!(s3.get(&job.index_key()).await, None, "{}", job.vector_path);
    }
    assert_eq!(s3.pending_uploads().await, 0);
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn encrypts_the_index_as_its_inputs() {
    let s3 = S3::new("kms").await;
    let vectors_key = "arn:aws:kms:us-east-1:000000000000:key/0b5c8a3e-6d1f-4f7a-9e2b-3c4d5e6f7a8b";
    let ids_key = "arn:aws:kms:us-east-1:000000000000:key/9f8e7d6c-5b4a-4321-8fed-cba987654321";
    let pipeline = s3.pipeline(FakeBuilder::new(Duration::ZERO));

    // The vectors' key wins over the doc ids'.
    let both = job(&s3.bucket, "both", 600, 16);
    s3.put_inputs(&both, &values(&both), Blob::F32, Some(vectors_key))
        .await;
    s3.put(&both.doc_id_path, le_ids(&ids(600)), Some(ids_key))
        .await;
    pipeline.submit(both.clone()).outcome().await.unwrap();
    assert_eq!(
        s3.kms_key(&both.index_key()).await.as_deref(),
        Some(vectors_key)
    );

    // Without one on the vectors, the doc ids' key.
    let ids_only = job(&s3.bucket, "ids-only", 600, 16);
    s3.put_inputs(&ids_only, &values(&ids_only), Blob::F32, None)
        .await;
    s3.put(&ids_only.doc_id_path, le_ids(&ids(600)), Some(ids_key))
        .await;
    pipeline.submit(ids_only.clone()).outcome().await.unwrap();
    assert_eq!(
        s3.kms_key(&ids_only.index_key()).await.as_deref(),
        Some(ids_key)
    );
    pipeline.close().await;
    s3.remove().await;
}
