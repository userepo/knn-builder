//! The upload stage's work: the index file, written straight into its
//! upload.

use std::io;

use knn_store::{Cancel, Store};
use knn_write::{IndexFile, IndexSpec};

use crate::load::Dataset;
use crate::{Job, JobError};

/// Writes the index of `job` from its inputs and `graph` (`degree` ids per
/// vector) into an upload to the job's index key, encrypted as its
/// inputs are. The writer's checks run before the upload starts. Once the
/// file is written, `verdict` says whether to store it, and the upload
/// completes only then: a verdict against it, or any failure after the
/// upload starts, aborts the upload, so nothing is stored.
pub(crate) fn write_index(
    store: &Store,
    job: &Job,
    dataset: &Dataset,
    graph: &[u32],
    degree: usize,
    cancel: &Cancel,
    verdict: impl FnOnce() -> Result<(), JobError>,
) -> Result<(), JobError> {
    if cancel.is_canceled() {
        return Err(JobError::Canceled);
    }
    let file = IndexFile::new(IndexSpec {
        dimension: job.dimension as usize,
        metric: job.file_metric(),
        graph,
        graph_degree: degree,
        vectors: dataset.vectors.as_write(),
        ids: &dataset.ids,
        ef_construction: job.ef_construction,
        ef_search: job.ef_search,
        skip_stored_vectors: job.skip_stored_vectors,
    })
    .map_err(|e| JobError::Write(e.to_string()))?;
    let mut upload = store.upload(
        &job.bucket,
        &job.index_key(),
        file.size(),
        dataset.kms_key.as_deref(),
        cancel,
    );
    match file.write_to(&mut upload) {
        Ok(()) => match verdict() {
            Ok(()) => upload.finish().map_err(JobError::upload),
            Err(error) => {
                // The verdict is why the job failed, and an abort that
                // fails stores no file either: it is warned of, and the
                // verdict told.
                if let Err(abort) = upload.abort() {
                    tracing::warn!(
                        vector_path = job.vector_path,
                        error = %abort,
                        "the upload was not aborted"
                    );
                }
                Err(error)
            }
        },
        Err(error) => {
            // The upload's own failure, when that is what stopped the
            // writer; otherwise the writer's, once the upload is given up.
            let cause = upload_failure(&error);
            upload.abort().map_err(JobError::upload)?;
            Err(cause.map_or_else(|| JobError::Write(error.to_string()), JobError::upload))
        }
    }
}

/// The upload's error inside an error a write returned.
fn upload_failure(error: &io::Error) -> Option<knn_store::Error> {
    error.get_ref()?.downcast_ref::<knn_store::Error>().cloned()
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use aws_sdk_s3::Client;
    use knn_store::{Cancel, Settings, Store, Transfers};

    use super::write_index;
    use crate::job::tests::job;
    use crate::load::Dataset;
    use crate::{FakeBuilder, JobError, Vectors};

    /// Aborts the one multipart upload in `bucket` once it is there, as
    /// another client or a lifecycle rule might.
    async fn abort_elsewhere(client: &Client, bucket: &str) {
        let deadline = Instant::now() + Duration::from_secs(30);
        let upload = loop {
            let listed = client
                .list_multipart_uploads()
                .bucket(bucket)
                .send()
                .await
                .unwrap();
            if let [upload] = listed.uploads() {
                break upload.clone();
            }
            assert!(Instant::now() < deadline, "no multipart upload started");
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        client
            .abort_multipart_upload()
            .bucket(bucket)
            .key(upload.key().unwrap())
            .upload_id(upload.upload_id().unwrap())
            .send()
            .await
            .unwrap();
    }

    /// A verdict against the file is why the job failed, though the abort
    /// that follows fails too: it is not the upload's error.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs LocalStack"]
    async fn tells_the_verdict_when_the_abort_fails() {
        let settings = Settings::from_env();
        assert!(
            settings.endpoint_url.is_some(),
            "needs LocalStack: set S3_ENDPOINT_URL and credentials (see the README)"
        );
        let store = Store::new(&settings, Transfers::default()).await;
        let client = store.sdk_client().clone();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let bucket = format!("knn-builder-write-{nanos}");
        client.create_bucket().bucket(&bucket).send().await.unwrap();

        // A file of 12 MB, over one part of 8 MiB: the first part goes up
        // as the file is written, the rest waits for the verdict.
        let request = job(&bucket, 3_000, 1_024);
        let key = request.index_key();
        #[expect(clippy::cast_precision_loss, reason = "below 2^24")]
        let dataset = Dataset {
            vectors: Vectors::F32((0..3_000 * 1_024).map(|i| i as f32).collect()),
            ids: (0..3_000).collect(),
            kms_key: None,
        };
        let graph = FakeBuilder::graph(3_000, 32);
        let against = JobError::LowRecall {
            recall: 0.5,
            min: 0.9,
            queries: 100,
            k: 10,
        };
        let handle = tokio::runtime::Handle::current();
        let (elsewhere, verdict) = ((client.clone(), bucket.clone()), against.clone());
        let written = tokio::task::spawn_blocking(move || {
            write_index(
                &store,
                &request,
                &dataset,
                &graph,
                32,
                &Cancel::new(),
                || {
                    // The upload's own abort, after the verdict, finds it gone.
                    handle.block_on(abort_elsewhere(&elsewhere.0, &elsewhere.1));
                    Err(verdict)
                },
            )
        })
        .await
        .unwrap();
        assert_eq!(written, Err(against));

        let stored = client.get_object().bucket(&bucket).key(&key).send().await;
        assert!(stored.is_err(), "nothing is stored");
        let listed = client
            .list_multipart_uploads()
            .bucket(&bucket)
            .send()
            .await
            .unwrap();
        assert!(listed.uploads().is_empty(), "{listed:?}");
        client.delete_bucket().bucket(&bucket).send().await.unwrap();
    }
}
