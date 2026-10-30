//! The upload stage's work: the index file, written straight into its
//! upload.

use std::io;

use knn_store::{Cancel, Store};
use knn_write::{IndexFile, IndexSpec};

use crate::load::Dataset;
use crate::{Job, JobError};

/// Writes the index of `job` from its inputs and `graph` (`degree` ids per
/// vector) into an upload to the job's index key, encrypted as its
/// inputs are. The writer's checks run before the upload starts; any
/// failure after aborts the upload, so nothing is stored.
pub(crate) fn write_index(
    store: &Store,
    job: &Job,
    dataset: &Dataset,
    graph: &[u32],
    degree: usize,
    cancel: &Cancel,
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
        Ok(()) => upload.finish().map_err(JobError::upload),
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
