use thiserror::Error;

use crate::metrics::Outcome;

/// Why a job failed: the message the status response carries.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum JobError {
    /// A blob could not be read, or holds another size than the request
    /// implies.
    #[error("downloading the inputs: {0}")]
    Download(knn_store::Error),
    /// A `half_float` vector blob of neither the fp16 size nor the fp32
    /// size; the service's message, word for word.
    #[error(
        "half_float vector blob of {size} bytes matches neither fp16 ({fp16}) nor fp32 ({fp32}) for doc_count={doc_count}, dimension={dimension}"
    )]
    HalfFloatSize {
        /// The blob's size.
        size: u64,
        /// The size of fp16 values.
        fp16: u64,
        /// The size of fp32 values.
        fp32: u64,
        /// The request's `doc_count`.
        doc_count: u32,
        /// The request's `dimension`.
        dimension: u32,
    },
    /// Inputs too large to allocate in host memory.
    #[error("{values} values of the job's inputs do not fit in host memory")]
    HostMemory {
        /// Values of the buffer that could not be allocated.
        values: u64,
    },
    /// A request parameter the service accepts but no build can take:
    /// an m below 0, say. The service fails such a job too.
    #[error("{name} {value} is outside {min} to {max}")]
    OutOfRange {
        /// The request's name for the parameter.
        name: &'static str,
        /// The request's value.
        value: i128,
        /// The least value a build takes.
        min: i128,
        /// The greatest value a build takes.
        max: i128,
    },
    /// The build parameters could not be derived from the request.
    #[error("deriving the build parameters: {0}")]
    Params(knn_gpu::Error),
    /// Too little device memory is free for the build, which would fail
    /// out of it partway: another process holds the rest.
    #[error(
        "building the graph: it needs about {needed_mib} MiB of device memory, {free_mib} MiB are free"
    )]
    DeviceMemory {
        /// The build's modeled peak, with a margin and what a process's
        /// first build takes besides, in MiB.
        needed_mib: u64,
        /// The device's free memory before the build, in MiB.
        free_mib: u64,
    },
    /// The builder failed.
    #[error("building the graph: {0}")]
    Build(String),
    /// The writer refused the graph or the inputs, or failed on its own.
    #[error("writing the index: {0}")]
    Write(String),
    /// Storing the index failed. Nothing was stored.
    #[error("uploading the index: {0}")]
    Upload(knn_store::Error),
    /// The recall check found the graph's recall below the least the
    /// service takes (`RECALL_CHECK_MIN`). Nothing was stored: the data
    /// node builds the segment itself.
    #[error(
        "the recall check failed: recall@{k} over {queries} sampled vectors is {recall:.4}, below the least of {min}"
    )]
    LowRecall {
        /// The recall found.
        recall: f64,
        /// The least taken.
        min: f64,
        /// Queries sampled.
        queries: usize,
        /// Neighbors each asked for.
        k: usize,
    },
    /// The job was canceled. Nothing was stored.
    #[error("the job was canceled")]
    Canceled,
    /// The forecast, corrected by the jobs that ended since this one was
    /// taken, has it seen ended only past the data node's timeout, so it
    /// was canceled before its build: the data node builds the segment
    /// itself at its next poll rather than after its timeout. Nothing was
    /// stored.
    #[error(
        "the job was dropped: forecast again as jobs ended, it would end past the data node's timeout of {seconds} s"
    )]
    WouldMissDeadline {
        /// The data node's timeout, in seconds.
        seconds: u64,
    },
    /// The data node stopped waiting for the job, and it was canceled.
    /// Nothing was stored.
    #[error("the data node's timeout of {seconds} s passed, so the job was dropped")]
    PastDeadline {
        /// The data node's timeout, in seconds.
        seconds: u64,
    },
    /// The pipeline stopped before the job ended.
    #[error("the pipeline stopped before the job ended")]
    Stopped,
    /// A bug: the job's work panicked in this stage. The job fails; the
    /// pipeline carries on with the next.
    #[error("the {stage} stage panicked")]
    Panicked {
        /// `download`, `build` or `upload`, the recall check's work
        /// counting as the upload's.
        stage: &'static str,
    },
}

impl JobError {
    /// How a job that failed so ended, as the metrics count it.
    pub(crate) fn outcome(&self) -> Outcome {
        match self {
            Self::Download(_) | Self::HalfFloatSize { .. } => Outcome::DownloadFailed,
            Self::HostMemory { .. } => Outcome::HostMemory,
            Self::OutOfRange { .. } | Self::Params(_) => Outcome::BadParameters,
            Self::DeviceMemory { .. } => Outcome::DeviceMemory,
            Self::Build(_) => Outcome::BuildFailed,
            Self::Write(_) => Outcome::WriteFailed,
            Self::Upload(_) => Outcome::UploadFailed,
            Self::LowRecall { .. } => Outcome::LowRecall,
            Self::Canceled | Self::Stopped => Outcome::Canceled,
            Self::WouldMissDeadline { .. } => Outcome::ForecastLate,
            Self::PastDeadline { .. } => Outcome::TimedOut,
            Self::Panicked { .. } => Outcome::Panicked,
        }
    }

    /// A download error; a canceled download as [`JobError::Canceled`].
    pub(crate) fn download(error: knn_store::Error) -> Self {
        match error {
            knn_store::Error::Canceled { .. } => Self::Canceled,
            error => Self::Download(error),
        }
    }

    /// An upload error; a canceled upload as [`JobError::Canceled`].
    pub(crate) fn upload(error: knn_store::Error) -> Self {
        match error {
            knn_store::Error::Canceled { .. } => Self::Canceled,
            error => Self::Upload(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::JobError;
    use crate::metrics::Outcome;

    #[test]
    fn labels_each_failure() {
        let object = || ("bucket".to_owned(), "a/_0_f.knnvec".to_owned());
        let s3 = || {
            let (bucket, key) = object();
            knn_store::Error::S3 {
                call: "GetObject",
                bucket,
                key,
                message: "refused".into(),
            }
        };
        let cases = [
            (JobError::Download(s3()), Outcome::DownloadFailed),
            (
                JobError::HalfFloatSize {
                    size: 3,
                    fp16: 4,
                    fp32: 8,
                    doc_count: 1,
                    dimension: 2,
                },
                Outcome::DownloadFailed,
            ),
            (JobError::HostMemory { values: 1 }, Outcome::HostMemory),
            (
                JobError::OutOfRange {
                    name: "m",
                    value: -1,
                    min: 0,
                    max: 1,
                },
                Outcome::BadParameters,
            ),
            (
                JobError::Params(knn_gpu::Error::Request {
                    doc_count: 0,
                    dimension: 8,
                    m: 16,
                }),
                Outcome::BadParameters,
            ),
            (
                JobError::DeviceMemory {
                    needed_mib: 2,
                    free_mib: 1,
                },
                Outcome::DeviceMemory,
            ),
            (JobError::Build("no GPU".into()), Outcome::BuildFailed),
            (JobError::Write("refused".into()), Outcome::WriteFailed),
            (JobError::Upload(s3()), Outcome::UploadFailed),
            (
                JobError::LowRecall {
                    recall: 0.5,
                    min: 0.9,
                    queries: 10,
                    k: 10,
                },
                Outcome::LowRecall,
            ),
            (JobError::Canceled, Outcome::Canceled),
            (JobError::Stopped, Outcome::Canceled),
            (
                JobError::WouldMissDeadline { seconds: 4 },
                Outcome::ForecastLate,
            ),
            (JobError::PastDeadline { seconds: 4 }, Outcome::TimedOut),
            (JobError::Panicked { stage: "build" }, Outcome::Panicked),
        ];
        for (error, outcome) in cases {
            assert_eq!(error.outcome(), outcome, "{error}");
        }
        // A canceled transfer is a cancel, whichever way it went.
        let (bucket, key) = object();
        let canceled = knn_store::Error::Canceled { bucket, key };
        assert_eq!(
            JobError::upload(canceled.clone()).outcome(),
            Outcome::Canceled
        );
        assert_eq!(JobError::download(canceled).outcome(), Outcome::Canceled);
    }
}
