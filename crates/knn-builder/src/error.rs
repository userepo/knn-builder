use thiserror::Error;

/// Why a job failed: the message the status response carries.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
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
        /// The build's modeled peak, with a margin, in MiB.
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
        /// `download`, `build` or `upload`.
        stage: &'static str,
    },
}

impl JobError {
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
