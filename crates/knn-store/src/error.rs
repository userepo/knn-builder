use thiserror::Error;

/// A transfer that failed. Each names the object, as `s3://bucket/key`.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// An S3 call failed, after the SDK's retries.
    #[error("{call} s3://{bucket}/{key}: {message}")]
    S3 {
        /// The call: `HeadObject`, `GetObject`, `PutObject`,
        /// `CreateMultipartUpload`, `UploadPart` or
        /// `CompleteMultipartUpload`.
        call: &'static str,
        /// The bucket.
        bucket: String,
        /// The key.
        key: String,
        /// The SDK's error, with its causes.
        message: String,
    },
    /// A blob of another size than the values asked for take.
    #[error("s3://{bucket}/{key} holds {actual} bytes, where {expected} were expected")]
    Size {
        /// The bucket.
        bucket: String,
        /// The key.
        key: String,
        /// The bytes the values take.
        expected: u64,
        /// The blob's size.
        actual: u64,
    },
    /// A GET that returned another number of bytes than it asked for.
    #[error(
        "GetObject s3://{bucket}/{key} returned {received} bytes for the {expected} at offset {offset}"
    )]
    Range {
        /// The bucket.
        bucket: String,
        /// The key.
        key: String,
        /// The first byte asked for.
        offset: u64,
        /// The bytes asked for.
        expected: u64,
        /// The bytes received, counted to the end of the frame that went
        /// past `expected` when one did.
        received: u64,
    },
    /// A download or upload stopped through its [`Cancel`](crate::Cancel);
    /// an upload so stopped stored nothing.
    #[error("the transfer of s3://{bucket}/{key} was canceled")]
    Canceled {
        /// The bucket.
        bucket: String,
        /// The key.
        key: String,
    },
    /// An upload whose writer stopped before the size it declared, or
    /// went past it. Nothing was stored.
    #[error("the upload of s3://{bucket}/{key} got {written} of its {size} bytes")]
    Incomplete {
        /// The bucket.
        bucket: String,
        /// The key.
        key: String,
        /// The bytes written.
        written: u64,
        /// The size declared.
        size: u64,
    },
    /// A multipart upload that failed and could not be aborted either: its
    /// parts stay in the bucket, and are billed, until a lifecycle rule
    /// removes them.
    #[error("{cause}; AbortMultipartUpload failed too: {message}")]
    NotAborted {
        /// Why the upload failed.
        cause: Box<Error>,
        /// Why the abort failed.
        message: String,
    },
    /// An upload whose task ended without a result, as when its runtime
    /// shuts down.
    #[error("the upload of s3://{bucket}/{key} stopped without a result")]
    Stopped {
        /// The bucket.
        bucket: String,
        /// The key.
        key: String,
    },
}
