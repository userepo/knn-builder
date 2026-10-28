//! The service's library half: the pipeline the HTTP API feeds. A
//! [`Job`] is downloaded from S3 into host memory, its CAGRA graph built
//! by a [`Builder`] (the GPU, or [`FakeBuilder`] in tests), and its index
//! file written straight into an upload, with no temporary file; a
//! [`Pipeline`] overlaps those three stages across jobs.

mod builder;
mod error;
mod job;
mod load;
mod pipeline;
mod write;

pub use builder::{BuildError, Builder, FakeBuilder, Vectors};
pub use error::JobError;
pub use job::{DataType, Job, SpaceType};
pub use pipeline::{Built, Canceler, JobHandle, Pipeline, Span, Timings};
