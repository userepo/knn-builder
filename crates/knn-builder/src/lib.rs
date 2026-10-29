//! The service's library half: the pipeline the HTTP API feeds. A
//! [`Job`] is downloaded from S3 into host memory, its CAGRA graph built
//! by a [`Builder`] (the GPU, or [`FakeBuilder`] in tests), and its index
//! file written straight into an upload, with no temporary file; a
//! [`Pipeline`] overlaps those three stages across jobs. [`request`]
//! reads a build request as the upstream service does, [`service`]
//! holds the jobs and feeds them to a pipeline, and [`api`] answers HTTP
//! requests with them, set up by [`config`].

pub mod api;
mod builder;
pub mod config;
pub mod cost;
mod error;
pub mod forecast;
mod gate;
mod job;
mod load;
mod pipeline;
pub mod request;
pub mod service;
mod write;

pub use builder::{BuildError, Builder, FakeBuilder, Vectors};
pub use error::JobError;
pub use job::{DataType, Job, SpaceType};
pub use pipeline::{Built, Canceler, Flights, InFlight, JobHandle, Pipeline, Span, Stage, Timings};
