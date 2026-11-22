//! The service's library half: the pipeline the HTTP API feeds. A
//! [`Job`] is downloaded from S3 into host memory, its CAGRA graph built
//! by a [`Builder`] (the GPU, or [`FakeBuilder`] in tests), its graph's
//! clusters linked when it asks (`project_link_to_base_layer`), and its
//! index file written straight into an upload, with no temporary file; a
//! [`Pipeline`] overlaps those three stages across jobs, the projection
//! running in the upload stage, and the [`recall`] check beside the write
//! when the service sets one. [`request`]
//! reads a build request as the upstream service does, [`service`]
//! holds the jobs and feeds them to a pipeline, and [`api`] answers HTTP
//! requests with them, set up by [`config`]. [`metrics`] holds the values
//! of the service's metrics and writes them as the page a Prometheus
//! scraper reads, with the host's and the GPU's memory as [`memory`]
//! reads it.

pub mod api;
mod builder;
pub mod config;
pub mod cost;
mod error;
pub mod forecast;
mod gate;
mod job;
mod load;
pub mod memory;
pub mod metrics;
mod pipeline;
mod project;
pub mod recall;
pub mod request;
pub mod service;
mod write;

pub use builder::{BuildError, Builder, FakeBuilder, Vectors};
pub use error::JobError;
pub use job::{DataType, GraphParams, Job, SpaceType};
pub use pipeline::{Built, Canceler, Flights, InFlight, JobHandle, Pipeline, Span, Stage, Timings};
pub use recall::{Checked, RecallCheck};
