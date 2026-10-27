//! Builds CAGRA graphs on the GPU through cuVS, with the parameter rules
//! of the upstream remote-vector-index-builder, and owns the per-GPU
//! resources the builds share.
//!
//! [`BuildParams`] (the service's rules, `docs/params.md`) and [`fp16`]
//! (its fp32 to fp16 conversion) build anywhere. The build itself, [`Gpu`],
//! is behind the `gpu` feature, which needs Linux, a C++20 compiler and
//! the conda environment in `environment.yml`.

mod error;
pub mod fp16;
#[cfg(feature = "gpu")]
mod gpu;
mod params;

pub use error::Error;
#[cfg(feature = "gpu")]
pub use gpu::{Gpu, Graph, HostVectors};
pub use params::{BuildParams, Metric};
