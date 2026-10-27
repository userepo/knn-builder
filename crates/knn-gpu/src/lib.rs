//! Builds CAGRA graphs on the GPU through cuVS, with the parameter rules
//! of the upstream remote-vector-index-builder, and owns the per-GPU
//! resources the builds share.
//!
//! The cuVS code is behind the `gpu` feature, which needs Linux and the
//! conda environment in `environment.yml`. Nothing is implemented yet; the
//! `cagra_smoke` test checks that the toolchain builds and searches a
//! graph.
