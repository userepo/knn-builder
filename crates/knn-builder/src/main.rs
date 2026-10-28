//! The service: the HTTP API OpenSearch k-NN calls to start a remote index
//! build and poll its status, and the job store, over the library's
//! pipeline.
//!
//! The HTTP API is not implemented yet; the binary only reports its
//! version.

fn main() {
    println!("knn-builder {}", env!("CARGO_PKG_VERSION"));
}
