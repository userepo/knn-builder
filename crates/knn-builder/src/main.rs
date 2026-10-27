//! The service: the HTTP API OpenSearch k-NN calls to start a remote index
//! build and poll its status, the job store, and the scheduling of
//! downloads, GPU builds and uploads.
//!
//! Nothing is implemented yet; the binary only reports its version.

fn main() {
    println!("knn-builder {}", env!("CARGO_PKG_VERSION"));
}
