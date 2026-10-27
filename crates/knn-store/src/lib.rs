//! Moves a build's data through S3: parallel ranged downloads of the
//! vector and doc-id blobs, straight into the buffers the build reads,
//! and a multipart upload of the index file as it is written, with no
//! temporary file.
//!
//! [`Store`] holds the client, set up from the upstream service's
//! environment variables ([`Settings`]). [`Store::head`] reads a blob's
//! size, [`Store::download`] decodes it into a slice on threads of its
//! own, and [`Store::upload`] returns an [`Upload`], the
//! [`std::io::Write`] that `knn-write` writes the index file to.

mod download;
mod error;
mod store;
#[cfg(test)]
mod testing;
mod upload;

pub use download::Cancel;
pub use error::Error;
pub use store::{Blob, Settings, Store, Transfers};
pub use upload::Upload;
