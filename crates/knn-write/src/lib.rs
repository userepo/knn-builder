//! Writes the Faiss index file the upstream remote-vector-index-builder
//! hands back to OpenSearch k-NN: an ID map over an `IndexHNSWCagra`,
//! from a CAGRA graph, the vectors and their doc ids, without Faiss. The
//! output matches the service's own file exactly (`docs/faiss-layout.md`
//! in the repository; the oracle in `tools/oracle` and the fixtures check
//! it).
//!
//! [`IndexFile::new`] checks an [`IndexSpec`] and knows the file's length
//! before [`IndexFile::write_to`] writes the first byte, so the file can
//! be streamed into an upload. Writing keeps no copy of the vectors or the
//! graph, and nothing else proportional to the vector count: the HNSW
//! level table Faiss draws for every node is drawn again for each array
//! that needs it ([`LevelTable`]).
//!
//! ```
//! use knn_write::{IndexFile, IndexSpec, Metric, Vectors};
//!
//! // Five 2-dimensional vectors, each linked to the other four.
//! let graph = [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3];
//! let vectors = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
//! let file = IndexFile::new(IndexSpec {
//!     dimension: 2,
//!     metric: Metric::L2,
//!     graph: &graph,
//!     graph_degree: 4,
//!     vectors: Vectors::F32(&vectors),
//!     ids: &[10, 20, 30, 40, 50],
//!     ef_construction: 100,
//!     ef_search: 100,
//!     skip_stored_vectors: false,
//! })?;
//! let mut out = Vec::with_capacity(usize::try_from(file.size())?);
//! file.write_to(&mut out)?;
//! assert_eq!(&out[..4], b"IxMp");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod error;
mod levels;
mod mt19937;
mod write;

pub use error::Error;
pub use levels::{LevelTable, Levels, NodeLevel};
pub use write::{IndexFile, IndexSpec, Metric, Vectors};
