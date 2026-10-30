//! What can go wrong in a build.

/// A build that cannot run or did not finish.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A request with a vector count, dimension or m of 0.
    #[error("doc_count {doc_count}, dimension {dimension}, m {m}: each must be at least 1")]
    Request {
        /// The vector count.
        doc_count: u32,
        /// The dimension.
        dimension: u32,
        /// The requested m.
        m: u32,
    },
    /// A binary dimension that is not whole bytes: Faiss packs 8 bits to a
    /// byte and refuses any other count.
    #[error("a binary dimension of {dimension} bits is not a multiple of 8")]
    Bits {
        /// The dimension, in bits.
        dimension: usize,
    },
    /// Vector values that are not whole rows of the dimension, or none.
    #[error("{values} vector values do not make rows of dimension {dimension}")]
    Vectors {
        /// Values given.
        values: usize,
        /// The dimension.
        dimension: usize,
    },
    /// cuVS, or the CUDA runtime, reported an error.
    #[error("{call} failed: {message}")]
    Cuvs {
        /// The step: `raft::resources`, `make_device_padded_dataset`,
        /// `cagra::build` or `raft::copy`; `knn_cagra_build` or
        /// `knn_cagra_build_binary` for arguments the shim refused,
        /// `the cuVS shim` for a status it did not document;
        /// `cudaMemGetInfo`, `cudaMalloc`, `cudaGetDevice` or
        /// `cudaDeviceGetAttribute` for the CUDA runtime's calls.
        call: &'static str,
        /// Its message.
        message: String,
    },
    /// A graph too large for host memory: rows x degree ids that overflow
    /// `usize` or cannot be allocated.
    #[error("a graph of {rows} x {degree} neighbor ids does not fit in host memory")]
    HostMemory {
        /// Vectors.
        rows: usize,
        /// Neighbors per vector.
        degree: usize,
    },
    /// The graph cuVS returned is not the n x degree array of `uint32` asked
    /// for.
    #[error("cuVS returned a graph of an unexpected shape or type: {detail}")]
    Graph {
        /// What did not match.
        detail: String,
    },
}
