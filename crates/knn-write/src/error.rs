//! Why an [`IndexSpec`](crate::IndexSpec) cannot be written.

/// An [`IndexSpec`](crate::IndexSpec) the writer refuses.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No ids, so no vectors.
    #[error("no vectors to write")]
    Empty,
    /// More vectors than the file's 32-bit neighbor ids can name.
    #[error("{count} vectors: the file's 32-bit neighbor ids name at most 2147483647")]
    TooManyVectors {
        /// The vector count, from the ids.
        count: usize,
    },
    /// A dimension of 0, or past Faiss's 32-bit `d`.
    #[error("dimension {dimension}: must be from 1 to 2147483647")]
    Dimension {
        /// The dimension given.
        dimension: usize,
    },
    /// A graph degree the HNSW layout cannot hold: odd, below 4 (Faiss's
    /// level table needs M = degree / 2 of at least 2), or so large that
    /// its cumulative neighbor counts overflow `i32`.
    #[error(
        "graph degree {degree}: must be even, at least 4, and small enough that its cumulative neighbor counts fit a 32-bit integer"
    )]
    GraphDegree {
        /// The degree given.
        degree: usize,
    },
    /// A graph whose length is not one row of `degree` per vector.
    #[error(
        "the graph holds {actual} entries, where {count} vectors of degree {degree} need {expected}"
    )]
    GraphLength {
        /// Entries given.
        actual: usize,
        /// Entries needed.
        expected: usize,
        /// The vector count.
        count: usize,
        /// The degree.
        degree: usize,
    },
    /// A neighbor id that names no vector.
    ///
    /// The upstream service writes such an id rather than refusing it:
    /// `GpuIndexCagra::copyTo` stores each `uint32` id from cuVS in Faiss's
    /// 32-bit signed neighbor slots, so an id from the vector count up to
    /// `i32::MAX` lands in the file as an invalid neighbor, and a larger
    /// one wraps to a negative value (`u32::MAX` to -1, Faiss's empty
    /// slot). Refusing it fails the build instead, and k-NN builds the
    /// segment on CPU.
    #[error("node {node} lists neighbor {neighbor}, but there are {count} vectors")]
    NeighborOutOfRange {
        /// The node whose row holds it.
        node: usize,
        /// The id.
        neighbor: u32,
        /// The vector count.
        count: usize,
    },
    /// Vector values that are not one row of `dimension` per id.
    #[error(
        "{actual} vector values, where {count} vectors of dimension {dimension} need {expected}"
    )]
    VectorsLength {
        /// Values given.
        actual: usize,
        /// Values needed.
        expected: usize,
        /// The vector count.
        count: usize,
        /// The dimension.
        dimension: usize,
    },
    /// A count past what the platform can hold: the file's length past
    /// `u64`, or the graph's or vectors' expected length past `usize`
    /// (possible only on a 32-bit target).
    #[error("the file, or the graph or vectors it holds, is too large to count")]
    TooLarge,
}
