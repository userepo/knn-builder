//! What builds the graph: the GPU through cuVS, or a stand-in for tests.

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use knn_gpu::BuildParams;

/// A job's vectors in host memory, row-major, as downloaded.
#[derive(Clone, PartialEq)]
pub enum Vectors {
    /// fp32 values.
    F32(Vec<f32>),
    /// IEEE half-precision bit patterns.
    F16(Vec<u16>),
    /// int8 values.
    I8(Vec<i8>),
}

impl Vectors {
    /// Values, all vectors together.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::I8(v) => v.len(),
        }
    }

    /// Whether there are no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Frees the values, keeping the type: all `knn-write` needs of
    /// vectors it does not store.
    pub fn release_values(&mut self) {
        match self {
            Self::F32(v) => *v = Vec::new(),
            Self::F16(v) => *v = Vec::new(),
            Self::I8(v) => *v = Vec::new(),
        }
    }

    /// The vectors as `knn-write` takes them.
    #[must_use]
    pub fn as_write(&self) -> knn_write::Vectors<'_> {
        match self {
            Self::F32(v) => knn_write::Vectors::F32(v),
            Self::F16(v) => knn_write::Vectors::F16(v),
            Self::I8(v) => knn_write::Vectors::I8(v),
        }
    }
}

/// The type and count only: a job's vectors run to millions of values.
impl fmt::Debug for Vectors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32(v) => write!(f, "F32({} values)", v.len()),
            Self::F16(v) => write!(f, "F16({} values)", v.len()),
            Self::I8(v) => write!(f, "I8({} values)", v.len()),
        }
    }
}

/// Why a build failed, as its builder reports it.
pub type BuildError = Box<dyn StdError + Send + Sync>;

/// Builds a job's CAGRA graph. The pipeline gives its builder a thread of
/// its own and calls it for one job at a time.
pub trait Builder: Send + 'static {
    /// The graph of `vectors`, rows of `dimension`: `params.graph_degree`
    /// neighbor ids per vector, row-major, row i for the i-th vector.
    ///
    /// # Errors
    ///
    /// Whatever stopped the build.
    fn build(
        &mut self,
        vectors: &Vectors,
        dimension: usize,
        params: &BuildParams,
    ) -> Result<Vec<u32>, BuildError>;

    /// The device memory free for the next build, in bytes, when the
    /// builder can tell; the pipeline fails a job whose build needs more
    /// at once. `None` by default: no check.
    fn free_device_memory(&self) -> Option<u64> {
        None
    }
}

/// The build itself: cuVS on the GPU, through `knn-gpu`.
#[cfg(feature = "gpu")]
impl Builder for knn_gpu::Gpu {
    fn free_device_memory(&self) -> Option<u64> {
        // A reading that fails leaves the build to fail on its own.
        knn_gpu::device_memory()
            .ok()
            .and_then(|memory| u64::try_from(memory.free).ok())
    }

    fn build(
        &mut self,
        vectors: &Vectors,
        dimension: usize,
        params: &BuildParams,
    ) -> Result<Vec<u32>, BuildError> {
        let host = match vectors {
            Vectors::F32(v) => knn_gpu::HostVectors::F32(v),
            Vectors::F16(v) => knn_gpu::HostVectors::F16(v),
            Vectors::I8(v) => knn_gpu::HostVectors::I8(v),
        };
        let graph = knn_gpu::Gpu::build(self, host, dimension, params)?;
        Ok(graph.into_neighbors())
    }
}

/// A stand-in for the GPU build, for tests of everything around it: it
/// takes a fixed time, then returns [`FakeBuilder::graph`], or fails as
/// it was made to. It counts its builds; clones share the count.
#[derive(Debug, Clone)]
pub struct FakeBuilder {
    delay: Duration,
    outcome: Fake,
    builds: Arc<AtomicUsize>,
    free_device_memory: Option<u64>,
}

#[derive(Debug, Clone)]
enum Fake {
    Graph,
    Fails(String),
    NamesMissingVectors,
}

impl FakeBuilder {
    /// Builds [`FakeBuilder::graph`] in `delay`.
    #[must_use]
    pub fn new(delay: Duration) -> Self {
        Self {
            delay,
            outcome: Fake::Graph,
            builds: Arc::default(),
            free_device_memory: None,
        }
    }

    /// The same builder, reporting `bytes` of device memory free before
    /// each build.
    #[must_use]
    pub fn with_free_device_memory(self, bytes: u64) -> Self {
        Self {
            free_device_memory: Some(bytes),
            ..self
        }
    }

    /// Fails every build after `delay`, with `message`.
    #[must_use]
    pub fn failing(delay: Duration, message: &str) -> Self {
        Self {
            delay,
            outcome: Fake::Fails(message.to_owned()),
            builds: Arc::default(),
            free_device_memory: None,
        }
    }

    /// Returns graphs whose first id names no vector, which `knn-write`
    /// refuses.
    #[must_use]
    pub fn naming_missing_vectors(delay: Duration) -> Self {
        Self {
            delay,
            outcome: Fake::NamesMissingVectors,
            builds: Arc::default(),
            free_device_memory: None,
        }
    }

    /// Builds started, by this builder and its clones.
    #[must_use]
    pub fn builds(&self) -> usize {
        self.builds.load(Ordering::SeqCst)
    }

    /// The fake's graph of `rows` vectors: each one linked to the next
    /// `degree`, wrapping around to the first.
    #[must_use]
    pub fn graph(rows: usize, degree: usize) -> Vec<u32> {
        (0..rows)
            .flat_map(|row| (1..=degree).map(move |k| (row + k) % rows))
            .map(|id| u32::try_from(id).unwrap_or(u32::MAX))
            .collect()
    }
}

impl Builder for FakeBuilder {
    fn free_device_memory(&self) -> Option<u64> {
        self.free_device_memory
    }

    fn build(
        &mut self,
        vectors: &Vectors,
        dimension: usize,
        params: &BuildParams,
    ) -> Result<Vec<u32>, BuildError> {
        self.builds.fetch_add(1, Ordering::SeqCst);
        thread::sleep(self.delay);
        let rows = vectors.len().checked_div(dimension).unwrap_or(0);
        let degree = params.graph_degree as usize;
        match &self.outcome {
            Fake::Graph => Ok(Self::graph(rows, degree)),
            Fake::Fails(message) => Err(message.clone().into()),
            Fake::NamesMissingVectors => {
                let mut graph = Self::graph(rows, degree);
                if let Some(first) = graph.first_mut() {
                    *first = u32::try_from(rows).unwrap_or(u32::MAX);
                }
                Ok(graph)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FakeBuilder;

    #[test]
    fn links_each_vector_to_the_next() {
        assert_eq!(FakeBuilder::graph(4, 2), [1, 2, 2, 3, 3, 0, 0, 1]);
        assert!(FakeBuilder::graph(10, 9).iter().all(|&id| id < 10));
    }
}
