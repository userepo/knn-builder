//! A build job, as the service's request describes it.

use std::num::NonZeroUsize;

use knn_write::projection;

use crate::RecallCheck;

/// How k-NN encoded the vectors: the request's `data_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// `float`: fp32 values.
    Float,
    /// `half_float`: fp16 values from a `half_float` field, or fp32 values
    /// from a float field with the fp16 scalar quantizer. The blob's size
    /// tells them apart, and fp32 values become fp16 on download, as the
    /// service converts them.
    HalfFloat,
    /// `byte`: int8 values.
    Byte,
    /// `binary`: packed bits, the dimension / 8 bytes of each vector, built
    /// through NN-descent in Hamming distance whatever the space type.
    Binary,
}

/// The request's `space_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpaceType {
    /// `l2`.
    L2,
    /// `innerproduct`.
    InnerProduct,
    /// `hamming`, binary vectors' space. Other vectors build in L2 under
    /// it, as the service's `configure_metric` falls back.
    Hamming,
}

impl SpaceType {
    /// The metric an IVF-PQ graph is built for.
    #[must_use]
    pub fn build_metric(self) -> knn_gpu::Metric {
        match self {
            Self::L2 | Self::Hamming => knn_gpu::Metric::L2,
            Self::InnerProduct => knn_gpu::Metric::InnerProduct,
        }
    }
}

/// What a job's graph is built with: IVF-PQ for float, fp16 and byte
/// vectors, NN-descent for binary ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphParams {
    /// An IVF-PQ graph build.
    IvfPq(knn_gpu::BuildParams),
    /// An NN-descent graph build in Hamming distance.
    Binary(knn_gpu::BinaryParams),
}

impl GraphParams {
    /// The neighbors each of `doc_count` vectors gets: the degree asked
    /// for, which for a binary build cuVS lowers to one less than the
    /// vector count when that is fewer.
    #[must_use]
    pub fn degree_for(&self, doc_count: u32) -> u32 {
        match self {
            Self::IvfPq(params) => params.graph_degree,
            Self::Binary(params) => params.graph_degree.min(doc_count.saturating_sub(1).max(1)),
        }
    }
}

/// One index to build: where its inputs are and what to build from them.
/// The service's request, validated: [`BuildRequest::job`](crate::request::BuildRequest::job),
/// with the recall check the service runs on it.
#[derive(Debug, Clone, PartialEq)]
pub struct Job {
    /// `container_name`: the bucket the blobs are in, and the index goes.
    pub bucket: String,
    /// `vector_path`: the vector blob's key, ending in `.knnvec`.
    pub vector_path: String,
    /// `doc_id_path`: the doc-id blob's key.
    pub doc_id_path: String,
    /// Values per vector.
    pub dimension: u32,
    /// Vectors, and doc ids.
    pub doc_count: u32,
    /// How the vector blob encodes the values.
    pub data_type: DataType,
    /// The distance the graph is built for.
    pub space_type: SpaceType,
    /// The HNSW `m`; the graph degree is m x 4.
    pub m: u32,
    /// Stored in the index file.
    pub ef_construction: i32,
    /// Stored in the index file.
    pub ef_search: i32,
    /// Leave the vectors out of the index file.
    pub skip_stored_vectors: bool,
    /// Link the graph's clusters through an HNSW layer 1 before the write
    /// (`project_link_to_base_layer`). Never for binary vectors, whose
    /// requests the service builds without it.
    pub project_link_to_base_layer: bool,
    /// The recall check run on its graph before its index is stored: the
    /// service's ([`Settings::recall_check`](crate::service::Settings::recall_check)),
    /// off for a job from [`BuildRequest::job`](crate::request::BuildRequest::job).
    pub recall_check: RecallCheck,
}

impl Job {
    /// The parameters the job's graph is built with, as the service derives
    /// them from its request.
    ///
    /// # Errors
    ///
    /// Returns knn-gpu's error for a count, dimension or m of 0, and for
    /// binary vectors a dimension that is not whole bytes.
    pub fn graph_params(&self) -> Result<GraphParams, knn_gpu::Error> {
        match self.data_type {
            DataType::Binary => knn_gpu::BinaryParams::new(self.doc_count, self.dimension, self.m)
                .map(GraphParams::Binary),
            DataType::Float | DataType::HalfFloat | DataType::Byte => knn_gpu::BuildParams::new(
                self.doc_count,
                self.dimension,
                self.m,
                self.space_type.build_metric(),
            )
            .map(GraphParams::IvfPq),
        }
    }

    /// The metric the index file records: Hamming for binary vectors, and
    /// otherwise the one the graph is built for.
    #[must_use]
    pub fn file_metric(&self) -> knn_write::Metric {
        match (self.data_type, self.space_type.build_metric()) {
            (DataType::Binary, _) => knn_write::Metric::Hamming,
            (_, knn_gpu::Metric::L2) => knn_write::Metric::L2,
            (_, knn_gpu::Metric::InnerProduct) => knn_write::Metric::InnerProduct,
        }
    }

    /// How the job's graph is projected, on `threads`: as the service
    /// projects it, with layer 1 of the m the graph was built with (a
    /// quarter of its degree, which may be lowered from the request's) and
    /// the request's efConstruction, in the index's metric. `None` when the
    /// job does not ask, or its parameters fail and it fails before a
    /// graph.
    pub(crate) fn projection(&self, threads: NonZeroUsize) -> Option<projection::Settings> {
        if !self.project_link_to_base_layer {
            return None;
        }
        let GraphParams::IvfPq(params) = self.graph_params().ok()? else {
            return None;
        };
        Some(projection::Settings {
            m: params.effective_m,
            // Faiss's int efConstruction meets the search's size_t heap
            // sizes as a size_t: one below 0 bounds no search.
            ef_construction: usize::try_from(self.ef_construction).unwrap_or(usize::MAX),
            metric: self.file_metric(),
            insertion: projection::Insertion::Batched { threads },
            ..projection::Settings::default()
        })
    }

    /// The neighbors each vector gets, or `None` when the parameters fail
    /// and the job fails before a graph.
    pub(crate) fn graph_degree(&self) -> Option<u32> {
        self.graph_params()
            .ok()
            .map(|params| params.degree_for(self.doc_count))
    }

    /// The bytes of one vector in its blob, as downloaded: fp16 for
    /// `half_float`, whatever the blob holds.
    pub(crate) fn vector_bytes(&self) -> u64 {
        let dimension = u64::from(self.dimension);
        match self.data_type {
            DataType::Float => dimension * 4,
            DataType::HalfFloat => dimension * 2,
            DataType::Byte => dimension,
            DataType::Binary => dimension / 8,
        }
    }

    /// The index file's key: the vector path with its last extension
    /// replaced by `.faiss`, next to the vectors, as the service names it.
    #[must_use]
    pub fn index_key(&self) -> String {
        let stem = self
            .vector_path
            .rsplit_once('.')
            .map_or(self.vector_path.as_str(), |(stem, _)| stem);
        format!("{stem}.faiss")
    }

    /// The index file's name, its key's last segment: the status
    /// response's `file_name`.
    #[must_use]
    pub fn file_name(&self) -> String {
        let key = self.index_key();
        key.rsplit_once('/')
            .map_or(key.as_str(), |(_, name)| name)
            .to_owned()
    }

    /// The host memory the job's buffers take, in bytes: the vectors as
    /// downloaded, the doc ids, and the graph; for a job that asks for the
    /// projection the most its buffers take at once, counted for as many
    /// threads as search at most; and the recall check's, when it has one.
    /// Saturating, past any memory.
    pub(crate) fn host_bytes(&self) -> u64 {
        let doc_count = u64::from(self.doc_count);
        // No graph when the parameters fail; the job fails before one.
        let degree = self.graph_degree().map_or(0, u64::from);
        let vectors = doc_count.saturating_mul(self.vector_bytes());
        let ids = doc_count * 4;
        let graph = doc_count.saturating_mul(degree).saturating_mul(4);
        let projection = self.projection(NonZeroUsize::MAX).map_or(0, |settings| {
            projection::host_bytes(doc_count, degree, u64::from(self.dimension), &settings)
        });
        vectors
            .saturating_add(ids)
            .saturating_add(graph)
            .saturating_add(projection)
            .saturating_add(self.recall_check.host_bytes(self))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::num::NonZeroUsize;

    use knn_write::projection;

    use super::{DataType, GraphParams, Job, SpaceType};
    use crate::RecallCheck;

    /// A job on `bucket` for `doc_count` x `dimension` fp32 vectors.
    pub(crate) fn job(bucket: &str, doc_count: u32, dimension: u32) -> Job {
        Job {
            bucket: bucket.to_owned(),
            vector_path: "segments/_0_target_field.knnvec".to_owned(),
            doc_id_path: "segments/_0_target_field.knndid".to_owned(),
            dimension,
            doc_count,
            data_type: DataType::Float,
            space_type: SpaceType::L2,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            skip_stored_vectors: false,
            project_link_to_base_layer: false,
            recall_check: RecallCheck::default(),
        }
    }

    #[test]
    fn maps_space_types_to_metrics() {
        // Spelled out here, so a test that derives its expected file through
        // these methods still rests on a fixed mapping.
        assert_eq!(SpaceType::L2.build_metric(), knn_gpu::Metric::L2);
        assert_eq!(
            SpaceType::InnerProduct.build_metric(),
            knn_gpu::Metric::InnerProduct
        );
        assert_eq!(SpaceType::Hamming.build_metric(), knn_gpu::Metric::L2);
        let mut job = job("bucket", 10, 8);
        assert_eq!(job.file_metric(), knn_write::Metric::L2);
        job.space_type = SpaceType::InnerProduct;
        assert_eq!(job.file_metric(), knn_write::Metric::InnerProduct);
        // The service builds float vectors under hamming in L2.
        job.space_type = SpaceType::Hamming;
        assert_eq!(job.file_metric(), knn_write::Metric::L2);
        // And binary vectors in Hamming distance under any space type.
        job.data_type = DataType::Binary;
        for space in [SpaceType::L2, SpaceType::InnerProduct, SpaceType::Hamming] {
            job.space_type = space;
            assert_eq!(job.file_metric(), knn_write::Metric::Hamming);
        }
    }

    #[test]
    fn builds_binary_vectors_through_nn_descent() {
        let mut job = job("bucket", 40, 256);
        job.data_type = DataType::Binary;
        let params = job.graph_params().unwrap();
        assert!(matches!(params, GraphParams::Binary(_)));
        // m 16 asks 64 neighbors of 40 vectors; cuVS gives 39.
        assert_eq!(params.degree_for(40), 39);
        assert_eq!(job.graph_degree(), Some(39));
        assert_eq!(job.vector_bytes(), 32);
        assert_eq!(job.host_bytes(), 40 * 32 + 40 * 4 + 40 * 39 * 4);
        job.dimension = 12;
        assert!(job.graph_params().is_err());
    }

    #[test]
    fn sizes_the_buffers_a_job_holds() {
        // The degree is knn-gpu's to derive (m 16 is lowered at 600).
        let mut job = job("bucket", 600, 16);
        let degree = knn_gpu::BuildParams::new(600, 16, 16, knn_gpu::Metric::L2)
            .unwrap()
            .graph_degree;
        let (ids, graph) = (600 * 4, 600 * u64::from(degree) * 4);
        assert_eq!(job.host_bytes(), 600 * 16 * 4 + ids + graph);
        job.data_type = DataType::Byte;
        assert_eq!(job.host_bytes(), 600 * 16 + ids + graph);
        job.data_type = DataType::HalfFloat;
        assert_eq!(job.host_bytes(), 600 * 16 * 2 + ids + graph);
        job.m = 0;
        assert_eq!(job.host_bytes(), 600 * 16 * 2 + ids, "no graph");
        let large = Job {
            doc_count: u32::MAX,
            dimension: u32::MAX,
            ..job
        };
        assert_eq!(large.host_bytes(), u64::MAX);
    }

    #[test]
    fn projects_as_the_service_does() {
        let mut job = job("bucket", 600, 16);
        assert_eq!(job.projection(NonZeroUsize::MIN), None, "not asked");
        let plain = job.host_bytes();
        job.project_link_to_base_layer = true;
        let settings = job.projection(NonZeroUsize::MIN).unwrap();
        // m 16, lowered to 15 at 600 vectors; the request's
        // efConstruction; the file's metric; the service's defaults else.
        assert_eq!(settings.m, 15);
        assert_eq!(settings.ef_construction, 100);
        assert_eq!(settings.metric, knn_write::Metric::L2);
        assert_eq!(settings.links, projection::LinkSettings::default());
        assert!((settings.skip_if_reachable - 0.99).abs() < f64::EPSILON);
        // Its buffers join the job's, counted for the most threads that
        // search.
        let most = job.projection(NonZeroUsize::MAX).unwrap();
        assert_eq!(
            job.host_bytes(),
            plain + projection::host_bytes(600, 60, 16, &most)
        );
        // Below 0, Faiss's efConstruction bounds no search; the buffers
        // stay those of a search over every promoted node.
        let asked = job.host_bytes();
        job.ef_construction = -1;
        let unbounded = job.projection(NonZeroUsize::MIN).unwrap();
        assert_eq!(unbounded.ef_construction, usize::MAX);
        assert!((asked..2 * asked).contains(&job.host_bytes()));
        // Binary vectors are not projected.
        job.data_type = DataType::Binary;
        job.dimension = 64;
        assert_eq!(job.projection(NonZeroUsize::MIN), None);
    }

    #[test]
    fn names_the_index_as_the_service_does() {
        let mut job = job("bucket", 10, 4);
        assert_eq!(job.index_key(), "segments/_0_target_field.faiss");
        assert_eq!(job.file_name(), "_0_target_field.faiss");
        // Only the last extension goes; dots elsewhere stay.
        job.vector_path = "a.b/c.d.knnvec".to_owned();
        assert_eq!(job.index_key(), "a.b/c.d.faiss");
        assert_eq!(job.file_name(), "c.d.faiss");
        job.vector_path = "flat.knnvec".to_owned();
        assert_eq!(job.file_name(), "flat.faiss");
    }
}
