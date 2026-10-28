//! A build job, as the service's request describes it.

/// How k-NN encoded the vectors: the request's `data_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// `float`: fp32 values.
    Float,
    /// `half_float`: fp16 values from a `half_float` field, or fp32 values
    /// from a float field with the fp16 scalar quantizer. The blob's size
    /// tells them apart, and fp32 values become fp16 on download, as the
    /// service converts them.
    HalfFloat,
}

/// The request's `space_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceType {
    /// `l2`.
    L2,
    /// `innerproduct`.
    InnerProduct,
}

impl SpaceType {
    /// The metric the graph is built for.
    #[must_use]
    pub fn build_metric(self) -> knn_gpu::Metric {
        match self {
            Self::L2 => knn_gpu::Metric::L2,
            Self::InnerProduct => knn_gpu::Metric::InnerProduct,
        }
    }

    /// The metric the index file records.
    #[must_use]
    pub fn file_metric(self) -> knn_write::Metric {
        match self {
            Self::L2 => knn_write::Metric::L2,
            Self::InnerProduct => knn_write::Metric::InnerProduct,
        }
    }
}

/// One index to build: where its inputs are and what to build from them.
/// The service's request, validated (M5 maps one to the other).
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

impl Job {
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
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{DataType, Job, SpaceType};

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
        assert_eq!(SpaceType::L2.file_metric(), knn_write::Metric::L2);
        assert_eq!(
            SpaceType::InnerProduct.file_metric(),
            knn_write::Metric::InnerProduct
        );
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
