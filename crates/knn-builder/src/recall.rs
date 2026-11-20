//! The recall check the service runs on each job when it is set: before
//! the index is stored, a sample of the job's own vectors searched in its
//! graph as k-NN will search the file, scored against their exact nearest
//! neighbors. knn-inspect's `knn_recall::host` does the work over the
//! vectors and graph in host memory, and gives the recall that
//! `knn-inspect recall --sample` gives on the file written from them, with
//! the same seed, k and count. A job that leaves its vectors out of the
//! file is the exception: k-NN searches its graph over the quantized
//! vectors Lucene keeps on the data node, which the service never holds,
//! so its check gives the graph's recall over the full vectors instead.
//!
//! It runs on the CPU, beside the write, on rayon's threads, one a
//! processor; the upload waits for it before it completes. A recall below
//! the least set fails the job with nothing stored, so its data node
//! builds the segment itself. A check that cannot run, on inputs the
//! writer refuses too, says so in a warning and judges nothing. A check
//! cannot be stopped partway: a job canceled meanwhile ends once its check
//! does.

use knn_recall::{Draw, host};

use crate::{Job, JobError, Vectors};

/// What the recall check does; off by default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallCheck {
    /// Vectors sampled as queries (`RECALL_CHECK_QUERIES`); 0, the
    /// default, turns the check off. A job of fewer vectors samples all
    /// of them.
    pub queries: u32,
    /// Neighbors each query asks for (`RECALL_CHECK_K`); 10 by default, as
    /// knn-inspect's.
    pub k: u32,
    /// The least recall a job may have (`RECALL_CHECK_MIN`), a fraction:
    /// below it, the job fails. Unset by default, and the recall is only
    /// logged.
    pub min_recall: Option<f64>,
}

impl Default for RecallCheck {
    fn default() -> Self {
        Self {
            queries: 0,
            k: 10,
            min_recall: None,
        }
    }
}

impl RecallCheck {
    /// Whether a job gets the check.
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.queries > 0
    }

    /// The queries a job of `doc_count` vectors samples.
    #[must_use]
    pub fn queries_for(&self, doc_count: u32) -> u32 {
        self.queries.min(doc_count)
    }

    /// The host memory the check takes for `job` beside the job's own
    /// buffers, in bytes: a table of the nodes seen for each thread that
    /// searches, 4 bytes a node, and the queries, their exact neighbors
    /// and their result lists. 0 when it is off. Saturating.
    pub(crate) fn host_bytes(&self, job: &Job) -> u64 {
        if !self.is_on() {
            return 0;
        }
        let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let threads = u64::try_from(threads).unwrap_or(u64::MAX);
        let nodes = u64::from(job.doc_count);
        let queries = u64::from(self.queries_for(job.doc_count));
        // A query as float32 values, or a binary one as its bytes.
        let row = match job.data_type {
            crate::DataType::Binary => job.vector_bytes(),
            _ => u64::from(job.dimension) * 4,
        };
        // A neighbor's distance and node, a hit's, and a list's label.
        let lists = u64::from(self.k).saturating_mul(16 + 16 + 8);
        threads
            .saturating_mul(nodes)
            .saturating_mul(4)
            .saturating_add(queries.saturating_mul(row.saturating_add(lists)))
    }
}

/// The seed of the sample and of the searches' entry draws: knn-inspect's
/// default.
pub const SEED: u64 = 1;

/// What a job's recall check found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Checked {
    /// Matches over all queries, as a share of those perfect lists would
    /// have.
    pub recall: f64,
    /// Queries sampled.
    pub queries: usize,
    /// Neighbors each query asked for.
    pub k: usize,
}

/// Checks `job`'s graph, `degree` ids a vector, over its `vectors`, as its
/// file holds them: the queries drawn from the vectors with [`SEED`], each
/// search entered through the random nodes the file stores the count of,
/// at the file's efSearch.
pub(crate) fn check(
    job: &Job,
    vectors: &Vectors,
    graph: &[u32],
    degree: usize,
) -> Result<Checked, knn_recall::Error> {
    let settings = job.recall_check;
    let dimension = job.dimension as usize;
    let space = match job.file_metric() {
        knn_write::Metric::InnerProduct => host::Space::InnerProduct,
        // Binary vectors take no space.
        _ => host::Space::L2,
    };
    let held = match vectors {
        Vectors::F32(values) => host::Vectors::float32(values, dimension, space),
        Vectors::F16(bits) => host::Vectors::fp16(bits, dimension, space),
        Vectors::I8(values) => host::Vectors::int8(values, dimension, space),
        Vectors::Bits(bits) => host::Vectors::binary(bits, dimension),
    }?;
    let graph = host::Graph::new(graph, degree)?
        .with_entry_points(usize::from(knn_write::BASE_LEVEL_ENTRY_POINTS));
    let count = usize::try_from(settings.queries_for(job.doc_count)).unwrap_or(usize::MAX);
    let (queries, _rows) = host::sample_queries(&held, count, SEED)?;
    let k = usize::try_from(settings.k).unwrap_or(usize::MAX);
    let truth = host::ground_truth(&held, &queries, k)?;
    // The file's efSearch; one below 0 searches as 0 does, as knn-inspect
    // reads it.
    let ef_search = usize::try_from(job.ef_search).unwrap_or(0);
    let draw = Draw {
        seed: SEED,
        count: None,
    };
    let lists: Vec<Vec<i64>> = host::search_all(&held, &graph, &queries, k, ef_search, draw)?
        .iter()
        .map(|hits| host::result_labels(&graph, hits, k))
        .collect();
    let recall = host::score(&held, &queries, &truth, &lists)?;
    Ok(Checked {
        recall: recall.recall,
        queries: recall.queries,
        k,
    })
}

/// Whether `checked` passes `settings`: a recall below the least set fails
/// the job.
pub(crate) fn verdict(settings: &RecallCheck, checked: &Checked) -> Result<(), JobError> {
    match settings.min_recall {
        Some(min) if checked.recall < min => Err(JobError::LowRecall {
            recall: checked.recall,
            min,
            queries: checked.queries,
            k: checked.k,
        }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use knn_recall::{Draw, Labels};
    use knn_write::{IndexFile, IndexSpec};

    use super::{Checked, RecallCheck, SEED, check, verdict};
    use crate::job::tests::job;
    use crate::{DataType, FakeBuilder, Job, JobError, SpaceType, Vectors};

    /// Vectors a job of `doc_count` x `dimension` holds of its type: values
    /// that spread, from a small linear congruential stream.
    fn vectors(job: &Job) -> Vectors {
        let mut state = 0x2545_F491_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state >> 8
        };
        let values = (job.doc_count * job.dimension) as usize;
        #[expect(clippy::cast_precision_loss, reason = "24 bits")]
        let mut float = move || next() as f32 / (1 << 24) as f32 - 0.5;
        match job.data_type {
            DataType::Float => Vectors::F32((0..values).map(|_| float()).collect()),
            DataType::HalfFloat => Vectors::F16(
                (0..values)
                    .map(|_| knn_gpu::fp16::f32_to_f16(float()))
                    .collect(),
            ),
            #[expect(clippy::cast_possible_truncation, reason = "within -128 to 127")]
            DataType::Byte => Vectors::I8((0..values).map(|_| (float() * 250.0) as i8).collect()),
            DataType::Binary => {
                let bytes = values / 8;
                #[expect(clippy::cast_possible_truncation, reason = "the low byte")]
                Vectors::Bits((0..bytes).map(|_| (float().to_bits() >> 3) as u8).collect())
            }
        }
    }

    /// What knn-inspect's recall gives on the file written from `job`'s
    /// inputs, its doc ids far from its node numbers.
    fn file_recall(job: &Job, vectors: &Vectors, graph: &[u32], degree: usize) -> f64 {
        let ids: Vec<i32> = (0..i32::try_from(job.doc_count).unwrap())
            .map(|i| 1_000 + 3 * i)
            .collect();
        let file = IndexFile::new(IndexSpec {
            dimension: job.dimension as usize,
            metric: job.file_metric(),
            graph,
            graph_degree: degree,
            vectors: vectors.as_write(),
            ids: &ids,
            ef_construction: job.ef_construction,
            ef_search: job.ef_search,
            skip_stored_vectors: false,
        })
        .unwrap();
        let mut bytes = Vec::new();
        file.write_to(&mut bytes).unwrap();
        let parsed = knn_format::parse(&bytes).unwrap();
        let index = parsed.index.hnsw().unwrap();
        let storage = index.storage.as_ref().unwrap();
        let settings = job.recall_check;
        let count = settings.queries_for(job.doc_count) as usize;
        let k = settings.k as usize;
        let (queries, _) = knn_recall::sample_queries(storage, count, SEED).unwrap();
        let truth = knn_recall::ground_truth(storage, &queries, k).unwrap();
        let ef_search = usize::try_from(job.ef_search).unwrap();
        let draw = Draw {
            seed: SEED,
            count: None,
        };
        let lists: Vec<Vec<i64>> = knn_recall::search_all(index, &queries, k, ef_search, draw)
            .unwrap()
            .iter()
            .map(|hits| knn_recall::result_labels(&parsed.index, hits, k))
            .collect();
        let labels = Labels::from_id_map(parsed.index.id_map).unwrap();
        knn_recall::score(storage, &labels, &queries, &truth, &lists)
            .unwrap()
            .recall
    }

    /// A job of 300 vectors in three runs no search leaves, checked with
    /// 50 queries.
    fn checked_job(data_type: DataType, space_type: SpaceType) -> Job {
        let mut job = job(
            "bucket",
            300,
            if data_type == DataType::Binary { 64 } else { 8 },
        );
        job.data_type = data_type;
        job.space_type = space_type;
        job.ef_search = 16;
        job.recall_check = RecallCheck {
            queries: 50,
            ..RecallCheck::default()
        };
        job
    }

    #[test]
    fn gives_knn_inspects_recall_on_the_file() {
        for (data_type, space_type, degree) in [
            (DataType::Float, SpaceType::L2, 16),
            (DataType::Float, SpaceType::InnerProduct, 16),
            (DataType::HalfFloat, SpaceType::L2, 16),
            (DataType::Byte, SpaceType::L2, 16),
            (DataType::Byte, SpaceType::InnerProduct, 16),
            // An odd degree, which only a binary build has: the file keeps
            // all but its last id.
            (DataType::Binary, SpaceType::Hamming, 15),
        ] {
            let job = checked_job(data_type, space_type);
            let vectors = vectors(&job);
            let graph = FakeBuilder::clustered_graph(300, degree, 3);
            let checked = check(&job, &vectors, &graph, degree).unwrap();
            let case = format!("{data_type:?} in {space_type:?}");
            assert_eq!(
                checked.recall,
                file_recall(&job, &vectors, &graph, degree),
                "{case}"
            );
            assert_eq!((checked.queries, checked.k), (50, 10), "{case}");
            // A search that stays in its query's run misses the rest.
            assert!(
                (0.2..0.9).contains(&checked.recall),
                "{case}: {}",
                checked.recall
            );
        }
    }

    #[test]
    fn asks_for_the_k_it_is_given() {
        let mut job = checked_job(DataType::Float, SpaceType::L2);
        job.recall_check.k = 25;
        let vectors = vectors(&job);
        let graph = FakeBuilder::clustered_graph(300, 16, 3);
        let checked = check(&job, &vectors, &graph, 16).unwrap();
        assert_eq!(checked.k, 25);
        assert_eq!(checked.recall, file_recall(&job, &vectors, &graph, 16));
        job.recall_check.k = 10;
        assert_ne!(
            check(&job, &vectors, &graph, 16).unwrap().recall,
            checked.recall
        );
    }

    #[test]
    fn finds_a_connected_graph_whole() {
        // Each vector linked to every other: a search reaches them all.
        let mut job = checked_job(DataType::Float, SpaceType::L2);
        job.doc_count = 40;
        job.ef_search = 100;
        let vectors = vectors(&job);
        let graph = FakeBuilder::graph(40, 39);
        // An odd degree, but the file's 38 still link every vector.
        let checked = check(&job, &vectors, &graph, 39).unwrap();
        assert!(
            (checked.recall - 1.0).abs() < f64::EPSILON,
            "{}",
            checked.recall
        );
        // A job of fewer vectors than queries samples all of them.
        assert_eq!(checked.queries, 40);
    }

    #[test]
    fn cannot_check_what_no_file_holds() {
        let job = checked_job(DataType::Float, SpaceType::L2);
        let vectors = vectors(&job);
        // A degree below 4, which the writer refuses too.
        let graph = FakeBuilder::graph(300, 3);
        assert!(check(&job, &vectors, &graph, 3).is_err());
        // A graph of other vectors than the job holds.
        let graph = FakeBuilder::graph(200, 16);
        assert!(check(&job, &vectors, &graph, 16).is_err());
    }

    #[test]
    fn fails_a_job_below_the_least_recall() {
        let checked = Checked {
            recall: 0.5,
            queries: 100,
            k: 10,
        };
        let least = |min| RecallCheck {
            queries: 100,
            min_recall: min,
            ..RecallCheck::default()
        };
        assert_eq!(verdict(&least(None), &checked), Ok(()));
        assert_eq!(verdict(&least(Some(0.5)), &checked), Ok(()));
        let failed = verdict(&least(Some(0.9)), &checked).unwrap_err();
        assert_eq!(
            failed,
            JobError::LowRecall {
                recall: 0.5,
                min: 0.9,
                queries: 100,
                k: 10,
            }
        );
        assert_eq!(
            failed.to_string(),
            "the recall check failed: recall@10 over 100 sampled vectors is 0.5000, below the least of 0.9"
        );
    }

    #[test]
    fn counts_the_memory_it_takes() {
        let mut job = job("bucket", 1_000, 16);
        let unchecked = job.host_bytes();
        assert_eq!(job.recall_check.host_bytes(&job), 0, "off");
        job.recall_check = RecallCheck {
            queries: 100,
            ..RecallCheck::default()
        };
        let threads = std::thread::available_parallelism().unwrap().get() as u64;
        // A table of 1,000 nodes a thread; 100 queries of 16 float values
        // and their lists of 10.
        let bytes = threads * 1_000 * 4 + 100 * (16 * 4 + 10 * 40);
        assert_eq!(job.recall_check.host_bytes(&job), bytes);
        assert_eq!(job.host_bytes(), unchecked + bytes);
        // No more queries than vectors.
        job.doc_count = 50;
        let rows = 50 * (16 * 4 + 10 * 40);
        assert_eq!(job.recall_check.host_bytes(&job), threads * 50 * 4 + rows);
    }
}
