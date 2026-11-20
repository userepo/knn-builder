//! What a job costs: each stage's seconds, its index file's size and the
//! device memory its build takes at its peak, as fitted to 58 builds on an
//! RTX 4070 Laptop GPU against LocalStack (`fixtures/grid/fits.json`, by
//! `tools/grid/fit.py`). The times are the forecast's starting point, which
//! completed jobs then correct. The sizes should hold on any GPU of 8 GB or
//! more running the same cuVS: its graph optimization batches a fixed
//! number of rows, and the workspace its IVF-PQ search batches by, a quarter
//! of the device's memory by default, is capped at 2 GiB (`knn_gpu::Gpu`),
//! about what the 4070 had; no larger GPU has been measured yet.
//!
//! The terms: n vectors of dimension d, s bytes a value on the device (2
//! for `half_float`, whose vectors are converted on download, 1 for
//! `byte`), graph degree g. The IVF-PQ build's k-means and its search
//! cost about n^1.5 d, the graph's optimization n g^2, moving the vectors
//! to the device, a batch at a time, n d s.
//!
//! The models were not fitted to byte builds, only checked against them
//! (`fixtures/grid/byte-runs.json`): their device peaks, stage times and
//! file sizes come as close as those of the builds fitted, but a byte build
//! takes about as long as a float one, so the build's time counts a byte
//! value as 4 bytes. A data node's own byte build takes as long as a float
//! one too (`byte-cpu.json`).
//!
//! A binary build, NN-descent in Hamming distance with d counting bits, has
//! models of its own, fitted to 30 builds by `tools/grid/fit.py binary`
//! (`fixtures/grid/binary-runs.json`, `binary-fits.json`): its time about
//! n d and n g^2 d, NN-descent comparing each node with pairs of its
//! candidates; its device peak the dataset, n d / 8 bytes, and the larger of
//! NN-descent's phase (a copy of the dataset and its graph) and the graph's
//! optimization (the graph and a buffer over at most 520,000 rows); its
//! file's size by terms of a float file's form, fitted anew. Its transfers are the float
//! models', a binary blob being n d / 8 bytes, and well within their spread.
//! A data node's own binary build, Faiss's binary HNSW, has its CPU model
//! too (`binary-cpu.json`), as fast on 4 threads as the GPU's NN-descent
//! at 500,000 vectors of 1,024 bits.
//!
//! Beside them, the build a data node runs itself when the service refuses
//! a segment: Faiss's HNSW over the float vectors (M 16,
//! `ef_construction` 100), timed with faiss-cpu 1.15.1 on the same machine's CPU on 1 and 4
//! threads. An insertion visits about log n nodes, each a distance over d
//! values and the bookkeeping around it. It under-estimates by 14 to 23% at
//! 500,000 vectors, where HNSW slows faster than n log n, and a data node's
//! CPU is not this one.
//!
//! And the projection a job may ask for, on the CPU between the build and
//! the write, timed under WSL on the same machine's i9-14900HX on 8
//! threads, a quarter of its 32 as the service takes, over 100,000 to
//! 1,000,000 vectors of 128 and 768 dimensions at m 2 to 32 in 20 clusters,
//! far apart and overlapping (`fixtures/grid/projection.json`, by
//! `knn-write`'s `projection_cost` test): the gate's search over the graph,
//! about n g; layer 1, about p ef sqrt(d) m^(1/3) for its p = n / m nodes;
//! and its links, about n g again. The whole is within 0.57 to 2.4 times
//! of each projection timed at m 4 or more, overlapping clusters the
//! slower, and 5.2 times the one at m 2, where a level's rows hold a link
//! or two and a search visits few nodes. Layer 1 runs 6.3 to 6.5 times as
//! fast on 8 threads as on one, a little more than the share of 5% done
//! on one thread that the model counts would give; on 16 and 32 threads,
//! this CPU's slower cores among them, it runs 8.4 to 10.2 times as fast,
//! which the model does not count. It is counted in full whenever a job
//! asks, though the gate leaves a graph that is already connected as
//! built after its search alone.
//!
//! And the recall check, on the CPU beside the write, timed with
//! knn-inspect's search on the same machine's 32 threads over 1M vectors
//! of knn-builder's files (cohere-768 and gist-960 as float, cohere-768 as
//! byte and as binary): its exact neighbors, a pass over every vector for
//! each query, about q n d for q queries, at 4.6e-11 to 4.7e-11 s a float
//! value, 3.3e-11 s a byte one and 3.3e-12 s a bit; and its searches,
//! about 1.3 ms a query at efSearch 100. A `half_float` value, decoded by
//! table, is counted as a float one, unmeasured. On 4 threads the exact
//! neighbors take 2.5 to 3.9 times as long, which the model does not count.

use std::num::NonZeroUsize;

use knn_gpu::BuildParams;

use crate::{DataType, GraphParams, Job};

/// The threads a data node builds a graph on itself
/// (`knn.algo_param.index_thread_qty`): the two its CPU build was measured
/// on, k-NN's defaults for a node of fewer than 32 processors and of 32 or
/// more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuThreads {
    /// One thread.
    One,
    /// Four threads.
    Four,
}

/// The fitted models, for one GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fitted {
    /// The device's shared memory a block may take, in bytes:
    /// `knn_gpu::shared_memory_per_block`.
    pub shared_memory_per_block: usize,
    /// The threads the projection builds layer 1 on.
    pub projection_threads: NonZeroUsize,
}

/// The machine the models were fitted on: the RTX 4070 Laptop GPU, and a
/// quarter of its CPU's 32 threads.
impl Default for Fitted {
    fn default() -> Self {
        Self {
            shared_memory_per_block: 101_376,
            projection_threads: NonZeroUsize::MIN.saturating_add(7),
        }
    }
}

/// Each stage's seconds, and the projection's and the recall check's apart
/// from the upload's they come before and beside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stages {
    /// Reading the blobs' sizes and downloading them.
    pub download: f64,
    /// Building the graph.
    pub build: f64,
    /// Projecting layer-1 links into the graph, in the upload stage before
    /// the write: 0 for a job that does not ask.
    pub projection: f64,
    /// Writing and uploading the index.
    pub upload: f64,
    /// The recall check, beside the write, in the upload stage: 0 for a
    /// job without one. The stage takes the longer of the two.
    pub check: f64,
}

impl Stages {
    /// The upload stage's seconds: the projection, then the longer of the
    /// write and the recall check.
    #[must_use]
    pub fn upload_stage(&self) -> f64 {
        self.projection + self.upload.max(self.check)
    }
}

/// A job's sizes, as the models take them.
#[derive(Debug, Clone, Copy)]
struct Shape {
    n: f64,
    d: f64,
    /// Bytes a value on the device and in the file.
    s: f64,
    /// Bytes a value as the build's time counts it.
    build_s: f64,
    /// Bytes a value in the vector blob.
    blob_s: f64,
    /// Bytes a value in the index file: none when it leaves the vectors
    /// out.
    file_s: f64,
    /// The graph degree.
    degree: u32,
    /// A binary build, NN-descent in Hamming distance, whose d counts bits.
    binary: bool,
    g: f64,
    pq_dim: u32,
}

impl Shape {
    fn of(job: &Job) -> Self {
        // A binary dimension counts bits, an eighth of a byte each.
        let s = match job.data_type {
            DataType::Float => 4.0,
            DataType::HalfFloat => 2.0,
            DataType::Byte => 1.0,
            DataType::Binary => 0.125,
        };
        // A job whose parameters fail is built by no one; its forecast
        // only has to be finite.
        let degree = job.graph_degree().unwrap_or(0);
        let pq_dim = match job.graph_params() {
            Ok(GraphParams::IvfPq(params)) => params.pq_dim,
            Ok(GraphParams::Binary(_)) | Err(_) => 0,
        };
        Self {
            n: f64::from(job.doc_count),
            d: f64::from(job.dimension),
            s,
            build_s: if job.data_type == DataType::Byte {
                4.0
            } else {
                s
            },
            // A byte blob holds a byte a value. A `half_float` blob holds
            // fp16 values from a half_float field or fp32 values from a
            // float field with the fp16 quantizer, which only its size
            // tells apart once the download starts: counted as fp32 until
            // then.
            blob_s: match job.data_type {
                DataType::Byte | DataType::Binary => s,
                DataType::Float | DataType::HalfFloat => 4.0,
            },
            file_s: if job.skip_stored_vectors { 0.0 } else { s },
            degree,
            binary: job.data_type == DataType::Binary,
            g: f64::from(degree),
            pq_dim,
        }
    }
}

/// How many candidates cuVS's IVF-PQ search may keep in the fused kernel
/// that also selects them (`kMaxCapacity`).
const FUSED_TOP_K: u32 = 128;

/// The most vectors the IVF-PQ build holds on the device at once, in their
/// own type, as it reads them from host memory. Where a float build and an
/// fp16 one of the same shape both peak in that phase, they peak exactly
/// this many rows of d values times the 2 bytes between them apart
/// (`fixtures/grid/runs.json`); builds of fewer vectors hold them all, and
/// 1,000 float and fp16 vectors peak alike (`fixtures/grid/first.json`).
const BATCH_ROWS: f64 = 65_536.0;

/// What a process's first build takes of the device besides a later one:
/// cuVS loads its kernels and creates its library handles then, and keeps
/// them: 40 to 42 MiB more at the peak, at every size measured
/// (`fixtures/grid/first.json`).
const FIRST_BUILD_BYTES: f64 = 44_040_192.0;

/// The candidates the IVF-PQ search that seeds a graph of `degree` over
/// `doc_count` vectors asks for at `refinement_rate`, as cuVS's CAGRA build
/// counts them: the degree times the rate, truncated, at least one more
/// than the degree, at most the vectors.
fn candidates(degree: u32, doc_count: u32, refinement_rate: f32) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "truncated as cuVS truncates it; a degree times a rate of a few is far below u32::MAX"
    )]
    let refined = (f64::from(degree) * f64::from(refinement_rate)) as u32;
    refined.max(degree.saturating_add(1)).min(doc_count)
}

impl Fitted {
    /// The stages' seconds for `job` on the RTX 4070 the models were
    /// fitted on, against LocalStack.
    #[must_use]
    pub fn stages(&self, job: &Job) -> Stages {
        let x = Shape::of(job);
        let (n, d, g) = (x.n, x.d, x.g);
        let slow = if self.slow_search(job) {
            n.powf(1.5) * d
        } else {
            0.0
        };
        let build = if x.binary {
            4.492_611_318_137_086e-8 * n * d
                + 1.561_331_592_921_353_3e-8 * n * g
                + 2.484_567_308_376_237e-11 * n * g * g
                + 1.498_931_781_089_467_2e-13 * n * g * g * d
        } else {
            0.535_603_221_132_059_7
                + 2.915_792_487_684_948e-9 * n * d * x.build_s
                + 9.566_176_532_461_707e-12 * n.powf(1.5) * d
                + 1.191_717_379_819_209_6e-10 * n * g * g
                + 8.814_974_178_064_082e-11 * slow
        };
        Stages {
            download: 0.006_905_632_182_188_668 + 2.321_054_281_931_872_5e-9 * n * d * x.blob_s,
            build,
            projection: self.projection_seconds(job),
            upload: 2.081_239_079_181_345e-9 * n * d * x.file_s
                + 6.751_823_325_351_644e-9 * n * g
                + 4.825_297_812_211_012e-8 * n,
            check: Self::check_seconds(job),
        }
    }

    /// The seconds `job`'s recall check takes on the 32 threads it was
    /// timed on: its exact neighbors and its searches. 0 for a job without
    /// one.
    #[must_use]
    pub fn check_seconds(job: &Job) -> f64 {
        let check = job.recall_check;
        if !check.is_on() {
            return 0.0;
        }
        let x = Shape::of(job);
        let q = f64::from(check.queries_for(job.doc_count));
        // A value's time, a binary dimension counting bits.
        let value = match job.data_type {
            DataType::Float | DataType::HalfFloat => 4.7e-11,
            DataType::Byte => 3.3e-11,
            DataType::Binary => 3.3e-12,
        };
        value * q * x.n * x.d + 1.3e-3 * q
    }

    /// The seconds `job`'s projection takes on [`Fitted::projection_threads`]
    /// when the gate lets it through: the gate's search, layer 1 and its
    /// links. 0 for a job that does not ask.
    #[must_use]
    pub fn projection_seconds(&self, job: &Job) -> f64 {
        let Some(settings) = job.projection(self.projection_threads) else {
            return 0.0;
        };
        let x = Shape::of(job);
        let (n, d, g) = (x.n, x.d, x.g);
        let gate = 1.179e-7 * n + 1.13e-9 * n * g;
        // An m below 2 promotes none: the gate alone.
        if settings.m < 2 {
            return gate;
        }
        let effective_m = f64::from(settings.m);
        let promoted = n / effective_m;
        // No search visits more nodes than layer 1 has.
        #[expect(
            clippy::cast_precision_loss,
            reason = "efConstruction, up to usize::MAX, then held to the promoted count, so its last digits change nothing"
        )]
        let ef = (settings.ef_construction as f64).min(promoted).max(1.0);
        #[expect(clippy::cast_precision_loss, reason = "a thread count, far below 2^53")]
        let threads = self.projection_threads.get().min(8) as f64;
        let speedup = 1.0 / (0.05 + 0.95 / threads);
        let layer1 =
            7.289e-6 * promoted * (ef / 100.0) * d.sqrt() * (effective_m / 16.0).cbrt() / speedup;
        let links = 1.613e-7 * n + 5.085e-10 * n * g;
        gate + layer1 + links
    }

    /// Whether cuVS runs the IVF-PQ search that seeds the graph in its slow
    /// form: its candidates (g + 1 at the build's refinement rate of 1) are
    /// too many for the kernel that also selects them, and its lookup table
    /// (fp32, `pq_dim` x 256 entries) does not fit a block's shared memory,
    /// so the distances go out to device memory and come back for the
    /// selection. On the RTX 4070 that is d 768 at m 32 alone, four times
    /// as slow.
    #[must_use]
    pub fn slow_search(&self, job: &Job) -> bool {
        let x = Shape::of(job);
        let lookup_table = (4 * u64::from(x.pq_dim)) << BuildParams::PQ_BITS;
        let shared = u64::try_from(self.shared_memory_per_block).unwrap_or(u64::MAX);
        candidates(x.degree, job.doc_count, BuildParams::REFINEMENT_RATE) > FUSED_TOP_K
            && lookup_table > shared
    }

    /// The index file's size in bytes: the vectors unless left out, the
    /// graph, the ids and the layout's own bytes. Exact on every build the
    /// models were fitted to.
    #[must_use]
    pub fn file_bytes(job: &Job) -> f64 {
        let x = Shape::of(job);
        if x.binary {
            return 6_901.013_298_726_097
                + 1.000_002_637_373_886 * x.n * x.d * x.file_s
                + 3.998_690_043_805_443 * x.n * x.g
                + 24.212_232_411_297_077 * x.n;
        }
        6_589.814_920_669_402
            + 1.000_000_492_050_695_3 * x.n * x.d * x.file_s
            + 3.998_668_335_023_911 * x.n * x.g
            + 24.214_104_760_075_195 * x.n
    }

    /// The device memory the build takes at its peak, in bytes: the larger
    /// of the IVF-PQ phase (the PQ codes, `pq_dim` bytes a vector, a batch
    /// of at most 65,536 vectors in their own type, which the build reads
    /// from host memory a batch at a time, and buffers of about 530 KB a
    /// dimension) and the graph's optimization (the graph, 4 g bytes a
    /// vector, and a buffer of g ids over at most 500,000 rows). The
    /// dataset is never on the device whole. The codes, the batch and the
    /// graph are floors the fitted terms only add to. For a binary build,
    /// the dataset and the larger of NN-descent's phase (a copy of the
    /// dataset and its graph) and the graph's optimization (the graph and a
    /// buffer of g ids over at most 520,000 rows), again on floors.
    #[must_use]
    pub fn device_peak_bytes(job: &Job) -> f64 {
        let x = Shape::of(job);
        let (n, d, g) = (x.n, x.d, x.g);
        if x.binary {
            let dataset = n * d * x.s;
            let nn_descent = dataset + 4.0 * n * g + 5_701_632.0;
            let graph = 4.0 * n * g
                + 5_741_063.172_462_075
                + 0.001_685_952_471_530_249_9 * n * g
                + 4.139_523_539_509_295 * n.min(520_000.0) * g;
            return dataset + nn_descent.max(graph);
        }
        let ivf_pq = n * f64::from(x.pq_dim)
            + n.min(BATCH_ROWS) * d * x.s
            + 530_468.456_779_138_1 * d
            + 0.035_995_253_966_641_88 * n * d;
        let graph = 4.0 * n * g
            + 0.060_478_834_177_069_97 * n * g
            + 4.198_149_930_919_303 * n.min(500_000.0) * g;
        ivf_pq.max(graph)
    }

    /// The device memory to have free for `job`'s build, in bytes: its
    /// peak, a margin of [`DEVICE_MARGIN`], and what the build takes
    /// besides if it is the process's first, which the forecast does not
    /// know.
    #[must_use]
    pub fn device_bytes_needed(job: &Job) -> u64 {
        let needed = Self::device_peak_bytes(job) * (1.0 + DEVICE_MARGIN) + FIRST_BUILD_BYTES;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a positive number of bytes, saturating past u64::MAX"
        )]
        let bytes = needed.ceil() as u64;
        bytes
    }

    /// The seconds a data node takes to build `job`'s graph itself on
    /// `threads`.
    #[must_use]
    pub fn cpu_seconds(job: &Job, threads: CpuThreads) -> f64 {
        let x = Shape::of(job);
        let n_log_n = x.n * x.n.max(1.0).log2();
        if x.binary {
            return match threads {
                CpuThreads::One => {
                    2.562_140_325_441_633_3e-6 * n_log_n
                        + 3.325_921_398_800_654_3e-9 * n_log_n * x.d
                }
                CpuThreads::Four => {
                    7.079_431_515_444_463e-7 * n_log_n + 8.526_998_939_344_604e-10 * n_log_n * x.d
                }
            };
        }
        match threads {
            CpuThreads::One => {
                3.236_285_214_401_488_3e-6 * n_log_n + 2.013_133_437_965_566e-8 * n_log_n * x.d
            }
            CpuThreads::Four => {
                7.841_646_090_633_016e-7 * n_log_n + 5.601_684_767_440_988e-9 * n_log_n * x.d
            }
        }
    }
}

/// The share of a build's modeled device peak kept free besides. The
/// models are under by at most 0.8% on their grids (float, fp16 and byte)
/// and 1.2% (binary); the margin stays the 6% it was when the float model
/// was under by up to 2.7%, for GPUs and sizes no grid has measured, and
/// for allocations a measurement can miss, as the 5 ms samples missed the
/// briefest builds' peaks (`fixtures/grid/tight.json`).
pub const DEVICE_MARGIN: f64 = 0.06;

#[cfg(test)]
mod tests {
    use super::{FUSED_TOP_K, Fitted, candidates};
    use crate::job::tests::job;
    use crate::{DataType, RecallCheck};

    #[test]
    fn times_the_recall_check_as_measured() {
        let mut cohere = job("bucket", 1_000_000, 768);
        assert!(Fitted::check_seconds(&cohere).abs() < f64::EPSILON, "off");
        cohere.recall_check = RecallCheck {
            queries: 100,
            ..RecallCheck::default()
        };
        // Measured: 3.6 s for the exact neighbors of 100 queries, 0.13 s
        // for their searches; 2.5 s and 0.25 s as byte and binary vectors.
        let within = |job: &crate::Job, measured: f64| {
            let modeled = Fitted::check_seconds(job);
            assert!(
                (modeled / measured - 1.0).abs() < 0.05,
                "{:?}: {modeled} for {measured}",
                job.data_type
            );
        };
        within(&cohere, 3.6 + 0.13);
        let stages = Fitted::default().stages(&cohere);
        assert!((stages.check - Fitted::check_seconds(&cohere)).abs() < f64::EPSILON);
        cohere.data_type = DataType::Byte;
        within(&cohere, 2.5 + 0.13);
        cohere.data_type = DataType::Binary;
        within(&cohere, 0.25 + 0.13);
        // No more queries than vectors.
        cohere.doc_count = 50;
        cohere.data_type = DataType::Float;
        let fifty = Fitted::check_seconds(&cohere);
        cohere.recall_check.queries = 50;
        assert!((Fitted::check_seconds(&cohere) - fifty).abs() < f64::EPSILON);
    }

    #[test]
    fn counts_the_search_candidates_as_cuvs_does() {
        // At the build's rate of 1, one more than the degree: m 32 is the
        // first past the fused kernel's 128.
        assert_eq!(candidates(124, 1_000_000, 1.0), 125);
        assert_eq!(candidates(128, 1_000_000, 1.0), 129);
        // At Faiss's own default of 2, twice the degree: m 17 is the first
        // past it.
        assert_eq!(candidates(64, 1_000_000, 2.0), FUSED_TOP_K);
        assert_eq!(candidates(68, 1_000_000, 2.0), 136);
        // Never more than the vectors.
        assert_eq!(candidates(128, 100, 2.0), 100);
    }
}
