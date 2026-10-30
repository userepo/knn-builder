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
//! cost about n^1.5 d, the graph's optimization n g^2, the dataset's copy
//! n d s.
//!
//! The models were not fitted to byte builds, only checked against them
//! (`fixtures/grid/byte-runs.json`): their device peaks are off by the
//! same bytes as the float builds of the same shape, and their stage times
//! and file sizes as close as those of the builds fitted, but a byte build
//! takes as long as a float one, 0.99 to 1.25 times on that grid, so the
//! build's time counts a byte value as 4 bytes. A data node's own byte
//! build takes as long as a float one too (`byte-cpu.json`).
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
}

/// The GPU the models were fitted on, the RTX 4070 Laptop GPU.
impl Default for Fitted {
    fn default() -> Self {
        Self {
            shared_memory_per_block: 101_376,
        }
    }
}

/// A stage's seconds, or all three.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stages {
    /// Reading the blobs' sizes and downloading them.
    pub download: f64,
    /// Building the graph.
    pub build: f64,
    /// Writing and uploading the index.
    pub upload: f64,
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
            4.484_168_419_630_195_4e-8 * n * d
                + 1.781_806_605_547_999_5e-8 * n * g
                + 3.253_617_429_798_503_6e-11 * n * g * g
                + 9.243_215_069_359_727e-14 * n * g * g * d
        } else {
            0.462_016_340_462_556_74
                + 2.646_071_502_409_124e-9 * n * d * x.build_s
                + 8.893_904_400_583_446e-12 * n.powf(1.5) * d
                + 1.473_025_728_723_046_4e-10 * n * g * g
                + 8.832_281_977_621_297e-11 * slow
        };
        Stages {
            download: 0.004_586_874_751_921_019 + 2.440_557_794_472_648_3e-9 * n * d * x.blob_s,
            build,
            upload: 2.384_391_263_270_578_8e-9 * n * d * x.file_s
                + 6.416_459_498_281_393_6e-9 * n * g
                + 1.330_050_367_178_579_5e-7 * n,
        }
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

    /// The device memory the build takes at its peak, in bytes: the
    /// dataset, and the larger of the IVF-PQ phase (the PQ codes, `pq_dim`
    /// bytes a vector, and a batch of about 127k rows of d floats) and the
    /// graph's optimization (the graph, 4 g bytes a vector, and a buffer of
    /// g ids over at most 500,000 rows). The codes and the graph are floors
    /// the fitted terms only add to. For a binary build, the dataset and the
    /// larger of NN-descent's phase (a copy of the dataset and its graph)
    /// and the graph's optimization (the graph and a buffer of g ids over at
    /// most 520,000 rows), again on floors.
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
        let ivf_pq =
            n * f64::from(x.pq_dim) + 509_470.100_899_898_9 * d + 0.067_215_460_416_540_28 * n * d;
        let graph = 4.0 * n * g
            + 318_821.859_684_872_73
            + 0.043_274_792_718_468_46 * n * g
            + 4.261_985_417_489_097 * n.min(500_000.0) * g;
        n * d * x.s + ivf_pq.max(graph)
    }

    /// The device memory to have free for `job`'s build, in bytes: its
    /// peak and a margin of [`DEVICE_MARGIN`].
    #[must_use]
    pub fn device_bytes_needed(job: &Job) -> u64 {
        let needed = Self::device_peak_bytes(job) * (1.0 + DEVICE_MARGIN);
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

/// The share of a build's modeled device peak kept free besides: twice
/// the model's worst under-estimate on the grid it was fitted to (2.7%),
/// and a little for what cuVS rounds its allocations up to. A byte build
/// is off by the same bytes as a float one of its shape, which is up to
/// 3.0% of its smaller peak; the binary model is under by at most 1.2%.
pub const DEVICE_MARGIN: f64 = 0.06;

#[cfg(test)]
mod tests {
    use super::{FUSED_TOP_K, candidates};

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
