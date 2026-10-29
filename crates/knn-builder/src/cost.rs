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
//! for `half_float`, whose vectors are converted on download), graph degree
//! g. The IVF-PQ build's k-means and its search cost about n^1.5 d, the
//! graph's optimization n g^2, the dataset's copy n d s.

use knn_gpu::BuildParams;

use crate::{DataType, Job};

/// The fitted models, for one GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fitted {
    /// The device's shared memory a block may take, in bytes:
    /// `knn_gpu::shared_memory_per_block`.
    pub shared_memory_per_block: usize,
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
    /// Bytes a value in the vector blob.
    blob_s: f64,
    /// Bytes a value in the index file: none when it leaves the vectors
    /// out.
    file_s: f64,
    /// The graph degree.
    degree: u32,
    g: f64,
    pq_dim: u32,
}

impl Shape {
    fn of(job: &Job) -> Self {
        let s = match job.data_type {
            DataType::Float => 4.0,
            DataType::HalfFloat => 2.0,
        };
        // A job whose parameters fail is built by no one; its forecast
        // only has to be finite.
        let params = BuildParams::new(
            job.doc_count,
            job.dimension,
            job.m,
            job.space_type.build_metric(),
        )
        .ok();
        Self {
            n: f64::from(job.doc_count),
            d: f64::from(job.dimension),
            s,
            // A `half_float` blob holds fp16 values from a half_float
            // field or fp32 values from a float field with the fp16
            // quantizer, which only its size tells apart once the download
            // starts: counted as fp32 until then.
            blob_s: 4.0,
            file_s: if job.skip_stored_vectors { 0.0 } else { s },
            degree: params.map_or(0, |p| p.graph_degree),
            g: params.map_or(0.0, |p| f64::from(p.graph_degree)),
            pq_dim: params.map_or(0, |p| p.pq_dim),
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
        Stages {
            download: 0.004_586_874_751_921_019 + 2.440_557_794_472_648_3e-9 * n * d * x.blob_s,
            build: 0.462_016_340_462_556_74
                + 2.646_071_502_409_124e-9 * n * d * x.s
                + 8.893_904_400_583_446e-12 * n.powf(1.5) * d
                + 1.473_025_728_723_046_4e-10 * n * g * g
                + 8.832_281_977_621_297e-11 * slow,
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
    /// the fitted terms only add to.
    #[must_use]
    pub fn device_peak_bytes(job: &Job) -> f64 {
        let x = Shape::of(job);
        let (n, d, g) = (x.n, x.d, x.g);
        let ivf_pq =
            n * f64::from(x.pq_dim) + 509_470.100_899_898_9 * d + 0.067_215_460_416_540_28 * n * d;
        let graph = 4.0 * n * g
            + 318_821.859_684_872_73
            + 0.043_274_792_718_468_46 * n * g
            + 4.261_985_417_489_097 * n.min(500_000.0) * g;
        n * d * x.s + ivf_pq.max(graph)
    }
}

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
