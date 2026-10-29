//! Builds on the GPU: the graph's shape and quality on random vectors, in
//! both element types and both metrics, and the input checks. Needs the
//! `gpu` feature, a CUDA GPU and the conda environment in
//! `environment.yml`.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::sync::Mutex;

use knn_gpu::{BuildParams, Error, Gpu, Graph, HostVectors, Metric, fp16};

/// Held through each test's build: cargo runs the tests on parallel
/// threads, and two builds at once can crash the process (see `Gpu`).
static GPU: Mutex<()> = Mutex::new(());

/// Builds on a `Gpu` of its own, one test at a time.
fn build(vectors: HostVectors<'_>, dimension: usize, params: &BuildParams) -> Result<Graph, Error> {
    // The callers check the result after the lock is released, so nothing
    // panics while it is held.
    let _gpu = GPU.lock().expect("a build panicked holding the GPU lock");
    let mut gpu = Gpu::new()?;
    gpu.build(vectors, dimension, params)
}

/// Values in [-0.5, 0.5) from a fixed-seed xorshift generator.
fn random_vectors(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[expect(clippy::cast_precision_loss, reason = "a 24-bit value is exact in f32")]
            let value = (state >> 40) as f32;
            value / 16_777_216.0 - 0.5
        })
        .collect()
}

fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum(),
        Metric::InnerProduct => -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
    }
}

/// Builds and checks: n x degree, every id in range and not the node
/// itself, and, over a sample of nodes, the nearest neighbor in most rows.
fn check(metric: Metric, half: bool) {
    const ROWS: usize = 5_000;
    const DIM: usize = 32;
    const SAMPLE: usize = 500;
    let values = random_vectors(ROWS * DIM, 0x9E37_79B9_7F4A_7C15);
    let f16s: Vec<u16> = values.iter().map(|&x| fp16::f32_to_f16(x)).collect();
    let vectors = if half {
        HostVectors::F16(&f16s)
    } else {
        HostVectors::F32(&values)
    };
    let params = BuildParams::new(5_000, 32, 16, metric).unwrap();
    let graph = build(vectors, DIM, &params).unwrap();

    let degree = params.graph_degree as usize;
    assert_eq!(graph.degree(), degree);
    assert_eq!(graph.neighbors().len(), ROWS * degree);
    assert_eq!(graph.out_of_range(), 0);

    let row = |i: usize| &values[i * DIM..(i + 1) * DIM];
    for (node, neighbors) in graph.neighbors().chunks_exact(degree).enumerate() {
        assert!(
            neighbors.iter().all(|&n| n as usize != node),
            "node {node} lists itself"
        );
    }
    let mut found = 0;
    for node in (0..ROWS).step_by(ROWS / SAMPLE) {
        let neighbors = &graph.neighbors()[node * degree..(node + 1) * degree];
        let nearest = (0..ROWS)
            .filter(|&other| other != node)
            .min_by(|&a, &b| {
                distance(metric, row(node), row(a)).total_cmp(&distance(metric, row(node), row(b)))
            })
            .unwrap();
        found += usize::from(neighbors.contains(&u32::try_from(nearest).unwrap()));
    }
    // A sanity bound, not a quality bar: a random graph of degree 64 holds
    // the nearest in about 1.3% of rows, while the service's settings (5 of
    // 70 lists probed, 8 PQ subspaces, no refinement) leave it out of about
    // a third. Parity with the service is measured against its baseline
    // (`examples/build_inputs.rs`), not here.
    assert!(
        found * 2 >= SAMPLE,
        "{found} of {SAMPLE} sampled rows hold the nearest"
    );
}

#[test]
fn builds_l2_float32() {
    check(Metric::L2, false);
}

#[test]
fn builds_inner_product_float32() {
    check(Metric::InnerProduct, false);
}

#[test]
fn builds_l2_float16() {
    check(Metric::L2, true);
}

#[test]
fn refuses_partial_rows() {
    let values = random_vectors(100, 1);
    let params = BuildParams::new(10, 10, 16, Metric::L2).unwrap();
    assert_eq!(
        build(HostVectors::F32(&values[..99]), 10, &params),
        Err(Error::Vectors {
            values: 99,
            dimension: 10
        })
    );
}
