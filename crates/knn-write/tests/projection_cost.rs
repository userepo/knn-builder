//! Measures what `project_upper_level` takes, its time and its peak host
//! memory, on synthetic graphs, into `fixtures/grid/projection.json`, which
//! `knn-builder`'s projection model and `projection::host_bytes` are held
//! to. Ignored; in release, where the times mean something, with the
//! fixture's path, from the workspace's root:
//!
//! ```sh
//! PROJECTION_COST_OUT=fixtures/grid/projection.json \
//!     cargo test --release -p knn-write --test projection_cost -- --ignored
//! ```
//!
//! A graph of n nodes, degree 4m, in clusters of n / clusters consecutive
//! nodes, each node linked to random nodes of its own cluster, so no search
//! leaves one; its vectors, of dimension d, are each value its cluster's
//! number times `spacing` plus a uniform draw from 0 to 1. Clusters 100
//! apart are far apart; 0.5 apart they overlap, and layer 1's searches run
//! longer. One cluster makes a connected graph, which the gate leaves as
//! built. A counting allocator takes the projection's peak above what was
//! held before it.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]
#![expect(
    unsafe_code,
    reason = "a global allocator that counts the bytes held, to measure the projection's peak"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use knn_write::projection::{Insertion, Outcome, Settings, project_upper_level};
use knn_write::{Metric, Vectors};
use serde_json::{Value, json};

/// The system allocator, counting the bytes held and their peak.
struct Counting;

static HELD: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let held = HELD.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(held, Ordering::SeqCst);
}

// SAFETY: every call goes to the system allocator with the caller's own
// arguments, so it keeps the system allocator's contract; the counting
// beside it touches no memory the allocator hands out.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's layout, as `GlobalAlloc::alloc` requires.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's layout, as `GlobalAlloc::alloc_zeroed`
        // requires.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: a block this allocator, the system's, handed out with
        // this layout, as the caller guarantees.
        unsafe { System.dealloc(pointer, layout) };
        HELD.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: a block the system allocator handed out with this layout,
        // and a new size the caller guarantees valid.
        let moved = unsafe { System.realloc(pointer, layout, size) };
        if !moved.is_null() {
            if size > layout.size() {
                grew(size - layout.size());
            } else {
                HELD.fetch_sub(layout.size() - size, Ordering::SeqCst);
            }
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The oracle's LCG: Knuth's MMIX constants, the top 31 bits a draw.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

/// One measurement's shape.
#[derive(Clone, Copy)]
struct Case {
    n: usize,
    dimension: usize,
    m: usize,
    ef_construction: usize,
    threads: usize,
    clusters: usize,
    spacing: f32,
}

/// The cases: 100,000 to 1,000,000 vectors of 128 and 768 dimensions at m
/// 8 to 32 on 8 threads, clusters apart and overlapping; efConstruction 50
/// to 400 and 1 to 32 threads at 1,000,000 x m 16; m 2 and 4; and the gate
/// alone on connected graphs.
fn cases() -> Vec<Case> {
    let base = Case {
        n: 1_000_000,
        dimension: 128,
        m: 16,
        ef_construction: 100,
        threads: 8,
        clusters: 20,
        spacing: 0.5,
    };
    let mut cases = Vec::new();
    for spacing in [0.5, 100.0] {
        for n in [100_000, 300_000, 1_000_000] {
            for dimension in [128, 768] {
                for m in [8, 16, 32] {
                    cases.push(Case {
                        n,
                        dimension,
                        m,
                        spacing,
                        ..base
                    });
                }
            }
        }
    }
    for dimension in [128, 768] {
        for ef_construction in [50, 200, 400] {
            cases.push(Case {
                dimension,
                ef_construction,
                ..base
            });
        }
        for threads in [1, 2, 4, 16, 32] {
            cases.push(Case {
                dimension,
                threads,
                ..base
            });
        }
    }
    for m in [2, 4] {
        cases.push(Case { m, ..base });
    }
    for (n, m) in [
        (300_000, 16),
        (1_000_000, 8),
        (1_000_000, 16),
        (1_000_000, 32),
        (3_000_000, 16),
    ] {
        cases.push(Case {
            n,
            dimension: 8,
            m,
            clusters: 1,
            ..base
        });
    }
    cases
}

fn measure(case: Case) -> Value {
    let Case {
        n,
        dimension,
        m,
        ef_construction,
        threads,
        clusters,
        spacing,
    } = case;
    let degree = 4 * m;
    let size = n.div_ceil(clusters);
    let mut draws = Lcg(u64::try_from(n * 1_000 + dimension).unwrap());
    let mut vectors = Vec::with_capacity(n * dimension);
    for i in 0..n {
        #[expect(clippy::cast_precision_loss, reason = "a cluster's number, small")]
        let center = (i / size) as f32 * spacing;
        for _ in 0..dimension {
            #[expect(clippy::cast_precision_loss, reason = "a draw of 12 bits")]
            vectors.push(center + (draws.next() % 4_096) as f32 / 4_096.0);
        }
    }
    let mut graph = Vec::with_capacity(n * degree);
    for i in 0..n {
        let low = i / size * size;
        let high = n.min(low + size);
        for _ in 0..degree {
            let id = low + usize::try_from(draws.next()).unwrap() % (high - low);
            graph.push(u32::try_from(id).unwrap());
        }
    }
    let settings = Settings {
        m: u32::try_from(m).unwrap(),
        ef_construction,
        metric: Metric::L2,
        insertion: Insertion::Batched {
            threads: NonZeroUsize::new(threads).unwrap(),
        },
        ..Settings::default()
    };
    let before = HELD.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    let start = Instant::now();
    let outcome = project_upper_level(
        &mut graph,
        degree,
        Vectors::F32(&vectors),
        dimension,
        &settings,
        &|| false,
    )
    .unwrap();
    let seconds = start.elapsed().as_secs_f64();
    let peak = PEAK.load(Ordering::SeqCst) - before;
    let (outcome, gate, layer1, links_s, promoted, links) = match outcome {
        Outcome::Connected { gate, .. } => ("connected", gate, None, None, 0, 0),
        Outcome::Projected {
            promoted,
            links,
            gate,
            layer1,
            projection,
            ..
        } => (
            "projected",
            gate,
            Some(layer1),
            Some(projection),
            promoted,
            links,
        ),
        other => panic!("{other:?}"),
    };
    let secs = |duration: std::time::Duration| (duration.as_secs_f64() * 1e4).round() / 1e4;
    let line = json!({
        "n": n,
        "dimension": dimension,
        "m": m,
        "degree": degree,
        "ef_construction": ef_construction,
        "threads": threads,
        "clusters": clusters,
        "spacing": spacing,
        "outcome": outcome,
        "gate_s": secs(gate),
        "layer1_s": layer1.map(secs),
        "links_s": links_s.map(secs),
        "seconds": (seconds * 1e4).round() / 1e4,
        "peak_bytes": peak,
        "promoted": promoted,
        "links": links,
    });
    eprintln!("{line}");
    line
}

#[test]
#[ignore = "a measurement of several minutes, for the fixture"]
fn measures_the_projection() {
    let records: Vec<Value> = cases().into_iter().map(measure).collect();
    if let Ok(out) = std::env::var("PROJECTION_COST_OUT") {
        // A relative path from the workspace's root, not the crate's, where
        // cargo runs the test.
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(out);
        let text = serde_json::to_string_pretty(&records).unwrap() + "\n";
        std::fs::write(&path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
}
