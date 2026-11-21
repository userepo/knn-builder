//! Writes the index a graph-only file would be with its vectors stored:
//! the file's own graph, doc ids and settings, and the vectors of the blob
//! the file was built from.
//!
//! A file built with `skip_stored_vectors` holds no vectors, so knn-inspect
//! cannot search its graph over them. `tools/bench/recall_parity.py` writes
//! this copy to hold knn-builder's recall check of such a job to
//! `knn-inspect recall`: the check searches the job's graph over its full
//! vectors, which is this file's search.
//!
//! ```text
//! with_vectors GRAPH_ONLY.faiss VECTORS.knnvec OUT.faiss
//! ```
//!
//! The blob holds what the build request named: fp32 values for a float
//! file, fp16 or fp32 values for an fp16 one (told apart by the blob's
//! size, fp32 converted as the service converts it), int8 values, or a
//! binary file's packed bits.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "a tool for the benchmark: failing loudly is the point"
)]

use std::fs::{self, File};
use std::io::{BufWriter, Write as _};
use std::process::ExitCode;

use knn_format::faiss::{HeaderKind, Metric as FileMetric};
use knn_write::{IndexFile, IndexSpec, Metric, Vectors};

/// Faiss's `NumericType` values, as the file stores its vectors' type.
const FLOAT32: i32 = 0;
const FLOAT16: i32 = 1;
const INT8: i32 = 3;

/// The blob's values as the file stores them.
enum Held {
    F32(Vec<f32>),
    F16(Vec<u16>),
    I8(Vec<i8>),
    Bits(Vec<u8>),
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [index, blob, out] = args.as_slice() else {
        eprintln!("usage: with_vectors GRAPH_ONLY.faiss VECTORS.knnvec OUT.faiss");
        return ExitCode::from(2);
    };
    let bytes = fs::read(index).unwrap();
    let parsed = knn_format::parse(&bytes).unwrap();
    let hnsw = parsed.index.hnsw().expect("a graph index");
    assert!(hnsw.storage.is_none(), "{index} stores its vectors already");
    let cagra = hnsw.kind.cagra().expect("a CAGRA graph");
    let nodes = hnsw.graph.node_count();
    let dimension = usize::try_from(hnsw.header.d).unwrap();
    let binary = matches!(hnsw.header.kind, HeaderKind::Binary { .. });

    // Level 0 as the file holds it: the degree's slots a node, each a node.
    let degree = hnsw.graph.neighbors_of(0, 0).unwrap().len();
    let mut graph = Vec::with_capacity(nodes * degree);
    for node in 0..nodes {
        let slots = hnsw.graph.neighbors_of(node, 0).unwrap();
        assert_eq!(slots.len(), degree, "node {node}");
        graph.extend(slots.iter().map(|slot| u32::try_from(slot.get()).unwrap()));
    }
    let ids: Vec<i32> = parsed
        .index
        .id_map
        .iter()
        .map(|id| i32::try_from(id.get()).unwrap())
        .collect();

    let values = nodes * dimension;
    let blob = fs::read(blob).unwrap();
    let held = if binary {
        Held::Bits(blob)
    } else {
        let typed = match cagra.numeric_type {
            Some(FLOAT32) => Held::F32(floats(&blob)),
            Some(FLOAT16) if blob.len() == values * 2 => Held::F16(
                blob.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&b| u16::from_le_bytes(b))
                    .collect(),
            ),
            Some(FLOAT16) => Held::F16(
                blob.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| half::f16::from_f32(f32::from_le_bytes(b)).to_bits())
                    .collect(),
            ),
            Some(INT8) => Held::I8(blob.iter().map(|&b| i8::from_le_bytes([b])).collect()),
            other => panic!("numeric type {other:?}"),
        };
        // Typed, the blob's bytes are no longer held.
        drop(blob);
        typed
    };
    let file = IndexFile::new(IndexSpec {
        dimension,
        metric: match hnsw.header.metric() {
            FileMetric::L2 => Metric::L2,
            FileMetric::InnerProduct => Metric::InnerProduct,
            FileMetric::Hamming => Metric::Hamming,
            FileMetric::Other(other) => panic!("metric {other}"),
        },
        graph: &graph,
        graph_degree: degree,
        vectors: match &held {
            Held::F32(values) => Vectors::F32(values),
            Held::F16(bits) => Vectors::F16(bits),
            Held::I8(values) => Vectors::I8(values),
            Held::Bits(bits) => Vectors::Bits(bits),
        },
        ids: &ids,
        ef_construction: hnsw.graph.ef_construction,
        ef_search: hnsw.graph.ef_search,
        skip_stored_vectors: false,
    })
    .unwrap();
    // Straight into the file: no copy of it in memory.
    let mut written = BufWriter::with_capacity(1 << 20, File::create(out).unwrap());
    file.write_to(&mut written).unwrap();
    written.flush().unwrap();
    ExitCode::SUCCESS
}

/// A blob's little-endian fp32 values.
fn floats(blob: &[u8]) -> Vec<f32> {
    blob.as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}
