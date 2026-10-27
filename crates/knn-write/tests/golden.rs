//! The writer against the files the upstream service built
//! (`fixtures/service/`): from each file only the CAGRA graph is taken,
//! everything else comes from the build input (`fixtures/inputs/`), as it
//! reaches the service, and the written file must equal the service's.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::fs;
use std::path::{Path, PathBuf};

use knn_format::faiss::Inner;
use knn_write::{IndexFile, IndexSpec, Metric, Vectors};
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// The level-0 graph of a service-built file, row-major.
fn graph_of(service: &[u8]) -> (Vec<u32>, usize) {
    let parsed = knn_format::parse(service).unwrap();
    let Inner::Hnsw(index) = parsed.index.index else {
        panic!("not an HNSW index");
    };
    let graph = index.graph;
    let degree = usize::try_from(graph.cum_nneighbor_per_level[1].get()).unwrap();
    let mut rows = Vec::new();
    for offset in &graph.offsets[..graph.offsets.len() - 1] {
        let start = usize::try_from(offset.get()).unwrap();
        rows.extend(
            graph.neighbors[start..start + degree]
                .iter()
                .map(|id| u32::try_from(id.get()).unwrap()),
        );
    }
    (rows, degree)
}

fn le_chunks<const N: usize>(bytes: &[u8]) -> impl Iterator<Item = [u8; N]> + '_ {
    let (chunks, rest) = bytes.as_chunks::<N>();
    assert!(
        rest.is_empty(),
        "a blob of {} bytes is not whole values",
        bytes.len()
    );
    chunks.iter().copied()
}

/// Writes the input `name` with the graph of its service-built file.
fn rewrite(name: &str) -> (Vec<u8>, Vec<u8>) {
    let input = fixtures().join("inputs").join(name);
    let meta: Value = serde_json::from_slice(&fs::read(input.join("input.json")).unwrap()).unwrap();
    let service = fs::read(fixtures().join("service").join(format!("{name}.faiss"))).unwrap();

    let blob = fs::read(input.join("vectors.knnvec")).unwrap();
    let f32s: Vec<f32> = le_chunks(&blob).map(f32::from_le_bytes).collect();
    let f16s: Vec<u16> = match (meta["data_type"].as_str(), meta["blob_dtype"].as_str()) {
        (Some("half_float"), Some("<f2")) => le_chunks(&blob).map(u16::from_le_bytes).collect(),
        // The service converts an fp32 blob under half_float on download,
        // with numpy's round to nearest even, as half does.
        (Some("half_float"), Some("<f4")) => f32s
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect(),
        _ => Vec::new(),
    };
    let vectors = if meta["data_type"] == "half_float" {
        Vectors::F16(&f16s)
    } else {
        Vectors::F32(&f32s)
    };
    let ids: Vec<i32> = le_chunks(&fs::read(input.join("doc_ids.knndid")).unwrap())
        .map(i32::from_le_bytes)
        .collect();
    let (graph, graph_degree) = graph_of(&service);
    let number = |key: &str| i32::try_from(meta[key].as_i64().unwrap()).unwrap();

    let file = IndexFile::new(IndexSpec {
        dimension: usize::try_from(meta["dimension"].as_u64().unwrap()).unwrap(),
        metric: match meta["space_type"].as_str().unwrap() {
            "l2" => Metric::L2,
            "innerproduct" => Metric::InnerProduct,
            other => panic!("space type {other}"),
        },
        graph: &graph,
        graph_degree,
        vectors,
        ids: &ids,
        ef_construction: number("ef_construction"),
        ef_search: number("ef_search"),
        skip_stored_vectors: meta["skip_stored_vectors"].as_bool().unwrap(),
    })
    .unwrap();
    let mut ours = Vec::new();
    file.write_to(&mut ours).unwrap();
    assert_eq!(ours.len() as u64, file.size(), "{name}: size");
    (ours, service)
}

fn assert_same(name: &str) {
    let (ours, theirs) = rewrite(name);
    if let Some(at) = ours.iter().zip(&theirs).position(|(a, b)| a != b) {
        panic!("{name}: first difference at byte {at}");
    }
    assert_eq!(ours.len(), theirs.len(), "{name}: length");
}

#[test]
fn l2_float32() {
    assert_same("l2-1000x64");
}

#[test]
fn inner_product_float32() {
    assert_same("ip-1000x64");
}

#[test]
fn lowered_m() {
    assert_same("l2-150x64");
}

#[test]
fn fp16_blob() {
    assert_same("l2-1000x64-fp16");
}

#[test]
fn fp16_converted_from_fp32() {
    assert_same("l2-1000x64-fp16from32");
}

#[test]
fn skip_stored_vectors() {
    assert_same("l2-1000x64-skip");
}

#[test]
fn every_fixture_is_covered() {
    let mut names: Vec<String> = fs::read_dir(fixtures().join("service"))
        .unwrap()
        .filter_map(|entry| {
            let name = entry.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".faiss").map(str::to_owned)
        })
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "ip-1000x64",
            "l2-1000x64",
            "l2-1000x64-fp16",
            "l2-1000x64-fp16from32",
            "l2-1000x64-skip",
            "l2-150x64"
        ]
    );
}
