//! The writer against the files the upstream service built
//! (`fixtures/service/`, and `fixtures/byte/service/` and
//! `fixtures/binary/service/` for byte and binary vectors): from each file
//! only the CAGRA graph is taken, everything else comes from the build
//! input (`fixtures/inputs/`, `fixtures/byte/inputs/`,
//! `fixtures/binary/inputs/`), as it reaches the service, and the written
//! file must equal the service's.

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
    if service.starts_with(b"IBMp") {
        return binary_graph_of(service);
    }
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

/// The level-0 graph of a binary file (`IBMp` > `IBHc`), from its bytes:
/// knn-format reads no binary graph without its vectors. The HNSW block
/// follows two 25-byte binary headers and the graph's three CAGRA fields.
fn binary_graph_of(service: &[u8]) -> (Vec<u32>, usize) {
    assert_eq!(&service[25..29], b"IBHc");
    let mut at = 25 + 25 + 1 + 1 + 4;
    let mut vector = |width: usize| {
        let count =
            usize::try_from(u64::from_le_bytes(service[at..at + 8].try_into().unwrap())).unwrap();
        let items = &service[at + 8..at + 8 + count * width];
        at += 8 + count * width;
        items
    };
    vector(8); // assign_probas
    let cum = vector(4);
    let degree = usize::try_from(i32::from_le_bytes(cum[4..8].try_into().unwrap())).unwrap();
    vector(4); // levels
    let offsets: Vec<usize> = le_chunks(vector(8))
        .map(|o| usize::try_from(u64::from_le_bytes(o)).unwrap())
        .collect();
    // Slots above level 0 hold -1; level 0 holds ids.
    let neighbors: Vec<i32> = le_chunks(vector(4)).map(i32::from_le_bytes).collect();
    let rows = offsets[..offsets.len() - 1]
        .iter()
        .flat_map(|&start| neighbors[start..start + degree].iter())
        .map(|&id| u32::try_from(id).unwrap())
        .collect();
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

/// Writes the input `name` of the fixture set `set` (`fixtures/`,
/// `fixtures/byte/` or `fixtures/binary/`) with the graph of its
/// service-built file.
fn rewrite(set: &Path, name: &str) -> (Vec<u8>, Vec<u8>) {
    let input = set.join("inputs").join(name);
    let meta: Value = serde_json::from_slice(&fs::read(input.join("input.json")).unwrap()).unwrap();
    let service = fs::read(set.join("service").join(format!("{name}.faiss"))).unwrap();

    let blob = fs::read(input.join("vectors.knnvec")).unwrap();
    let f32s: Vec<f32> = if meta["blob_dtype"] == "<f4" {
        le_chunks(&blob).map(f32::from_le_bytes).collect()
    } else {
        Vec::new()
    };
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
    let i8s: Vec<i8> = blob.iter().map(|&b| b.cast_signed()).collect();
    let vectors = match meta["data_type"].as_str().unwrap() {
        "half_float" => Vectors::F16(&f16s),
        "byte" => Vectors::I8(&i8s),
        "binary" => Vectors::Bits(&blob),
        _ => Vectors::F32(&f32s),
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
            "hamming" => Metric::Hamming,
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
    assert_same_in(&fixtures(), name);
}

fn assert_same_byte(name: &str) {
    assert_same_in(&fixtures().join("byte"), name);
}

fn assert_same_binary(name: &str) {
    assert_same_in(&fixtures().join("binary"), name);
}

fn assert_same_in(set: &Path, name: &str) {
    let (ours, theirs) = rewrite(set, name);
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
fn l2_byte() {
    assert_same_byte("l2-1000x64-byte");
}

#[test]
fn inner_product_byte() {
    assert_same_byte("ip-1000x64-byte");
}

#[test]
fn lowered_m_byte() {
    assert_same_byte("l2-150x64-byte");
}

#[test]
fn skip_stored_vectors_byte() {
    assert_same_byte("l2-1000x64-byte-skip");
}

#[test]
fn binary_uniform() {
    assert_same_binary("hamming-1000x256-binary");
}

#[test]
fn binary_clustered() {
    assert_same_binary("hamming-1000x256-binary-clustered");
}

#[test]
fn binary_1024_bits() {
    assert_same_binary("hamming-1000x1024-binary-clustered");
}

#[test]
fn skip_stored_vectors_binary() {
    assert_same_binary("hamming-1000x256-binary-skip");
}

#[test]
fn binary_kept_m() {
    assert_same_binary("hamming-150x256-binary");
}

/// cuVS lowered the degree to 39; level 0 holds 38. The file gives only
/// those 38 of each node's neighbors, so this rewrites an even graph; that
/// `copyTo` kept the first 38 of the 39 rests on reading it, and
/// `writes_an_odd_binary_degree_as_its_even_part` holds the writer to that.
#[test]
fn binary_lowered_degree() {
    assert_same_binary("hamming-40x256-binary");
}

/// The service-built files of a fixture set, by input name.
fn service_files(set: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(set.join("service"))
        .unwrap()
        .filter_map(|entry| {
            let name = entry.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".faiss").map(str::to_owned)
        })
        .collect();
    names.sort();
    names
}

#[test]
fn every_fixture_is_covered() {
    assert_eq!(
        service_files(&fixtures()),
        [
            "ip-1000x64",
            "l2-1000x64",
            "l2-1000x64-fp16",
            "l2-1000x64-fp16from32",
            "l2-1000x64-skip",
            "l2-150x64"
        ]
    );
    assert_eq!(
        service_files(&fixtures().join("byte")),
        [
            "ip-1000x64-byte",
            "l2-1000x64-byte",
            "l2-1000x64-byte-skip",
            "l2-150x64-byte"
        ]
    );
    assert_eq!(
        service_files(&fixtures().join("binary")),
        [
            "hamming-1000x1024-binary-clustered",
            "hamming-1000x256-binary",
            "hamming-1000x256-binary-clustered",
            "hamming-1000x256-binary-skip",
            "hamming-150x256-binary",
            "hamming-40x256-binary"
        ]
    );
}
