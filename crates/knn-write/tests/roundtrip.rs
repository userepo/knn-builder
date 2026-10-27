//! Random small specs, written and read back with knn-inspect's reader
//! (knn-format) and checks (knn-check): every field must come back as
//! given or as the layout fixes it, and `verify` must find nothing.
//!
//! knn-format was validated against Faiss independently of this crate, so
//! the round trip is not circular; the golden tests hold the writer to the
//! service's own files.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use knn_check::{Options, Status};
use knn_format::faiss::{CodeFormat, GraphKind, HnswGraph, Inner, Storage};
use knn_write::{IndexFile, IndexSpec, LevelTable, Metric, Vectors};
use proptest::prelude::*;

/// A small generator for the data a case needs from its seed.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).unwrap()
    }

    fn unit(&mut self) -> f32 {
        // 24 random bits, exact in f32.
        let bits = u32::try_from(self.next() >> 40).unwrap();
        #[expect(clippy::cast_precision_loss, reason = "a 24-bit value is exact in f32")]
        let value = bits as f32;
        value / 16_777_216.0 - 0.5
    }
}

#[derive(Debug, Clone)]
struct Case {
    n: usize,
    degree: usize,
    dimension: usize,
    metric: Metric,
    fp16: bool,
    skip: bool,
    ef_construction: i32,
    ef_search: i32,
    seed: u64,
}

fn cases() -> impl Strategy<Value = Case> {
    (6_usize..=40)
        .prop_flat_map(|n| {
            (
                Just(n),
                // Degree 4 up to n - 1, even.
                (2..=((n - 1) / 2).min(8)).prop_map(|m| m * 2),
                1_usize..=12,
                prop_oneof![Just(Metric::L2), Just(Metric::InnerProduct)],
                any::<bool>(),
                any::<bool>(),
                1_i32..=512,
                1_i32..=512,
                1_u64..,
            )
        })
        .prop_map(
            |(n, degree, dimension, metric, fp16, skip, ef_construction, ef_search, seed)| Case {
                n,
                degree,
                dimension,
                metric,
                fp16,
                skip,
                ef_construction,
                ef_search,
                seed,
            },
        )
}

/// A graph of distinct neighbors other than the node itself, vectors and
/// distinct ids, all from the case's seed.
fn data(case: &Case) -> (Vec<u32>, Vec<f32>, Vec<u16>, Vec<i32>) {
    let mut rng = XorShift(case.seed);
    let mut graph = Vec::with_capacity(case.n * case.degree);
    for node in 0..case.n {
        let mut others: Vec<u32> = (0..case.n)
            .filter(|&o| o != node)
            .map(|o| u32::try_from(o).unwrap())
            .collect();
        for i in 0..case.degree {
            let pick = i + rng.below(others.len() - i);
            others.swap(i, pick);
        }
        graph.extend_from_slice(&others[..case.degree]);
    }
    let f32s: Vec<f32> = (0..case.n * case.dimension).map(|_| rng.unit()).collect();
    let f16s = f32s
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let mut ids: Vec<i32> = (0..case.n)
        .map(|i| i32::try_from(i * 3 + 7).unwrap())
        .collect();
    for i in (1..ids.len()).rev() {
        ids.swap(i, rng.below(i + 1));
    }
    (graph, f32s, f16s, ids)
}

fn check(case: &Case) {
    let (graph, f32s, f16s, ids) = data(case);
    let vectors = if case.fp16 {
        Vectors::F16(&f16s)
    } else {
        Vectors::F32(&f32s)
    };
    let file = IndexFile::new(IndexSpec {
        dimension: case.dimension,
        metric: case.metric,
        graph: &graph,
        graph_degree: case.degree,
        vectors,
        ids: &ids,
        ef_construction: case.ef_construction,
        ef_search: case.ef_search,
        skip_stored_vectors: case.skip,
    })
    .unwrap();
    let mut bytes = Vec::new();
    file.write_to(&mut bytes).unwrap();
    assert_eq!(bytes.len() as u64, file.size());

    let n = i64::try_from(case.n).unwrap();
    let d = i32::try_from(case.dimension).unwrap();
    let metric = match case.metric {
        Metric::L2 => 1,
        Metric::InnerProduct => 0,
    };
    let parsed = knn_format::parse(&bytes).unwrap();
    assert_eq!(parsed.trailing, 0);
    assert_eq!(parsed.payload_len, bytes.len());

    // The ID map.
    let map = &parsed.index;
    assert_eq!(map.fourcc.0, *b"IxMp");
    assert_eq!((map.header.d, map.header.ntotal), (d, n));
    assert!(!map.header.is_trained);
    assert_eq!(map.header.metric_type, metric);
    let map_ids: Vec<i64> = map.id_map.iter().map(|id| id.get()).collect();
    let given: Vec<i64> = ids.iter().copied().map(i64::from).collect();
    assert_eq!(map_ids, given);

    // The graph and its CAGRA fields.
    let Inner::Hnsw(index) = &map.index else {
        panic!("not an HNSW index");
    };
    assert_eq!(index.fourcc.0, *b"IHc2");
    assert_eq!((index.header.d, index.header.ntotal), (d, n));
    assert!(index.header.is_trained);
    assert_eq!(index.header.metric_type, metric);
    let GraphKind::Cagra(fields) = &index.kind else {
        panic!("not a CAGRA graph");
    };
    assert!(fields.keep_max_size_level0 && fields.base_level_only);
    assert_eq!(fields.search_entry_points, 32);
    assert_eq!(fields.numeric_type, Some(i32::from(case.fp16)));

    check_hnsw(case, &index.graph, &graph);

    // The storage.
    match (&index.storage, case.skip, case.fp16) {
        (None, true, _) => {}
        (Some(Storage::Flat(flat)), false, false) => {
            let fourcc = if metric == 1 { *b"IxF2" } else { *b"IxFI" };
            assert_eq!(flat.fourcc.0, fourcc);
            assert_eq!((flat.header.d, flat.header.ntotal), (d, n));
            assert!(flat.header.is_trained);
            assert_eq!(flat.header.metric_type, metric);
            let codes: Vec<f32> = flat.codes.iter().map(|c| c.get()).collect();
            assert_eq!(codes, f32s);
        }
        (Some(Storage::Sq(sq)), false, true) => {
            assert_eq!(sq.format, CodeFormat::Fp16);
            assert_eq!((sq.header.d, sq.header.ntotal), (d, n));
            assert!(sq.header.is_trained);
            assert_eq!(sq.header.metric_type, metric);
            assert_eq!(sq.quantizer.d, case.dimension as u64);
            assert_eq!(sq.quantizer.code_size, 2 * case.dimension as u64);
            let codes: Vec<u8> = f16s.iter().flat_map(|x| x.to_le_bytes()).collect();
            assert_eq!(sq.codes, &codes[..]);
        }
        (storage, skip, fp16) => panic!("storage {storage:?} for skip {skip}, fp16 {fp16}"),
    }

    let report = knn_check::verify(&bytes, &Options::default());
    assert_eq!(report.status, Status::Clean, "{report:#?}");
}

/// The HNSW block: Faiss's level table, the graph on level 0, nothing
/// above, and the fields the layout fixes.
fn check_hnsw(case: &Case, hnsw: &HnswGraph<'_>, graph: &[u32]) {
    let table = LevelTable::new(u32::try_from(case.degree / 2).unwrap()).unwrap();
    let probas: Vec<f64> = hnsw.assign_probas.iter().map(|p| p.get()).collect();
    assert_eq!(probas, table.assign_probas());
    let cum: Vec<i32> = hnsw
        .cum_nneighbor_per_level
        .iter()
        .map(|c| c.get())
        .collect();
    assert_eq!(cum, table.cum_nneighbor_per_level());
    let levels: Vec<i32> = hnsw.levels.iter().map(|l| l.get()).collect();
    let drawn: Vec<i32> = table.levels(case.n).map(|node| node.stored).collect();
    assert_eq!(levels, drawn);
    for (node, row) in graph.chunks_exact(case.degree).enumerate() {
        let start = usize::try_from(hnsw.offsets[node].get()).unwrap();
        let end = usize::try_from(hnsw.offsets[node + 1].get()).unwrap();
        let slots: Vec<i32> = hnsw.neighbors[start..end].iter().map(|s| s.get()).collect();
        let row_ids: Vec<i32> = row.iter().map(|&id| i32::try_from(id).unwrap()).collect();
        assert_eq!(slots[..case.degree], row_ids[..], "node {node}: level 0");
        assert!(
            slots[case.degree..].iter().all(|&s| s == -1),
            "node {node}: upper levels"
        );
    }
    assert_eq!((hnsw.entry_point, hnsw.max_level), (-1, -1));
    assert_eq!(
        (hnsw.ef_construction, hnsw.ef_search),
        (case.ef_construction, case.ef_search)
    );
    assert_eq!(hnsw.upper_beam, 1);
}

proptest! {
    #[test]
    fn written_files_read_back(case in cases()) {
        check(&case);
    }
}
