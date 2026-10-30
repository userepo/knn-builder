//! Random small specs, written and read back with knn-inspect's reader
//! (knn-format) and checks (knn-check): every field must come back as
//! given or as the layout fixes it, and `verify` must find nothing. Binary
//! specs take odd degrees too, and keep their vectors: knn-format reads no
//! binary graph without them.
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
use knn_format::faiss::{CodeFormat, GraphKind, HeaderKind, HnswGraph, Inner, Storage};
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

/// The vectors' type, and so the storage's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    F32,
    F16,
    I8,
    /// Packed bits; the case's dimension counts bits.
    Bits,
}

#[derive(Debug, Clone)]
struct Case {
    n: usize,
    degree: usize,
    dimension: usize,
    metric: Metric,
    kind: Kind,
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
                prop_oneof![Just(Kind::F32), Just(Kind::F16), Just(Kind::I8)],
                any::<bool>(),
                1_i32..=512,
                1_i32..=512,
                1_u64..,
            )
        })
        .prop_map(
            |(n, degree, dimension, metric, kind, skip, ef_construction, ef_search, seed)| Case {
                n,
                degree,
                dimension,
                metric,
                kind,
                skip,
                ef_construction,
                ef_search,
                seed,
            },
        )
}

/// A case's graph, its vectors in every type, and its ids.
struct Data {
    graph: Vec<u32>,
    f32s: Vec<f32>,
    f16s: Vec<u16>,
    i8s: Vec<i8>,
    bits: Vec<u8>,
    ids: Vec<i32>,
}

/// Binary cases: Hamming, any degree from 4 to the vectors less one, odd
/// ones too, and 8 to 96 bits.
fn binary_cases() -> impl Strategy<Value = Case> {
    (6_usize..=40)
        .prop_flat_map(|n| {
            (
                Just(n),
                4..=(n - 1).min(17),
                1_usize..=12,
                1_i32..=512,
                1_i32..=512,
                1_u64..,
            )
        })
        .prop_map(
            |(n, degree, bytes, ef_construction, ef_search, seed)| Case {
                n,
                degree,
                dimension: bytes * 8,
                metric: Metric::Hamming,
                kind: Kind::Bits,
                skip: false,
                ef_construction,
                ef_search,
                seed,
            },
        )
}

/// A graph of distinct neighbors other than the node itself, vectors and
/// distinct ids, all from the case's seed.
fn data(case: &Case) -> Data {
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
    // Every int8 value, -128 and 127 included, from 8 random bits each.
    let i8s = (0..case.n * case.dimension)
        .map(|_| i8::from_le_bytes([u8::try_from(rng.next() >> 56).unwrap()]))
        .collect();
    let bits = (0..case.n * case.dimension / 8)
        .map(|_| u8::try_from(rng.next() >> 56).unwrap())
        .collect();
    let mut ids: Vec<i32> = (0..case.n)
        .map(|i| i32::try_from(i * 3 + 7).unwrap())
        .collect();
    for i in (1..ids.len()).rev() {
        ids.swap(i, rng.below(i + 1));
    }
    Data {
        graph,
        f32s,
        f16s,
        i8s,
        bits,
        ids,
    }
}

fn check(case: &Case) {
    let data = data(case);
    let (graph, ids) = (&data.graph, &data.ids);
    let vectors = match case.kind {
        Kind::F32 => Vectors::F32(&data.f32s),
        Kind::F16 => Vectors::F16(&data.f16s),
        Kind::I8 => Vectors::I8(&data.i8s),
        Kind::Bits => Vectors::Bits(&data.bits),
    };
    let binary = case.kind == Kind::Bits;
    let file = IndexFile::new(IndexSpec {
        dimension: case.dimension,
        metric: case.metric,
        graph,
        graph_degree: case.degree,
        vectors,
        ids,
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
    // A binary index records Faiss's default, METRIC_L2.
    let metric = match case.metric {
        Metric::L2 | Metric::Hamming => 1,
        Metric::InnerProduct => 0,
    };
    let header_kind = |kind: HeaderKind| match kind {
        HeaderKind::Binary { code_size } => Some(code_size),
        HeaderKind::Float { .. } => None,
    };
    let code_size = binary.then(|| i32::try_from(case.dimension / 8).unwrap());
    let parsed = knn_format::parse(&bytes).unwrap();
    assert_eq!(parsed.trailing, 0);
    assert_eq!(parsed.payload_len, bytes.len());

    // The ID map.
    let map = &parsed.index;
    assert_eq!(map.fourcc.0, if binary { *b"IBMp" } else { *b"IxMp" });
    assert_eq!(header_kind(map.header.kind), code_size);
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
    assert_eq!(index.fourcc.0, if binary { *b"IBHc" } else { *b"IHc2" });
    assert_eq!(header_kind(index.header.kind), code_size);
    assert_eq!((index.header.d, index.header.ntotal), (d, n));
    assert!(index.header.is_trained);
    assert_eq!(index.header.metric_type, metric);
    let GraphKind::Cagra(fields) = &index.kind else {
        panic!("not a CAGRA graph");
    };
    // copyTo leaves keep_max_size_level0 false for a binary graph alone.
    assert_eq!(fields.keep_max_size_level0, !binary);
    assert!(fields.base_level_only);
    assert_eq!(fields.search_entry_points, 32);
    // Faiss's NumericType: Float32 0, Float16 1, Int8 3; a binary graph
    // records none.
    let numeric = match case.kind {
        Kind::F32 => Some(0),
        Kind::F16 => Some(1),
        Kind::I8 => Some(3),
        Kind::Bits => None,
    };
    assert_eq!(fields.numeric_type, numeric);

    check_hnsw(case, &index.graph, graph);

    check_storage(case, index.storage.as_ref(), &data);

    let report = knn_check::verify(&bytes, &Options::default());
    assert_eq!(report.status, Status::Clean, "{report:#?}");
}

/// The HNSW block: Faiss's level table, the graph on level 0, nothing
/// above, and the fields the layout fixes.
/// The storage, or its absence, as the case says the file keeps it.
fn check_storage(case: &Case, storage: Option<&Storage<'_>>, data: &Data) {
    let n = i64::try_from(case.n).unwrap();
    let d = i32::try_from(case.dimension).unwrap();
    // A binary index records Faiss's default, METRIC_L2.
    let metric = match case.metric {
        Metric::L2 | Metric::Hamming => 1,
        Metric::InnerProduct => 0,
    };
    let header_kind = |kind: HeaderKind| match kind {
        HeaderKind::Binary { code_size } => Some(code_size),
        HeaderKind::Float { .. } => None,
    };
    let code_size = (case.kind == Kind::Bits).then(|| i32::try_from(case.dimension / 8).unwrap());
    match (storage, case.skip, case.kind) {
        (None, true, _) => {}
        (Some(Storage::Flat(flat)), false, Kind::F32) => {
            let fourcc = if metric == 1 { *b"IxF2" } else { *b"IxFI" };
            assert_eq!(flat.fourcc.0, fourcc);
            assert_eq!((flat.header.d, flat.header.ntotal), (d, n));
            assert!(flat.header.is_trained);
            assert_eq!(flat.header.metric_type, metric);
            let codes: Vec<f32> = flat.codes.iter().map(|c| c.get()).collect();
            assert_eq!(codes, data.f32s);
        }
        (Some(Storage::Sq(sq)), false, kind @ (Kind::F16 | Kind::I8)) => {
            let (format, width, codes): (_, u64, Vec<u8>) = if kind == Kind::F16 {
                let codes = data.f16s.iter().flat_map(|x| x.to_le_bytes()).collect();
                (CodeFormat::Fp16, 2, codes)
            } else {
                // Each value plus 128, computed apart from the writer's xor.
                let codes = data
                    .i8s
                    .iter()
                    .map(|&x| u8::try_from(i16::from(x) + 128).unwrap())
                    .collect();
                (CodeFormat::Byte, 1, codes)
            };
            assert_eq!(sq.format, format);
            assert_eq!((sq.header.d, sq.header.ntotal), (d, n));
            assert!(sq.header.is_trained);
            assert_eq!(sq.header.metric_type, metric);
            assert_eq!(sq.quantizer.d, case.dimension as u64);
            assert_eq!(sq.quantizer.code_size, width * case.dimension as u64);
            assert_eq!(sq.codes, &codes[..]);
        }
        (Some(Storage::Binary(storage)), false, Kind::Bits) => {
            assert_eq!((storage.header.d, storage.header.ntotal), (d, n));
            assert!(storage.header.is_trained);
            assert_eq!(storage.header.metric_type, 1);
            assert_eq!(header_kind(storage.header.kind), code_size);
            assert_eq!(storage.codes, &data.bits[..]);
        }
        (storage, skip, kind) => panic!("storage {storage:?} for skip {skip}, {kind:?}"),
    }
}

/// Level 0 holds the first `case.degree / 2 * 2` of each row: all of an
/// even one, all but the last of an odd binary one.
fn check_hnsw(case: &Case, hnsw: &HnswGraph<'_>, graph: &[u32]) {
    let table = LevelTable::new(u32::try_from(case.degree / 2).unwrap()).unwrap();
    let kept = case.degree / 2 * 2;
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
        assert_eq!(slots[..kept], row_ids[..kept], "node {node}: level 0");
        assert!(
            slots[kept..].iter().all(|&s| s == -1),
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

    #[test]
    fn written_binary_files_read_back(case in binary_cases()) {
        check(&case);
    }
}
