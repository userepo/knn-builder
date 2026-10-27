//! The build parameters and the fp16 conversion against the upstream
//! service's own results, from `tools/oracle/build_tables.py`.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use knn_gpu::{BuildParams, Metric, fp16};

const PARAMS: &str = include_str!("data/params.txt");
const FP16: &str = include_str!("data/fp16.txt");

fn rows(data: &'static str, kind: &'static str) -> impl Iterator<Item = Vec<u32>> {
    data.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .filter(move |fields| fields.first() == Some(&kind))
        .map(|fields| fields[1..].iter().map(|f| f.parse().unwrap()).collect())
}

#[test]
fn lists_and_lowered_m_match_the_service() {
    let m_values = rows(PARAMS, "m").next().unwrap();
    let mut checked = 0;
    for row in rows(PARAMS, "n") {
        let (doc_count, n_lists, effective) = (row[0], row[1], &row[2..]);
        assert_eq!(effective.len(), m_values.len());
        for (&m, &expected) in m_values.iter().zip(effective) {
            let params = BuildParams::new(doc_count, 128, m, Metric::L2).unwrap();
            assert_eq!(params.n_lists, n_lists, "doc_count {doc_count}: n_lists");
            assert_eq!(
                params.effective_m, expected,
                "doc_count {doc_count}, m {m}: effective m"
            );
            assert_eq!(params.graph_degree, expected * 4);
            checked += 1;
        }
    }
    assert!(checked > 40_000, "{checked} cases");
}

#[test]
fn pq_dim_matches_the_service() {
    let mut checked = 0;
    for row in rows(PARAMS, "pq") {
        let params = BuildParams::new(1000, row[0], 16, Metric::L2).unwrap();
        assert_eq!(params.pq_dim, row[1], "dimension {}", row[0]);
        checked += 1;
    }
    assert_eq!(checked, 15);
}

#[test]
fn fp16_matches_numpy() {
    let mut checked = 0;
    for line in FP16.lines().filter(|line| !line.starts_with('#')) {
        let (from, to) = line.split_once(' ').unwrap();
        let value = f32::from_bits(u32::from_str_radix(from, 16).unwrap());
        let expected = u16::from_str_radix(to, 16).unwrap();
        assert_eq!(
            fp16::f32_to_f16(value),
            expected,
            "{value:e} ({from}): {expected:04x} expected"
        );
        checked += 1;
    }
    assert!(checked > 6_900, "{checked} values");
}
