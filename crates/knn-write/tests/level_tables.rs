//! The level tables against `tests/data/level_tables.txt`, which
//! `tools/oracle/level_tables.py` computed with glibc's `exp`, as the
//! service's files were written: every M the service can produce (m from
//! 1 to 2048), and level histograms for a spread of M.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use knn_write::LevelTable;

const DATA: &str = include_str!("data/level_tables.txt");

fn lines(kind: &str) -> impl Iterator<Item = Vec<&'static str>> {
    DATA.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .filter(move |fields| fields.first() == Some(&kind))
}

fn number<T: std::str::FromStr>(field: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    field.parse().unwrap()
}

#[test]
fn probabilities_match_for_every_m() {
    let mut count = 0;
    for fields in lines("probas") {
        let m: u32 = number(fields[1]);
        let expected: Vec<f64> = fields[2..]
            .iter()
            .map(|bits| f64::from_bits(u64::from_str_radix(bits, 16).unwrap()))
            .collect();
        let table = LevelTable::new(m).unwrap();
        let ours = table.assign_probas();
        assert_eq!(ours.len(), expected.len(), "M = {m}: level count");
        for (level, (a, b)) in ours.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "M = {m}, level {level}: {a:e} vs {b:e}"
            );
        }
        count += 1;
    }
    assert_eq!(count, 2048);
}

#[test]
fn neighbor_counts_match_for_every_m() {
    let mut count = 0;
    for fields in lines("cum") {
        let m: u32 = number(fields[1]);
        let expected: Vec<i32> = fields[2..].iter().map(|c| number(c)).collect();
        let table = LevelTable::new(m).unwrap();
        assert_eq!(table.cum_nneighbor_per_level(), expected, "M = {m}");
        count += 1;
    }
    assert_eq!(count, 2048);
}

#[test]
fn level_draws_match() {
    let mut count = 0;
    for fields in lines("levels") {
        let m: u32 = number(fields[1]);
        let n: usize = number(fields[2]);
        let slots: u64 = number(fields[3]);
        let checksum: u64 = number(fields[4]);
        let histogram: Vec<u64> = fields[5..].iter().map(|c| number(c)).collect();

        let table = LevelTable::new(m).unwrap();
        let mut ours_histogram = vec![0_u64; histogram.len()];
        let (mut ours_slots, mut ours_checksum) = (0_u64, 0_u64);
        for (i, node) in table.levels(n).enumerate() {
            let stored = u64::try_from(node.stored).unwrap();
            ours_histogram[usize::try_from(stored - 1).unwrap()] += 1;
            ours_slots += u64::from(node.slots);
            ours_checksum = ours_checksum.wrapping_add((i as u64 + 1).wrapping_mul(stored));
        }
        assert_eq!(ours_histogram, histogram, "M = {m}: nodes per level");
        assert_eq!(ours_slots, slots, "M = {m}: neighbor slots");
        assert_eq!(ours_checksum, checksum, "M = {m}: level checksum");
        count += 1;
    }
    assert_eq!(count, 7);
}
