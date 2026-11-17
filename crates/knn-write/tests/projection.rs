//! The projection's steps against `tests/data/projection.json`, which
//! `tools/oracle/projection.py` wrote from the upstream service's own
//! module and the numpy and Faiss it runs on.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use knn_write::projection::{
    Insertion, LinkSettings, Settings, build_layer1, host_bytes, max_reachable_share, probes,
    project_links, promoted,
};
use knn_write::{LevelTable, Metric};
use serde_json::Value;

fn data() -> Value {
    serde_json::from_str(include_str!("data/projection.json")).unwrap()
}

fn u32s(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

fn field(record: &Value, key: &str) -> u64 {
    record[key].as_u64().unwrap()
}

/// The oracle's `lcg`: Knuth's MMIX constants, the top 31 bits a draw.
struct Lcg(u64);

impl Iterator for Lcg {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        Some(self.0 >> 33)
    }
}

/// The oracle's `graph`, its empty slots as `u32::MAX` (Faiss's -1).
fn graph(record: &Value) -> (Vec<u32>, usize) {
    let n = field(record, "n");
    let degree = field(record, "degree");
    let clusters = field(record, "clusters");
    let bridge_every = field(record, "bridge_every");
    let empty_every = field(record, "empty_every");
    let mut draws = Lcg(field(record, "seed"));
    let size = n.div_ceil(clusters);
    let mut out = Vec::new();
    for i in 0..n {
        let low = i / size * size;
        let high = n.min(low + size);
        for s in 0..degree {
            let slot = i * degree + s;
            let draw = draws.next().unwrap();
            out.push(if empty_every != 0 && slot.is_multiple_of(empty_every) {
                u32::MAX
            } else if bridge_every != 0 && slot.is_multiple_of(bridge_every) {
                u32::try_from(draw % n).unwrap()
            } else {
                u32::try_from(low + draw % (high - low)).unwrap()
            });
        }
    }
    (out, usize::try_from(degree).unwrap())
}

#[test]
fn probes_the_nodes_numpy_draws() {
    let data = data();
    let expected = data["probes"].as_object().unwrap();
    assert!(expected.len() > 260);
    for (n, want) in expected {
        let n: u32 = n.parse().unwrap();
        assert_eq!(probes(n), u32s(want), "n = {n}");
    }
}

#[test]
fn promotes_the_nodes_faiss_draws() {
    let data = data();
    let mut checked = 0;
    for record in data["promoted"].as_array().unwrap() {
        let (n, m) = (field(record, "n"), field(record, "m"));
        let ids = promoted(u32::try_from(n).unwrap(), u32::try_from(m).unwrap());
        if let Some(want) = record.get("ids") {
            assert_eq!(ids, u32s(want), "n = {n}, m = {m}");
        } else {
            let label = format!("n = {n}, m = {m}");
            assert_eq!(ids.len() as u64, field(record, "count"), "{label}");
            let sum: u64 = ids.iter().map(|&id| u64::from(id)).sum();
            assert_eq!(sum, field(record, "sum"), "{label}");
            assert_eq!(ids[..10], u32s(&record["first"]), "{label}");
            assert_eq!(ids[ids.len() - 10..], u32s(&record["last"]), "{label}");
        }
        checked += 1;
    }
    assert_eq!(checked, 21);
}

#[test]
fn reaches_the_share_the_service_measures() {
    let data = data();
    let mut checked = 0;
    for record in data["gate"].as_array().unwrap() {
        let (graph, degree) = graph(record);
        // The graph is the oracle's: its first row matches.
        let first: Vec<u32> = record["first_row"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let id = v.as_i64().unwrap();
                u32::try_from(id).unwrap_or(u32::MAX)
            })
            .collect();
        assert_eq!(graph[..degree], first, "{record}");
        for (stop_at, want) in record["shares"].as_object().unwrap() {
            let stop_at: f64 = stop_at.parse().unwrap();
            let share = max_reachable_share(&graph, degree, stop_at).unwrap();
            assert_eq!(
                share.to_bits(),
                want.as_f64().unwrap().to_bits(),
                "n = {}, clusters = {}, stop at {stop_at}: {share}",
                record["n"],
                record["clusters"]
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 30);
}

#[test]
fn promotes_with_faiss_s_probability_for_every_m() {
    // Layer 1's M is the request's m, odd ones too, where the file's level
    // table is for 2m; the level-0 probability alone decides a promotion.
    let data = data();
    let expected = data["level0"].as_object().unwrap();
    assert_eq!(expected.len(), 2047);
    for (m, bits) in expected {
        let m: u32 = m.parse().unwrap();
        let want = u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap();
        let table = LevelTable::new(m).unwrap();
        assert_eq!(table.assign_probas()[0].to_bits(), want, "m = {m}");
    }
}

/// A string of integers, space separated, as the oracle writes long lists;
/// negative ones (Faiss's empty slots) as `u32::MAX`.
fn ids(value: &Value) -> Vec<u32> {
    value
        .as_str()
        .unwrap()
        .split_whitespace()
        .map(|v| {
            let v: i64 = v.parse().unwrap();
            u32::try_from(v).unwrap_or(u32::MAX)
        })
        .collect()
}

#[test]
fn projects_the_links_the_service_projects() {
    let data = data();
    let mut checked = 0;
    for case in data["links"].as_array().unwrap() {
        let (base, degree) = graph(case);
        let promoted = ids(&case["promoted"]);
        let layer1 = ids(&case["layer1"]);
        let layer1_degree = usize::try_from(field(case, "layer1_degree")).unwrap();
        for result in case["results"].as_array().unwrap() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the oracle's shares are float32 values"
            )]
            let settings = LinkSettings {
                max_share: result["max_share"].as_f64().unwrap() as f32,
                protect_last_in_edge: result["protect_last_in_edge"].as_bool().unwrap(),
            };
            let label = format!("{}, {settings:?}", case["name"]);
            let mut projected = base.clone();
            let written = project_links(
                &mut projected,
                degree,
                &promoted,
                &layer1,
                layer1_degree,
                &settings,
            )
            .unwrap();
            assert_eq!(written as u64, field(result, "written"), "{label}");
            let changes: Vec<u32> = projected
                .iter()
                .zip(&base)
                .enumerate()
                .filter(|(_, (after, before))| after != before)
                .flat_map(|(index, (&after, _))| [u32::try_from(index).unwrap(), after])
                .collect();
            assert_eq!(changes, ids(&result["changes"]), "{label}");
            checked += 1;
        }
    }
    assert_eq!(checked, 24);
}

/// The oracle's `vectors`: row after row, each value from an LCG draw r.
fn vectors(record: &Value) -> Vec<f32> {
    let n = field(record, "n");
    let d = field(record, "dimension");
    let clusters = field(record, "clusters");
    let kind = record["kind"].as_str().unwrap();
    let mut draws = Lcg(field(record, "seed"));
    let mut out = Vec::new();
    for i in 0..n {
        #[expect(clippy::cast_precision_loss, reason = "small whole numbers")]
        let cluster = (i % clusters) as f32;
        for _ in 0..d {
            let r = draws.next().unwrap();
            #[expect(clippy::cast_precision_loss, reason = "small whole numbers")]
            let value = match kind {
                "float" => cluster * 7.0 + ((r % 20001) as f32 - 10000.0) / 997.0,
                "coarse" => cluster * 7.0 + ((r % 4097) as f32 - 2048.0) / 64.0,
                _ => cluster * 20.0 + (r % 256) as f32 - 128.0,
            };
            out.push(value);
        }
    }
    out
}

#[test]
fn builds_the_layer_faiss_builds_on_one_thread() {
    let data = data();
    let mut checked = 0;
    for record in data["layer1"].as_array().unwrap() {
        let x = vectors(record);
        let d = usize::try_from(field(record, "dimension")).unwrap();
        let first: Vec<f32> = record["first_row"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the oracle wrote float32 values"
                )]
                let value = v.as_f64().unwrap() as f32;
                value
            })
            .collect();
        assert_eq!(x[..d], first, "{record}");
        let metric = match record["metric"].as_str().unwrap() {
            "l2" => Metric::L2,
            _ => Metric::InnerProduct,
        };
        let m = usize::try_from(field(record, "m")).unwrap();
        let ef = usize::try_from(field(record, "ef_construction")).unwrap();
        let graph = build_layer1(&x, d, metric, m, ef, Insertion::Sequential).unwrap();
        let want = ids(&record["graph"]);
        let label = format!(
            "n {}, d {d}, {} {}, m {m}, ef {ef}",
            record["n"], record["kind"], record["metric"]
        );
        if let Some(row) = (0..want.len() / m).find(|&r| graph[r * m..][..m] != want[r * m..][..m])
        {
            panic!(
                "{label}: row {row} is {:?}, Faiss's {:?}",
                &graph[row * m..][..m],
                &want[row * m..][..m]
            );
        }
        assert_eq!(graph, want, "{label}");
        checked += 1;
    }
    assert_eq!(checked, 13);
}

/// Batched layer 1 on `threads` threads for the oracle's case `record`.
fn batched(record: &Value, x: &[f32], threads: usize) -> Vec<u32> {
    let d = usize::try_from(field(record, "dimension")).unwrap();
    let m = usize::try_from(field(record, "m")).unwrap();
    let ef = usize::try_from(field(record, "ef_construction")).unwrap();
    let metric = if record["metric"] == "l2" {
        Metric::L2
    } else {
        Metric::InnerProduct
    };
    let threads = std::num::NonZeroUsize::new(threads).unwrap();
    build_layer1(x, d, metric, m, ef, Insertion::Batched { threads }).unwrap()
}

#[test]
fn batches_to_one_layer_whatever_the_threads() {
    let data = data();
    for record in data["layer1"].as_array().unwrap() {
        let x = vectors(record);
        let one = batched(record, &x, 1);
        for threads in [2, 3, 8] {
            assert_eq!(
                batched(record, &x, threads),
                one,
                "{threads} threads, {record}"
            );
        }
    }
}

#[test]
fn batches_to_a_layer_close_to_faiss_s() {
    // Faiss's own threads see part of each other's work, so its layer is
    // not one graph either; held to it on what the projection uses: as
    // many links, as close, reaching as far. Not for M below 4 at a tiny
    // efConstruction, where any change of order makes another graph.
    let data = data();
    let mut checked = 0;
    for record in data["layer1"].as_array().unwrap() {
        let width = usize::try_from(field(record, "m")).unwrap();
        let count = usize::try_from(field(record, "n")).unwrap();
        if width < 4 || count < 100 {
            continue;
        }
        let values = vectors(record);
        let dim = usize::try_from(field(record, "dimension")).unwrap();
        let l2 = record["metric"] == "l2";
        let faiss = ids(&record["graph"]);
        let ours = batched(record, &values, 4);
        let score = |a: usize, b: usize| -> f64 {
            let (va, vb) = (&values[a * dim..][..dim], &values[b * dim..][..dim]);
            va.iter()
                .zip(vb)
                .map(|(p, q)| {
                    if l2 {
                        f64::from(p - q).powi(2)
                    } else {
                        f64::from(p * q)
                    }
                })
                .sum()
        };
        let links = |g: &[u32]| -> Vec<(usize, usize)> {
            (0..count)
                .flat_map(|r| {
                    g[r * width..][..width]
                        .iter()
                        .filter(|&&v| v != u32::MAX)
                        .map(move |&v| (r, v as usize))
                })
                .collect()
        };
        #[expect(clippy::cast_precision_loss, reason = "counts of a few thousand")]
        let mean = |g: &[u32]| {
            let pairs = links(g);
            pairs.iter().map(|&(a, b)| score(a, b)).sum::<f64>() / pairs.len() as f64
        };
        #[expect(clippy::cast_precision_loss, reason = "counts of a few thousand")]
        let reach = |g: &[u32]| {
            let mut seen = vec![false; count];
            seen[0] = true;
            let mut stack = vec![0];
            let mut reached = 1_u32;
            while let Some(u) = stack.pop() {
                for &v in &g[u * width..][..width] {
                    if v != u32::MAX && !seen[v as usize] {
                        seen[v as usize] = true;
                        reached += 1;
                        stack.push(v as usize);
                    }
                }
            }
            f64::from(reached) / count as f64
        };
        let label = format!("{record}");
        #[expect(clippy::cast_precision_loss, reason = "counts of a few thousand")]
        let ratio = links(&ours).len() as f64 / links(&faiss).len() as f64;
        assert!((0.98..=1.02).contains(&ratio), "links {ratio}: {label}");
        // Worse by at most 5%: larger for L2, smaller for inner product.
        let (ours_mean, faiss_mean) = (mean(&ours), mean(&faiss));
        let worse = if l2 {
            ours_mean / faiss_mean - 1.0
        } else {
            1.0 - ours_mean / faiss_mean
        };
        assert!(
            worse <= 0.05,
            "mean link score {ours_mean} vs {faiss_mean}: {label}"
        );
        assert!(
            (reach(&ours) - reach(&faiss)).abs() <= 0.02,
            "reach: {label}"
        );
        checked += 1;
    }
    assert_eq!(checked, 9);
}

/// What `tests/projection_cost.rs` measured the projection to take.
const COST: &str = include_str!("../../../fixtures/grid/projection.json");

#[test]
fn bounds_the_memory_it_was_measured_to_take() {
    let records: Value = serde_json::from_str(COST).unwrap();
    let mut checked = 0;
    for record in records.as_array().unwrap() {
        let number = |key: &str| field(record, key);
        let settings = Settings {
            m: u32::try_from(number("m")).unwrap(),
            ef_construction: usize::try_from(number("ef_construction")).unwrap(),
            insertion: Insertion::Batched {
                threads: std::num::NonZeroUsize::new(usize::try_from(number("threads")).unwrap())
                    .unwrap(),
            },
            ..Settings::default()
        };
        let bound = host_bytes(
            number("n"),
            number("degree"),
            number("dimension"),
            &settings,
        );
        let peak = number("peak_bytes");
        assert!(bound >= peak, "{bound} under the peak: {record}");
        // Within four times of it where layer 1 was built; a connected graph
        // takes the gate's buffers alone, where the bound counts layer 1's.
        if record["outcome"] == "projected" {
            assert!(bound <= 4 * peak, "{bound}: {record}");
        }
        checked += 1;
    }
    assert!(checked >= 55, "{checked} records");
}
