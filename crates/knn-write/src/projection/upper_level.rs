//! The projection as the service runs it on a graph, `project_upper_level`:
//! the gate, the draw, layer 1 and its links, in that order, with each
//! step's time for the log.

use std::time::{Duration, Instant};

use super::{
    Insertion, LinkSettings, ProjectionError, SKIP_IF_REACHABLE, Stop, gate, layer1, links,
    max_reachable_share, node_count, project_links, promoted,
};
use crate::{LevelTable, Metric, Vectors};

/// How a graph is projected. The default is the service's for a request's
/// defaults (m 16, efConstruction 100, L2), on one thread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// The HNSW m the graph was built with, a quarter of its degree: layer
    /// 1 takes one node in m, and links each to m others.
    pub m: u32,
    /// Layer 1's efConstruction: the request's.
    pub ef_construction: usize,
    /// The index's metric. Binary vectors' Hamming is not projected.
    pub metric: Metric,
    /// How layer 1's links go into the graph.
    pub links: LinkSettings,
    /// The share of the nodes reachable at which the graph is left as
    /// built (`skip_if_reachable`); above 1, the gate does not run and the
    /// graph is always projected.
    pub skip_if_reachable: f64,
    /// How layer 1's nodes go in.
    pub insertion: Insertion,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
            metric: Metric::L2,
            links: LinkSettings::default(),
            skip_if_reachable: SKIP_IF_REACHABLE,
            insertion: Insertion::Sequential,
        }
    }
}

/// What the projection did to a graph.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Outcome {
    /// The gate found the graph connected enough, and left it as built.
    Connected {
        /// The largest share of the nodes a probe reached.
        reachable: f64,
        /// The gate's time.
        gate: Duration,
    },
    /// Fewer than two nodes promoted: no link to write, the graph as built.
    TooFewPromoted {
        /// The largest share of the nodes a probe reached; NaN when the
        /// gate did not run.
        reachable: f64,
        /// The nodes promoted.
        promoted: usize,
        /// The gate's time.
        gate: Duration,
    },
    /// Layer 1 built and its links written.
    Projected {
        /// The largest share of the nodes a probe reached before; NaN when
        /// the gate did not run.
        reachable: f64,
        /// The nodes promoted.
        promoted: usize,
        /// The links written.
        links: usize,
        /// The gate's time.
        gate: Duration,
        /// The draw, the promoted vectors' float32 copy and layer 1's
        /// build.
        layer1: Duration,
        /// Writing layer 1's links into the graph.
        projection: Duration,
    },
}

/// Projects layer-1 links into `graph`, rows of `degree` ids over the
/// nodes of `vectors` (rows of `dimension` values), as the service's
/// `project_upper_level` does under `settings`: when the gate finds less
/// than `skip_if_reachable` of the nodes reachable, it promotes one node
/// in m, builds layer 1 over float32 copies of their vectors (half floats
/// and bytes converted exactly, as numpy's `astype(float32)` converts
/// them), and writes its links both ways into their rows. `stop` is asked
/// before each step, every few thousand rows of the copy and before each
/// of layer 1's batches, so a projection whose job is canceled is given up
/// within a batch of layer 1, or once the step under way ends: the gate's
/// search or the links, a pass over the graph each, about 0.2 s for a
/// million nodes.
///
/// # Errors
///
/// Returns a [`ProjectionError`], with `graph` unchanged, when the graph is
/// not whole rows or names a node past the last, `vectors` are not one row
/// of `dimension` a node, they are binary or the metric is Hamming, a
/// buffer cannot be allocated, a step refuses its input, or `stop` said
/// to give up ([`ProjectionError::Stopped`]).
pub fn project_upper_level(
    graph: &mut [u32],
    degree: usize,
    vectors: Vectors<'_>,
    dimension: usize,
    settings: &Settings,
    stop: Stop<'_>,
) -> Result<Outcome, ProjectionError> {
    if stop() {
        return Err(ProjectionError::Stopped);
    }
    let n = node_count(graph, degree)?;
    if matches!(vectors, Vectors::Bits(_)) || settings.metric == Metric::Hamming {
        return Err(ProjectionError::Hamming);
    }
    if dimension == 0 || Some(vectors.len()) != (n as usize).checked_mul(dimension) {
        return Err(ProjectionError::Vectors {
            len: vectors.len(),
            dimension,
            nodes: n as usize,
        });
    }

    let start = Instant::now();
    let reachable = if settings.skip_if_reachable <= 1.0 {
        max_reachable_share(graph, degree, settings.skip_if_reachable)?
    } else {
        f64::NAN
    };
    let gate = start.elapsed();
    if reachable >= settings.skip_if_reachable {
        return Ok(Outcome::Connected { reachable, gate });
    }

    if stop() {
        return Err(ProjectionError::Stopped);
    }
    let start = Instant::now();
    let ids = promoted(n, settings.m);
    if ids.len() < 2 {
        return Ok(Outcome::TooFewPromoted {
            reachable,
            promoted: ids.len(),
            gate,
        });
    }
    let m = settings.m as usize;
    let copy = promoted_rows(vectors, dimension, n as usize, &ids, stop)?;
    let layer1 = layer1::build_layer1_until(
        &copy,
        dimension,
        settings.metric,
        m,
        settings.ef_construction,
        settings.insertion,
        stop,
    )?;
    drop(copy);
    let layer1_time = start.elapsed();

    if stop() {
        return Err(ProjectionError::Stopped);
    }
    let start = Instant::now();
    let links = project_links(graph, degree, &ids, &layer1, m, &settings.links)?;
    Ok(Outcome::Projected {
        reachable,
        promoted: ids.len(),
        links,
        gate,
        layer1: layer1_time,
        projection: start.elapsed(),
    })
}

/// The most host memory [`project_upper_level`] takes at once besides the
/// graph and the vectors, in bytes, for a graph of `nodes` full rows of
/// `degree` (as CAGRA builds them) over vectors of `dimension`, under
/// `settings`: the largest of its steps' buffers, the gate's, layer 1's
/// with the promoted vectors' float32 copy, and the links', with as many
/// nodes promoted as a draw could promote.
#[must_use]
pub fn host_bytes(nodes: u64, degree: u64, dimension: u64, settings: &Settings) -> u64 {
    let gate = gate::host_bytes(nodes);
    let count = promoted_bound(nodes, settings.m);
    if count < 2 {
        return gate;
    }
    let m = u64::from(settings.m);
    let ids = count.saturating_mul(4);
    let ef = u64::try_from(settings.ef_construction).unwrap_or(u64::MAX);
    let layer1 = ids
        .saturating_add(count.saturating_mul(dimension).saturating_mul(4))
        .saturating_add(layer1::host_bytes(count, m, ef, settings.insertion));
    let links = ids
        .saturating_add(count.saturating_mul(m).saturating_mul(4))
        .saturating_add(links::host_bytes(
            nodes,
            degree,
            count,
            m,
            settings.links.max_share,
        ));
    gate.max(layer1).max(links)
}

/// At most the nodes [`promoted`] promotes of `nodes`: their expected count
/// and six standard deviations past it, which a draw passes with a chance
/// below one in a billion.
fn promoted_bound(nodes: u64, m: u32) -> u64 {
    let Some(table) = LevelTable::new(m) else {
        return 0;
    };
    // A node is promoted where its draw is at least level 0's probability.
    let share = 1.0 - table.assign_probas().first().copied().unwrap_or(1.0);
    #[expect(
        clippy::cast_precision_loss,
        reason = "a node count, whose expected share needs no exact integer"
    )]
    let mean = nodes as f64 * share;
    let bound = mean + 6.0 * mean.sqrt() + 6.0;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a positive count, saturating, then held to the node count"
    )]
    let bound = bound.ceil() as u64;
    bound.min(nodes)
}

/// The rows a copy of the promoted vectors takes between two asks of
/// `stop`: a few milliseconds' copying, at most.
const ROWS_A_STOP: usize = 4_096;

/// The rows of `ids`, in that order, as float32: `vectors[ids].astype(float32)`.
/// `vectors` holds one row of `dimension` values for every id. `stop` is
/// asked every [`ROWS_A_STOP`] rows.
fn promoted_rows(
    vectors: Vectors<'_>,
    dimension: usize,
    nodes: usize,
    ids: &[u32],
    stop: Stop<'_>,
) -> Result<Vec<f32>, ProjectionError> {
    let len = ids.len().saturating_mul(dimension);
    let mut out: Vec<f32> = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| ProjectionError::HostMemory {
            bytes: len.saturating_mul(4),
        })?;
    for (i, &id) in ids.iter().enumerate() {
        if i.is_multiple_of(ROWS_A_STOP) && i > 0 && stop() {
            return Err(ProjectionError::Stopped);
        }
        let first = id as usize * dimension;
        let rows = first..first + dimension;
        let missing = || ProjectionError::Vectors {
            len: vectors.len(),
            dimension,
            nodes,
        };
        match vectors {
            Vectors::F32(values) => out.extend_from_slice(values.get(rows).ok_or_else(missing)?),
            Vectors::F16(values) => out.extend(
                values
                    .get(rows)
                    .ok_or_else(missing)?
                    .iter()
                    .map(|&half| f16_to_f32(half)),
            ),
            Vectors::I8(values) => out.extend(
                values
                    .get(rows)
                    .ok_or_else(missing)?
                    .iter()
                    .map(|&byte| f32::from(byte)),
            ),
            Vectors::Bits(_) => return Err(ProjectionError::Hamming),
        }
    }
    Ok(out)
}

/// An IEEE half-precision bit pattern as the float32 of the same value,
/// which holds every one exactly; a NaN keeps its payload.
fn f16_to_f32(half: u16) -> f32 {
    let sign = u32::from(half >> 15) << 31;
    let exponent = u32::from((half >> 10) & 0x1f);
    let fraction = u32::from(half & 0x3ff);
    let magnitude = match exponent {
        // Zero, or a subnormal: the fraction times 2^-24, a float32
        // normal number, exact.
        0 => (f32::from(half & 0x3ff) * f32::from_bits(0x3380_0000)).to_bits(),
        // Infinity, or a NaN.
        0x1f => 0x7f80_0000 | (fraction << 13),
        // Rebiased from 15 to 127.
        _ => ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(sign | magnitude)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::num::NonZeroUsize;

    use super::{Outcome, ROWS_A_STOP, Settings, f16_to_f32, project_upper_level, promoted_rows};
    use crate::projection::{Insertion, ProjectionError, max_reachable_share};
    use crate::{Metric, Vectors};

    #[test]
    fn widens_every_half_float_exactly() {
        for bits in 0..=u16::MAX {
            let want = half::f16::from_bits(bits).to_f32();
            let got = f16_to_f32(bits);
            if want.is_nan() {
                assert!(got.is_nan(), "{bits:#06x}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "{bits:#06x}");
            }
        }
    }

    /// `clusters` clusters of `size` nodes, `dimension` values a node, far
    /// apart: node i in cluster i / size, each value within one unit above
    /// its center, the cluster's number times 100. Its graph links each
    /// node to the next `degree` of its cluster, so no search leaves one.
    fn clustered(
        clusters: usize,
        size: usize,
        dimension: usize,
        degree: usize,
    ) -> (Vec<f32>, Vec<u32>) {
        let mut state: u64 = 7;
        let mut vectors = Vec::new();
        for i in 0..clusters * size {
            #[expect(clippy::cast_precision_loss, reason = "small whole numbers")]
            let center = (i / size) as f32 * 100.0;
            for _ in 0..dimension {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                #[expect(clippy::cast_precision_loss, reason = "a draw of 24 bits")]
                vectors.push(center + (state >> 40) as f32 / 16_777_216.0);
            }
        }
        let graph = (0..clusters * size)
            .flat_map(|i| {
                let first = i / size * size;
                (1..=degree).map(move |k| first + (i - first + k) % size)
            })
            .map(|id| u32::try_from(id).unwrap())
            .collect();
        (vectors, graph)
    }

    #[test]
    fn joins_clusters_a_search_could_not_leave() {
        let (vectors, mut graph) = clustered(8, 250, 8, 16);
        let before = max_reachable_share(&graph, 16, 1.0).unwrap();
        assert!((before - 0.125).abs() < 1e-9, "{before}");
        let settings = Settings {
            m: 4,
            ef_construction: 40,
            ..Settings::default()
        };
        let outcome = project_upper_level(
            &mut graph,
            16,
            Vectors::F32(&vectors),
            8,
            &settings,
            &|| false,
        )
        .unwrap();
        let Outcome::Projected {
            reachable,
            promoted,
            links,
            ..
        } = outcome
        else {
            panic!("{outcome:?}");
        };
        assert!((reachable - 0.125).abs() < 1e-9, "{reachable}");
        assert!((400..600).contains(&promoted), "{promoted}");
        assert!(links > 0);
        let after = max_reachable_share(&graph, 16, 1.0).unwrap();
        assert!(after > 0.99, "{after}");
    }

    #[test]
    fn projects_half_floats_and_bytes_as_their_float_values() {
        // Whole numbers, so all three types hold the same values.
        let (vectors, graph) = clustered(4, 100, 4, 8);
        let whole: Vec<f32> = vectors.iter().map(|v| v.floor()).collect();
        let halves: Vec<u16> = whole
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "whole numbers from -40 to 35"
        )]
        let bytes: Vec<i8> = whole.iter().map(|&v| (v / 4.0 - 40.0) as i8).collect();
        let byte_values: Vec<f32> = bytes.iter().map(|&b| f32::from(b)).collect();
        let settings = Settings {
            m: 4,
            ef_construction: 20,
            ..Settings::default()
        };
        let project = |vectors: Vectors<'_>| {
            let mut graph = graph.clone();
            project_upper_level(&mut graph, 8, vectors, 4, &settings, &|| false).unwrap();
            graph
        };
        assert_eq!(
            project(Vectors::F16(&halves)),
            project(Vectors::F32(&whole))
        );
        assert_eq!(
            project(Vectors::I8(&bytes)),
            project(Vectors::F32(&byte_values))
        );
    }

    #[test]
    fn leaves_a_connected_graph_as_built() {
        let (vectors, _) = clustered(1, 300, 4, 8);
        // A ring: every node reaches every other.
        let mut graph: Vec<u32> = (0..300_u32)
            .flat_map(|i| (1..=8).map(move |k| (i + k) % 300))
            .collect();
        let built = graph.clone();
        let outcome = project_upper_level(
            &mut graph,
            8,
            Vectors::F32(&vectors),
            4,
            &Settings::default(),
            &|| false,
        )
        .unwrap();
        assert!(
            matches!(outcome, Outcome::Connected { reachable, .. } if reachable.to_bits() == 1.0_f64.to_bits()),
            "{outcome:?}"
        );
        assert_eq!(graph, built);
        // Past 1, the gate does not run and the graph is projected anyway.
        let always = Settings {
            m: 4,
            skip_if_reachable: 1.5,
            ..Settings::default()
        };
        let outcome =
            project_upper_level(&mut graph, 8, Vectors::F32(&vectors), 4, &always, &|| false)
                .unwrap();
        assert!(
            matches!(outcome, Outcome::Projected { reachable, .. } if reachable.is_nan()),
            "{outcome:?}"
        );
    }

    #[test]
    fn writes_nothing_with_fewer_than_two_promoted() {
        let (vectors, mut graph) = clustered(2, 50, 4, 8);
        let built = graph.clone();
        // m 1 promotes none: Faiss's level table for it has no level.
        let settings = Settings {
            m: 1,
            ..Settings::default()
        };
        let outcome =
            project_upper_level(&mut graph, 8, Vectors::F32(&vectors), 4, &settings, &|| {
                false
            })
            .unwrap();
        assert!(
            matches!(outcome, Outcome::TooFewPromoted { promoted: 0, .. }),
            "{outcome:?}"
        );
        assert_eq!(graph, built);
    }

    #[test]
    fn gives_up_when_told_to_stop() {
        let (vectors, graph) = clustered(8, 250, 8, 16);
        // The most `stop` is asked: before the gate, the draw and the links,
        // and before each batch of layer 1.
        let asked = |settings: &Settings| {
            let asked = Cell::new(0_u32);
            let stop = || {
                asked.set(asked.get() + 1);
                false
            };
            let mut projected = graph.clone();
            project_upper_level(
                &mut projected,
                16,
                Vectors::F32(&vectors),
                8,
                settings,
                &stop,
            )
            .unwrap();
            asked.get()
        };
        let batched = Settings {
            m: 4,
            ef_construction: 40,
            insertion: Insertion::Batched {
                threads: NonZeroUsize::MIN.saturating_add(1),
            },
            ..Settings::default()
        };
        let sequential = Settings {
            insertion: Insertion::Sequential,
            ..batched
        };
        for settings in [batched, sequential] {
            let most = asked(&settings);
            assert!(most > 10, "{most}");
            // Told to stop at any point, it gives up with the graph as built.
            for stop_at in [1, 2, 3, most / 2, most] {
                let asked = Cell::new(0_u32);
                let stop = || {
                    asked.set(asked.get() + 1);
                    asked.get() >= stop_at
                };
                let mut projected = graph.clone();
                assert_eq!(
                    project_upper_level(
                        &mut projected,
                        16,
                        Vectors::F32(&vectors),
                        8,
                        &settings,
                        &stop
                    ),
                    Err(ProjectionError::Stopped),
                    "stop at {stop_at}"
                );
                assert_eq!(projected, graph, "stop at {stop_at}");
                assert_eq!(asked.get(), stop_at, "asked no more once told");
            }
        }
        // In order, it is asked before every node but the first.
        assert!(asked(&sequential) > 400);
    }

    #[test]
    fn asks_to_stop_every_few_thousand_rows_of_the_copy() {
        let values: Vec<f32> = (0..10_000_u16).map(f32::from).collect();
        let ids: Vec<u32> = (0..10_000).collect();
        let asked = Cell::new(0_u32);
        let stop = || {
            asked.set(asked.get() + 1);
            true
        };
        // One batch of rows, and no ask.
        let copy = promoted_rows(Vectors::F32(&values), 1, 10_000, &ids[..ROWS_A_STOP], &stop);
        assert_eq!(copy.map(|rows| rows.len()), Ok(ROWS_A_STOP));
        assert_eq!(asked.get(), 0);
        // Past it, asked before the next row, and given up.
        let copy = promoted_rows(Vectors::F32(&values), 1, 10_000, &ids, &stop);
        assert_eq!(copy, Err(ProjectionError::Stopped));
        assert_eq!(asked.get(), 1);
    }

    #[test]
    fn refuses_what_it_cannot_project() {
        let (vectors, mut graph) = clustered(2, 50, 4, 8);
        let settings = Settings {
            m: 4,
            ..Settings::default()
        };
        let mut project = |vectors: Vectors<'_>, dimension, settings: &Settings| {
            project_upper_level(&mut graph, 8, vectors, dimension, settings, &|| false)
        };
        assert_eq!(
            project(Vectors::F32(&vectors[4..]), 4, &settings),
            Err(ProjectionError::Vectors {
                len: 396,
                dimension: 4,
                nodes: 100
            })
        );
        assert!(matches!(
            project(Vectors::F32(&vectors), 0, &settings),
            Err(ProjectionError::Vectors { .. })
        ));
        assert_eq!(
            project(Vectors::Bits(&[0; 50]), 4, &settings),
            Err(ProjectionError::Hamming)
        );
        let hamming = Settings {
            metric: Metric::Hamming,
            ..settings
        };
        assert_eq!(
            project(Vectors::F32(&vectors), 4, &hamming),
            Err(ProjectionError::Hamming)
        );
    }
}
