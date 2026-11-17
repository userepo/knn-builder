//! Writes the links of the level built over the promoted nodes into their
//! level-0 rows, as the service's `project_links` does in one vectorized
//! pass; here row by row, to the same result.

use super::{ProjectionError, node_count, slot};

/// How the links go in. The default is the service's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkSettings {
    /// The most of a full row the links may overwrite, its last slots
    /// first (`max_share`, from 0 to 1). A row with an empty slot takes
    /// links into every slot it can give up.
    pub max_share: f32,
    /// Never overwrite a slot that may hold its target's last in-edge
    /// (`protect_last_in_edge`), and check before writing that no node
    /// loses its last one.
    pub protect_last_in_edge: bool,
}

impl Default for LinkSettings {
    /// The service's: half a full row, last in-edges kept.
    fn default() -> Self {
        Self {
            max_share: 0.5,
            protect_last_in_edge: true,
        }
    }
}

/// Writes every link a-b of `layer1` into `graph` both ways, a -> b and
/// b -> a, as the service's `project_links` does, and returns the links
/// written.
///
/// `graph` is the level-0 graph, rows of `degree` ids; `promoted` the
/// promoted node ids, ascending; `layer1` the level over them, a row of
/// `layer1_degree` positions into `promoted` for each, with empty slots as
/// in `graph`. Only the promoted nodes' rows change:
///
/// - A link whose target the row already holds is dropped, and the slots
///   holding it are kept.
/// - With `protect_last_in_edge`, every slot holding a node that no row
///   outside the promoted ones links to is kept, so no node can lose its
///   last in-edge.
/// - Each row takes its links in ascending target id: into its empty slots
///   from the first, then into its other slots that are not kept, from
///   the last; a full row gives up at most `max_share` of its slots,
///   truncated as numpy 1.26 truncates `float32(max_share) * degree`
///   (computed in float64), and the links past what a row can take are
///   dropped.
///
/// Nothing is written until every row is placed and, with
/// `protect_last_in_edge`, checked. The protection keeps every slot the
/// check could find, so the check never fails; it stays, as the service's
/// does.
///
/// # Errors
///
/// Returns a [`ProjectionError`], with `graph` unchanged, when the graph
/// is not whole rows or names a node past the last; `promoted` is not
/// ascending or names a node past the last; `layer1` is not one row a
/// promoted node or names a position past the last; `max_share` is not
/// from 0 to 1; or the check finds a node that would lose its last
/// in-edge.
pub fn project_links(
    graph: &mut [u32],
    degree: usize,
    promoted: &[u32],
    layer1: &[u32],
    layer1_degree: usize,
    settings: &LinkSettings,
) -> Result<usize, ProjectionError> {
    let n = node_count(graph, degree)?;
    check_graph(graph, degree, n)?;
    check_promoted(promoted, n)?;
    if !(0.0..=1.0).contains(&settings.max_share) {
        return Err(ProjectionError::MaxShare {
            max_share: settings.max_share,
        });
    }
    let links = Links::new(layer1, layer1_degree, promoted.len())?;
    let pass = Pass {
        promoted,
        promoted_set: Bits::of(n, promoted.iter().copied()),
        protection: settings
            .protect_last_in_edge
            .then(|| Protection::new(graph, degree, n, promoted)),
        cap: full_row_cap(settings.max_share, degree),
    };

    let mut writes: Vec<Write> = Vec::new();
    for (row, &node) in promoted.iter().enumerate() {
        let start = node as usize * degree;
        if let Some(slots) = graph.get(start..start + degree) {
            pass.plan_row(slots, start, &links.of(row), &mut writes);
        }
    }

    if let Some(protection) = &pass.protection {
        let stranded = protection.stranded(&writes);
        if stranded > 0 {
            return Err(ProjectionError::WouldStrand { nodes: stranded });
        }
    }
    for write in &writes {
        if let Some(slot) = graph.get_mut(write.index) {
            *slot = write.id;
        }
    }
    Ok(writes.len())
}

/// The most host memory [`project_links`] takes besides its inputs, in
/// bytes, for a graph of `nodes` full rows of `degree` (as CAGRA builds
/// them), `promoted` of them, and layer-1 rows of `layer1_degree`: the
/// links both ways as keys, two bits and an in-degree a node, and the
/// writes, at most a full row's cap each, in a vector that may double,
/// then their old and new ids; a row's own buffers besides.
pub(super) fn host_bytes(
    nodes: u64,
    degree: u64,
    promoted: u64,
    layer1_degree: u64,
    max_share: f32,
) -> u64 {
    let cap = usize::try_from(degree).map_or(degree, |degree| {
        full_row_cap(max_share.clamp(0.0, 1.0), degree) as u64
    });
    let writes = promoted.saturating_mul(cap);
    let keys = promoted.saturating_mul(layer1_degree).saturating_mul(2 * 8);
    let bits = nodes.div_ceil(64).saturating_mul(2 * 8);
    let in_degree = nodes.saturating_mul(4);
    let row = degree.saturating_mul(1 + 8 + 8);
    // A write is 16 bytes, and the vector holding them up to twice as
    // large as they are; then 4 bytes for each old id and each new one.
    keys.saturating_add(bits)
        .saturating_add(in_degree)
        .saturating_add(writes.saturating_mul(2 * 16 + 2 * 4))
        .saturating_add(row)
}

/// One planned write: the graph entry, what it held, what it gets.
#[derive(Debug, Clone, Copy)]
struct Write {
    index: usize,
    old: u32,
    id: u32,
}

/// What every promoted row's placement reads.
struct Pass<'a> {
    promoted: &'a [u32],
    promoted_set: Bits,
    protection: Option<Protection>,
    /// The slots a full row gives up at most.
    cap: usize,
}

impl Pass<'_> {
    /// Places the links of one promoted row, `slots` its level-0 row at
    /// `start` in the graph and `wanted` its targets' positions, ascending.
    fn plan_row(&self, slots: &[u32], start: usize, wanted: &[u64], writes: &mut Vec<Write>) {
        let degree = slots.len();
        let mut keep = vec![false; degree];
        // The targets the row already holds, as positions: their links drop.
        let mut present: Vec<u64> = Vec::new();
        for (&id, k) in slots.iter().zip(keep.iter_mut()) {
            let Some(node) = slot(id) else {
                continue;
            };
            if self.promoted_set.contains(node)
                && let Ok(position) = self.promoted.binary_search(&node)
                && wanted.binary_search(&(position as u64)).is_ok()
            {
                // Every slot holding it is kept, each found against the
                // links as they came.
                *k = true;
                present.push(position as u64);
            }
            if self
                .protection
                .as_ref()
                .is_some_and(|p| !p.has_outside_in_edge(node))
            {
                *k = true;
            }
        }

        let empty = slots.iter().filter(|&&id| slot(id).is_none()).count();
        let kept = keep.iter().filter(|&&k| k).count();
        let overwritable = degree - empty - kept;
        let row_cap = if empty > 0 { degree } else { self.cap };
        let budget = empty + row_cap.min(overwritable);

        // Empty slots from the first, then the others not kept from the last.
        let empties = (0..degree).filter(|&s| slots.get(s).is_some_and(|&id| slot(id).is_none()));
        let others = (0..degree).rev().filter(|&s| {
            keep.get(s) == Some(&false) && slots.get(s).is_some_and(|&id| slot(id).is_some())
        });
        let targets = wanted
            .iter()
            .filter(|position| !present.contains(position))
            .filter_map(|&position| self.promoted.get(usize::try_from(position).ok()?).copied());
        for (s, target) in empties.chain(others).zip(targets).take(budget) {
            if let Some(&old) = slots.get(s) {
                writes.push(Write {
                    index: start + s,
                    old,
                    id: target,
                });
            }
        }
    }
}

/// How many slots of a full row of `degree` the links may take: numpy
/// 1.26's `int(np.float32(max_share) * degree)`, which widens the float32
/// share to float64 for the product and truncates it.
fn full_row_cap(max_share: f32, degree: usize) -> usize {
    #[expect(clippy::cast_precision_loss, reason = "a row's degree, far below 2^53")]
    let product = f64::from(max_share) * degree as f64;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "truncation is numpy's int(); the share is from 0 to 1, so the product from 0 to degree"
    )]
    let cap = product as usize;
    cap
}

/// Every id names a node or is an empty slot.
fn check_graph(graph: &[u32], degree: usize, n: u32) -> Result<(), ProjectionError> {
    match graph
        .iter()
        .enumerate()
        .find(|&(_, &id)| slot(id).is_some_and(|node| node >= n))
    {
        Some((index, &neighbor)) => Err(ProjectionError::NeighborOutOfRange {
            node: index / degree,
            neighbor,
            count: n as usize,
        }),
        None => Ok(()),
    }
}

/// The promoted ids ascend, each once, and name nodes.
fn check_promoted(promoted: &[u32], n: u32) -> Result<(), ProjectionError> {
    let ascending = promoted.windows(2).all(|pair| pair.first() < pair.get(1));
    if ascending && promoted.last().is_none_or(|&last| last < n) {
        Ok(())
    } else {
        Err(ProjectionError::PromotedIds { count: n as usize })
    }
}

/// The links of layer 1 both ways, as `row * promoted + target` in
/// positions, ascending and each once, so grouped by row: the service's
/// `keys`.
struct Links {
    keys: Vec<u64>,
    promoted: u64,
}

impl Links {
    fn new(layer1: &[u32], degree: usize, promoted: usize) -> Result<Self, ProjectionError> {
        let shape = ProjectionError::Layer1Shape {
            len: layer1.len(),
            rows: promoted,
            degree,
        };
        if promoted.checked_mul(degree) != Some(layer1.len()) {
            return Err(shape);
        }
        let count = promoted as u64;
        let mut keys = Vec::with_capacity(2 * layer1.len());
        if degree > 0 {
            for (row, targets) in (0_u64..).zip(layer1.chunks_exact(degree)) {
                for &target in targets {
                    let Some(target) = slot(target) else {
                        continue;
                    };
                    let target = u64::from(target);
                    if target >= count {
                        return Err(ProjectionError::Layer1OutOfRange {
                            row: usize::try_from(row).unwrap_or(usize::MAX),
                            target: u32::try_from(target).unwrap_or(u32::MAX),
                            count: promoted,
                        });
                    }
                    if target != row {
                        keys.push(row * count + target);
                        keys.push(target * count + row);
                    }
                }
            }
        }
        keys.sort_unstable();
        keys.dedup();
        Ok(Self {
            keys,
            promoted: count,
        })
    }

    /// The targets of row `row`, as positions, ascending.
    fn of(&self, row: usize) -> Vec<u64> {
        let row = row as u64;
        let first = self.keys.partition_point(|&key| key / self.promoted < row);
        let last = self.keys.partition_point(|&key| key / self.promoted <= row);
        self.keys
            .get(first..last)
            .unwrap_or_default()
            .iter()
            .map(|key| key % self.promoted)
            .collect()
    }
}

/// What the last-in-edge protection needs: every node's in-degree, and
/// which nodes a row outside the promoted ones links to.
struct Protection {
    in_degree: Vec<u32>,
    outside: Bits,
}

impl Protection {
    fn new(graph: &[u32], degree: usize, n: u32, promoted: &[u32]) -> Self {
        let mut in_degree = vec![0_u32; n as usize];
        let mut outside = Bits::new(n);
        let mut next_promoted = promoted.iter().copied().peekable();
        for (row, slots) in (0_u32..).zip(graph.chunks_exact(degree)) {
            let row_promoted = next_promoted.next_if_eq(&row).is_some();
            for node in slots.iter().filter_map(|&id| slot(id)) {
                if let Some(count) = in_degree.get_mut(node as usize) {
                    *count += 1;
                }
                if !row_promoted {
                    outside.insert(node);
                }
            }
        }
        Self { in_degree, outside }
    }

    fn has_outside_in_edge(&self, node: u32) -> bool {
        self.outside.contains(node)
    }

    /// The nodes that had an in-edge and would have none after `writes`.
    fn stranded(&self, writes: &[Write]) -> usize {
        let mut removed: Vec<u32> = Vec::with_capacity(writes.len());
        removed.extend(writes.iter().filter_map(|w| slot(w.old)));
        removed.sort_unstable();
        let mut added: Vec<u32> = writes.iter().map(|w| w.id).collect();
        added.sort_unstable();
        let count_of = |sorted: &[u32], node: u32| {
            sorted.partition_point(|&v| v <= node) - sorted.partition_point(|&v| v < node)
        };
        let mut stranded = 0;
        let mut previous = None;
        for &node in &removed {
            if previous == Some(node) {
                continue;
            }
            previous = Some(node);
            let before = self.in_degree.get(node as usize).copied().unwrap_or(0) as usize;
            let after = before + count_of(&added, node);
            if before > 0 && after <= count_of(&removed, node) {
                stranded += 1;
            }
        }
        stranded
    }
}

/// A bit a node.
struct Bits(Vec<u64>);

impl Bits {
    fn new(n: u32) -> Self {
        Self(vec![0; n.div_ceil(64) as usize])
    }

    fn of(n: u32, nodes: impl Iterator<Item = u32>) -> Self {
        let mut bits = Self::new(n);
        for node in nodes {
            bits.insert(node);
        }
        bits
    }

    fn insert(&mut self, node: u32) {
        if let Some(word) = self.0.get_mut((node / 64) as usize) {
            *word |= 1 << (node % 64);
        }
    }

    fn contains(&self, node: u32) -> bool {
        self.0
            .get((node / 64) as usize)
            .is_some_and(|word| word & (1 << (node % 64)) != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::{LinkSettings, ProjectionError, Protection, Write, full_row_cap, project_links};

    /// Four nodes of degree 2: 0 -> 1 2, 1 -> 0 3, 2 -> 3 0, 3 -> 2 1.
    const GRAPH: [u32; 8] = [1, 2, 0, 3, 3, 0, 2, 1];

    #[test]
    fn writes_each_link_both_ways_from_the_end_of_a_full_row() {
        // Nodes 0 and 3 promoted; layer 1 links them. Row 0 gives its last
        // slot to 3, row 3 its last to 0; half of 2 is 1 slot each.
        let mut graph = GRAPH;
        let settings = LinkSettings {
            protect_last_in_edge: false,
            ..LinkSettings::default()
        };
        let written = project_links(&mut graph, 2, &[0, 3], &[1, u32::MAX], 1, &settings);
        assert_eq!(written, Ok(2));
        assert_eq!(graph, [1, 3, 0, 3, 3, 0, 2, 0]);
    }

    #[test]
    fn keeps_a_slot_that_may_hold_a_last_in_edge() {
        // 0 -> 1 2, 1 -> 0 2, 2 -> 3 0, 3 -> 2 1, nodes 0 and 3 promoted:
        // only their rows link to node 1, so its slots stay. Row 0 gives up
        // its last slot, row 3 the one before node 1.
        let mut graph = [1, 2, 0, 2, 3, 0, 2, 1];
        let written = project_links(
            &mut graph,
            2,
            &[0, 3],
            &[1, u32::MAX],
            1,
            &LinkSettings::default(),
        );
        assert_eq!(written, Ok(2));
        assert_eq!(graph, [1, 3, 0, 2, 3, 0, 0, 1]);
        // In GRAPH, only promoted rows link to nodes 1 and 2: nothing goes.
        let mut graph = GRAPH;
        let written = project_links(&mut graph, 2, &[0, 3], &[1, 0], 1, &LinkSettings::default());
        assert_eq!(written, Ok(0));
        assert_eq!(graph, GRAPH);
    }

    /// Promoted ids, layer 1, its degree, the settings, and the refusal.
    type BadInput<'a> = (&'a [u32], &'a [u32], usize, LinkSettings, ProjectionError);

    #[test]
    fn leaves_the_graph_as_built_on_bad_input() {
        let defaults = LinkSettings::default();
        let cases: [BadInput<'_>; 5] = [
            (
                &[3, 0],
                &[1, 0],
                1,
                defaults,
                ProjectionError::PromotedIds { count: 4 },
            ),
            (
                &[0, 4],
                &[1, 0],
                1,
                defaults,
                ProjectionError::PromotedIds { count: 4 },
            ),
            (
                &[0, 3],
                &[1],
                1,
                defaults,
                ProjectionError::Layer1Shape {
                    len: 1,
                    rows: 2,
                    degree: 1,
                },
            ),
            (
                &[0, 3],
                &[2, 0],
                1,
                defaults,
                ProjectionError::Layer1OutOfRange {
                    row: 0,
                    target: 2,
                    count: 2,
                },
            ),
            (
                &[0, 3],
                &[1, 0],
                1,
                LinkSettings {
                    max_share: 1.5,
                    ..defaults
                },
                ProjectionError::MaxShare { max_share: 1.5 },
            ),
        ];
        for (promoted, layer1, degree, settings, error) in cases {
            let mut graph = GRAPH;
            assert_eq!(
                project_links(&mut graph, 2, promoted, layer1, degree, &settings),
                Err(error)
            );
            assert_eq!(graph, GRAPH);
        }
        let mut graph = GRAPH;
        graph[3] = 7;
        assert_eq!(
            project_links(&mut graph, 2, &[0, 3], &[1, 0], 1, &defaults),
            Err(ProjectionError::NeighborOutOfRange {
                node: 1,
                neighbor: 7,
                count: 4
            })
        );
        let nan = LinkSettings {
            max_share: f32::NAN,
            ..defaults
        };
        assert!(matches!(
            project_links(&mut GRAPH.clone(), 2, &[0, 3], &[1, 0], 1, &nan),
            Err(ProjectionError::MaxShare { .. })
        ));
        // A degree whose row count overflows is refused, and its message
        // formats without multiplying.
        let error = project_links(
            &mut GRAPH.clone(),
            2,
            &[0, 3],
            &[1, 0],
            usize::MAX,
            &defaults,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "layer 1 holds 2 entries, which are not 2 rows of degree {}",
                usize::MAX
            )
        );
    }

    #[test]
    fn finds_a_node_that_would_lose_its_last_in_edge() {
        // Unreachable through project_links, whose protection keeps such
        // slots; the check stays, as the service's does.
        let protection = Protection::new(&GRAPH, 2, 4, &[]);
        // Node 1's in-edges are slots 0 and 7.
        let both = [
            Write {
                index: 0,
                old: 1,
                id: 3,
            },
            Write {
                index: 7,
                old: 1,
                id: 3,
            },
        ];
        assert_eq!(protection.stranded(&both), 1);
        assert_eq!(protection.stranded(&both[..1]), 0);
        // Losing both and gaining one back is not stranded.
        let back = [
            both[0],
            both[1],
            Write {
                index: 2,
                old: 0,
                id: 1,
            },
        ];
        assert_eq!(protection.stranded(&back), 0);
    }

    #[test]
    fn rounds_the_cap_as_numpy_1_26() {
        // float32(0.7) * 10 in float64 is 6.9999998..., so 6 (numpy 2's
        // float32 product rounds to 7).
        assert_eq!(full_row_cap(0.7, 10), 6);
        assert_eq!(full_row_cap(0.5, 64), 32);
        assert_eq!(full_row_cap(0.0, 64), 0);
        assert_eq!(full_row_cap(1.0, 64), 64);
    }
}
