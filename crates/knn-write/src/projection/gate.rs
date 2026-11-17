//! The gate: how much of a graph a search from random entry points
//! reaches, which decides whether the projection runs at all.

use super::{ProjectionError, node_count, pcg64, slot};

/// The entry points the gate tries (`probes`).
pub const PROBES: usize = 3;

/// The seed of the numpy generator that draws them.
pub const PROBE_SEED: u64 = 12345;

/// The largest share of the nodes of `graph`, rows of `degree` ids, that a
/// breadth-first search reaches from one of [`PROBES`] entry points, as
/// the service's `max_reachable_share` measures it: the entry points are
/// those `numpy.random.default_rng(12345).choice(n, 3, replace=False)`
/// draws, tried in that order until one reaches `stop_at`. That is what a
/// search confined to level 0 can see. A graph of no nodes reaches 0.
///
/// # Errors
///
/// Returns a [`ProjectionError`] when the graph is not whole rows of
/// `degree`, has more nodes than Faiss's ids can name, or a row the search
/// visits names a node past the last, where the service's search fails.
pub fn max_reachable_share(
    graph: &[u32],
    degree: usize,
    stop_at: f64,
) -> Result<f64, ProjectionError> {
    let n = node_count(graph, degree)?;
    let mut best = 0.0_f64;
    for start in probes(n) {
        let reached = reachable_count(graph, degree, n, start)?;
        best = best.max(f64::from(reached) / f64::from(n));
        if best >= stop_at {
            break;
        }
    }
    Ok(best)
}

/// The most host memory [`max_reachable_share`] takes for a graph of
/// `nodes` nodes, in bytes: a bit a node, and a stack that holds each node
/// once at most.
pub(super) fn host_bytes(nodes: u64) -> u64 {
    nodes
        .div_ceil(64)
        .saturating_mul(8)
        .saturating_add(nodes.saturating_mul(4))
}

/// The entry points the gate tries on a graph of `n` nodes, in order:
/// those `numpy.random.default_rng(12345).choice(n, min(3, n),
/// replace=False)` draws.
#[must_use]
pub fn probes(n: u32) -> Vec<u32> {
    pcg64::choose_few(PROBE_SEED, u64::from(n), PROBES)
        .into_iter()
        // Every draw is below n, so fits u32.
        .filter_map(|start| u32::try_from(start).ok())
        .collect()
}

/// The nodes reachable from `start`, itself included, following every
/// link that is not an empty slot.
fn reachable_count(
    graph: &[u32],
    degree: usize,
    n: u32,
    start: u32,
) -> Result<u32, ProjectionError> {
    let mut visited = Visited::new(n);
    visited.insert(start);
    let mut stack = vec![start];
    let mut count = 1_u32;
    while let Some(node) = stack.pop() {
        let first = node as usize * degree;
        let Some(row) = graph.get(first..first + degree) else {
            continue;
        };
        for &id in row {
            let Some(next) = slot(id) else {
                continue;
            };
            if next >= n {
                return Err(ProjectionError::NeighborOutOfRange {
                    node: node as usize,
                    neighbor: id,
                    count: n as usize,
                });
            }
            if visited.insert(next) {
                count += 1;
                stack.push(next);
            }
        }
    }
    Ok(count)
}

/// A bit a node.
struct Visited(Vec<u64>);

impl Visited {
    fn new(n: u32) -> Self {
        Self(vec![0; n.div_ceil(64) as usize])
    }

    /// Marks `node`; whether it was unmarked.
    fn insert(&mut self, node: u32) -> bool {
        let bit = 1_u64 << (node % 64);
        match self.0.get_mut((node / 64) as usize) {
            Some(word) if *word & bit == 0 => {
                *word |= bit;
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ProjectionError, max_reachable_share, reachable_count};

    /// Two triangles, 0-1-2 and 3-4-5, joined one way by 2 -> 3, and node
    /// 6 alone; degree 2, an empty slot as `u32::MAX`.
    const GRAPH: [u32; 14] = [1, 2, 0, 2, 0, 3, 4, 5, 3, 5, 3, 4, u32::MAX, u32::MAX];

    #[test]
    fn counts_what_a_search_reaches() {
        assert_eq!(reachable_count(&GRAPH, 2, 7, 0), Ok(6));
        assert_eq!(reachable_count(&GRAPH, 2, 7, 4), Ok(3));
        assert_eq!(reachable_count(&GRAPH, 2, 7, 6), Ok(1));
    }

    #[test]
    fn takes_the_best_probe() {
        // n = 7: numpy draws 1, 5 and 3 (default_rng(12345), numpy 2.5.3);
        // the first reaches 6 of 7, the others 3.
        let share = max_reachable_share(&GRAPH, 2, 1.0).unwrap();
        assert!((share - 6.0 / 7.0).abs() < f64::EPSILON, "{share}");
        assert_eq!(max_reachable_share(&[], 2, 1.0), Ok(0.0));
    }

    #[test]
    fn fails_on_a_neighbor_past_the_last_node() {
        let mut graph = GRAPH;
        graph[1] = 9;
        assert_eq!(
            reachable_count(&graph, 2, 7, 0),
            Err(ProjectionError::NeighborOutOfRange {
                node: 0,
                neighbor: 9,
                count: 7
            })
        );
    }
}
