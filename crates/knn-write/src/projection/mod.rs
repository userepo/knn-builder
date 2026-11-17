//! The upper-level projection the upstream service runs on a float,
//! `half_float` or byte graph when a request sets
//! `project_link_to_base_layer`, ported from its
//! `cagra_upper_level_projection.py`.
//!
//! On vectors in well-separated clusters, CAGRA's graph splits into
//! components a search from random entry points cannot leave. The service
//! first measures how much of the graph a breadth-first search reaches
//! ([`max_reachable_share`]); when that is less than 99%, it promotes the
//! nodes a Faiss HNSW of M = m would put on level 1 ([`promoted`]), builds
//! an HNSW level over them ([`build_layer1`]), and writes that level's
//! links into their level-0 rows ([`project_links`]). The file's format
//! does not change. [`project_upper_level`] runs the four in turn.
//!
//! A graph here is the level-0 graph the file holds: `degree` ids a node,
//! row after row. An id of 2^31 or more is an empty slot, as Faiss's
//! 32-bit signed slots read it once it wraps negative.

mod gate;
mod kernels;
mod layer1;
mod links;
mod pcg64;
mod upper_level;

pub use gate::{PROBE_SEED, PROBES, max_reachable_share, probes};
pub use layer1::{Insertion, build_layer1};
pub use links::{LinkSettings, project_links};
pub use upper_level::{Outcome, Settings, host_bytes, project_upper_level};

use crate::LevelTable;

/// Asked between a projection's steps, its copy's rows and layer 1's
/// batches: whether to give it up, as when its job is canceled.
pub type Stop<'a> = &'a dyn Fn() -> bool;

/// The share of the nodes reachable at which the service leaves the graph
/// as built (`skip_if_reachable`).
pub const SKIP_IF_REACHABLE: f64 = 0.99;

/// Why the projection leaves a graph as built.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ProjectionError {
    /// A graph that is not whole rows of `degree` ids, or of degree 0.
    #[error("{len} graph entries are not whole rows of degree {degree}")]
    GraphShape {
        /// Entries given.
        len: usize,
        /// The degree given.
        degree: usize,
    },
    /// More nodes than Faiss's 32-bit signed ids can name.
    #[error("{count} nodes: Faiss's 32-bit ids name at most 2147483647")]
    TooManyNodes {
        /// The node count.
        count: usize,
    },
    /// A neighbor id that names no node and is not an empty slot, where the
    /// service's numpy indexing fails and leaves the graph as built.
    #[error("node {node} lists neighbor {neighbor}, but there are {count} nodes")]
    NeighborOutOfRange {
        /// The node whose row holds it.
        node: usize,
        /// The id.
        neighbor: u32,
        /// The node count.
        count: usize,
    },
    /// Promoted ids that do not ascend, repeat one, or name a node past
    /// the last.
    #[error("the promoted ids must ascend, each once, below {count}")]
    PromotedIds {
        /// The node count.
        count: usize,
    },
    /// A layer-1 graph that is not one row of `degree` a promoted node.
    #[error("layer 1 holds {len} entries, which are not {rows} rows of degree {degree}")]
    Layer1Shape {
        /// Entries given.
        len: usize,
        /// The promoted nodes.
        rows: usize,
        /// The degree given.
        degree: usize,
    },
    /// A layer-1 link to a position past the last promoted node.
    #[error("layer-1 row {row} links to position {target}, but {count} nodes are promoted")]
    Layer1OutOfRange {
        /// The row.
        row: usize,
        /// The position.
        target: u32,
        /// The promoted nodes.
        count: usize,
    },
    /// A share of a row outside 0 to 1.
    #[error("max_share {max_share}: must be from 0 to 1")]
    MaxShare {
        /// The share given.
        max_share: f32,
    },
    /// Layer-1 vectors that are not whole rows, or a dimension or m of 0.
    #[error(
        "{len} layer-1 values are not whole rows of dimension {dimension}, or the dimension or m ({m}) is 0"
    )]
    Layer1Input {
        /// Values given.
        len: usize,
        /// The dimension given.
        dimension: usize,
        /// The m given.
        m: usize,
    },
    /// Binary vectors in Hamming space, which the service does not project.
    #[error("the projection is not defined for binary vectors")]
    Hamming,
    /// Writing the links would leave nodes that had an in-edge with none.
    #[error("the projection would leave {nodes} nodes without an in-edge")]
    WouldStrand {
        /// Those nodes.
        nodes: usize,
    },
    /// Vectors that are not one row of `dimension` a node, or a dimension
    /// of 0.
    #[error("{len} vector values are not {nodes} rows of dimension {dimension}")]
    Vectors {
        /// Values given.
        len: usize,
        /// The dimension given.
        dimension: usize,
        /// The graph's nodes.
        nodes: usize,
    },
    /// A buffer the projection could not allocate.
    #[error("cannot allocate {bytes} bytes")]
    HostMemory {
        /// Its size.
        bytes: usize,
    },
    /// Given up when asked to stop, the graph as built.
    #[error("the projection was stopped")]
    Stopped,
}

/// The ids promoted to the HNSW level the projection builds, in id order:
/// the nodes a Faiss `HNSW(m)` puts on level 1 or above when it draws the
/// levels of `n` nodes from its own generator (seed 12345), as
/// `draw_promoted` takes them. Each node is promoted with probability
/// 1 / m. An m below 2 promotes none: Faiss's level table for it has no
/// level.
#[must_use]
pub fn promoted(n: u32, m: u32) -> Vec<u32> {
    let Some(table) = LevelTable::new(m) else {
        return Vec::new();
    };
    let levels = table.levels(n as usize);
    // A level of 1 is stored as 2.
    (0..n)
        .zip(levels)
        .filter(|(_, level)| level.stored >= 2)
        .map(|(id, _)| id)
        .collect()
}

/// The node count of `graph`, rows of `degree` ids, checked.
fn node_count(graph: &[u32], degree: usize) -> Result<u32, ProjectionError> {
    if degree == 0 || !graph.len().is_multiple_of(degree) {
        return Err(ProjectionError::GraphShape {
            len: graph.len(),
            degree,
        });
    }
    let count = graph.len() / degree;
    i32::try_from(count)
        .map(i32::unsigned_abs)
        .map_err(|_| ProjectionError::TooManyNodes { count })
}

/// The node an id names, `None` for an empty slot (2^31 or more).
fn slot(id: u32) -> Option<u32> {
    i32::try_from(id).ok().map(i32::unsigned_abs)
}

#[cfg(test)]
mod tests {
    use super::{ProjectionError, node_count, promoted, slot};

    #[test]
    fn promotes_about_one_node_in_m() {
        let ids = promoted(100_000, 16);
        assert!(ids.is_sorted());
        assert!((5_800..6_700).contains(&ids.len()), "{}", ids.len());
        assert_eq!(promoted(1_000, 1), [] as [u32; 0]);
        assert_eq!(promoted(1_000, 0), [] as [u32; 0]);
        assert_eq!(promoted(0, 16), [] as [u32; 0]);
    }

    #[test]
    fn reads_wrapped_ids_as_empty_slots() {
        assert_eq!(slot(7), Some(7));
        assert_eq!(slot(i32::MAX.unsigned_abs()), Some(i32::MAX.unsigned_abs()));
        assert_eq!(slot(1 << 31), None);
        assert_eq!(slot(u32::MAX), None);
    }

    #[test]
    fn refuses_a_graph_of_partial_rows() {
        assert_eq!(node_count(&[0; 6], 3), Ok(2));
        assert_eq!(
            node_count(&[0; 5], 3),
            Err(ProjectionError::GraphShape { len: 5, degree: 3 })
        );
        assert_eq!(
            node_count(&[], 0),
            Err(ProjectionError::GraphShape { len: 0, degree: 0 })
        );
    }
}
