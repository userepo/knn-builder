//! The CAGRA build parameters the upstream service derives from a request;
//! `docs/params.md` in the repository lists each rule and its source line,
//! and `tests/data/params.txt` holds the service's own results.

use crate::Error;

/// The field's space type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// `l2`: cuVS's `L2Expanded`.
    L2,
    /// `innerproduct`: cuVS's `InnerProduct`.
    InnerProduct,
}

/// What one build asks of cuVS: the graph's degree and the IVF-PQ graph
/// build's size, computed from the request, and the settings the service
/// fixes for every build (the associated constants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildParams {
    /// The space type.
    pub metric: Metric,
    /// IVF-PQ lists: the truncated square root of the vector count.
    pub n_lists: u32,
    /// The request's m, lowered when the vector count cannot supply its
    /// degree.
    pub effective_m: u32,
    /// The graph degree and the intermediate graph degree, equal: m x 4.
    pub graph_degree: u32,
    /// PQ subspaces: the dimension / 4, truncated; 0 leaves the choice to
    /// cuVS.
    pub pq_dim: u32,
}

impl BuildParams {
    /// The service's degree per m (`GRAPH_DEGREE_PER_M`).
    pub const GRAPH_DEGREE_PER_M: u32 = 4;
    /// IVF-PQ lists probed per vector while seeding the graph.
    pub const N_PROBES: u32 = 5;
    /// k-means iterations of the IVF-PQ build.
    pub const KMEANS_N_ITERS: u32 = 20;
    /// Share of the vectors the k-means trains on.
    pub const KMEANS_TRAINSET_FRACTION: f64 = 0.1;
    /// Bits per PQ code.
    pub const PQ_BITS: u32 = 8;
    /// The IVF-PQ build rotates the vectors randomly.
    pub const FORCE_RANDOM_ROTATION: bool = true;
    /// The IVF-PQ lists are allocated without headroom.
    pub const CONSERVATIVE_MEMORY_ALLOCATION: bool = true;
    /// No refinement of the IVF-PQ neighbors.
    pub const REFINEMENT_RATE: f32 = 1.0;

    /// The parameters for `doc_count` vectors of `dimension` and a request
    /// for `m`, as `FaissIndexBuildService.build_index` computes them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Request`] for a count, dimension or m of 0.
    pub fn new(doc_count: u32, dimension: u32, m: u32, metric: Metric) -> Result<Self, Error> {
        if doc_count == 0 || dimension == 0 || m == 0 {
            return Err(Error::Request {
                doc_count,
                dimension,
                m,
            });
        }
        // calculate_ivf_pq_n_lists: int(math.sqrt(doc_count)), a correctly
        // rounded double square root, truncated; exact for every u32.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the square root of a u32 is a non-negative value below 65536"
        )]
        let n_lists = f64::from(doc_count).sqrt() as u32;
        let effective_m = effective_m(m, doc_count, n_lists);
        Ok(Self {
            metric,
            n_lists,
            effective_m,
            graph_degree: effective_m * Self::GRAPH_DEGREE_PER_M,
            // int(dimension / PQ_DIM_COMPRESSION_FACTOR)
            pq_dim: dimension / 4,
        })
    }
}

/// `calculate_effective_m`: m lowered until the graph degree fits half the
/// candidates an IVF-PQ search of `N_PROBES` lists sees, and the vector
/// count less one; at least 1.
fn effective_m(m: u32, doc_count: u32, n_lists: u32) -> u32 {
    let n_lists = u64::from(n_lists.max(1));
    let doc_count = u64::from(doc_count);
    let candidates = u64::from(BuildParams::N_PROBES).min(n_lists) * doc_count / n_lists;
    let max_degree = (candidates / 2).min(doc_count - 1);
    let fitting = max_degree / u64::from(BuildParams::GRAPH_DEGREE_PER_M);
    // min(m, fitting) is at most m, a u32.
    u32::try_from(u64::from(m).min(fitting)).unwrap_or(m).max(1)
}

#[cfg(test)]
mod tests {
    use super::{BuildParams, Metric};
    use crate::Error;

    #[test]
    fn lowers_m_for_a_small_count() {
        // docs/params.md: 150 vectors, m = 16 give 12 lists and m = 7.
        let params = BuildParams::new(150, 64, 16, Metric::L2).unwrap();
        assert_eq!(
            (
                params.n_lists,
                params.effective_m,
                params.graph_degree,
                params.pq_dim
            ),
            (12, 7, 28, 16)
        );
    }

    #[test]
    fn refuses_zeros() {
        assert!(matches!(
            BuildParams::new(0, 64, 16, Metric::L2),
            Err(Error::Request { .. })
        ));
        assert!(BuildParams::new(10, 0, 16, Metric::L2).is_err());
        assert!(BuildParams::new(10, 64, 0, Metric::L2).is_err());
    }
}
