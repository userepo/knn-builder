//! The HNSW level table: the level `GpuIndexCagra::copyTo` draws for every
//! node, which sets the file's `levels`, `offsets` and neighbor slots even
//! though only level 0 is filled. See `docs/faiss-layout.md`, "Levels".

use crate::mt19937::Mt19937;

/// The seed `HNSW`'s generator gets at construction (`impl/HNSW.cpp:59`);
/// `copyTo` draws from it before anything else does.
const SEED: u32 = 12345;

/// More levels than any M can produce (M = 2, the smallest, has 29): a
/// bound that keeps the table finite whatever the floating point does.
const MAX_LEVELS: u16 = 1000;

/// Faiss's level probabilities and cumulative neighbor counts for one M,
/// as `HNSW::set_default_probas(M, 1.0 / log(M))` sets them.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelTable {
    assign_probas: Vec<f64>,
    cum_nneighbor_per_level: Vec<i32>,
}

impl LevelTable {
    /// The table for an HNSW `M` of `m`, half the graph degree.
    ///
    /// The probabilities go through the platform's `exp` and `ln`, as
    /// Faiss's go through the C library's. For every even `m` from 2 to
    /// 4096 (a field's m up to 2048), the tests hold them to the values
    /// glibc gives, which wrote the service's files, on Linux and Windows;
    /// a larger `m` is computed the same way but unverified.
    ///
    /// Returns `None` when `m` is below 2 (Faiss's `1 / log(1)` leaves no
    /// level to draw) or the neighbor counts overflow `i32`.
    #[must_use]
    pub fn new(m: u32) -> Option<Self> {
        if m < 2 {
            return None;
        }
        // set_default_probas(int M, float levelMult) is called with
        // 1.0 / log(M): computed in double, passed as a float.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Faiss passes levelMult as a float"
        )]
        let level_mult = (1.0 / f64::from(m).ln()) as f32;
        let m = i32::try_from(m).ok()?;
        let mut assign_probas = Vec::new();
        let mut cum_nneighbor_per_level = vec![0];
        let mut total: i32 = 0;
        for level in 0..MAX_LEVELS {
            // float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult)):
            // float quotients, C's double exp and product, one rounding.
            let a = f64::from(-f32::from(level) / level_mult);
            let b = f64::from(-1.0_f32 / level_mult);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "Faiss stores the product in a float"
            )]
            let proba = (a.exp() * (1.0 - b.exp())) as f32;
            if f64::from(proba) < 1e-9 {
                return Some(Self {
                    assign_probas,
                    cum_nneighbor_per_level,
                });
            }
            assign_probas.push(f64::from(proba));
            total = total.checked_add(if level == 0 { m.checked_mul(2)? } else { m })?;
            cum_nneighbor_per_level.push(total);
        }
        None
    }

    /// `assign_probas`, as the file stores them.
    #[must_use]
    pub fn assign_probas(&self) -> &[f64] {
        &self.assign_probas
    }

    /// `cum_nneighbor_per_level`, as the file stores them: 0, then the
    /// neighbor slots of a node through each level.
    #[must_use]
    pub fn cum_nneighbor_per_level(&self) -> &[i32] {
        &self.cum_nneighbor_per_level
    }

    /// The levels of `n` nodes, in node order, as `prepare_level_tab(n,
    /// false)` draws them on a fresh `HNSW`. Each costs one draw, so the
    /// table can be walked again instead of stored.
    #[must_use]
    pub fn levels(&self, n: usize) -> Levels<'_> {
        Levels {
            table: self,
            rng: Mt19937::new(SEED),
            remaining: n,
        }
    }

    /// One node's level: `HNSW::random_level` on the draw
    /// `RandomGenerator::rand_float` makes, `mt() / float(mt.max())` in
    /// float.
    fn draw(&self, rng: &mut Mt19937) -> NodeLevel {
        #[expect(
            clippy::cast_precision_loss,
            reason = "Faiss converts the draw and the maximum to float"
        )]
        let mut f = f64::from(rng.next_u32() as f32 / u32::MAX as f32);
        // Level l is stored as l + 1, and cum_nneighbor_per_level[l + 1]
        // is the node's slot count.
        let mut last = NodeLevel {
            stored: 0,
            slots: 0,
        };
        for ((stored, &proba), &cum) in (1..)
            .zip(&self.assign_probas)
            .zip(self.cum_nneighbor_per_level.iter().skip(1))
        {
            last = NodeLevel {
                stored,
                slots: cum.unsigned_abs(),
            };
            if f < proba {
                return last;
            }
            f -= proba;
        }
        // Past every level, Faiss returns the last.
        last
    }
}

/// A node's entry in the level table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeLevel {
    /// The value `levels` stores: the node's level + 1.
    pub stored: i32,
    /// Its neighbor slots, `cum_nneighbor_per_level[stored]`.
    pub slots: u32,
}

/// The nodes' levels, in node order; see [`LevelTable::levels`].
#[derive(Debug, Clone)]
pub struct Levels<'a> {
    table: &'a LevelTable,
    rng: Mt19937,
    remaining: usize,
}

impl Iterator for Levels<'_> {
    type Item = NodeLevel;

    fn next(&mut self) -> Option<NodeLevel> {
        self.remaining = self.remaining.checked_sub(1)?;
        Some(self.table.draw(&mut self.rng))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for Levels<'_> {}

#[cfg(test)]
mod tests {
    use super::LevelTable;

    #[test]
    fn refuses_an_m_without_levels() {
        assert_eq!(LevelTable::new(0), None);
        assert_eq!(LevelTable::new(1), None);
    }

    #[test]
    fn matches_the_fixtures_m() {
        // M = 32: the table every fixture built with m = 16 holds.
        let table = LevelTable::new(32).unwrap();
        assert_eq!(
            table.cum_nneighbor_per_level(),
            [0, 64, 96, 128, 160, 192, 224]
        );
        assert_eq!(table.assign_probas()[0], 0.96875);
        assert_eq!(table.assign_probas()[1], 0.030_273_444_950_580_597);
    }
}
