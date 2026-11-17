//! numpy's `default_rng(seed)`: PCG64, the XSL-RR 128/64 generator, seeded
//! through `SeedSequence`, and the draws `Generator.choice(n, k,
//! replace=False)` makes for a small k, so the gate probes the nodes the
//! service probes.

/// PCG's 128-bit multiplier (`PCG_DEFAULT_MULTIPLIER_128`).
const MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

// SeedSequence's hash constants (numpy/random/bit_generator.pyx).
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

/// The generator `numpy.random.default_rng(seed)` wraps.
#[derive(Debug, Clone)]
pub(crate) struct Pcg64 {
    state: u128,
    increment: u128,
    /// The high half of the last 64-bit output, which the next 32-bit draw
    /// returns (`has_uint32`, `uinteger`).
    spare: Option<u32>,
}

impl Pcg64 {
    /// `PCG64(SeedSequence(seed))`: four 64-bit words from the sequence,
    /// the first two the initial state and the last two the stream, as
    /// `pcg64_set_seed` takes them.
    pub(crate) fn from_seed(seed: u64) -> Self {
        let [s0, s1, i0, i1] = seed_sequence_state(seed);
        let initial = (u128::from(s0) << 64) | u128::from(s1);
        let stream = (u128::from(i0) << 64) | u128::from(i1);
        // pcg_setseq_128_srandom_r.
        let mut rng = Self {
            state: 0,
            increment: (stream << 1) | 1,
            spare: None,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initial);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.increment);
    }

    /// The next 64-bit output: a step, then XSL-RR on the new state.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.step();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the state's halves, taken apart on purpose"
        )]
        let (high, low) = ((self.state >> 64) as u64, self.state as u64);
        let rotation = (high >> 58) as u32;
        (high ^ low).rotate_right(rotation)
    }

    /// The next 32-bit output: the low half of a 64-bit one, then its high
    /// half (`pcg64_next32`).
    pub(crate) fn next_u32(&mut self) -> u32 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let next = self.next_u64();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the output's halves, taken apart on purpose"
        )]
        let (high, low) = ((next >> 32) as u32, next as u32);
        self.spare = Some(high);
        low
    }

    /// A draw from 0 to `max`, both included, as
    /// `random_bounded_uint64(state, 0, max, 0, false)` makes it: Lemire's
    /// method on a 32-bit draw where `max` fits 32 bits, else on a 64-bit
    /// one.
    pub(crate) fn bounded(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        if let Ok(max) = u32::try_from(max) {
            if max == u32::MAX {
                return u64::from(self.next_u32());
            }
            let excluded = u64::from(max) + 1;
            let mut product = u64::from(self.next_u32()) * excluded;
            let mut leftover = product & u64::from(u32::MAX);
            if leftover < excluded {
                let threshold = u64::from((u32::MAX - max) % (max + 1));
                while leftover < threshold {
                    product = u64::from(self.next_u32()) * excluded;
                    leftover = product & u64::from(u32::MAX);
                }
            }
            return product >> 32;
        }
        if max == u64::MAX {
            return self.next_u64();
        }
        let excluded = u128::from(max) + 1;
        let mut product = u128::from(self.next_u64()) * excluded;
        let mut leftover = product & u128::from(u64::MAX);
        if leftover < excluded {
            let threshold = u128::from((u64::MAX - max) % (max + 1));
            while leftover < threshold {
                product = u128::from(self.next_u64()) * excluded;
                leftover = product & u128::from(u64::MAX);
            }
        }
        // The high half of a 64 x 64-bit product fits 64 bits.
        u64::try_from(product >> 64).unwrap_or(u64::MAX)
    }
}

/// `SeedSequence(entropy).generate_state(4, np.uint64)`.
fn seed_sequence_state(entropy: u64) -> [u64; 4] {
    // The entropy as 32-bit words, least significant first, at least one.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the entropy's 32-bit words, taken apart on purpose"
    )]
    let words: &[u32] = &if entropy >> 32 == 0 {
        vec![entropy as u32]
    } else {
        vec![entropy as u32, (entropy >> 32) as u32]
    };

    // mix_entropy: at most two words, so all of them go in with the first
    // pass and none is left for the last.
    let mut hash = INIT_A;
    let mut hashmix = |value: u32| {
        let mut value = value ^ hash;
        hash = hash.wrapping_mul(MULT_A);
        value = value.wrapping_mul(hash);
        value ^ (value >> XSHIFT)
    };
    let mut pool = [0_u32; POOL_SIZE];
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(words.get(i).copied().unwrap_or(0));
    }
    let mix = |x: u32, y: u32| {
        let result = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        result ^ (result >> XSHIFT)
    };
    for source in 0..POOL_SIZE {
        for destination in 0..POOL_SIZE {
            if source != destination {
                let (Some(&from), Some(&to)) = (pool.get(source), pool.get(destination)) else {
                    continue;
                };
                let mixed = mix(to, hashmix(from));
                if let Some(slot) = pool.get_mut(destination) {
                    *slot = mixed;
                }
            }
        }
    }

    // generate_state: eight 32-bit words cycling over the pool, paired
    // little end first into four 64-bit ones.
    let mut hash = INIT_B;
    let mut out = [0_u64; 4];
    for (i, word) in out.iter_mut().enumerate() {
        let mut halves = [0_u32; 2];
        for (j, half) in halves.iter_mut().enumerate() {
            let mut value = pool
                .get((2 * i + j) % POOL_SIZE)
                .copied()
                .unwrap_or_default();
            value ^= hash;
            hash = hash.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash);
            *half = value ^ (value >> XSHIFT);
        }
        let [low, high] = halves;
        *word = u64::from(low) | (u64::from(high) << 32);
    }
    out
}

/// `default_rng(seed).choice(n, min(k, n), replace=False)` for a k of at
/// most 3: Floyd's algorithm, then the shuffle `choice` ends with. For so
/// few draws numpy always takes Floyd's path; its other, a shuffle of the
/// whole range, starts only past 10,000 values and n / 50 draws.
pub(crate) fn choose_few(seed: u64, n: u64, k: usize) -> Vec<u64> {
    debug_assert!(k <= 3, "numpy takes another path for many draws");
    let k = usize::try_from(n).map_or(k, |n| k.min(n));
    let mut rng = Pcg64::from_seed(seed);
    let mut chosen: Vec<u64> = Vec::with_capacity(k);
    // j runs over the last k values below n.
    let first = n - k as u64;
    for j in first..n {
        let value = rng.bounded(j);
        chosen.push(if chosen.contains(&value) { j } else { value });
    }
    // _shuffle_int(k, 1, ...): from the last position down to the second.
    for i in (1..k).rev() {
        let j = usize::try_from(rng.bounded(i as u64)).unwrap_or(i);
        chosen.swap(i, j);
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::{Pcg64, choose_few};

    #[test]
    fn draws_as_numpy_does() {
        // numpy.random.default_rng(12345).bit_generator.random_raw().
        assert_eq!(
            Pcg64::from_seed(12345).next_u64(),
            4_193_609_425_186_963_869
        );
        // default_rng(12345).choice(n, min(3, n), replace=False), from
        // numpy 2.5.3; the oracle's fixture holds many more n from the
        // service's numpy.
        for (n, want) in [
            (1, vec![0]),
            (2, vec![1, 0]),
            (3, vec![1, 0, 2]),
            (10, vec![2, 7, 5]),
            (1000, vec![227, 788, 697]),
            (1_000_000, vec![227_335, 788_646, 699_213]),
            // Past 32 bits, the draws are 64-bit.
            (
                4_294_967_301,
                vec![976_400_782, 3_424_658_566, 1_360_466_711],
            ),
        ] {
            assert_eq!(choose_few(12345, n, 3), want, "n = {n}");
        }
        assert_eq!(choose_few(12345, 0, 3), [] as [u64; 0]);
    }
}
