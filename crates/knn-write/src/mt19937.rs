//! `std::mt19937`, the 32-bit Mersenne Twister Faiss's `RandomGenerator`
//! wraps (`utils/random.h`), so the level draws match the service's.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// The generator, seeded as `std::mt19937(seed)` is.
#[derive(Debug, Clone)]
pub(crate) struct Mt19937 {
    state: [u32; N],
    next: usize,
}

impl Mt19937 {
    #[expect(
        clippy::indexing_slicing,
        reason = "i runs from 1 to N - 1 over an array of N"
    )]
    pub(crate) fn new(seed: u32) -> Self {
        let mut state = [0_u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            // i < 624, so the cast is exact.
            #[expect(clippy::cast_possible_truncation, reason = "i < 624")]
            let index = i as u32;
            state[i] = 1_812_433_253_u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(index);
        }
        Self { state, next: N }
    }

    /// The next 32-bit output, as `std::mt19937::operator()` returns it.
    #[expect(clippy::indexing_slicing, reason = "next < N once the twist has run")]
    pub(crate) fn next_u32(&mut self) -> u32 {
        if self.next >= N {
            self.twist();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "every index is i, (i + 1) % N or (i + M) % N with i < N"
    )]
    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mag = if y & 1 == 0 { 0 } else { MATRIX_A };
            self.state[i] = self.state[(i + M) % N] ^ (y >> 1) ^ mag;
        }
        self.next = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::Mt19937;

    #[test]
    fn matches_the_standards_check_value() {
        // C++ [rand.predef]: the 10000th output of a default-constructed
        // std::mt19937 (seed 5489) is 4123659995.
        let mut rng = Mt19937::new(5489);
        let mut last = 0;
        for _ in 0..10_000 {
            last = rng.next_u32();
        }
        assert_eq!(last, 4_123_659_995);
    }

    #[test]
    fn matches_the_seed_faiss_uses() {
        // std::mt19937(12345), as numpy's legacy RandomState(12345) draws it.
        let mut rng = Mt19937::new(12345);
        let first: Vec<u32> = (0..4).map(|_| rng.next_u32()).collect();
        assert_eq!(
            first,
            [3_992_670_690, 3_823_185_381, 1_358_822_685, 561_383_553]
        );
    }
}
