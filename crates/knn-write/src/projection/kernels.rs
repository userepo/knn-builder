//! The float32 distances Faiss's flat storage computes for the layer-1
//! build, summed in the order the service's library sums them, so equal
//! inputs give equal bits.
//!
//! The service's Faiss is built for generic x86-64 (`FAISS_OPT_LEVEL=generic`,
//! SSE2), where its distance kernels are plain loops GCC vectorizes under an
//! "imprecise" pragma. What GCC made of them, read off the compiled library
//! by comparing its results with these orders on random vectors:
//!
//! - every kernel adds the terms in four lanes over the largest multiple of
//!   4 of the dimension, lane k taking terms k, k + 4, ...;
//! - the lanes reduce as (lane 0 + lane 2) + (lane 1 + lane 3) and the
//!   remaining terms add one by one, except where a two-lane epilogue
//!   takes the first two of them: the lanes fold to two, (lane 0 + lane 2)
//!   and (lane 1 + lane 3), each takes one of the two terms, the two reduce,
//!   and a third remaining term adds last;
//! - the single L2 distance takes that epilogue for two or three remaining
//!   terms, the single inner product never, and the four-at-a-time forms of
//!   both only for three.
//!
//! A term is `(x - y) * (x - y)` or `x * y`, each rounded to float32; the
//! generic build has no fused multiply-add.

/// When the two-lane epilogue takes over the remaining terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Epilogue {
    /// For two or three remaining terms.
    TwoOrThree,
    /// For three remaining terms only.
    Three,
    /// Never.
    Never,
}

/// The sum of `term(x[i], y[i])` over the vectors' common length, in the
/// order above.
#[inline]
fn sum(x: &[f32], y: &[f32], epilogue: Epilogue, term: impl Fn(f32, f32) -> f32) -> f32 {
    let d = x.len().min(y.len());
    let (x, y) = (x.split_at(d).0, y.split_at(d).0);
    let (xs, x_rest) = x.as_chunks::<4>();
    let (ys, y_rest) = y.as_chunks::<4>();
    let mut lanes = [0.0_f32; 4];
    for (a, b) in xs.iter().zip(ys) {
        for ((lane, &a), &b) in lanes.iter_mut().zip(a).zip(b) {
            *lane += term(a, b);
        }
    }
    let mut rest = x_rest.iter().zip(y_rest).map(|(&a, &b)| term(a, b));
    let two_lanes = match epilogue {
        Epilogue::TwoOrThree => x_rest.len() >= 2,
        Epilogue::Three => x_rest.len() == 3,
        Epilogue::Never => false,
    };
    let [l0, l1, l2, l3] = lanes;
    if two_lanes && let (Some(first), Some(second)) = (rest.next(), rest.next()) {
        let low = (l0 + l2) + first;
        let high = (l1 + l3) + second;
        return rest.fold(low + high, |total, t| total + t);
    }
    rest.fold((l0 + l2) + (l1 + l3), |total, t| total + t)
}

fn l2_term(a: f32, b: f32) -> f32 {
    let t = a - b;
    t * t
}

fn ip_term(a: f32, b: f32) -> f32 {
    a * b
}

/// The squared L2 distance of two vectors of the same length
/// (`fvec_L2sqr`).
pub(crate) fn l2(x: &[f32], y: &[f32]) -> f32 {
    sum(x, y, Epilogue::TwoOrThree, l2_term)
}

/// The inner product of two vectors of the same length
/// (`fvec_inner_product`).
pub(crate) fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    sum(x, y, Epilogue::Never, ip_term)
}

/// Four squared L2 distances from `x` (`fvec_L2sqr_batch_4`).
pub(crate) fn l2_batch_4(x: &[f32], ys: [&[f32]; 4]) -> [f32; 4] {
    batch_4(x, ys, l2_term)
}

/// Four inner products with `x` (`fvec_inner_product_batch_4`).
pub(crate) fn inner_product_batch_4(x: &[f32], ys: [&[f32]; 4]) -> [f32; 4] {
    batch_4(x, ys, ip_term)
}

/// The four-at-a-time forms' sums, each written out: the level's searches
/// spend most of their time here, and an array's `map` is not always
/// inlined into them.
#[inline]
fn batch_4(x: &[f32], ys: [&[f32]; 4], term: impl Fn(f32, f32) -> f32 + Copy) -> [f32; 4] {
    let [y0, y1, y2, y3] = ys;
    [
        sum(x, y0, Epilogue::Three, term),
        sum(x, y1, Epilogue::Three, term),
        sum(x, y2, Epilogue::Three, term),
        sum(x, y3, Epilogue::Three, term),
    ]
}

#[cfg(test)]
mod tests {
    use super::{inner_product, inner_product_batch_4, l2, l2_batch_4};

    #[test]
    fn sums_in_four_lanes_then_the_rest() {
        // Terms 1e8, 1, -1e8, 1 then 3: lanes (1e8 + -1e8) + (1 + 1) = 2,
        // then + 3. A plain left-to-right sum would lose the ones.
        let x = [1e8_f32, 1.0, -1e8, 1.0, 3.0];
        let ones = [1.0_f32; 5];
        assert_eq!(inner_product(&x, &ones), 5.0);
        assert_eq!(inner_product_batch_4(&x, [&ones; 4]), [5.0; 4]);
        // Squares: 16 + 0 + 16 + 0, then a single remaining term of 4.
        let y = [4.0_f32, 0.0, 4.0, 0.0, 2.0];
        let zero = [0.0_f32; 5];
        assert_eq!(l2(&y, &zero), 36.0);
        assert_eq!(l2_batch_4(&y, [&zero; 4]), [36.0; 4]);
    }

    #[test]
    fn takes_the_two_lane_epilogue_where_faiss_does() {
        // Every value here is what the service's Faiss returns for the same
        // inputs.
        //
        // Squares 2^24, 1, 0, 0 in the lanes, then 1, 1 remaining. The
        // two-lane epilogue adds a 1 to each folded lane, (2^24 + 1) + (1 +
        // 1), which keeps 2; the plain order adds each 1 to 2^24 and rounds
        // it away. Float32 holds only even numbers there.
        let zero = [0.0_f32; 7];
        let two = [4096.0_f32, 1.0, 0.0, 0.0, 1.0, 1.0];
        assert_eq!(l2(&two, &zero[..6]), 16_777_218.0);
        assert_eq!(l2_batch_4(&two, [&zero[..6]; 4]), [16_777_216.0; 4]);
        // Three remaining: both take the epilogue, the third 1 rounding the
        // total up to 2^24 + 4.
        let three = [4096.0_f32, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(l2(&three, &zero), 16_777_220.0);
        assert_eq!(l2_batch_4(&three, [&zero; 4]), [16_777_220.0; 4]);

        // Products 1e8, -1e8, 0, 0 in the lanes, then 1s: the plain order
        // cancels the lanes first and keeps every 1; the epilogue adds a 1
        // to each of 1e8 and -1e8, where it rounds away.
        let ones = [1.0_f32; 7];
        let x2 = [1e8_f32, -1e8, 0.0, 0.0, 1.0, 1.0];
        assert_eq!(inner_product(&x2, &ones[..6]), 2.0);
        assert_eq!(inner_product_batch_4(&x2, [&ones[..6]; 4]), [2.0; 4]);
        let x3 = [1e8_f32, -1e8, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(inner_product(&x3, &ones), 3.0);
        assert_eq!(inner_product_batch_4(&x3, [&ones; 4]), [1.0; 4]);
    }
}
