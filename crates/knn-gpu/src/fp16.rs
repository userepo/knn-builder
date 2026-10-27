//! fp32 to fp16, as the upstream service converts an fp32 blob sent under
//! `data_type: half_float` (a float field with the fp16 scalar quantizer)
//! on download: numpy's float32 to float16 cast, rounding to nearest even,
//! values past float16's range becoming infinity. `tests/data/fp16.txt`
//! holds numpy's results.

/// The IEEE half-precision bit pattern nearest `value`, ties to even.
///
/// NaN keeps its sign and stays NaN, but its payload may differ from
/// numpy's; k-NN rejects NaN vectors before a build.
#[must_use]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    // The sign moves from bit 31 to bit 15.
    let sign = low_u16((bits >> 16) & 0x8000);
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x7f_ffff;

    if exponent == 0xff {
        // Infinity, or NaN with the top of its payload and the quiet bit.
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00 | low_u16(mantissa >> 13);
    }
    // The exponent rebiased from float's 127 to half's 15.
    let half_exponent = i32::from(u8::try_from(exponent).unwrap_or(0)) - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        // Subnormal in half precision, or zero: below 2^-25 everything
        // rounds to zero (2^-25 itself ties to the even zero).
        if half_exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x80_0000;
        let shift = (14 - half_exponent).unsigned_abs();
        return sign | round_shifted(significand, shift);
    }
    let exponent_bits = half_exponent.unsigned_abs() << 10;
    // A carry out of the mantissa raises the exponent, and past the largest
    // finite value lands on infinity, as rounding should.
    sign | low_u16(exponent_bits + u32::from(round_shifted(mantissa, 13)))
}

/// `value >> shift`, rounded to nearest, ties to even; `shift` from 13 to
/// 24, so the result fits 11 bits.
fn round_shifted(value: u32, shift: u32) -> u16 {
    let kept = value >> shift;
    let rest = value & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let up = rest > half || (rest == half && kept & 1 == 1);
    low_u16(kept + u32::from(up))
}

/// The low 16 bits; every caller's value is below 2^16.
fn low_u16(value: u32) -> u16 {
    u16::try_from(value & 0xffff).unwrap_or(0)
}

/// Converts `values` into `out`, element by element, as far as the shorter
/// of the two reaches; callers size `out` to `values`.
pub fn convert(values: &[f32], out: &mut [u16]) {
    for (to, &from) in out.iter_mut().zip(values) {
        *to = f32_to_f16(from);
    }
}

#[cfg(test)]
mod tests {
    use super::f32_to_f16;

    #[test]
    fn rounds_ties_to_even_and_overflows_to_infinity() {
        // 1 + 2^-11 lies halfway between 1 and the next half, 1 + 2^-10:
        // the even one, 1, wins.
        assert_eq!(f32_to_f16(1.000_488_3), 0x3c00);
        assert_eq!(f32_to_f16(65_520.0), 0x7c00);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        // The smallest subnormal half, 2^-24, and the tie below it.
        assert_eq!(f32_to_f16(5.960_464_5e-8), 0x0001);
        assert_eq!(f32_to_f16(2.980_232_2e-8), 0x0000);
        assert_eq!(f32_to_f16(f32::NAN) & 0x7e00, 0x7e00);
    }
}
