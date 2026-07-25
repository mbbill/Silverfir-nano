//! The float operations `core` does not provide, in software.
//!
//! The slow path must be able to execute every op the target's handler set
//! declines, and two configurations make that a real requirement rather
//! than a formality: a `no_std` embedding (an MCU has no libm, and the
//! project takes no dependencies), and the x86-64 baseline, which has no
//! `roundss`/`roundsd` and therefore no native ceil/floor/trunc/nearest.
//!
//! These are the software forms in every build, not just the `no_std`
//! ones: a fallback that only compiles where it is never exercised is a
//! fallback nobody has tested. `tests` below pins them against libstd.
//!
//! `abs` and `copysign` are not here — they are sign-bit masks, which
//! `core` does expose.

/// Make a NaN quiet, leaving everything else alone.
///
/// Every wasm float operation that takes a NaN operand must produce an
/// ARITHMETIC NaN — one with the payload's high bit set. Hardware rounding
/// instructions do that for free, which is why this only shows up in the
/// software forms: an f32 `trunc` of `nan:0x200000` that passes the operand
/// through unchanged returns a signalling NaN and fails the spec suite.
fn quiet64(x: f64) -> f64 {
    if x.is_nan() {
        f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000)
    } else {
        x
    }
}

fn quiet32(x: f32) -> f32 {
    if x.is_nan() {
        f32::from_bits(x.to_bits() | 0x0040_0000)
    } else {
        x
    }
}

/// Truncate toward zero by clearing the fractional bits of the
/// significand. Correct for every class: subnormals go to a signed zero,
/// and infinities and NaNs are already "integral" by exponent.
pub(super) fn trunc64(x: f64) -> f64 {
    let x = quiet64(x);
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    if e < 0 {
        return f64::from_bits(bits & 0x8000_0000_0000_0000);
    }
    if e >= 52 {
        return x;
    }
    let mask = (1u64 << (52 - e)) - 1;
    f64::from_bits(bits & !mask)
}

pub(super) fn trunc32(x: f32) -> f32 {
    let x = quiet32(x);
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    if e < 0 {
        return f32::from_bits(bits & 0x8000_0000);
    }
    if e >= 23 {
        return x;
    }
    let mask = (1u32 << (23 - e)) - 1;
    f32::from_bits(bits & !mask)
}

/// `trunc` differs from `floor` only below zero, and the correction is
/// exact: `trunc(x) != x` implies `|x| < 2^52`, where adding one is exact.
/// A NaN compares false both ways and falls through unchanged.
pub(super) fn floor64(x: f64) -> f64 {
    let t = trunc64(x);
    if t > x {
        t - 1.0
    } else {
        t
    }
}

pub(super) fn floor32(x: f32) -> f32 {
    let t = trunc32(x);
    if t < x {
        t
    } else if t > x {
        t - 1.0
    } else {
        t
    }
}

pub(super) fn ceil64(x: f64) -> f64 {
    let t = trunc64(x);
    if t < x {
        t + 1.0
    } else {
        t
    }
}

pub(super) fn ceil32(x: f32) -> f32 {
    let t = trunc32(x);
    if t < x {
        t + 1.0
    } else {
        t
    }
}

/// Whether an integral float is odd. Only meaningful for `|t| < 2^52`,
/// which is the only range the tie case in [`nearest64`] can reach.
fn odd64(t: f64) -> bool {
    let bits = t.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    if !(0..52).contains(&e) {
        return false;
    }
    let m = (bits & 0x000f_ffff_ffff_ffff) | (1u64 << 52);
    (m >> (52 - e)) & 1 == 1
}

fn odd32(t: f32) -> bool {
    let bits = t.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    if !(0..23).contains(&e) {
        return false;
    }
    let m = (bits & 0x007f_ffff) | (1u32 << 23);
    (m >> (23 - e)) & 1 == 1
}

/// Round to nearest, ties to even — wasm's `nearest`.
///
/// Written as explicit comparisons rather than the usual "add and subtract
/// 2^52" trick because that trick is only correct under a round-to-nearest
/// FPU mode, and on a soft-float target the mode belongs to
/// `compiler_builtins` rather than to us. Every arithmetic step here is
/// exact: `x - t`, `t + 1` and `t - 1` all stay inside `|x| < 2^52`.
pub(super) fn nearest64(x: f64) -> f64 {
    let t = trunc64(x);
    if t == x {
        return x;
    }
    let diff = x - t;
    let r = if diff > 0.5 {
        t + 1.0
    } else if diff < -0.5 {
        t - 1.0
    } else if diff == 0.5 {
        if odd64(t) {
            t + 1.0
        } else {
            t
        }
    } else if diff == -0.5 {
        if odd64(t) {
            t - 1.0
        } else {
            t
        }
    } else {
        t
    };
    // `nearest(-0.3)` is -0.0, and `t + 1.0` style arithmetic loses that.
    if r == 0.0 {
        f64::from_bits(x.to_bits() & 0x8000_0000_0000_0000)
    } else {
        r
    }
}

pub(super) fn nearest32(x: f32) -> f32 {
    let t = trunc32(x);
    if t == x {
        return x;
    }
    let diff = x - t;
    let r = if diff > 0.5 {
        t + 1.0
    } else if diff < -0.5 {
        t - 1.0
    } else if diff == 0.5 {
        if odd32(t) {
            t + 1.0
        } else {
            t
        }
    } else if diff == -0.5 {
        if odd32(t) {
            t - 1.0
        } else {
            t
        }
    } else {
        t
    };
    if r == 0.0 {
        f32::from_bits(x.to_bits() & 0x8000_0000)
    } else {
        r
    }
}

/// Integer square root with remainder, by the classic restoring
/// bit-pair algorithm. Returns `(floor(sqrt(n)), n - floor(sqrt(n))^2)`.
fn isqrt_rem(n: u128) -> (u128, u128) {
    if n == 0 {
        return (0, 0);
    }
    let mut bit: u128 = 1u128 << 126;
    while bit > n {
        bit >>= 2;
    }
    let mut res: u128 = 0;
    let mut num = n;
    while bit != 0 {
        if num >= res + bit {
            num -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    (res, num)
}

/// Correctly rounded IEEE-754 square root.
///
/// The result significand is `sqrt(M << 52)` where `M` is the operand's
/// significand scaled to an even exponent. That value is never exactly
/// halfway between two representable results — `(Q + 1/2)^2` is not an
/// integer — so rounding to nearest needs no tie rule at all: round up
/// exactly when the remainder exceeds the root.
pub(super) fn sqrt64(x: f64) -> f64 {
    let bits = x.to_bits();
    if x.is_nan() {
        return quiet64(x);
    }
    if bits & 0x7fff_ffff_ffff_ffff == 0 {
        return x; // +-0
    }
    if bits >> 63 == 1 {
        return f64::from_bits(0x7ff8_0000_0000_0000); // negative -> canonical NaN
    }
    if x.is_infinite() {
        return x;
    }
    let be = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    let (mut sig, mut unb) = if be == 0 {
        let sh = frac.leading_zeros() - 11;
        (frac << sh, -1022 - sh as i32)
    } else {
        (frac | (1u64 << 52), be - 1023)
    };
    if unb & 1 != 0 {
        sig <<= 1;
        unb -= 1;
    }
    let (q, r) = isqrt_rem((sig as u128) << 52);
    let mut res = if r > q { q + 1 } else { q };
    let mut rexp = unb / 2;
    if res >= 1u128 << 53 {
        res >>= 1;
        rexp += 1;
    }
    f64::from_bits(((rexp + 1023) as u64) << 52 | (res as u64 & 0x000f_ffff_ffff_ffff))
}

/// f32 square root via the f64 one. Double rounding is safe here: 53 bits
/// is more than the `2 * 24 + 2` a correctly-rounded square root needs to
/// survive a second rounding to 24.
pub(super) fn sqrt32(x: f32) -> f32 {
    if x.is_nan() {
        return quiet32(x);
    }
    if x.to_bits() >> 31 == 1 && x != 0.0 {
        return f32::from_bits(0x7fc0_0000);
    }
    sqrt64(x as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every interesting class, plus a deterministic sweep. Pinned against
    /// libstd, which is the reference these replace.
    fn probes64() -> alloc::vec::Vec<f64> {
        let mut v = alloc::vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            -2.5,
            0.49999999999999994,
            -0.49999999999999994,
            1.0,
            -1.0,
            4503599627370495.5,
            -4503599627370495.5,
            4503599627370496.0,
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE / 4.0,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            2.0,
            3.0,
            1e300,
            1e-300,
            f64::from_bits(0x7ff4_0000_0000_0000),
            f64::from_bits(0xfff4_0000_0000_0000),
            f64::from_bits(0x7ff8_0000_0000_0000),
            f64::from_bits(0xfff8_0000_0000_0000),
        ];
        // A reproducible spread over exponents and mantissas.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..4000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = f64::from_bits(s);
            if !x.is_nan() {
                v.push(x);
            }
            v.push((s as i64 as f64) / 65536.0);
        }
        v
    }

    /// Every rounding op must return a QUIET NaN for a NaN operand —
    /// wasm's `nan:arithmetic`. libstd's forms lower to hardware rounding
    /// instructions, which quiet for free, so this is checked directly
    /// rather than through the comparison below.
    #[test]
    fn rounding_ops_quiet_signalling_nan() {
        let s64 = f64::from_bits(0x7ff4_0000_0000_0000);
        let s32 = f32::from_bits(0x7fa0_0000);
        for got in [
            trunc64(s64),
            floor64(s64),
            ceil64(s64),
            nearest64(s64),
            sqrt64(s64),
        ] {
            assert!(got.is_nan());
            assert_ne!(got.to_bits() & 0x0008_0000_0000_0000, 0, "quiet bit");
        }
        for got in [
            trunc32(s32),
            floor32(s32),
            ceil32(s32),
            nearest32(s32),
            sqrt32(s32),
        ] {
            assert!(got.is_nan());
            assert_ne!(got.to_bits() & 0x0040_0000, 0, "quiet bit");
        }
    }

    #[test]
    fn rounding_ops_match_libstd() {
        for x in probes64() {
            if x.is_nan() {
                continue; // covered by the quieting test above
            }
            assert_eq!(trunc64(x).to_bits(), x.trunc().to_bits(), "trunc {x:e}");
            assert_eq!(floor64(x).to_bits(), x.floor().to_bits(), "floor {x:e}");
            assert_eq!(ceil64(x).to_bits(), x.ceil().to_bits(), "ceil {x:e}");
            assert_eq!(
                nearest64(x).to_bits(),
                x.round_ties_even().to_bits(),
                "nearest {x:e}"
            );
            let y = x as f32;
            if !y.is_nan() {
                assert_eq!(trunc32(y).to_bits(), y.trunc().to_bits(), "trunc32 {y:e}");
                assert_eq!(floor32(y).to_bits(), y.floor().to_bits(), "floor32 {y:e}");
                assert_eq!(ceil32(y).to_bits(), y.ceil().to_bits(), "ceil32 {y:e}");
                assert_eq!(
                    nearest32(y).to_bits(),
                    y.round_ties_even().to_bits(),
                    "nearest32 {y:e}"
                );
            }
        }
    }

    #[test]
    fn sqrt_matches_libstd() {
        for x in probes64() {
            if x.is_nan() {
                continue; // covered by the quieting test above
            }
            let got = sqrt64(x);
            let want = x.sqrt();
            if want.is_nan() {
                assert!(got.is_nan(), "sqrt {x:e}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "sqrt {x:e}");
            }
            let y = x as f32;
            if !y.is_nan() {
                let g = sqrt32(y);
                let w = y.sqrt();
                if w.is_nan() {
                    assert!(g.is_nan(), "sqrt32 {y:e}");
                } else {
                    assert_eq!(g.to_bits(), w.to_bits(), "sqrt32 {y:e}");
                }
            }
        }
    }
}
