//! Riemann zeta function ζ(s) on the real line.

use core::f64::consts::PI;

/// Bernoulli numbers B₂, B₄, …, B₁₆.
const BERNOULLI_EVEN: [f64; 8] = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
];

/// Exact value of ζ(2n) = (−1)^{n+1} B_{2n} (2π)^{2n} / (2 (2n)!).
fn zeta_even_integer(n: u64) -> f64 {
    let mut fact = 1.0;
    for i in 2..=2 * n {
        fact *= i as f64;
    }
    let b = BERNOULLI_EVEN[(n - 1) as usize];
    let sign = if n % 2 == 0 { -1.0 } else { 1.0 }; // (-1)^{n+1}
    let two_pi = 2.0 * PI;
    sign * b * two_pi.powi(2 * n as i32) / (2.0 * fact)
}

/// Riemann zeta function ζ(s) for real s > 1.
///
/// Uses direct summation with a tail estimate via the Euler–Maclaurin
/// formula. Returns `+∞` at `s = 1` and `NaN` for `s ≤ 1`.
///
/// ```
/// use mathverse_special::zeta;
/// assert!((zeta(2.0) - core::f64::consts::PI * core::f64::consts::PI / 6.0).abs() < 1e-10);
/// assert!((zeta(3.0) - 1.202_056_903_159_594).abs() < 1e-10);
/// assert!(zeta(1.0).is_infinite());
/// ```
pub fn zeta(s: f64) -> f64 {
    if s.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY;
    }
    if s <= 1.0 {
        return f64::NAN;
    }
    if s.fract() == 0.0 && s > 1.0 && s <= 16.0 {
        let n = s as u64;
        if n % 2 == 0 {
            return zeta_even_integer(n / 2);
        }
    }
    // Direct summation up to N, then approximate the tail.
    const N: u64 = 1000;
    let mut sum = 0.0;
    for n in 1..=N {
        sum += (n as f64).powf(-s);
    }
    // Tail estimate: Σ_{n=N+1}^{∞} n^{-s} ≈ ∫_N^∞ x^{-s} dx + ½·N^{-s}
    // = N^{1-s}/(s-1) + ½·N^{-s}
    let nm = N as f64;
    sum += nm.powf(1.0 - s) / (s - 1.0);
    sum += 0.5 * nm.powf(-s);
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeta_known_values() {
        let z3 = zeta(3.0);
        let expected = 1.202_056_903_159_594;
        eprintln!("zeta(3.0) = {z3:.15}, expected = {expected:.15}, diff = {}", (z3 - expected).abs());
        assert!((z3 - expected).abs() < 1e-10, "zeta(3.0)={z3}, expected={expected}");
        assert!((zeta(4.0) - 1.082_323_233_711_138).abs() < 1e-12);
        assert!((zeta(6.0) - 1.017_343_061_984_449).abs() < 1e-10);
        assert!((zeta(8.0) - 1.004_077_356_197_944).abs() < 1e-10);
        assert!((zeta(3.0) - 1.202_056_903_159_594).abs() < 1e-10);
        assert!((zeta(5.0) - 1.036_927_755_143_370).abs() < 1e-10);
        assert!((zeta(1.5) - 2.612_375_348_685_488).abs() < 1e-9);
        assert!((zeta(10.0) - 1.000_994_575_127_818).abs() < 1e-10);
    }

    #[test]
    fn zeta_edges() {
        assert!(zeta(1.0).is_infinite());
        assert!(zeta(0.5).is_nan());
        assert!(zeta(-2.0).is_nan());
        assert!(zeta(f64::NAN).is_nan());
    }
}