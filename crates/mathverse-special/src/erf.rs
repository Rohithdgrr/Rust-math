//! Error function [`erf`] and complementary error function [`erfc`].

/// Error function erf(x) = (2/√π) ∫₀ˣ e^(−t²) dt.
///
/// Abramowitz & Stegun 7.1.26 approximation, absolute error ≤ 1.5e-7 for all
/// real `x`. Saturates to ±1 for |x| ≥ 4.
///
/// ```
/// use mathverse_special::erf;
/// assert!((erf(0.0) - 0.0).abs() < 1e-15);
/// assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 2e-7);
/// assert!((erf(-1.0) + 0.842_700_792_949_714_9).abs() < 2e-7);
/// assert!((erf(4.0) - 1.0).abs() < 2e-7);
/// ```
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    if x.abs() > 4.0 {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    // A&S 7.1.26: erf(x) = 1 − (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)e^(−x²), t = 1/(1+px)
    const P: f64 = 0.327_591_1;
    const A: [f64; 5] = [
        0.254_829_592,
        -0.284_496_736,
        1.421_413_741,
        -1.453_152_027,
        1.061_405_429,
    ];
    let t = 1.0 / (1.0 + P * ax);
    let mut poly = 0.0;
    for c in A.iter().rev() {
        poly = poly * t + c;
    }
    sign * (1.0 - poly * t * (-ax * ax).exp())
}

/// Complementary error function erfc(x) = 1 − erf(x).
///
/// Uses the A&S 7.1.26 polynomial directly on `|x|` so large arguments do
/// not suffer cancellation from `1 − erf(x)`.
///
/// ```
/// use mathverse_special::erfc;
/// assert!((erfc(1.0) - 0.157_299_207_050_285_1).abs() < 2e-7);
/// assert!((erfc(4.0) - 0.000_000_015_797).abs() < 2e-8);
/// assert!((erfc(0.0) - 1.0).abs() < 1e-15);
/// ```
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    let ax = x.abs();
    if ax > 4.0 {
        // direct asymptotic-ish A&S evaluation: poly(t) e^(−x²), t = 1/(1+px)
        const P: f64 = 0.327_591_1;
        const A: [f64; 5] = [
            0.254_829_592,
            -0.284_496_736,
            1.421_413_741,
            -1.453_152_027,
            1.061_405_429,
        ];
        let t = 1.0 / (1.0 + P * ax);
        let mut poly = 0.0;
        for c in A.iter().rev() {
            poly = poly * t + c;
        }
        let small = poly * t * (-ax * ax).exp();
        return if x > 0.0 { small } else { 2.0 - small };
    }
    if x >= 0.0 {
        1.0 - erf(x)
    } else {
        1.0 + erf(-x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_identity() {
        assert!((erf(0.0) - 0.0).abs() < 1e-8);
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 2e-7);
        assert!((erf(-1.0) + 0.842_700_792_949_714_9).abs() < 2e-7);
        assert!((erf(2.0) - 0.995_322_265).abs() < 2e-7);
    }

    #[test]
    fn erfc_identity() {
        for x in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, -1.0] {
            assert!((erfc(x) + erf(x) - 1.0).abs() < 2e-7, "mismatch at x={x}");
        }
        // Large positive argument: no cancellation in erfc itself.
        assert!(erfc(5.0) > 0.0 && erfc(5.0) < 1e-10);
        assert!((erfc(0.0) - 1.0).abs() < 1e-8);
    }
}
