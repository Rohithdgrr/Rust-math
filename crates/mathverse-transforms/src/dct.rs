//! Discrete Cosine Transform (Type II) and its inverse.

/// Discrete Cosine Transform, Type II (orthonormal).
///
/// `X[k] = c(k)·Σᵢ x[i]·cos(π·(i+0.5)·k/N)` with `c(0) = √(1/N)` and
/// `c(k) = √(2/N)` otherwise. Inverted by [`idct2`].
pub fn dct2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|k| {
        let s: f64 = (0..n).map(|i| x[i] * (core::f64::consts::PI * (i as f64 + 0.5) * k as f64 / n as f64).cos()).sum();
        let c = if k == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        c * s
    }).collect()
}

/// Inverse Discrete Cosine Transform, Type II (orthonormal).
///
/// Inverts [`dct2`]; the orthonormal scaling makes it an exact round trip.
pub fn idct2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|i| {
        let mut s = x[0] / (n as f64).sqrt();
        for (k, &xk) in x.iter().enumerate().skip(1) {
            s += (2.0 / n as f64).sqrt() * xk * (core::f64::consts::PI * (i as f64 + 0.5) * k as f64 / n as f64).cos();
        }
        s
    }).collect()
}

/// Discrete Cosine Transform, Type I (orthonormal).
///
/// `X[k] = √(2/(N−1))·u(k)·Σᵢ u(i)·x[i]·cos(π·i·k/(N−1))` where `u(0) =
/// u(N−1) = 1/√2` and `u = 1` elsewhere. The matrix is symmetric and
/// orthogonal, so DCT-I is its own inverse. Requires `N ≥ 2`; shorter
/// inputs are returned unchanged.
pub fn dct1(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 { return x.to_vec(); }
    (0..n).map(|k| {
        let s: f64 = (0..n).map(|i| {
            let scale = if i == 0 || i == n - 1 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            scale * x[i] * (core::f64::consts::PI * i as f64 * k as f64 / (n - 1) as f64).cos()
        }).sum();
        let c = if k == 0 || k == n - 1 {
            (1.0 / (n - 1) as f64).sqrt()
        } else {
            (2.0 / (n - 1) as f64).sqrt()
        };
        c * s
    }).collect()
}

/// Discrete Cosine Transform, Type III (orthonormal).
///
/// `X[k] = Σᵢ w(i)·x[i]·cos(π·(k+0.5)·i/N)` with `w(0) = √(1/N)` and
/// `w(i) = √(2/N)` otherwise — the exact transpose (inverse) of the
/// orthonormal [`dct2`].
pub fn dct3(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|k| {
        (0..n).map(|i| {
            let w = if i == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
            w * x[i] * (core::f64::consts::PI * (k as f64 + 0.5) * i as f64 / n as f64).cos()
        }).sum()
    }).collect()
}

/// Discrete Cosine Transform, Type IV (orthonormal).
///
/// `X[k] = √(2/N)·Σᵢ x[i]·cos(π·(i+0.5)·(k+0.5)/N)`. Self-inverse up to
/// floating-point rounding (DCT-IV is an involution).
pub fn dct4(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let scale = (2.0 / n as f64).sqrt();
    (0..n).map(|k| {
        let s: f64 = (0..n).map(|i| x[i] * (core::f64::consts::PI * (i as f64 + 0.5) * (k as f64 + 0.5) / n as f64).cos()).sum();
        scale * s
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct2_roundtrip() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let back = idct2(&dct2(&x));
        for (a, b) in x.iter().zip(&back) { assert!((a - b).abs() < 1e-12); }
    }
}
