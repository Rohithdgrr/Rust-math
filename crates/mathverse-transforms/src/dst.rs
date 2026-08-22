//! Discrete Sine Transform (Type I) and its inverse.

/// Discrete Sine Transform, Type I (orthonormal).
///
/// `X[k] = √(2/(N+1))·Σᵢ x[i]·sin(π·(i+1)·(k+1)/(N+1))`.
pub fn dst1(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let scale = (2.0 / (n + 1) as f64).sqrt();
    (0..n).map(|k| {
        let s: f64 = (0..n).map(|i| x[i] * (core::f64::consts::PI * (i + 1) as f64 * (k + 1) as f64 / (n + 1) as f64).sin()).sum();
        scale * s
    }).collect()
}

/// Discrete Sine Transform, Type II (orthonormal).
///
/// `X[k] = c(k)·Σᵢ x[i]·sin(π·(i+0.5)·(k+1)/N)` with `c(N-1) = √(1/N)` and
/// `c(k) = √(2/N)` otherwise.
pub fn dst2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|k| {
        let c = if k == n - 1 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        let s: f64 = (0..n).map(|i| x[i] * (core::f64::consts::PI * (i as f64 + 0.5) * (k + 1) as f64 / n as f64).sin()).sum();
        c * s
    }).collect()
}

/// Discrete Sine Transform, Type III (orthonormal).
///
/// `X[k] = Σᵢ w(i)·x[i]·sin(π·(k+0.5)·(i+1)/N)` with `w(N−1) = √(1/N)` and
/// `w(i) = √(2/N)` otherwise — the exact transpose (inverse) of the
/// orthonormal [`dst2`].
pub fn dst3(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|k| {
        (0..n).map(|i| {
            let w = if i == n - 1 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
            w * x[i] * (core::f64::consts::PI * (k as f64 + 0.5) * (i + 1) as f64 / n as f64).sin()
        }).sum()
    }).collect()
}

/// Discrete Sine Transform, Type IV (orthonormal).
///
/// `X[k] = √(2/N)·Σᵢ x[i]·sin(π·(i+0.5)·(k+0.5)/N)`. Self-inverse up to
/// floating-point rounding (DST-IV is an involution).
pub fn dst4(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let scale = (2.0 / n as f64).sqrt();
    (0..n).map(|k| {
        let s: f64 = (0..n).map(|i| x[i] * (core::f64::consts::PI * (i as f64 + 0.5) * (k as f64 + 0.5) / n as f64).sin()).sum();
        scale * s
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dst1_test() {
        let x = [1.0, 2.0, 3.0];
        let y = dst1(&x);
        assert_eq!(y.len(), 3);
    }
}
