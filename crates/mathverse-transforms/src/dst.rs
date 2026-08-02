//! Discrete Sine Transform (Type I) and its inverse.

pub fn dst1(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let scale = (2.0 / (n + 1) as f64).sqrt();
    (0..n).map(|k| {
        let s: f64 = (0..n).map(|i| x[i] * (core::f64::consts::PI * (i + 1) as f64 * (k + 1) as f64 / (n + 1) as f64).sin()).sum();
        scale * s
    }).collect()
}

pub fn dst2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|k| {
        let c = if k == n - 1 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        let s: f64 = (0..n).map(|i| x[i] * (core::f64::consts::PI * (i as f64 + 0.5) * (k + 1) as f64 / n as f64).sin()).sum();
        c * s
    }).collect()
}

pub fn dst3(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|k| {
        let c = if k == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        let s: f64 = (0..n).map(|i| x[i] * (core::f64::consts::PI * i as f64 * (k as f64 + 0.5) / n as f64).sin()).sum();
        c * s
    }).collect()
}

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
