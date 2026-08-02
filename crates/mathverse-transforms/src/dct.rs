//! Discrete Cosine Transform (Type II) and its inverse.

pub fn dct2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|k| {
        let s: f64 = (0..n).map(|i| x[i] * (core::f64::consts::PI * (i as f64 + 0.5) * k as f64 / n as f64).cos()).sum();
        let c = if k == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        c * s
    }).collect()
}

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

pub fn dct1(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 { return x.to_vec(); }
    (0..n).map(|k| {
        let s: f64 = (0..n).map(|i| {
            let scale = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
            scale * x[i] * (core::f64::consts::PI * i as f64 * k as f64 / (n - 1) as f64).cos()
        }).sum();
        let c = if k == 0 || k == n - 1 { (1.0 / (2.0 * (n - 1) as f64)).sqrt() } else { (1.0 / (n - 1) as f64).sqrt() };
        c * s
    }).collect()
}

pub fn dct3(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|k| {
        let c = if k == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        let s: f64 = (0..n).map(|i| x[i] * (core::f64::consts::PI * i as f64 * (k as f64 + 0.5) / n as f64).cos()).sum();
        c * s
    }).collect()
}

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
