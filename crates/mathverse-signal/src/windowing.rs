//! Window functions: Hamming, Hanning, Blackman, Blackman-Harris, Kaiser, rectangular.

pub fn window_hamming(len: usize) -> Vec<f64> {
    if len <= 1 { return vec![1.0; len]; }
    (0..len).map(|i| { let n = i as f64 / (len - 1) as f64; 0.54 - 0.46 * (2.0 * core::f64::consts::PI * n).cos() }).collect()
}

pub fn window_hanning(len: usize) -> Vec<f64> {
    if len <= 1 { return vec![1.0; len]; }
    (0..len).map(|i| { let n = i as f64 / (len - 1) as f64; 0.5 * (1.0 - (2.0 * core::f64::consts::PI * n).cos()) }).collect()
}

pub fn window_blackman(len: usize) -> Vec<f64> {
    if len <= 1 { return vec![1.0; len]; }
    (0..len).map(|i| { let n = i as f64 / (len - 1) as f64; 0.42 - 0.5 * (2.0 * core::f64::consts::PI * n).cos() + 0.08 * (4.0 * core::f64::consts::PI * n).cos() }).collect()
}

pub fn window_bartlett(len: usize) -> Vec<f64> {
    if len == 1 { return vec![1.0]; }
    (0..len).map(|i| { let n = i as f64 / (len - 1) as f64; if n <= 0.5 { 2.0 * n } else { 2.0 * (1.0 - n) } }).collect()
}

pub fn window_rectangular(len: usize) -> Vec<f64> { vec![1.0; len] }

pub fn window_flat_top(len: usize) -> Vec<f64> {
    if len <= 1 { return vec![1.0; len]; }
    (0..len).map(|i| { let n = i as f64 / (len - 1) as f64;
        0.21557895 - 0.41663158 * (2.0 * core::f64::consts::PI * n).cos() + 0.277263158 * (4.0 * core::f64::consts::PI * n).cos()
        - 0.083578947 * (6.0 * core::f64::consts::PI * n).cos() + 0.006947368 * (8.0 * core::f64::consts::PI * n).cos()
    }).collect()
}

pub fn window_kaiser(len: usize, beta: f64) -> Vec<f64> {
    let n = len as f64;
    (0..len).map(|i| {
        let x = 2.0 * i as f64 / (n - 1.0) - 1.0;
        bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta)
    }).collect()
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x2 = x * x / 4.0;
    for k in 1..=25 { term *= x2 / (k as f64 * k as f64); sum += term; }
    sum
}

pub fn window_gaussian(len: usize, sigma: f64) -> Vec<f64> {
    let n = len as f64;
    (0..len).map(|i| { let x = (i as f64 - (n - 1.0) / 2.0) / (sigma * (n - 1.0) / 2.0); (-x * x / 2.0).exp() }).collect()
}

pub fn apply_window(signal: &[f64], window: &[f64]) -> Vec<f64> {
    signal.iter().zip(window).map(|(s, w)| s * w).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows() {
        assert_eq!(window_hamming(64).len(), 64);
        assert_eq!(window_hanning(64).len(), 64);
        assert_eq!(window_blackman(64).len(), 64);
        assert_eq!(window_bartlett(64).len(), 64);
        assert_eq!(window_kaiser(64, 3.0).len(), 64);
    }
}
