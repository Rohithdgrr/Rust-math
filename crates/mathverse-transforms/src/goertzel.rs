//! Goertzel algorithm for single-frequency tone detection.

pub fn goertzel(x: &[f64], k: usize) -> Option<(f64, f64)> {
    let n = x.len();
    if n == 0 || k >= n { return None; }
    let w = 2.0 * core::f64::consts::PI * k as f64 / n as f64;
    let cosine = w.cos();
    let coeff = 2.0 * cosine;
    let (mut s_prev, mut s_prev2) = (0.0, 0.0);
    for &xi in x {
        let s = xi + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    let real = s_prev - cosine * s_prev2;
    let imag = w.sin() * s_prev2;
    Some((real, imag))
}

pub fn goertzel_magnitude(x: &[f64], k: usize) -> f64 {
    goertzel(x, k).map(|(re, im)| (re * re + im * im).sqrt()).unwrap_or(0.0)
}

pub fn goertzel_batch(x: &[f64], ks: &[usize]) -> Vec<(f64, f64)> {
    ks.iter().map(|&k| goertzel(x, k).unwrap_or((0.0, 0.0))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goertzel_test() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let (re, im) = goertzel(&x, 0).unwrap();
        assert!((re - 10.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }
}
