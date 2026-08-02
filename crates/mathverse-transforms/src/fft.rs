//! Radix-2 Cooley-Tukey FFT using complex numbers.

use mathverse_complex::Complex;

pub fn fft(x: &[Complex]) -> mathverse_core::error::MathResult<Vec<Complex>> {
    let n = x.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(mathverse_core::error::MathError::InvalidArgument("fft: length must be nonzero power of two"));
    }
    let mut a = x.to_vec();
    for i in 0..n { let j = i.reverse_bits() >> (usize::BITS - n.trailing_zeros()); if j > i { a.swap(i, j); } }
    let mut len = 2;
    while len <= n {
        let wlen = Complex::polar(1.0, -2.0 * core::f64::consts::PI / len as f64);
        let mut i = 0;
        while i < n {
            let mut w = Complex::real(1.0);
            for k in 0..len / 2 {
                let u = a[i + k];
                let v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w = w * wlen;
            }
            i += len;
        }
        len *= 2;
    }
    Ok(a)
}

pub fn ifft(x: &[Complex]) -> mathverse_core::error::MathResult<Vec<Complex>> {
    let n = x.len();
    let conj: Vec<Complex> = x.iter().map(Complex::conjugate).collect();
    let y = fft(&conj)?;
    Ok(y.iter().map(|c| c.conjugate() / Complex::real(n as f64)).collect())
}

pub fn dft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    (0..n).map(|k| {
        (0..n).map(|i| {
            let angle = -2.0 * core::f64::consts::PI * k as f64 * i as f64 / n as f64;
            x[i] * Complex::polar(1.0, angle)
        }).fold(Complex::zero(), |a, b| a + b)
    }).collect()
}

pub fn idft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    (0..n).map(|k| {
        (0..n).map(|i| {
            let angle = 2.0 * core::f64::consts::PI * k as f64 * i as f64 / n as f64;
            x[i] * Complex::polar(1.0, angle)
        }).fold(Complex::zero(), |a, b| a + b) / Complex::real(n as f64)
    }).collect()
}

pub fn fft_real(x: &[f64]) -> Vec<Complex> {
    let xc: Vec<Complex> = x.iter().map(|&v| Complex::real(v)).collect();
    fft(&xc).unwrap_or_default()
}

pub fn power_spectrum(x: &[f64]) -> Vec<f64> {
    fft_real(x).iter().map(Complex::norm_sq).collect()
}

pub fn cross_correlation(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len()).next_power_of_two();
    let mut fa = vec![Complex::zero(); n];
    let mut fb = vec![Complex::zero(); n];
    for i in 0..a.len() { fa[i] = Complex::real(a[i]); }
    for i in 0..b.len() { fb[i] = Complex::real(b[i]); }
    let pa = fft(&fa).unwrap();
    let pb = fft(&fb).unwrap();
    let product: Vec<Complex> = pa.iter().zip(pb.iter()).map(|(a, b)| *a * b.conjugate()).collect();
    let result = ifft(&product).unwrap();
    result.iter().map(|c| c.re).collect()
}

pub fn convolution(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = (a.len() + b.len() - 1).next_power_of_two();
    let mut fa = vec![Complex::zero(); n];
    let mut fb = vec![Complex::zero(); n];
    for i in 0..a.len() { fa[i] = Complex::real(a[i]); }
    for i in 0..b.len() { fb[i] = Complex::real(b[i]); }
    let pa = fft(&fa).unwrap();
    let pb = fft(&fb).unwrap();
    let product: Vec<Complex> = pa.iter().zip(&pb).map(|(a, b)| a * b).collect();
    let result = ifft(&product).unwrap();
    result.iter().map(|c| c.re).take(a.len() + b.len() - 1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_roundtrip() {
        let x: Vec<Complex> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let y = fft(&x).unwrap();
        let back = ifft(&y).unwrap();
        for (a, b) in x.iter().zip(&back) { assert!((*a - *b).norm() < 1e-12); }
    }
}
