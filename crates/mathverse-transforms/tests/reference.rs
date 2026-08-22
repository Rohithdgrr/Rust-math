//! Reference-data verification for every transform.
//!
//! These tests pin each implementation against independently computed values
//! (direct O(N²) definitions, textbook constants, and physical invariants
//! like Parseval's theorem), so a silent algorithmic regression cannot pass.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // test lengths < 2^53

use mathverse_complex::Complex;
use mathverse_transforms::{
    dct1, dct2, dct3, dct4, dft, dst1, dst4, fft, goertzel_magnitude,
    haar_dwt, haar_dwt_multi, haar_idwt_multi, idct2, ifft,
};

fn c(re: f64) -> Complex {
    Complex::real(re)
}

// ---- FFT -----------------------------------------------------------------

/// The radix-2 FFT must agree with the direct DFT definition.
#[test]
fn fft_matches_direct_dft() {
    let x: Vec<Complex> = [0.7, -1.3, 2.2, 4.1, -0.5, 3.3, 1.9, -2.8]
        .iter()
        .map(|&v| c(v))
        .collect();
    let fast = fft(&x).unwrap();
    let slow = dft(&x);
    for (a, b) in fast.iter().zip(&slow) {
        assert!((*a - *b).norm() < 1e-10, "fft {a} vs dft {b}");
    }
}

/// A pure cosine sampled over an integer number of periods has energy only
/// at bins k=1 and k=N-1, each of magnitude N/2.
#[test]
fn fft_sinusoid_energy_in_correct_bins() {
    let n = 64;
    let signal: Vec<Complex> = (0..n)
        .map(|i| c((2.0 * core::f64::consts::PI * i as f64 / n as f64).cos()))
        .collect();
    let spectrum = fft(&signal).unwrap();
    for (k, bin) in spectrum.iter().enumerate() {
        let expected = if k == 1 || k == n - 1 { n as f64 / 2.0 } else { 0.0 };
        assert!(
            (bin.norm() - expected).abs() < 1e-8,
            "bin {k}: got {}, expected {expected}",
            bin.norm()
        );
    }
}

/// Parseval: sum |x[n]|² == (1/N) sum |X[k]|².
#[test]
fn fft_parseval() {
    let x: Vec<Complex> = [1.0, 2.0, -3.0, 4.5, 0.25, -1.75, 8.0, -6.2]
        .iter()
        .map(|&v| c(v))
        .collect();
    let energy: f64 = x.iter().map(Complex::norm_sq).sum();
    let spec = fft(&x).unwrap();
    let spectral: f64 = spec.iter().map(Complex::norm_sq).sum::<f64>() / 8.0;
    assert!((energy - spectral).abs() < 1e-9);
}

// ---- Goertzel ------------------------------------------------------------

/// Goertzel must reproduce the FFT magnitude at every bin.
#[test]
fn goertzel_matches_fft_bins() {
    let x = [0.3, -1.1, 2.7, 4.0, -2.2, 1.5, 0.9, -3.6];
    let xc: Vec<Complex> = x.iter().map(|&v| c(v)).collect();
    let spec = fft(&xc).unwrap();
    for k in 0..x.len() {
        let g = goertzel_magnitude(&x, k);
        assert!(
            (g - spec[k].norm()).abs() < 1e-8,
            "bin {k}: goertzel {g} vs fft {}",
            spec[k].norm()
        );
    }
}

// ---- DCT -----------------------------------------------------------------

/// Orthonormal DCT-II of [1, 2, 3, 4], cross-checked against reference
/// values (`scipy.fft.dct(..., norm="ortho")`).
#[test]
fn dct2_reference_values() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = dct2(&x);
    // Exact pins: DC carries the whole sum (√(1/4)·10), k=2 cancels exactly.
    assert!((y[0] - 5.0).abs() < 1e-12);
    assert!(y[2].abs() < 1e-12);
    // Rounded reference values.
    assert!((y[1] - (-2.230_442_5)).abs() < 1e-7);
    assert!((y[3] - (-0.158_512_7)).abs() < 1e-7);
}

/// Constant input: all DCT-II energy in the DC coefficient (√N·c).
#[test]
fn dct2_constant_input() {
    let y = dct2(&[3.0; 8]);
    assert!((y[0] - 3.0 * (8.0_f64).sqrt()).abs() < 1e-12);
    for v in &y[1..] {
        assert!(v.abs() < 1e-12);
    }
}

/// Orthonormal transforms are involutive or pair-invertible; check every
/// DCT flavour round-trips (DCT-I and DCT-IV are their own inverses).
#[test]
fn dct_roundtrips_and_involution() {
    let x = [0.5, -1.25, 3.75, 2.0, -4.0, 1.125];

    let back = idct2(&dct2(&x));
    for (a, b) in x.iter().zip(&back) {
        assert!((a - b).abs() < 1e-12);
    }

    // DCT-I is orthonormal and symmetric: applying it twice returns the input.
    let i1 = dct1(&dct1(&x));
    for (a, b) in x.iter().zip(&i1) {
        assert!((a - b).abs() < 1e-12, "dct1 involution");
    }

    // DCT-IV is an involution.
    let i4 = dct4(&dct4(&x));
    for (a, b) in x.iter().zip(&i4) {
        assert!((a - b).abs() < 1e-12, "dct4 involution");
    }

    // DCT-III is the transpose (inverse) of the un-normalized DCT-II;
    // with orthonormal scaling, dct3(dct2(x)) == x.
    let via3 = dct3(&dct2(&x));
    for (a, b) in x.iter().zip(&via3) {
        assert!((a - b).abs() < 1e-12, "dct3∘dct2");
    }
}

/// Parseval holds for the orthonormal DCT-II.
#[test]
fn dct2_parseval() {
    let x = [1.0, 2.0, -3.0, 4.5, 0.25, -1.75];
    let e_in: f64 = x.iter().map(|v| v * v).sum();
    let y = dct2(&x);
    let e_out: f64 = y.iter().map(|v| v * v).sum();
    assert!((e_in - e_out).abs() < 1e-12);
}

// ---- DST -----------------------------------------------------------------

/// DST-I is orthonormal and self-inverse.
#[test]
fn dst1_involution_and_parseval() {
    let x = [1.0, 2.0, 3.0, -2.5];
    let twice = dst1(&dst1(&x));
    for (a, b) in x.iter().zip(&twice) {
        assert!((a - b).abs() < 1e-12, "dst1 involution");
    }
    let e_in: f64 = x.iter().map(|v| v * v).sum();
    let e_out: f64 = dst1(&x).iter().map(|v| v * v).sum();
    assert!((e_in - e_out).abs() < 1e-12);
}

/// DST-IV is an involution.
#[test]
fn dst4_involution() {
    let x = [0.25, -2.0, 5.5, 1.0, -0.75, 3.25];
    let twice = dst4(&dst4(&x));
    for (a, b) in x.iter().zip(&twice) {
        assert!((a - b).abs() < 1e-12, "dst4 involution");
    }
}

/// DST-II followed by DST-III is the identity (exact transpose pair).
#[test]
fn dst3_inverts_dst2() {
    let x = [0.25, -2.0, 5.5, 1.0];
    let back = mathverse_transforms::dst3(&mathverse_transforms::dst2(&x));
    for (a, b) in x.iter().zip(&back) {
        assert!((a - b).abs() < 1e-12, "dst3∘dst2 {a} vs {b}");
    }
    // Parseval.
    let e_in: f64 = x.iter().map(|v| v * v).sum();
    let e_out: f64 = mathverse_transforms::dst2(&x).iter().map(|v| v * v).sum();
    assert!((e_in - e_out).abs() < 1e-12);
}

// ---- Haar wavelet ----------------------------------------------------------

/// Single-level transform of [1,2,3,4]: averages (√(3/2), √(98)/2·…) — exact
/// values from the definition ((a+b)/√2, (a−b)/√2).
#[test]
fn haar_reference_values() {
    let coeff = haar_dwt(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let s = core::f64::consts::FRAC_1_SQRT_2;
    let expected = [3.0 * s, 7.0 * s, -s, -s];
    for (got, want) in coeff.iter().zip(expected) {
        assert!((got - want).abs() < 1e-15, "haar {got} vs {want}");
    }
}

/// Parseval: orthonormal Haar preserves energy exactly.
#[test]
fn haar_parseval() {
    let x = [4.0, -2.5, 1.125, 8.75, -0.5, 3.0, 2.25, -6.125];
    let e_in: f64 = x.iter().map(|v| v * v).sum();
    let e_out: f64 = haar_dwt(&x).unwrap().iter().map(|v| v * v).sum();
    assert!((e_in - e_out).abs() < 1e-12);
}

/// Multi-level forward followed by multi-level inverse reconstructs the
/// original signal; the packed layout preserves every detail block.
#[test]
fn haar_multi_level_roundtrip() {
    let x = [1.0, 3.0, -2.0, 5.0, 0.5, 4.5, -1.5, 2.5];
    let coeffs = haar_dwt_multi(&x, 3).unwrap();
    assert_eq!(coeffs.len(), 8); // full pyramid: [app_3 | det_3 | det_2 | det_1]
    let back = haar_idwt_multi(&coeffs, 3).unwrap();
    for (a, b) in x.iter().zip(&back) {
        assert!((a - b).abs() < 1e-12);
    }
    // Parseval holds across the whole multi-level transform.
    let e_in: f64 = x.iter().map(|v| v * v).sum();
    let e_out: f64 = coeffs.iter().map(|v| v * v).sum();
    assert!((e_in - e_out).abs() < 1e-12);
}

/// A constant signal's pyramid: the approximation block holds c·(√2)^levels
/// per entry; every detail block is exactly zero.
#[test]
fn haar_constant_signal_scaling() {
    let c_val = 3.5;
    for levels in 1..=3usize {
        let coeff = haar_dwt_multi(&[c_val; 8], levels).unwrap();
        assert_eq!(coeff.len(), 8);
        let app_len = 8 >> levels;
        let want = c_val * (2.0_f64).powf(levels as f64 / 2.0);
        for v in &coeff[..app_len] {
            assert!((v - want).abs() < 1e-12, "level {levels} app {v} vs {want}");
        }
        for v in &coeff[app_len..] {
            assert!(v.abs() < 1e-12, "level {levels} stray detail {v}");
        }
    }
}

/// Level-1 detail block of a linear ramp is constant (Haar sees the slope):
/// each pair contributes (x[2i] − x[2i+1])/√2 = −1/√2.
#[test]
fn haar_linear_ramp_details() {
    let x: Vec<f64> = (0..8).map(f64::from).collect();
    let coeffs = haar_dwt_multi(&x, 3).unwrap();
    for &d in &coeffs[4..8] {
        assert!(
            (d + core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12,
            "detail {d}"
        );
    }
}
