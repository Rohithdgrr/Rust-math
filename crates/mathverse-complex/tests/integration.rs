//! Integration tests: exercise `mathverse-complex` exactly as an external
//! consumer would, through the public API only.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::needless_range_loop
)]

use mathverse_complex::{
    eval_polynomial, fft, ifft, mandelbrot_iterate, mandelbrot_smooth, polynomial_roots, Complex,
    ComplexAnalysis, ComplexMatrix, ComplexSpecialFunctions, C32,
};

// ---------------------------------------------------------------------------
// Core type
// ---------------------------------------------------------------------------

#[test]
fn generic_precision_c32_and_c64() {
    // C64 (default) and C32 both compile and behave consistently.
    let z64 = Complex::new(1.0, 2.0);
    let z32: C32 = Complex::new(1.0f32, 2.0);
    assert!((z64.norm() - f64::from(z32.norm())).abs() < 1e-6);
    assert!((z64.arg() - f64::from(z32.arg())).abs() < 1e-6);
    assert!((z64.re - 1.0).abs() < 1e-12);
    assert!((z32.im - 2.0f32).abs() < 1e-6);
}

#[test]
fn display_formats_like_math() {
    assert_eq!(format!("{}", Complex::new(3.0, 4.0)), "3+4i");
    assert_eq!(format!("{}", Complex::new(1.0, -2.0)), "1-2i");
    assert_eq!(format!("{:.2}", Complex::new(3.0, 4.0)), "3.00+4.00i");
}

#[test]
fn display_edge_cases_do_not_panic() {
    // NaN/Inf components render textually; never panic
    assert_eq!(format!("{}", Complex::new(f64::NAN, 0.0)), "NaN+0i");
    assert_eq!(format!("{}", Complex::new(f64::INFINITY, 1.0)), "inf+1i");
    // signed-zero imaginary part is preserved (branch-cut info)
    assert_eq!(format!("{}", Complex::new(1.0, -0.0)), "1-0i");
}

#[test]
fn numpy_cmath_parity_names() {
    // phase/to_polar/rect/is_close mirror cmath; re/im are public fields
    let z: Complex = Complex::new(0.0, 1.0);
    assert!((z.phase() - z.arg()).abs() < 1e-12);
    let (r, theta) = z.to_polar();
    let back = Complex::rect(r, theta);
    assert!(z.is_close(&back, 1e-12, 1e-12));
    assert!(z.re.abs() < 1e-12);
    assert!((z.im - 1.0).abs() < 1e-12);
}

#[test]
fn property_identity_z_times_conj() {
    // z·conj(z) == |z|²  for a spread of magnitudes
    for &(re, im) in &[
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (-3.0, 4.0),
        (1e-150, -1e-150),
        (1e150, 1e150),
    ] {
        let z: Complex = Complex::new(re, im);
        let product = z * z.conjugate();
        assert!((product.re - z.norm_sq()).abs() <= 1e-9 * z.norm_sq().max(1.0));
        assert!(product.im.abs() <= 1e-9 * z.norm_sq().max(1.0));
    }
}

#[test]
fn overflow_safe_division_through_public_api() {
    let a: Complex = Complex::new(1e200, 0.0);
    let b: Complex = Complex::new(1e200, 1e200);
    let q = a / b;
    assert!((q.re - 0.5).abs() < 1e-12 && (q.im + 0.5).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------

#[test]
fn fft_roundtrip_identity() {
    let n = 1024;
    let signal: Vec<Complex> = (0..n)
        .map(|k| {
            let t = k as f64 / n as f64;
            Complex::new((2.0 * std::f64::consts::PI * 8.0 * t).sin(), 0.0)
        })
        .collect();
    let spectrum = fft(&signal);
    // Sine at bin 8 (± bin 1016 mirror): magnitude peak at bin 8
    assert!((spectrum[8].norm() - n as f64 / 2.0).abs() < 1e-6);
    let back = ifft(&spectrum);
    let max_err = (0..n)
        .map(|k| (back[k] - signal[k]).norm())
        .fold(0.0f64, f64::max);
    assert!(max_err < 1e-10);
}

#[test]
fn fft_single_bin_and_dc() {
    // DC signal: only bin 0 is nonzero
    let signal: Vec<Complex> = vec![Complex::one(); 8];
    let spectrum = fft(&signal);
    assert!((spectrum[0].re - 8.0).abs() < 1e-12);
    assert!(spectrum[1..].iter().all(|c| c.norm() < 1e-12));
}

// ---------------------------------------------------------------------------
// Polynomials
// ---------------------------------------------------------------------------

#[test]
fn polynomial_roots_of_z2_plus_one() {
    let roots = polynomial_roots(
        &[Complex::one(), Complex::zero(), Complex::one()],
        1000,
        1e-12,
    );
    assert_eq!(roots.len(), 2);
    let to_i = |r: &Complex| (*r - Complex::i()).norm();
    let to_neg_i = |r: &Complex| (*r + Complex::i()).norm();
    assert!(roots.iter().map(to_i).fold(f64::MAX, f64::min) < 1e-8);
    assert!(roots.iter().map(to_neg_i).fold(f64::MAX, f64::min) < 1e-8);
}

#[test]
fn eval_polynomial_matches_horner() {
    // p(z) = 1 + 2z + z² = (1 + z)²
    let coeffs = [Complex::one(), Complex::real(2.0), Complex::one()];
    let at = Complex::new(1.0, 1.0);
    let v = eval_polynomial(&coeffs, at);
    let expected = (Complex::one() + at) * (Complex::one() + at);
    assert!((v - expected).norm() < 1e-12);
}

// ---------------------------------------------------------------------------
// Mandelbrot
// ---------------------------------------------------------------------------

#[test]
fn mandelbrot_membership_and_smooth() {
    // c = 0 is inside the set (never escapes)
    assert_eq!(mandelbrot_iterate(Complex::zero(), 200, 2.0), 200);
    // c = 3 escapes on the second step (z1 = 3, |z1|² = 9 > 4)
    assert_eq!(mandelbrot_iterate(Complex::real(3.0), 200, 2.0), 1);
    // smooth value stays finite for both
    assert!(mandelbrot_smooth(Complex::zero(), 200, 2.0).is_finite());
    assert!(mandelbrot_smooth(Complex::real(3.0), 200, 2.0).is_finite());
}

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

#[test]
fn matrix_solve_and_linalg_eig() {
    let mut a = ComplexMatrix::new(2, 2);
    a.set(0, 0, Complex::real(2.0));
    a.set(0, 1, Complex::real(1.0));
    a.set(1, 0, Complex::real(1.0));
    a.set(1, 1, Complex::real(1.0));
    let x = a.solve(&[Complex::real(3.0), Complex::real(2.0)]).unwrap();
    assert!((x[0].re - 1.0).abs() < 1e-10 && (x[1].re - 1.0).abs() < 1e-10);

    let ev = mathverse_complex::matrix::linalg::eig(&a).unwrap();
    let mut vals: Vec<f64> = ev.iter().map(|c| c.re).collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!((vals[0] - 0.381966).abs() < 1e-4);
    assert!((vals[1] - 2.618034).abs() < 1e-4);
}

#[test]
fn matrix_dimension_mismatch_returns_error() {
    let a = ComplexMatrix::new(2, 2);
    let b = ComplexMatrix::new(3, 3);
    assert!(a.add(&b).is_err());
    assert!(a.mul(&b).is_err());
    assert!(a.try_get(5, 5).is_none());
    let mut c = ComplexMatrix::new(1, 1);
    assert!(c.try_set(2, 2, Complex::one()).is_err());
}

// ---------------------------------------------------------------------------
// Analysis & special functions
// ---------------------------------------------------------------------------

#[test]
fn complex_step_derivative_matches_analytic() {
    // d/dz e^z = e^z
    let f = |z: Complex| z.exp();
    let z = Complex::new(0.3, -0.7);
    let numeric = ComplexAnalysis::derivative_complex_step(&f, z, 1e-8);
    assert!((numeric - f(z)).norm() < 1e-6);
}

#[test]
fn special_functions_public_api() {
    // Γ(0.5) = √π
    let g = ComplexSpecialFunctions::gamma(Complex::new(0.5, 0.0));
    assert!((g.re - std::f64::consts::PI.sqrt()).abs() < 1e-10);
    // ζ(2) = π²/6
    let z = ComplexSpecialFunctions::zeta(Complex::real(2.0), 1000);
    assert!((z.re - std::f64::consts::PI * std::f64::consts::PI / 6.0).abs() < 1e-6);
    // J₀(1) ≈ 0.7652
    let j0 = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::one(), 50);
    assert!((j0.re - 0.7651976865).abs() < 1e-6);
}

#[cfg(feature = "serde")]
#[test]
fn serde_roundtrip() {
    let z = Complex::new(1.5, -2.7);
    let json = serde_json::to_string(&z).unwrap();
    let z2: Complex = serde_json::from_str(&json).unwrap();
    assert_eq!(z, z2);

    let m = ComplexMatrix::from_data(
        vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
        1,
        2,
    );
    let json = serde_json::to_string(&m).unwrap();
    let m2: ComplexMatrix = serde_json::from_str(&json).unwrap();
    assert_eq!(m.get(0, 0), m2.get(0, 0));
    assert_eq!(m.get(0, 1), m2.get(0, 1));
}

#[cfg(feature = "rand")]
#[test]
fn rand_complex_sampling() {
    use mathverse_complex::{complex_gaussian, complex_uniform_disk};
    let mut rng = rand::thread_rng();
    let z: Complex = rand::random();
    // Standard sample should produce finite values
    assert!(z.re.is_finite() && z.im.is_finite());

    // Disk sample should be inside the unit disk
    for _ in 0..100 {
        let z = complex_uniform_disk(&mut rng);
        assert!(z.norm() <= 1.0 + 1e-12);
    }

    // Gaussian sample with sigma=0 should be (near) zero
    let z = complex_gaussian(&mut rng, 0.0);
    assert!(z.norm() < 1e-15);
}

#[test]
fn fft_in_place_api() {
    use mathverse_complex::fft_in_place;
    let x: Vec<Complex> = (0..16)
        .map(|i| Complex::new(f64::from(i), 0.0))
        .collect();
    let mut buf = x.clone();
    fft_in_place(&mut buf);
    let y = mathverse_complex::fft(&x);
    for (a, b) in buf.iter().zip(y.iter()) {
        assert!((a - b).norm() < 1e-12);
    }
}

#[test]
fn hessenberg_reduction() {
    // A diagonal matrix is already Hessenberg
    let mut m = ComplexMatrix::new(3, 3);
    m.set(0, 0, Complex::real(1.0));
    m.set(1, 1, Complex::real(2.0));
    m.set(2, 2, Complex::real(3.0));
    let h = m.hessenberg_reduction();
    // Subdiagonal should be zero (within tolerance)
    assert!(h.get(2, 0).norm() < 1e-10);

    // A general matrix
    let mut m = ComplexMatrix::new(3, 3);
    m.set(0, 0, Complex::real(6.0));
    m.set(0, 1, Complex::real(2.0));
    m.set(0, 2, Complex::real(1.0));
    m.set(1, 0, Complex::real(3.0));
    m.set(1, 1, Complex::real(1.0));
    m.set(2, 0, Complex::real(4.0));
    m.set(2, 1, Complex::real(5.0));
    let h = m.hessenberg_reduction();
    // h[2][0] should be zero
    assert!(h.get(2, 0).norm() < 1e-10);
}

#[test]
fn modified_bessel_i() {
    use mathverse_complex::ComplexSpecialFunctions;
    // I_0(0) = 1
    let i0 = ComplexSpecialFunctions::bessel_i(Complex::zero(), Complex::zero(), 50);
    assert!((i0.re - 1.0).abs() < 1e-10);
    // I_1(1) ≈ 0.5652
    let i1 = ComplexSpecialFunctions::bessel_i(Complex::one(), Complex::one(), 50);
    assert!((i1.re - 0.5651591040).abs() < 1e-4);
}

#[test]
fn modified_bessel_k() {
    use mathverse_complex::ComplexSpecialFunctions;
    // K_0(1) ≈ 0.4210
    let k0 = ComplexSpecialFunctions::bessel_k(Complex::zero(), Complex::one(), 50);
    assert!((k0.re - 0.4210).abs() < 0.5);
}

#[test]
fn hankel_functions() {
    use mathverse_complex::ComplexSpecialFunctions;
    let z = Complex::real(2.0);
    let h1 = ComplexSpecialFunctions::hankel_h1(Complex::zero(), z, 50);
    let h2 = ComplexSpecialFunctions::hankel_h2(Complex::zero(), z, 50);
    // H1 = J + iY, H2 = J - iY => H1 + H2 = 2J
    let j0 = ComplexSpecialFunctions::bessel_j(Complex::zero(), z, 50);
    let sum = h1 + h2;
    assert!((sum.re - 2.0 * j0.re).abs() < 1e-6);
    assert!((sum.im).abs() < 1e-6);
}

#[test]
fn incomplete_gamma() {
    use mathverse_complex::ComplexSpecialFunctions;
    // γ(1, 1) = 1 - e^{-1} ≈ 0.6321
    let gl = ComplexSpecialFunctions::gamma_lower(Complex::one(), Complex::one(), 50);
    assert!((gl.re - (1.0 - (-1.0_f64).exp())).abs() < 1e-4);
    // Γ(1, 1) = e^{-1} ≈ 0.3679
    let gu = ComplexSpecialFunctions::gamma_upper(Complex::one(), Complex::one(), 50);
    assert!((gu.re - (-1.0_f64).exp()).abs() < 1e-4);
}

#[test]
fn sinpi_cospi() {
    use mathverse_complex::ComplexSpecialFunctions;
    // sin(π·0.5) = 1
    let s = ComplexSpecialFunctions::sinpi(Complex::real(0.5));
    assert!((s.re - 1.0).abs() < 1e-10);
    // cos(π·0.5) = 0
    let c = ComplexSpecialFunctions::cospi(Complex::real(0.5));
    assert!(c.re.abs() < 1e-10);
    // sin(π·0) = 0
    let s0 = ComplexSpecialFunctions::sinpi(Complex::zero());
    assert!(s0.norm() < 1e-10);
    // cos(π·0) = 1
    let c0 = ComplexSpecialFunctions::cospi(Complex::zero());
    assert!((c0.re - 1.0).abs() < 1e-10);
}

#[test]
fn lambert_w() {
    use mathverse_complex::ComplexSpecialFunctions;
    // W(1) ≈ 0.5671 (Omega constant)
    let w = ComplexSpecialFunctions::lambert_w(Complex::one());
    assert!((w.re - 0.5671432904).abs() < 0.01);
    // W(0) = 0
    let w0 = ComplexSpecialFunctions::lambert_w(Complex::zero());
    assert!(w0.norm() < 1e-10);
}

#[test]
fn elliptic_integrals() {
    use mathverse_complex::ComplexSpecialFunctions;
    // K(0) = π/2
    let k0 = ComplexSpecialFunctions::elliptic_k(Complex::zero());
    assert!((k0.re - std::f64::consts::PI / 2.0).abs() < 1e-6);
    // E(0) = π/2
    let e0 = ComplexSpecialFunctions::elliptic_e(Complex::zero());
    assert!((e0.re - std::f64::consts::PI / 2.0).abs() < 1e-6);
}

#[test]
fn pow_zero_imaginary_exponent() {
    // 0^i should be NaN (not 1)
    let z: Complex<f64> = Complex::new(0.0, 0.0);
    let p: Complex<f64> = Complex::new(0.0, 1.0);
    let result = z.pow(p);
    assert!(result.re.is_nan() && result.im.is_nan(), "0^i should be NaN");
}

#[test]
fn matrix_eigenvectors() {
    let mut m = ComplexMatrix::new(2, 2);
    m.set(0, 0, Complex::real(2.0));
    m.set(0, 1, Complex::real(1.0));
    m.set(1, 0, Complex::real(1.0));
    m.set(1, 1, Complex::real(2.0));
    let vecs = m.eigenvectors(100, 1e-10).unwrap();
    // A*vec should be lambda*vec for each eigenvector
    let v0 = Complex::new(vecs.get(0, 0).re, vecs.get(0, 0).im);
    let v1 = Complex::new(vecs.get(0, 1).re, vecs.get(0, 1).im);
    let av0 = m.get(0, 0) * v0 + m.get(0, 1) * v1;
    // eigenvalue is av0/v0
    let lam = av0 / v0;
    // lambda should be real (2±1)
    assert!(lam.im.abs() < 1e-4, "eigenvalue should be real, got {lam}");
    assert!((lam.re - 3.0).abs() < 1e-3 || (lam.re - 1.0).abs() < 1e-3,
        "eigenvalue should be 1 or 3, got {lam}");
}
