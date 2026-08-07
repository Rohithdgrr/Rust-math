//! Integration tests: exercise `mathverse-complex` exactly as an external
//! consumer would, through the public API only.

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
    assert!((z64.norm() - z32.norm() as f64).abs() < 1e-6);
    assert!((z64.arg() - z32.arg() as f64).abs() < 1e-6);
    assert_eq!(z64.re, 1.0);
    assert_eq!(z32.im, 2.0f32);
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
    assert_eq!(z.phase(), z.arg());
    let (r, theta) = z.to_polar();
    let back = Complex::rect(r, theta);
    assert!(z.is_close(&back, 1e-12, 1e-12));
    assert_eq!(z.re, 0.0);
    assert_eq!(z.im, 1.0);
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
