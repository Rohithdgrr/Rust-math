//! Criterion benchmarks: arithmetic, FFT (vs naive DFT), matrix ops, roots,
//! and special functions.
//!
//! Run with: `cargo bench -p mathverse-complex`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mathverse_complex::{fft, polynomial_roots, Complex, ComplexMatrix, ComplexSpecialFunctions};

fn bench_arithmetic(c: &mut Criterion) {
    let z1 = Complex::new(1.234, -5.678);
    let z2 = Complex::new(9.876, 0.543);
    c.bench_function("complex_add", |b| b.iter(|| z1 + z2));
    c.bench_function("complex_mul", |b| b.iter(|| z1 * z2));
    c.bench_function("complex_div", |b| b.iter(|| z1 / z2));
    c.bench_function("complex_exp", |b| b.iter(|| z1.exp()));
    c.bench_function("complex_sqrt", |b| b.iter(|| z1.sqrt()));
}

/// Naive O(n²) DFT used as the FFT reference point.
fn naive_dft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let mut out = vec![Complex::zero(); n];
    for (k, o) in out.iter_mut().enumerate() {
        for (j, &x) in input.iter().enumerate() {
            let theta = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
            *o = *o + x * Complex::polar(1.0, theta);
        }
    }
    out
}

fn bench_fft(c: &mut Criterion) {
    for &n in &[64usize, 256, 1024] {
        let signal: Vec<Complex> = (0..n)
            .map(|k| Complex::new((k as f64 * 0.1).sin(), 0.0))
            .collect();
        c.bench_with_input(BenchmarkId::new("fft", n), &signal, |b, s| {
            b.iter(|| fft(s))
        });
        c.bench_with_input(BenchmarkId::new("dft_naive", n), &signal, |b, s| {
            b.iter(|| naive_dft(s))
        });
    }
}

fn bench_matrix(c: &mut Criterion) {
    let mut a = ComplexMatrix::identity(8);
    for i in 0..8 {
        for j in 0..8 {
            let _ = a.try_set(
                i,
                j,
                Complex::new((i as f64 * 0.5).sin(), (j as f64 * 0.3).cos()),
            );
        }
    }
    let b = ComplexMatrix::identity(8);
    // unwrap: a bench should fail loudly (panic) if the op ever returns Err
    c.bench_function("matrix_mul_8x8", |bencher| {
        bencher.iter(|| a.mul(&b).unwrap())
    });
    c.bench_function("matrix_det_8x8", |bencher| bencher.iter(|| a.determinant()));
    c.bench_function("matrix_eig_8x8", |bencher| {
        bencher.iter(|| a.eigenvalues(500, 1e-10).unwrap())
    });
}

fn bench_polynomial(c: &mut Criterion) {
    // x^5 - 1 has 5 well-separated roots
    let coeffs: Vec<Complex> = vec![
        Complex::real(-1.0),
        Complex::zero(),
        Complex::zero(),
        Complex::zero(),
        Complex::zero(),
        Complex::one(),
    ];
    c.bench_function("polynomial_roots_deg5", |b| {
        b.iter(|| polynomial_roots(&coeffs, 100, 1e-12))
    });
}

fn bench_special_functions(c: &mut Criterion) {
    let z = Complex::new(0.7, -1.3);
    c.bench_function("gamma", |b| b.iter(|| ComplexSpecialFunctions::gamma(z)));
    c.bench_function("zeta", |b| {
        b.iter(|| ComplexSpecialFunctions::zeta(Complex::real(2.0), 100))
    });
    c.bench_function("erf", |b| b.iter(|| ComplexSpecialFunctions::erf(z, 60)));
    c.bench_function("bessel_j", |b| {
        b.iter(|| ComplexSpecialFunctions::bessel_j(Complex::zero(), z, 50))
    });
}

criterion_group!(
    benches,
    bench_arithmetic,
    bench_fft,
    bench_matrix,
    bench_polynomial,
    bench_special_functions
);
criterion_main!(benches);
