//! Complex probability distributions: Gaussian (circular symmetric),
//! uniform on the unit disk, and Wishart.
//!
//! These distributions model noise in radar, communications channel
//! estimation, speckle in SAR imagery, and complex-valued stochastic
//! processes.
//!
//! # Module overview
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`complex_gaussian_pdf`] | Probability density of circular complex Gaussian |
//! | [`complex_uniform_disk_sample`] | Sample uniformly from the unit disk |
//! | [`complex_gaussian_sample`] | Sample from circular complex Gaussian N(0, σ²I) |
//! | [`wishart_sample`] | Sample from complex Wishart distribution W(n, p, Σ) |
//! | [`wishart_mean`] | Expected value of Wishart distribution |
//! | [`complex_normality_test`] | Jarque-Bera test for circular normality |

use crate::Complex;
use crate::matrix::ComplexMatrix;
#[cfg(feature = "rand")]
use mathverse_core::error::{MathError, MathResult};

/// Probability density of the circular complex Gaussian distribution.
///
/// `p(z) = (1 / πσ²) · exp(−|z − μ|² / σ²)`
///
/// where `μ` is the mean and `σ²` is the variance per real dimension.
/// The total variance is `2σ²`.
///
/// # Arguments
/// * `z` — complex value at which to evaluate the density
/// * `mu` — mean of the distribution
/// * `sigma_sq` — variance per real dimension (must be > 0)
pub fn complex_gaussian_pdf(z: Complex, mu: Complex, sigma_sq: f64) -> f64 {
    if sigma_sq <= 0.0 {
        return 0.0;
    }
    let diff_sq = (z - mu).norm_sq();
    (1.0 / (core::f64::consts::PI * sigma_sq)) * (-diff_sq / sigma_sq).exp()
}

/// Sample a complex number uniformly distributed on the unit disk.
///
/// Uses rejection sampling: generate pairs in `[-1, 1]²` and reject
/// those outside the unit circle.
#[cfg(feature = "rand")]
pub fn complex_uniform_disk_sample(rng: &mut impl rand::Rng) -> Complex {
    loop {
        let re: f64 = rng.gen_range(-1.0..1.0);
        let im: f64 = rng.gen_range(-1.0..1.0);
        let r2 = re * re + im * im;
        if r2 <= 1.0 {
            return Complex::new(re, im);
        }
    }
}

/// Sample from a circular complex Gaussian distribution `N(μ, σ²I)`.
///
/// Both real and imaginary parts are independent `N(Re(μ), σ²)` and
/// `N(Im(μ), σ²)`.
#[cfg(feature = "rand")]
pub fn complex_gaussian_sample(
    rng: &mut impl rand::Rng,
    mu: Complex,
    sigma: f64,
) -> Complex {
    let re: f64 = rng.sample(rand_distr::StandardNormal);
    let im: f64 = rng.sample(rand_distr::StandardNormal);
    Complex::new(mu.re + sigma * re, mu.im + sigma * im)
}

/// Sample a complex Wishart matrix `W_p(n, Σ)`.
///
/// The complex Wishart distribution is the distribution of `Xᴴ·X`
/// where `X` is an `n × p` matrix with i.i.d. `CN(0, Σ)` rows.
///
/// # Arguments
/// * `n` — degrees of freedom (number of samples, must be ≥ p)
/// * `sigma` — scale matrix (p × p, must be positive definite)
/// * `rng` — random number generator
///
/// # Returns
/// A `p × p` complex Wishart-distributed matrix.
#[cfg(feature = "rand")]
pub fn wishart_sample(
    n: usize,
    sigma: &ComplexMatrix,
    rng: &mut impl rand::Rng,
) -> MathResult<ComplexMatrix> {
    let p = sigma.rows;
    if sigma.cols != p {
        return Err(MathError::DimensionMismatch);
    }
    if n < p {
        return Err(MathError::InvalidArgument(
            "Degrees of freedom must be ≥ dimension",
        ));
    }

    // Compute Cholesky factor L such that Σ = L·Lᴴ
    let l = sigma.cholesky()?;

    // Generate n samples from CN(0, I): X is n × p
    let mut x_data = Vec::with_capacity(n * p);
    for _ in 0..n {
        for _ in 0..p {
            let re: f64 = rng.sample(rand_distr::StandardNormal);
            let im: f64 = rng.sample(rand_distr::StandardNormal);
            x_data.push(Complex::new(re, im));
        }
    }
    let x = ComplexMatrix::from_data(x_data, n, p);

    // Scale by Cholesky factor: Y = X·Lᴴ
    let lh = l.hermitian();
    let y = x.mul(&lh)?;

    // Wishart = Yᴴ·Y
    let yh = y.hermitian();
    yh.mul(&y)
}

/// Expected value of the complex Wishart distribution: `E[W] = n · Σ`.
pub fn wishart_mean(n: usize, sigma: &ComplexMatrix) -> ComplexMatrix {
    let mut result = sigma.clone();
    let scale = Complex::real(n as f64);
    for v in &mut result.data {
        *v = *v * scale;
    }
    result
}

/// Jarque-Bera test for circular normality of a complex signal.
///
/// Tests whether the real and imaginary parts of the signal are jointly
/// Gaussian. Returns `(test_statistic, p_value_approx)`.
///
/// The test statistic is: `JB = n/6 · (S² + (K−3)²/4)` where `S` is the
/// skewness and `K` is the kurtosis, computed over the pooled real and
/// imaginary components (both are individually `N(μ, σ²)` under circular
/// Gaussianity; the magnitude is Rayleigh-distributed and must not be used).
///
/// A low p-value (< 0.05) suggests the data is not circular Gaussian.
pub fn complex_normality_test(signal: &[Complex]) -> (f64, f64) {
    let n = signal.len() * 2; // pooled real + imaginary components
    if n < 16 {
        return (f64::NAN, f64::NAN);
    }

    let mean = signal.iter().map(|z| z.re + z.im).sum::<f64>() / n as f64;
    let var = signal
        .iter()
        .map(|z| (z.re - mean).powi(2) + (z.im - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let std = var.sqrt();

    if std < 1e-15 {
        return (0.0, 1.0);
    }

    let skewness = signal
        .iter()
        .map(|z| ((z.re - mean) / std).powi(3) + ((z.im - mean) / std).powi(3))
        .sum::<f64>()
        / n as f64;
    let kurtosis = signal
        .iter()
        .map(|z| ((z.re - mean) / std).powi(4) + ((z.im - mean) / std).powi(4))
        .sum::<f64>()
        / n as f64;

    // Jarque-Bera statistic
    let jb = n as f64 / 6.0 * (skewness.powi(2) + (kurtosis - 3.0).powi(2) / 4.0);

    // Approximate p-value using chi-squared with 2 degrees of freedom
    let p_value = (-jb / 2.0).exp();

    (jb, p_value)
}

/// Compute the complex circular autocorrelation matrix of a signal.
///
/// `R = (1/n) · Xᴴ · X` where `X` is the data matrix.
pub fn autocorrelation_matrix(signal: &[Complex]) -> ComplexMatrix {
    let n = signal.len();
    if n == 0 {
        return ComplexMatrix::zeros(0, 0);
    }

    // X is n×1 so R = (1/n)·Xᴴ·X is 1×1 with R[0][0] = (1/n)·Σ|xᵢ|².
    // Compute the scalar directly instead of a (never-failing) matrix
    // multiplication, keeping the function infallible.
    let mean_sq: f64 = signal.iter().map(Complex::norm_sq).sum::<f64>() / n as f64;
    let mut result = ComplexMatrix::zeros(1, 1);
    result.set(0, 0, Complex::real(mean_sq));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn gaussian_pdf_at_mean() {
        let pdf = complex_gaussian_pdf(Complex::zero(), Complex::zero(), 1.0);
        assert!((pdf - 1.0 / core::f64::consts::PI).abs() < EPS);
    }

    #[test]
    fn gaussian_pdf_symmetry() {
        let z1 = Complex::new(1.0, 2.0);
        let z2 = Complex::new(-1.0, -2.0);
        let mu = Complex::zero();
        let pdf1 = complex_gaussian_pdf(z1, mu, 1.0);
        let pdf2 = complex_gaussian_pdf(z2, mu, 1.0);
        assert!((pdf1 - pdf2).abs() < EPS);
    }

    #[test]
    fn gaussian_pdf_zero_variance() {
        let pdf = complex_gaussian_pdf(Complex::zero(), Complex::zero(), 0.0);
        assert!((pdf - 0.0).abs() < EPS);
    }

    #[test]
    fn gaussian_pdf_far_from_mean() {
        let z = Complex::new(100.0, 100.0);
        let pdf = complex_gaussian_pdf(z, Complex::zero(), 1.0);
        assert!(pdf < 1e-100);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn uniform_disk_always_inside() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..1000 {
            let z = complex_uniform_disk_sample(&mut rng);
            assert!(
                z.norm() <= 1.0 + EPS,
                "Sample outside unit disk: |z| = {}",
                z.norm()
            );
        }
    }

    #[cfg(feature = "rand")]
    #[test]
    fn gaussian_sample_mean_near_zero() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let samples: Vec<Complex> = (0..1000)
            .map(|_| complex_gaussian_sample(&mut rng, Complex::zero(), 1.0))
            .collect();

        let mean_re: f64 = samples.iter().map(|z| z.re).sum::<f64>() / 1000.0;
        let mean_im: f64 = samples.iter().map(|z| z.im).sum::<f64>() / 1000.0;
        assert!(mean_re.abs() < 0.2, "Mean re: {mean_re}");
        assert!(mean_im.abs() < 0.2, "Mean im: {mean_im}");
    }

    #[cfg(feature = "rand")]
    #[test]
    fn gaussian_sample_variance_correct() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sigma = 2.0;
        let samples: Vec<Complex> = (0..2000)
            .map(|_| complex_gaussian_sample(&mut rng, Complex::zero(), sigma))
            .collect();

        let var_re: f64 = samples.iter().map(|z| z.re * z.re).sum::<f64>() / 2000.0;
        let var_im: f64 = samples.iter().map(|z| z.im * z.im).sum::<f64>() / 2000.0;
        assert!(
            (var_re - sigma * sigma).abs() < 0.5,
            "Var re: {var_re}"
        );
        assert!(
            (var_im - sigma * sigma).abs() < 0.5,
            "Var im: {var_im}"
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn wishart_sample_shape() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sigma = ComplexMatrix::identity(2);
        let w = wishart_sample(10, &sigma, &mut rng).unwrap();
        assert_eq!(w.rows, 2);
        assert_eq!(w.cols, 2);

        // Wishart matrix should be Hermitian
        for i in 0..2 {
            for j in 0..2 {
                let diff = (w[(i, j)] - w[(j, i)].conjugate()).norm();
                assert!(diff < 1e-10, "Not Hermitian at ({i}, {j})");
            }
        }
    }

    #[test]
    fn wishart_mean_correct() {
        let sigma = ComplexMatrix::identity(3);
        let expected = wishart_mean(5, &sigma);
        // E[W] = n·Σ = 5·I
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((expected[(i, j)].re - 5.0).abs() < EPS);
                } else {
                    assert!(expected[(i, j)].norm() < EPS);
                }
            }
        }
    }

    #[cfg(feature = "rand")]
    #[test]
    fn wishart_insufficient_dof() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sigma = ComplexMatrix::identity(3);
        assert!(wishart_sample(2, &sigma, &mut rng).is_err());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn normality_test_gaussian() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let signal: Vec<Complex> = (0..200)
            .map(|_| complex_gaussian_sample(&mut rng, Complex::zero(), 1.0))
            .collect();
        let (jb, p_value) = complex_normality_test(&signal);
        // Gaussian data should not be rejected (p > 0.05)
        assert!(p_value > 0.01, "JB={jb}, p={p_value}");
    }

    #[cfg(feature = "rand")]
    #[test]
    fn normality_test_non_gaussian() {
        // Uniform data should be detected as non-Gaussian
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let signal: Vec<Complex> = (0..200)
            .map(|_| {
                let re: f64 = rng.gen_range(-3.0..3.0);
                let im: f64 = rng.gen_range(-3.0..3.0);
                Complex::new(re, im)
            })
            .collect();
        let (_, p_value) = complex_normality_test(&signal);
        // Non-Gaussian data should have lower p-value
        assert!(p_value < 0.1, "p={p_value}");
    }

    #[test]
    fn autocorrelation_matrix_identity() {
        let signal = vec![Complex::real(1.0)];
        let r = autocorrelation_matrix(&signal);
        assert_eq!(r.rows, 1);
        assert_eq!(r.cols, 1);
        assert!((r[(0, 0)].re - 1.0).abs() < EPS);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn autocorrelation_matrix_hermitian() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let signal: Vec<Complex> = (0..10)
            .map(|_| complex_gaussian_sample(&mut rng, Complex::zero(), 1.0))
            .collect();
        let r = autocorrelation_matrix(&signal);
        assert_eq!(r.rows, 1);
        assert_eq!(r.cols, 1);
        assert!(r[(0, 0)].im.abs() < EPS);
    }
}
