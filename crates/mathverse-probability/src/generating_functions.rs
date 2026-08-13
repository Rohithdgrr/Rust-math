//! Probability generating functions: PGF, MGF, characteristic functions, Laplace transforms, Z-transforms.

/// Probability generating function (PGF) for discrete distributions.
#[must_use]
pub struct ProbabilityGeneratingFunction;

impl ProbabilityGeneratingFunction {
    /// PGF: G(s) = E[s^X] = Σ p_k * s^k.
    #[must_use]
    pub fn compute(pmf: &[f64], s: f64) -> f64 {
        pmf.iter()
            .enumerate()
            .map(|(k, &p)| p * s.powi(k as i32))
            .sum()
    }

    /// PGF for Bernoulli distribution.
    #[must_use]
    pub fn bernoulli(p: f64, s: f64) -> f64 {
        (1.0 - p) + p * s
    }

    /// PGF for Binomial distribution.
    #[must_use]
    pub fn binomial(n: u64, p: f64, s: f64) -> f64 {
        ((1.0 - p) + p * s).powi(n as i32)
    }

    /// PGF for Poisson distribution.
    #[must_use]
    pub fn poisson(lambda: f64, s: f64) -> f64 {
        (lambda * (s - 1.0)).exp()
    }

    /// PGF for Geometric distribution.
    #[must_use]
    pub fn geometric(p: f64, s: f64) -> f64 {
        let numerator = p * s;
        let denominator = 1.0 - (1.0 - p) * s;
        if denominator.abs() < 1e-10 {
            f64::INFINITY
        } else {
            numerator / denominator
        }
    }

    /// PGF for Negative Binomial distribution.
    #[must_use]
    pub fn negative_binomial(r: f64, p: f64, s: f64) -> f64 {
        let numerator = (p * s).powf(r);
        let denominator = (1.0 - (1.0 - p) * s).powf(r);
        if denominator.abs() < 1e-10 {
            f64::INFINITY
        } else {
            numerator / denominator
        }
    }

    /// Mean from PGF: E\[X\] = G'(1).  (First derivative of PGF at 1.)
    #[must_use]
    pub fn mean_from_pgfd(pmf: &[f64]) -> f64 {
        let epsilon = 1e-6;
        let g_plus = Self::compute(pmf, 1.0 + epsilon);
        let g_minus = Self::compute(pmf, 1.0 - epsilon);
        (g_plus - g_minus) / (2.0 * epsilon)
    }

    /// Variance from PGF: Var(X) = G''(1) + G'(1) - [G'(1)]².
    #[must_use]
    pub fn variance_from_pgfd(pmf: &[f64]) -> f64 {
        let epsilon = 1e-6;
        let g_plus = Self::compute(pmf, 1.0 + epsilon);
        let g_minus = Self::compute(pmf, 1.0 - epsilon);
        let first_derivative = (g_plus - g_minus) / (2.0 * epsilon);

        let g_plus_plus = Self::compute(pmf, 1.0 + 2.0 * epsilon);
        let g_minus_minus = Self::compute(pmf, 1.0 - 2.0 * epsilon);
        let second_derivative = (g_plus_plus - 2.0 * Self::compute(pmf, 1.0) + g_minus_minus)
            / (4.0 * epsilon * epsilon);

        second_derivative + first_derivative - first_derivative * first_derivative
        // Note: this is (d²G/dt²) + (dG/dt) - (dG/dt)², correct for the MGF second derivative formula
    }
}

/// Moment generating function (MGF).
#[must_use]
pub struct MomentGeneratingFunction;

impl MomentGeneratingFunction {
    /// MGF: M(t) = E[e^(tX)].
    #[must_use]
    pub fn compute(pmf: &[f64], t: f64) -> f64 {
        pmf.iter()
            .enumerate()
            .map(|(k, &p)| p * (t * k as f64).exp())
            .sum()
    }

    /// MGF for continuous distribution (numerical integration).
    #[must_use]
    pub fn continuous(pdf: impl Fn(f64) -> f64, t: f64, a: f64, b: f64, n: usize) -> f64 {
        let dx = (b - a) / n as f64;
        let mut integral = 0.0;

        for i in 0..n {
            let x = a + (i as f64 + 0.5) * dx;
            integral += pdf(x) * (t * x).exp() * dx;
        }

        integral
    }

    /// MGF for Normal distribution.
    #[must_use]
    pub fn normal(mu: f64, sigma: f64, t: f64) -> f64 {
        (mu * t + 0.5 * sigma * sigma * t * t).exp()
    }

    /// MGF for Exponential distribution.
    #[must_use]
    pub fn exponential(lambda: f64, t: f64) -> f64 {
        if t < lambda {
            lambda / (lambda - t)
        } else {
            f64::INFINITY
        }
    }

    /// MGF for Poisson distribution.
    #[must_use]
    pub fn poisson(lambda: f64, t: f64) -> f64 {
        (lambda * ((t).exp() - 1.0)).exp()
    }

    /// MGF for Binomial distribution.
    #[must_use]
    pub fn binomial(n: u64, p: f64, t: f64) -> f64 {
        ((1.0 - p) + p * t.exp()).powi(n as i32)
    }

    /// MGF for Gamma distribution.
    #[must_use]
    pub fn gamma(shape: f64, rate: f64, t: f64) -> f64 {
        if t < rate {
            (rate / (rate - t)).powf(shape)
        } else {
            f64::INFINITY
        }
    }

    /// nth moment from MGF: E[X^n] = M^(n)(0).
    #[must_use]
    pub fn nth_moment(mgf: impl Fn(f64) -> f64, n: u32, epsilon: f64) -> f64 {
        if n == 0 {
            return mgf(0.0);
        }

        let mut result = mgf(0.0);
        for _ in 0..n {
            let f_plus = mgf(epsilon);
            let f_minus = mgf(-epsilon);
            result = (f_plus - f_minus) / (2.0 * epsilon);
        }

        result
    }
}

/// Characteristic function.
#[must_use]
pub struct CharacteristicFunction;

impl CharacteristicFunction {
    /// Characteristic function: φ(t) = E[e^(itX)].
    #[must_use]
    pub fn compute(pmf: &[f64], t: f64) -> f64 {
        let mut real_part = 0.0;
        let mut _imag_part = 0.0;

        for (k, &p) in pmf.iter().enumerate() {
            let angle = t * k as f64;
            real_part += p * angle.cos();
            _imag_part += p * angle.sin();
        }

        real_part // Return real part for simplicity
    }

    /// Characteristic function for continuous distribution.
    #[must_use]
    pub fn continuous(pdf: impl Fn(f64) -> f64, t: f64, a: f64, b: f64, n: usize) -> f64 {
        let dx = (b - a) / n as f64;
        let mut real_integral = 0.0;

        for i in 0..n {
            let x = a + (i as f64 + 0.5) * dx;
            let angle = t * x;
            real_integral += pdf(x) * angle.cos() * dx;
        }

        real_integral
    }

    /// Characteristic function for Normal distribution.
    #[must_use]
    pub fn normal(mu: f64, sigma: f64, t: f64) -> f64 {
        (mu * t - 0.5 * sigma * sigma * t * t).cos()
    }

    /// Characteristic function for Poisson distribution.
    #[must_use]
    pub fn poisson(lambda: f64, t: f64) -> f64 {
        (lambda * (t.cos() - 1.0)).exp()
    }

    /// Characteristic function for Exponential distribution.
    #[must_use]
    pub fn exponential(lambda: f64, t: f64) -> f64 {
        lambda / (lambda - t) // Simplified (real part)
    }

    /// Inversion formula to recover PMF from characteristic function.
    #[must_use]
    pub fn invert_to_pmf(phi: impl Fn(f64) -> f64, k: i64, n: usize) -> f64 {
        let dt = 2.0 * core::f64::consts::PI / n as f64;
        let mut integral = 0.0;

        for i in 0..n {
            let t = i as f64 * dt;
            integral += phi(t) * (-t * k as f64).cos() * dt;
        }

        integral / (2.0 * core::f64::consts::PI)
    }
}

/// Laplace transform.
#[must_use]
pub struct LaplaceTransform;

impl LaplaceTransform {
    /// Laplace transform: L{f}(s) = ∫₀^∞ f(t) e^(-st) dt.
    #[must_use]
    pub fn compute(f: impl Fn(f64) -> f64, s: f64, max_time: f64, n: usize) -> f64 {
        let dt = max_time / n as f64;
        let mut integral = 0.0;

        for i in 0..n {
            let t = (i as f64 + 0.5) * dt;
            integral += f(t) * (-s * t).exp() * dt;
        }

        integral
    }

    /// Laplace transform of exponential function.
    #[must_use]
    pub fn exponential(lambda: f64, s: f64) -> f64 {
        if s > -lambda {
            1.0 / (s + lambda)
        } else {
            f64::INFINITY
        }
    }

    /// Laplace transform of unit step.
    #[must_use]
    pub fn unit_step(s: f64) -> f64 {
        if s > 0.0 {
            1.0 / s
        } else {
            f64::INFINITY
        }
    }

    /// Laplace transform of t^n.
    #[must_use]
    pub fn power_t(n: u32, s: f64) -> f64 {
        if s > 0.0 {
            let factorial = (1..=n).product::<u32>() as f64;
            factorial / s.powf(n as f64 + 1.0)
        } else {
            f64::INFINITY
        }
    }

    /// Inverse Laplace transform (numerical).
    #[must_use]
    pub fn inverse(laplace: impl Fn(f64) -> f64, t: f64, sigma: f64, n: usize) -> f64 {
        let omega = 10.0;
        let domega = omega / n as f64;
        let mut integral = 0.0;

        for i in 0..n {
            let omega_i = i as f64 * domega;
            let s = sigma + omega_i;
            let value = laplace(s) * (omega_i * t).cos();
            integral += value * domega;
        }

        (integral / core::f64::consts::PI) * (sigma * t).exp() / 2.0
    }
}

/// Z-transform (discrete-time).
#[must_use]
pub struct ZTransform;

impl ZTransform {
    /// Z-transform: X(z) = sum of x\[n\] z^(-n).
    #[must_use]
    pub fn compute(sequence: &[f64], z: f64) -> f64 {
        sequence
            .iter()
            .enumerate()
            .map(|(n, &x)| x * z.powi(-(n as i32)))
            .sum()
    }

    /// Z-transform of unit step sequence.
    #[must_use]
    pub fn unit_step(z: f64) -> f64 {
        if z.abs() > 1.0 {
            z / (z - 1.0)
        } else {
            f64::INFINITY
        }
    }

    /// Z-transform of exponential sequence.
    #[must_use]
    pub fn exponential(a: f64, z: f64) -> f64 {
        if z.abs() > a.abs() {
            z / (z - a)
        } else {
            f64::INFINITY
        }
    }

    /// Z-transform of geometric sequence.
    #[must_use]
    pub fn geometric(a: f64, z: f64) -> f64 {
        if z.abs() > a.abs() {
            z / (z - a)
        } else {
            f64::INFINITY
        }
    }

    /// Inverse Z-transform (numerical).
    #[must_use]
    pub fn inverse(x_func: impl Fn(f64) -> f64, n: usize) -> Vec<f64> {
        let mut sequence = Vec::with_capacity(n);

        for k in 0..n {
            let integral = Self::inverse_integral(&x_func, k as f64);
            sequence.push(integral);
        }

        sequence
    }

    fn inverse_integral(x_func: impl Fn(f64) -> f64, k: f64) -> f64 {
        let n = 1000;
        let dtheta = 2.0 * core::f64::consts::PI / n as f64;
        let mut integral = 0.0;

        for i in 0..n {
            let theta = i as f64 * dtheta;
            let z = theta.cos();
            let value = x_func(z) * (theta * k).cos();
            integral += value * dtheta;
        }

        integral / (2.0 * core::f64::consts::PI)
    }
}

/// Cumulant generating function.
#[must_use]
pub struct CumulantGeneratingFunction;

impl CumulantGeneratingFunction {
    /// CGF: K(t) = ln(M(t)).
    #[must_use]
    pub fn from_mgf(mgf: impl Fn(f64) -> f64, t: f64) -> f64 {
        mgf(t).ln()
    }

    /// nth cumulant from CGF: κ_n = K^(n)(0).
    #[must_use]
    pub fn nth_cumulant(cgf: impl Fn(f64) -> f64, n: u32, epsilon: f64) -> f64 {
        if n == 0 {
            return 0.0;
        }

        let mut result = cgf(0.0);
        for _ in 0..n {
            let f_plus = cgf(epsilon);
            let f_minus = cgf(-epsilon);
            result = (f_plus - f_minus) / (2.0 * epsilon);
        }

        result
    }

    /// First cumulant = mean.
    #[must_use]
    pub fn mean(cgf: impl Fn(f64) -> f64, epsilon: f64) -> f64 {
        Self::nth_cumulant(cgf, 1, epsilon)
    }

    /// Second cumulant = variance.
    #[must_use]
    pub fn variance(cgf: impl Fn(f64) -> f64, epsilon: f64) -> f64 {
        Self::nth_cumulant(cgf, 2, epsilon)
    }

    /// Third cumulant = skewness * variance^(3/2).
    #[must_use]
    pub fn third_cumulant(cgf: impl Fn(f64) -> f64, epsilon: f64) -> f64 {
        Self::nth_cumulant(cgf, 3, epsilon)
    }

    /// Fourth cumulant = kurtosis * variance^2.
    #[must_use]
    pub fn fourth_cumulant(cgf: impl Fn(f64) -> f64, epsilon: f64) -> f64 {
        Self::nth_cumulant(cgf, 4, epsilon)
    }
}

/// Bivariate generating functions.
#[must_use]
pub struct BivariateGeneratingFunctions;

impl BivariateGeneratingFunctions {
    /// Joint PGF: G(s,t) = E[s^X t^Y].
    #[must_use]
    pub fn joint_pgf(joint_pmf: &[Vec<f64>], s: f64, t: f64) -> f64 {
        joint_pmf
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(j, &p)| p * s.powi(i as i32) * t.powi(j as i32))
            })
            .sum()
    }

    /// PGF of sum: G_{X+Y}(s) = G(s,s).
    #[must_use]
    pub fn sum_pgf(joint_pmf: &[Vec<f64>], s: f64) -> f64 {
        Self::joint_pgf(joint_pmf, s, s)
    }

    /// Covariance from joint PGF.
    #[must_use]
    pub fn covariance_from_pgf(joint_pmf: &[Vec<f64>]) -> f64 {
        let epsilon = 1e-6;

        let _g_11 = Self::joint_pgf(joint_pmf, 1.0, 1.0);
        let _g_10 = Self::joint_pgf(joint_pmf, 1.0, 0.0);
        let _g_01 = Self::joint_pgf(joint_pmf, 0.0, 1.0);

        let mean_x = (Self::joint_pgf(joint_pmf, 1.0 + epsilon, 1.0)
            - Self::joint_pgf(joint_pmf, 1.0 - epsilon, 1.0))
            / (2.0 * epsilon);
        let mean_y = (Self::joint_pgf(joint_pmf, 1.0, 1.0 + epsilon)
            - Self::joint_pgf(joint_pmf, 1.0, 1.0 - epsilon))
            / (2.0 * epsilon);

        let mean_xy = (Self::joint_pgf(joint_pmf, 1.0 + epsilon, 1.0 + epsilon)
            - Self::joint_pgf(joint_pmf, 1.0 - epsilon, 1.0 + epsilon)
            - Self::joint_pgf(joint_pmf, 1.0 + epsilon, 1.0 - epsilon)
            + Self::joint_pgf(joint_pmf, 1.0 - epsilon, 1.0 - epsilon))
            / (4.0 * epsilon * epsilon);

        mean_xy - mean_x * mean_y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pgf_bernoulli() {
        let pgf = ProbabilityGeneratingFunction::bernoulli(0.5, 0.5);
        assert!((pgf - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_pgf_binomial() {
        let pgf = ProbabilityGeneratingFunction::binomial(10, 0.5, 1.0);
        assert!((pgf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mgf_normal() {
        let mgf = MomentGeneratingFunction::normal(0.0, 1.0, 0.0);
        assert!((mgf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mgf_exponential() {
        let mgf = MomentGeneratingFunction::exponential(2.0, 1.0);
        assert!((mgf - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_characteristic_normal() {
        let cf = CharacteristicFunction::normal(0.0, 1.0, 0.0);
        assert!((cf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_exponential() {
        let laplace = LaplaceTransform::exponential(2.0, 1.0);
        assert!((laplace - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_z_transform_unit_step() {
        let zt = ZTransform::unit_step(2.0);
        assert!((zt - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulant_mean() {
        let mgf = |t: f64| -> f64 { MomentGeneratingFunction::normal(5.0, 2.0, t) };
        let cgf = |t: f64| -> f64 { CumulantGeneratingFunction::from_mgf(mgf, t) };
        let mean = CumulantGeneratingFunction::mean(cgf, 1e-6);
        assert!((mean - 5.0).abs() < 1e-5);
    }
}
