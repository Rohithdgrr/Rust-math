//! Statistical distributions: Normal, t, Chi-squared, F, Binomial, Poisson.
//! Each provides PDF, CDF, and PPF (inverse CDF).

use std::f64::consts::{PI, SQRT_2};

const TAU: f64 = 2.0 * PI;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun approximation
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let poly =
        0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3 - 1.453152027 * t4 + 1.061405429 * t5;
    sign * (1.0 - poly * (-x * x).exp())
}

fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

fn gamma_ln(x: f64) -> f64 {
    // Lanczos approximation
    let g = 7.0;
    let c = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - gamma_ln(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = c[0];
    let tu = x + g + 0.5;
    for (i, c_i) in c.iter().enumerate().skip(1) {
        a += c_i / (x + i as f64);
    }
    0.5 * (TAU).ln() + (x + g + 0.5).ln() * (x + 0.5) - tu + a.ln()
}

fn gamma(x: f64) -> f64 {
    gamma_ln(x).exp()
}

fn incomplete_gamma_lower(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 2.0 {
        // Series expansion for γ(a, x) = x^a * e^{-x} * Σ
        let mut sum = 1.0 / a;
        let mut term = 1.0 / a;
        for n in 1..200 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }
        sum * (-x + a * x.ln()).exp()
    } else {
        // Continued fraction (Lentz's method) for upper incomplete gamma Γ(a,x)
        let b = x + 1.0 - a;
        let mut d = 1.0 / b;
        let mut c = d;
        let mut f = d;
        for i in 1..200 {
            let an = -(i as f64) * (i as f64 - a);
            let bn = b + 2.0 * i as f64;
            let dn = bn + an * d;
            if dn.abs() < 1e-30 {
                d = 1e-30;
            } else {
                d = 1.0 / dn;
            }
            let cn = bn + an / c;
            if cn.abs() < 1e-30 {
                c = 1e-30;
            } else {
                c = cn;
            }
            let delta = d * c;
            f *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        gamma(a) * (1.0 - f * (-x + a * x.ln() - gamma_ln(a)).exp())
    }
}

fn incomplete_gamma_regularized(a: f64, x: f64) -> f64 {
    incomplete_gamma_lower(a, x) / gamma(a)
}

fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    // Continued fraction (incomplete beta)
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = gamma_ln(a) + gamma_ln(b) - gamma_ln(a + b);
    let prefix = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut f = d;
    for i in 1..200 {
        let m = i as f64;
        let numerator = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + numerator / d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + numerator / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        f *= d * c;

        let numerator = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + numerator / d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + numerator / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }
    prefix * f
}

/// Student's t probability density function with `df` degrees of freedom.
#[must_use]
pub fn student_t_pdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    let dt = df + 1.0;
    let coeff = (gamma_ln(dt / 2.0) - gamma_ln(df / 2.0) - 0.5 * (df * PI).ln()).exp();
    coeff * (1.0 + t * t / df).powf(-dt / 2.0)
}

/// Student's t cumulative distribution function.
#[must_use]
pub fn student_t_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    let ib = beta_inc(df / 2.0, 0.5, x);
    if t >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

/// Student's t percent-point function (inverse CDF).
#[must_use]
pub fn student_t_ppf(p: f64, df: f64) -> f64 {
    // Newton-Raphson starting from normal approximation
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let mut x = normal_ppf(p);
    for _ in 0..50 {
        let cdf = student_t_cdf(x, df);
        let pdf = student_t_pdf(x, df);
        if pdf < 1e-30 {
            break;
        }
        let dx = (cdf - p) / pdf;
        x -= dx;
        if dx.abs() < 1e-12 {
            break;
        }
    }
    x
}

/// Chi-squared probability density function with `k` degrees of freedom.
#[must_use]
pub fn chi_squared_pdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 || k <= 0.0 {
        return 0.0;
    }
    let half_k = k / 2.0;
    ((half_k - 1.0) * x.ln() - x / 2.0 - half_k * 2.0_f64.ln() - gamma_ln(half_k)).exp()
}

/// Chi-squared cumulative distribution function.
#[must_use]
pub fn chi_squared_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 || k <= 0.0 {
        return 0.0;
    }
    incomplete_gamma_regularized(k / 2.0, x / 2.0)
}

/// Chi-squared percent-point function (inverse CDF).
#[must_use]
pub fn chi_squared_ppf(p: f64, k: f64) -> f64 {
    let mut x = k;
    for _ in 0..50 {
        let cdf = chi_squared_cdf(x, k);
        let pdf = chi_squared_pdf(x, k);
        if pdf < 1e-30 {
            break;
        }
        let dx = (cdf - p) / pdf;
        x -= dx;
        if dx.abs() < 1e-12 {
            break;
        }
    }
    x.max(0.0)
}

/// F-distribution probability density function with `d1`, `d2` degrees of freedom.
#[must_use]
pub fn f_pdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return 0.0;
    }
    let ln = d1.ln() * d1 / 2.0 + d2.ln() * d2 / 2.0 + gamma_ln((d1 + d2) / 2.0)
        - gamma_ln(d1 / 2.0)
        - gamma_ln(d2 / 2.0)
        + (d1 / 2.0 - 1.0) * x.ln()
        - (d1 * x + d2).ln() * (d1 + d2) / 2.0;
    ln.exp()
}

/// F-distribution cumulative distribution function.
#[must_use]
pub fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let z = d1 * x / (d1 * x + d2);
    beta_inc(d1 / 2.0, d2 / 2.0, z)
}

/// F-distribution percent-point function (inverse CDF).
#[must_use]
pub fn f_ppf(p: f64, d1: f64, d2: f64) -> f64 {
    let mut x = 1.0;
    for _ in 0..50 {
        let cdf = f_cdf(x, d1, d2);
        let pdf = f_pdf(x, d1, d2);
        if pdf < 1e-30 {
            break;
        }
        let dx = (cdf - p) / pdf;
        x -= dx;
        if dx.abs() < 1e-12 {
            break;
        }
    }
    x.max(0.0)
}

// ---------------------------------------------------------------------------
// Public: Normal
// ---------------------------------------------------------------------------

/// Standard normal distribution (mean=0, std=1).
pub struct Normal;

impl Normal {
    /// Normal probability density function: `φ(x) = e^{-x²/2} / √(2π)`.
    #[must_use]
    pub fn pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
    }
    /// Normal cumulative distribution function.
    #[must_use]
    pub fn cdf(x: f64) -> f64 {
        0.5 * erfc(-x / SQRT_2)
    }
    /// Normal percent-point function (inverse CDF).
    #[must_use]
    pub fn ppf(p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        // Rational approximation (Abramowitz & Stegun 26.2.23)
        let t = if p < 0.5 {
            (-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p).ln()).sqrt()
        };
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;
        let approx = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
        if p < 0.5 {
            -approx
        } else {
            approx
        }
    }
}

/// Normal PDF: `φ(x)`.
#[must_use]
pub fn normal_pdf(x: f64) -> f64 {
    Normal::pdf(x)
}

/// Normal CDF: `Φ(x) = ½(1 + erf(x/√2))`.
#[must_use]
pub fn normal_cdf(x: f64) -> f64 {
    Normal::cdf(x)
}

/// Normal PPF (quantile): inverse of `normal_cdf`.
#[must_use]
pub fn normal_ppf(p: f64) -> f64 {
    Normal::ppf(p)
}

// ---------------------------------------------------------------------------
// Public: Student's t
// ---------------------------------------------------------------------------

/// Student's t distribution.
pub struct StudentT;

impl StudentT {
    /// Student's t probability density function.
    #[must_use]
    pub fn pdf(t: f64, df: f64) -> f64 {
        student_t_pdf(t, df)
    }
    /// Student's t cumulative distribution function.
    #[must_use]
    pub fn cdf(t: f64, df: f64) -> f64 {
        student_t_cdf(t, df)
    }
    /// Student's t percent-point function (inverse CDF).
    #[must_use]
    pub fn ppf(p: f64, df: f64) -> f64 {
        student_t_ppf(p, df)
    }
}

// ---------------------------------------------------------------------------
// Public: Chi-squared
// ---------------------------------------------------------------------------

/// Chi-squared distribution.
pub struct ChiSquared;

impl ChiSquared {
    /// Chi-squared probability density function.
    #[must_use]
    pub fn pdf(x: f64, k: f64) -> f64 {
        chi_squared_pdf(x, k)
    }
    /// Chi-squared cumulative distribution function.
    #[must_use]
    pub fn cdf(x: f64, k: f64) -> f64 {
        chi_squared_cdf(x, k)
    }
    /// Chi-squared percent-point function (inverse CDF).
    #[must_use]
    pub fn ppf(p: f64, k: f64) -> f64 {
        chi_squared_ppf(p, k)
    }
}

// ---------------------------------------------------------------------------
// Public: F
// ---------------------------------------------------------------------------

/// F-distribution.
pub struct FDist;

impl FDist {
    /// F-distribution probability density function.
    #[must_use]
    pub fn pdf(x: f64, d1: f64, d2: f64) -> f64 {
        f_pdf(x, d1, d2)
    }
    /// F-distribution cumulative distribution function.
    #[must_use]
    pub fn cdf(x: f64, d1: f64, d2: f64) -> f64 {
        f_cdf(x, d1, d2)
    }
    /// F-distribution percent-point function (inverse CDF).
    #[must_use]
    pub fn ppf(p: f64, d1: f64, d2: f64) -> f64 {
        f_ppf(p, d1, d2)
    }
}

// ---------------------------------------------------------------------------
// Public: Binomial
// ---------------------------------------------------------------------------

/// Binomial distribution.
pub struct Binomial;

impl Binomial {
    /// Binomial probability mass function: `P(X = k)`.
    #[must_use]
    pub fn pmf(k: u64, n: u64, p: f64) -> f64 {
        binomial_coeff(n, k) * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
    }
    /// Binomial cumulative distribution function.
    #[must_use]
    pub fn cdf(k: u64, n: u64, p: f64) -> f64 {
        (0..=k).map(|i| Self::pmf(i, n, p)).sum()
    }
    /// Binomial mean: `n * p`.
    #[must_use]
    pub fn mean(n: u64, p: f64) -> f64 {
        n as f64 * p
    }
    /// Binomial variance: `n * p * (1 - p)`.
    #[must_use]
    pub fn variance(n: u64, p: f64) -> f64 {
        n as f64 * p * (1.0 - p)
    }
}

/// Binomial PMF: `P(X = k) = C(n,k) * p^k * (1-p)^(n-k)`.
#[must_use]
pub fn binomial_pmf(k: u64, n: u64, p: f64) -> f64 {
    Binomial::pmf(k, n, p)
}
/// Binomial CDF: `P(X ≤ k)`.
#[must_use]
pub fn binomial_cdf(k: u64, n: u64, p: f64) -> f64 {
    Binomial::cdf(k, n, p)
}

// ---------------------------------------------------------------------------
// Public: Poisson
// ---------------------------------------------------------------------------

/// Poisson distribution.
pub struct Poisson;

impl Poisson {
    /// Poisson probability mass function: `P(X = k) = e^{-λ} * λ^k / k!`.
    #[must_use]
    pub fn pmf(k: u64, lambda: f64) -> f64 {
        (-lambda + k as f64 * lambda.ln() - gamma_ln(k as f64 + 1.0)).exp()
    }
    /// Poisson cumulative distribution function.
    #[must_use]
    pub fn cdf(k: u64, lambda: f64) -> f64 {
        (0..=k).map(|i| Self::pmf(i, lambda)).sum()
    }
    /// Poisson mean: `λ`.
    #[must_use]
    pub fn mean(lambda: f64) -> f64 {
        lambda
    }
    /// Poisson variance: `λ`.
    #[must_use]
    pub fn variance(lambda: f64) -> f64 {
        lambda
    }
}

/// Poisson PMF: `P(X = k) = e^{-λ} * λ^k / k!`.
#[must_use]
pub fn poisson_pmf(k: u64, lambda: f64) -> f64 {
    Poisson::pmf(k, lambda)
}
/// Poisson CDF: `P(X ≤ k)`.
#[must_use]
pub fn poisson_cdf(k: u64, lambda: f64) -> f64 {
    Poisson::cdf(k, lambda)
}

// ---------------------------------------------------------------------------
// Binomial coefficient (for Binomial PMF)
// ---------------------------------------------------------------------------

fn binomial_coeff(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn normal_test() {
        assert!((Normal::pdf(0.0) - 1.0 / (TAU).sqrt()).abs() < EPS);
        assert!((Normal::cdf(0.0) - 0.5).abs() < EPS);
        assert!((Normal::cdf(1.96) - 0.975).abs() < 5e-3);
        assert!((Normal::ppf(0.975) - 1.96).abs() < 0.01);
        assert!((Normal::ppf(0.5)).abs() < EPS);
    }

    #[test]
    fn student_t_test() {
        assert!((StudentT::cdf(0.0, 10.0) - 0.5).abs() < EPS);
        // t with df=∞ → normal
        assert!((StudentT::cdf(1.96, 1000.0) - Normal::cdf(1.96)).abs() < 0.05);
    }

    #[test]
    fn chi_squared_test() {
        assert!((ChiSquared::cdf(0.0, 3.0)).abs() < EPS);
        assert!((ChiSquared::cdf(6.251, 3.0) - 0.9).abs() < 0.01);
    }

    #[test]
    fn chi_squared_pdf_formula_test() {
        // Test the corrected formula - for k=2, should be exp(-x/2)/2
        let x = 1.0;
        let k = 2.0;
        let pdf = chi_squared_pdf(x, k);
        let expected = (-x / 2.0).exp() / 2.0;
        assert!((pdf - expected).abs() < 1e-10);
    }

    #[test]
    fn chi_squared_pdf_invalid_k() {
        // Test validation for invalid k
        assert_eq!(chi_squared_pdf(1.0, 0.0), 0.0);
        assert_eq!(chi_squared_pdf(1.0, -1.0), 0.0);
    }

    #[test]
    fn f_dist_test() {
        assert!((FDist::cdf(0.0, 5.0, 10.0)).abs() < EPS);
    }

    #[test]
    fn f_pdf_formula_test() {
        // Test the corrected formula - should be around 0.375 for F(4,4) at x=1
        let x = 1.0;
        let d1 = 4.0;
        let d2 = 4.0;
        let pdf = f_pdf(x, d1, d2);
        assert!((pdf - 0.375).abs() < 0.1);
    }

    #[test]
    fn f_pdf_invalid_params() {
        // Test validation for invalid parameters
        assert_eq!(f_pdf(1.0, 0.0, 4.0), 0.0);
        assert_eq!(f_pdf(1.0, 4.0, 0.0), 0.0);
        assert_eq!(f_pdf(1.0, -1.0, 4.0), 0.0);
    }

    #[test]
    fn binomial_test() {
        assert!((Binomial::pmf(3, 10, 0.5) - 0.1171875).abs() < EPS);
        assert!((Binomial::cdf(0, 10, 0.5) - Binomial::pmf(0, 10, 0.5)).abs() < EPS);
        assert!((Binomial::mean(10, 0.5) - 5.0).abs() < EPS);
    }

    #[test]
    fn poisson_test() {
        assert!((Poisson::pmf(0, 1.0) - (-1.0_f64).exp()).abs() < EPS);
        assert!((Poisson::pmf(1, 1.0) - (-1.0_f64).exp()).abs() < EPS);
        assert!((Poisson::mean(3.0) - 3.0).abs() < EPS);
    }
}
