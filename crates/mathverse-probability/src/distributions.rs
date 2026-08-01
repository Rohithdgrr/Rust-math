//! Distributions: moments, pmf/pdf, cdf. Sampling lives in the parent module.

use crate::{rng::Rng, special::ln_gamma, F64Ext};

/// Common moment API for every distribution.
pub trait Distribution {
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

pub trait DiscreteDist: Distribution {
    fn pmf(&self, k: i64) -> f64;
    fn cdf(&self, k: i64) -> f64;
    /// Inverse-CDF sampling.
    fn sample(&self, rng: &mut Rng) -> i64 {
        let u = rng.uniform();
        let mut k = 0;
        while self.cdf(k) < u {
            k += 1;
        }
        k
    }
}

pub trait ContinuousDist: Distribution {
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
}

/// `P(X = 1) = p`.
#[must_use]
pub struct Bernoulli {
    pub p: f64,
}
impl Distribution for Bernoulli {
    fn mean(&self) -> f64 {
        self.p
    }
    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
}
impl DiscreteDist for Bernoulli {
    fn pmf(&self, k: i64) -> f64 {
        match k {
            0 => 1.0 - self.p,
            1 => self.p,
            _ => 0.0,
        }
    }
    fn cdf(&self, k: i64) -> f64 {
        match k {
            k if k < 0 => 0.0,
            0 => 1.0 - self.p,
            _ => 1.0,
        }
    }
}

/// `Binomial(n, p)`: successes in `n` independent trials.
#[must_use]
pub struct Binomial {
    pub n: u64,
    pub p: f64,
}
impl Distribution for Binomial {
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
}
impl DiscreteDist for Binomial {
    /// Exact via integer binomial (n ≤ 34 keeps `u128` exact).
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 || k as u64 > self.n {
            return 0.0;
        }
        let k_u = k as u64;
        if self.p == 0.0 {
            return if k_u == 0 { 1.0 } else { 0.0 };
        }
        if self.p == 1.0 {
            return if k_u == self.n { 1.0 } else { 0.0 };
        }

        let k_eff = k_u.min(self.n - k_u);
        let ln_coeff = ln_gamma(self.n as f64 + 1.0)
            - ln_gamma(k_eff as f64 + 1.0)
            - ln_gamma((self.n - k_eff) as f64 + 1.0);
        let ln_pmf =
            ln_coeff + k_u as f64 * self.p.ln() + (self.n - k_u) as f64 * (1.0 - self.p).ln();
        ln_pmf.exp()
    }
    fn cdf(&self, k: i64) -> f64 {
        (0..=k).map(|i| self.pmf(i)).sum()
    }
}

/// `Poisson(λ)`: count of events in a fixed interval.
#[must_use]
pub struct Poisson {
    pub lambda: f64,
}
impl Distribution for Poisson {
    fn mean(&self) -> f64 {
        self.lambda
    }
    fn variance(&self) -> f64 {
        self.lambda
    }
}
impl DiscreteDist for Poisson {
    /// Multiplication recurrence avoids `k!` overflow.
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        if self.lambda < 0.0 || !self.lambda.is_finite() {
            return f64::NAN;
        }
        if self.lambda == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        let kf = k as f64;
        (kf * self.lambda.ln() - self.lambda - ln_gamma(kf + 1.0)).exp()
    }
    fn cdf(&self, k: i64) -> f64 {
        (0..=k).map(|i| self.pmf(i)).sum()
    }
}

/// `Uniform(a, b)` on `[a, b)`.
#[must_use]
pub struct Uniform {
    pub a: f64,
    pub b: f64,
}
impl Distribution for Uniform {
    fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }
    fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
}
impl ContinuousDist for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x >= self.a && x < self.b {
            1.0 / (self.b - self.a)
        } else {
            0.0
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        ((x - self.a) / (self.b - self.a)).clamp(0.0, 1.0)
    }
}

/// `Normal(μ, σ)`. CDF via Abramowitz–Stegun erf approximation (|err| < 1.5e-7).
#[must_use]
pub struct Normal {
    pub mu: f64,
    pub sigma: f64,
}
impl Distribution for Normal {
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }
}
impl ContinuousDist for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        (-0.5 * z * z).exp() / (self.sigma * (2.0 * core::f64::consts::PI).sqrt())
    }
    fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf((x - self.mu) / (self.sigma * core::f64::consts::SQRT_2)))
    }
}
impl Normal {
    /// Box–Muller sampling.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        let u1 = rng.uniform().max(1e-300);
        let u2 = rng.uniform();
        self.mu + self.sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

pub fn erf(x: f64) -> f64 {
    crate::special::erf(x)
}

/// Exponential distribution with rate λ.
#[must_use]
pub struct Exponential {
    pub lambda: f64,
}
impl Distribution for Exponential {
    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }
    fn variance(&self) -> f64 {
        1.0 / (self.lambda * self.lambda)
    }
}
impl ContinuousDist for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.lambda * (-self.lambda * x).exp()
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.lambda * x).exp()
        }
    }
}
impl Exponential {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        -rng.uniform().ln() / self.lambda
    }
}

/// Log-normal distribution.
#[must_use]
pub struct LogNormal {
    pub mu: f64,
    pub sigma: f64,
}
impl Distribution for LogNormal {
    fn mean(&self) -> f64 {
        (self.mu + 0.5 * self.sigma * self.sigma).exp()
    }
    fn variance(&self) -> f64 {
        let m = (self.mu + 0.5 * self.sigma * self.sigma).exp();
        m * m * ((self.sigma * self.sigma).exp() - 1.0)
    }
}
impl ContinuousDist for LogNormal {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            let ln_x = x.ln();
            let z = (ln_x - self.mu) / self.sigma;
            (-0.5 * z * z).exp() / (x * self.sigma * (2.0 * core::f64::consts::PI).sqrt())
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            0.5 * (1.0 + erf((x.ln() - self.mu) / (self.sigma * core::f64::consts::SQRT_2)))
        }
    }
}
impl LogNormal {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        let n = Normal {
            mu: self.mu,
            sigma: self.sigma,
        };
        n.sample(rng).exp()
    }
}

/// Weibull distribution.
#[must_use]
pub struct Weibull {
    pub shape: f64,
    pub scale: f64,
}
impl Distribution for Weibull {
    fn mean(&self) -> f64 {
        self.scale * (1.0 + 1.0 / self.shape).gamma()
    }
    fn variance(&self) -> f64 {
        let g1 = (1.0 + 1.0 / self.shape).gamma();
        let g2 = (1.0 + 2.0 / self.shape).gamma();
        self.scale * self.scale * (g2 - g1 * g1)
    }
}
impl ContinuousDist for Weibull {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            (self.shape / self.scale)
                * (x / self.scale).powf(self.shape - 1.0)
                * (-(x / self.scale).powf(self.shape)).exp()
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-(x / self.scale).powf(self.shape)).exp()
        }
    }
}
impl Weibull {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        self.scale * (-rng.uniform().ln()).powf(1.0 / self.shape)
    }
}

/// Chi-squared distribution with k degrees of freedom.
#[must_use]
pub struct ChiSquared {
    pub k: f64,
}
impl Distribution for ChiSquared {
    fn mean(&self) -> f64 {
        self.k
    }
    fn variance(&self) -> f64 {
        2.0 * self.k
    }
}
impl ContinuousDist for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            let k_half = self.k / 2.0;
            let num = x.powf(k_half - 1.0) * (-0.5 * x).exp();
            let den = 2.0_f64.powf(k_half) * k_half.gamma();
            num / den
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            let k_half = self.k / 2.0;
            lower_gamma(k_half, 0.5 * x) / k_half.gamma()
        }
    }
}
impl ChiSquared {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        Gamma {
            shape: self.k / 2.0,
            rate: 0.5,
        }
        .sample(rng)
    }
}

/// Student's t-distribution with ν degrees of freedom.
#[must_use]
pub struct StudentsT {
    pub nu: f64,
}
impl Distribution for StudentsT {
    fn mean(&self) -> f64 {
        if self.nu > 1.0 {
            0.0
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        if self.nu > 2.0 {
            self.nu / (self.nu - 2.0)
        } else {
            f64::NAN
        }
    }
}
impl ContinuousDist for StudentsT {
    fn pdf(&self, x: f64) -> f64 {
        let nu = self.nu;
        let coeff = ((nu + 1.0) / 2.0).gamma()
            / (nu.sqrt() * core::f64::consts::PI.sqrt() * (nu / 2.0).gamma());
        coeff * (1.0 + x * x / nu).powf(-(nu + 1.0) / 2.0)
    }
    fn cdf(&self, x: f64) -> f64 {
        let nu = self.nu;
        let u = nu / (nu + x * x);
        let ib = (nu / 2.0, 0.5).beta_inc(u);
        if x >= 0.0 {
            1.0 - 0.5 * ib
        } else {
            0.5 * ib
        }
    }
}

/// F-distribution with d1 and d2 degrees of freedom.
#[must_use]
pub struct FDistribution {
    pub d1: f64,
    pub d2: f64,
}
impl Distribution for FDistribution {
    fn mean(&self) -> f64 {
        if self.d2 > 2.0 {
            self.d2 / (self.d2 - 2.0)
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        if self.d2 > 4.0 {
            let d2 = self.d2;
            let d1 = self.d1;
            2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0).powi(2) * (d2 - 4.0))
        } else {
            f64::NAN
        }
    }
}
impl ContinuousDist for FDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            let d1 = self.d1;
            let d2 = self.d2;
            let coeff = ((d1 + d2) / 2.0).gamma() / ((d1 / 2.0).gamma() * (d2 / 2.0).gamma())
                * (d1 / d2).powf(d1 / 2.0);
            coeff * x.powf(d1 / 2.0 - 1.0) * (1.0 + d1 * x / d2).powf(-(d1 + d2) / 2.0)
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            let d1 = self.d1;
            let d2 = self.d2;
            let u = d1 * x / (d1 * x + d2);
            (d1 / 2.0, d2 / 2.0).beta_inc(u)
        }
    }
}

/// Geometric distribution (number of trials until first success).
#[must_use]
pub struct Geometric {
    pub p: f64,
}
impl Distribution for Geometric {
    fn mean(&self) -> f64 {
        1.0 / self.p
    }
    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }
}
impl DiscreteDist for Geometric {
    fn pmf(&self, k: i64) -> f64 {
        if k < 1 {
            0.0
        } else {
            self.p * (1.0 - self.p).powi(k as i32 - 1)
        }
    }
    fn cdf(&self, k: i64) -> f64 {
        if k < 1 {
            0.0
        } else {
            1.0 - (1.0 - self.p).powi(k as i32)
        }
    }
}

/// Negative binomial distribution.
#[must_use]
pub struct NegativeBinomial {
    pub r: f64,
    pub p: f64,
}
impl Distribution for NegativeBinomial {
    fn mean(&self) -> f64 {
        self.r * (1.0 - self.p) / self.p
    }
    fn variance(&self) -> f64 {
        self.r * (1.0 - self.p) / (self.p * self.p)
    }
}
impl DiscreteDist for NegativeBinomial {
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 {
            0.0
        } else {
            let k = k as f64;
            let coeff = (self.r + k - 1.0).gamma() / (k.gamma() * self.r.gamma());
            coeff * self.p.powf(self.r) * (1.0 - self.p).powf(k)
        }
    }
    fn cdf(&self, k: i64) -> f64 {
        (0..=k).map(|i| self.pmf(i)).sum()
    }
}

/// Hypergeometric distribution.
#[must_use]
pub struct Hypergeometric {
    pub n: u64,
    pub k: u64,
    pub n_draws: u64,
}
impl Distribution for Hypergeometric {
    fn mean(&self) -> f64 {
        self.n_draws as f64 * self.k as f64 / self.n as f64
    }
    fn variance(&self) -> f64 {
        let n = self.n as f64;
        let k = self.k as f64;
        let draws = self.n_draws as f64;
        draws * k / n * (1.0 - k / n) * ((n - draws) / (n - 1.0))
    }
}
impl DiscreteDist for Hypergeometric {
    fn pmf(&self, x: i64) -> f64 {
        if x < 0 {
            return 0.0;
        }
        let x = x as u64;

        let min_x = self.n_draws.saturating_sub(self.n - self.k);
        let max_x = self.k.min(self.n_draws);
        if x < min_x || x > max_x {
            return 0.0;
        }

        let n_minus_k = self.n - self.k;
        let ln_num = ln_gamma(self.k as f64 + 1.0)
            - ln_gamma(x as f64 + 1.0)
            - ln_gamma((self.k - x) as f64 + 1.0)
            + ln_gamma(n_minus_k as f64 + 1.0)
            - ln_gamma((self.n_draws - x) as f64 + 1.0)
            - ln_gamma((n_minus_k - self.n_draws + x) as f64 + 1.0);
        let ln_den = ln_gamma(self.n as f64 + 1.0)
            - ln_gamma(self.n_draws as f64 + 1.0)
            - ln_gamma((self.n - self.n_draws) as f64 + 1.0);
        (ln_num - ln_den).exp()
    }
    fn cdf(&self, k: i64) -> f64 {
        (0..=k).map(|i| self.pmf(i)).sum()
    }
}

/// Cauchy distribution.
#[must_use]
pub struct Cauchy {
    pub x0: f64,
    pub gamma: f64,
}
impl Distribution for Cauchy {
    fn mean(&self) -> f64 {
        f64::NAN
    }
    fn variance(&self) -> f64 {
        f64::NAN
    }
}
impl ContinuousDist for Cauchy {
    fn pdf(&self, x: f64) -> f64 {
        1.0 / (core::f64::consts::PI * self.gamma * (1.0 + ((x - self.x0) / self.gamma).powi(2)))
    }
    fn cdf(&self, x: f64) -> f64 {
        0.5 + (1.0 / core::f64::consts::PI) * ((x - self.x0) / self.gamma).atan()
    }
}
impl Cauchy {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        self.x0 + self.gamma * (core::f64::consts::PI * (rng.uniform() - 0.5)).tan()
    }
}

/// Laplace distribution.
#[must_use]
pub struct Laplace {
    pub mu: f64,
    pub b: f64,
}
impl Distribution for Laplace {
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        2.0 * self.b * self.b
    }
}
impl ContinuousDist for Laplace {
    fn pdf(&self, x: f64) -> f64 {
        (-(x - self.mu).abs() / self.b).exp() / (2.0 * self.b)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < self.mu {
            0.5 * ((x - self.mu) / self.b).exp()
        } else {
            1.0 - 0.5 * (-(x - self.mu) / self.b).exp()
        }
    }
}
impl Laplace {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        let u = rng.uniform() - 0.5;
        self.mu - self.b * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

/// Gumbel distribution.
#[must_use]
pub struct Gumbel {
    pub mu: f64,
    pub beta: f64,
}
impl Distribution for Gumbel {
    fn mean(&self) -> f64 {
        self.mu + self.beta * 0.577_215_664_901_532_9_f64
    }
    fn variance(&self) -> f64 {
        (core::f64::consts::PI * core::f64::consts::PI / 6.0) * self.beta * self.beta
    }
}
impl ContinuousDist for Gumbel {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.beta;
        (-z - (-z).exp()).exp() / self.beta
    }
    fn cdf(&self, x: f64) -> f64 {
        (-((x - self.mu) / self.beta).exp()).exp()
    }
}
impl Gumbel {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        self.mu - self.beta * (-(-rng.uniform().ln()).ln())
    }
}

/// Pareto distribution.
#[must_use]
pub struct Pareto {
    pub xm: f64,
    pub alpha: f64,
}
impl Distribution for Pareto {
    fn mean(&self) -> f64 {
        if self.alpha > 1.0 {
            self.alpha * self.xm / (self.alpha - 1.0)
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        if self.alpha > 2.0 {
            self.xm * self.xm * self.alpha / ((self.alpha - 1.0).powi(2) * (self.alpha - 2.0))
        } else {
            f64::NAN
        }
    }
}
impl ContinuousDist for Pareto {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.xm {
            0.0
        } else {
            self.alpha * self.xm.powf(self.alpha) / x.powf(self.alpha + 1.0)
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < self.xm {
            0.0
        } else {
            1.0 - (self.xm / x).powf(self.alpha)
        }
    }
}
impl Pareto {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        self.xm / (1.0 - rng.uniform()).powf(1.0 / self.alpha)
    }
}

/// Triangular distribution.
#[must_use]
pub struct Triangular {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}
impl Distribution for Triangular {
    fn mean(&self) -> f64 {
        (self.a + self.b + self.c) / 3.0
    }
    fn variance(&self) -> f64 {
        let a = self.a;
        let b = self.b;
        let c = self.c;
        (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
    }
}
impl ContinuousDist for Triangular {
    fn pdf(&self, x: f64) -> f64 {
        let a = self.a;
        let b = self.b;
        let c = self.c;
        if x < a || x > b {
            0.0
        } else if x < c {
            2.0 * (x - a) / ((b - a) * (c - a))
        } else {
            2.0 * (b - x) / ((b - a) * (b - c))
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        let a = self.a;
        let b = self.b;
        let c = self.c;
        if x <= a {
            0.0
        } else if x < c {
            (x - a).powi(2) / ((b - a) * (c - a))
        } else if x < b {
            1.0 - (b - x).powi(2) / ((b - a) * (b - c))
        } else {
            1.0
        }
    }
}

/// Beta distribution.
#[must_use]
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}
impl Distribution for Beta {
    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
    fn variance(&self) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        a * b / ((a + b).powi(2) * (a + b + 1.0))
    }
}
impl ContinuousDist for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            0.0
        } else {
            let a = self.alpha;
            let b = self.beta;
            x.powf(a - 1.0) * (1.0 - x).powf(b - 1.0) / (a, b).beta()
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            (self.alpha, self.beta).beta_inc(x)
        }
    }
}

/// Gamma distribution.
#[must_use]
pub struct Gamma {
    pub shape: f64,
    pub rate: f64,
}
impl Distribution for Gamma {
    fn mean(&self) -> f64 {
        self.shape / self.rate
    }
    fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }
}
impl ContinuousDist for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            let k = self.shape;
            let theta = 1.0 / self.rate;
            x.powf(k - 1.0) * (-x / theta).exp() / (k.gamma() * theta.powf(k))
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            lower_gamma(self.shape, self.rate * x) / self.shape.gamma()
        }
    }
}
impl Gamma {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        if self.shape >= 1.0 {
            marsaglia_tsang_gamma(self.shape, rng) / self.rate
        } else {
            let u = rng.uniform();
            marsaglia_tsang_gamma(1.0 + self.shape, rng) / self.rate * u.powf(1.0 / self.shape)
        }
    }
}

fn marsaglia_tsang_gamma(shape: f64, rng: &mut Rng) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = Normal {
            mu: 0.0,
            sigma: 1.0,
        }
        .sample(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.uniform();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

pub fn lower_gamma(a: f64, x: f64) -> f64 {
    crate::special::lower_gamma(a, x)
}

pub(crate) trait BetaFunc {
    fn beta(self) -> f64;
    fn beta_inc(self, x: f64) -> f64;
}

impl BetaFunc for (f64, f64) {
    fn beta(self) -> f64 {
        let (a, b) = self;
        a.gamma() * b.gamma() / (a + b).gamma()
    }

    fn beta_inc(self, x: f64) -> f64 {
        let (a, b) = self;
        if x == 0.0 {
            0.0
        } else if x == 1.0 {
            1.0
        } else if !(a > 0.0 && b > 0.0) {
            f64::NAN
        } else {
            const MAX_ITERS: usize = 200;
            const EPS: f64 = 3.0e-14;
            const FPMIN: f64 = 1.0e-300;

            fn betacf(a: f64, b: f64, x: f64) -> f64 {
                const MAX_ITERS: usize = 200;
                const EPS: f64 = 3.0e-14;
                const FPMIN: f64 = 1.0e-300;

                let qab = a + b;
                let qap = a + 1.0;
                let qam = a - 1.0;
                let mut c = 1.0;
                let mut d = 1.0 - qab * x / qap;
                if d.abs() < FPMIN {
                    d = FPMIN;
                }
                d = 1.0 / d;
                let mut h = d;

                for m in 1..=MAX_ITERS {
                    let m2 = 2.0 * m as f64;
                    let mut aa = m as f64 * (b - m as f64) * x / ((qam + m2) * (a + m2));
                    d = 1.0 + aa * d;
                    if d.abs() < FPMIN {
                        d = FPMIN;
                    }
                    c = 1.0 + aa / c;
                    if c.abs() < FPMIN {
                        c = FPMIN;
                    }
                    d = 1.0 / d;
                    h *= d * c;

                    aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2) * (qap + m2));
                    d = 1.0 + aa * d;
                    if d.abs() < FPMIN {
                        d = FPMIN;
                    }
                    c = 1.0 + aa / c;
                    if c.abs() < FPMIN {
                        c = FPMIN;
                    }
                    d = 1.0 / d;
                    let delta = d * c;
                    h *= delta;

                    if (delta - 1.0).abs() < EPS {
                        break;
                    }
                }

                h
            }

            let ln_bt =
                ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
            let bt = ln_bt.exp();
            let threshold = (a + 1.0) / (a + b + 2.0);
            if x < threshold {
                bt * betacf(a, b, x) / a
            } else {
                1.0 - bt * betacf(b, a, 1.0 - x) / b
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_match_closed_forms() {
        assert!((Bernoulli { p: 0.25 }.mean() - 0.25).abs() < 1e-12);
        assert!((Bernoulli { p: 0.25 }.variance() - 0.1875).abs() < 1e-12);
        let b = Binomial { n: 10, p: 0.5 };
        assert!((b.mean() - 5.0).abs() < 1e-12);
        assert!((b.variance() - 2.5).abs() < 1e-12);
        let p = Poisson { lambda: 3.0 };
        assert!((p.mean() - 3.0).abs() < 1e-12);
        assert!((p.variance() - 3.0).abs() < 1e-12);
        let u = Uniform { a: 0.0, b: 4.0 };
        assert!((u.mean() - 2.0).abs() < 1e-12);
        assert!((u.variance() - 16.0 / 12.0).abs() < 1e-12);
        let n = Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        assert!((n.mean() - 0.0).abs() < 1e-12);
        assert!((n.variance() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pmf_pdf_cdf() {
        let b = Binomial { n: 10, p: 0.5 };
        assert!((b.pmf(5) - 252.0 / 1024.0).abs() < 1e-12);
        assert!((b.cdf(5) - 638.0 / 1024.0).abs() < 1e-12);
        assert!(Binomial { n: 70, p: 0.5 }.pmf(35).is_finite());
        let p = Poisson { lambda: 2.0 };
        assert!((p.pmf(0) - (-2.0f64).exp()).abs() < 1e-12);
        assert!(Poisson { lambda: 1000.0 }.pmf(1000).is_finite());
        let n = Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        assert!((n.cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((n.cdf(1.0) - 0.841344746).abs() < 1e-6);
        assert!((n.cdf(-1.0) - 0.158655254).abs() < 1e-6);
        assert!((n.pdf(0.0) - 1.0 / (2.0 * core::f64::consts::PI).sqrt()).abs() < 1e-12);
        let t = StudentsT { nu: 1.0 };
        assert!((t.pdf(0.0) - 1.0 / core::f64::consts::PI).abs() < 1e-12);
        assert!((t.cdf(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn hypergeometric_large_parameters_stay_finite() {
        let h = Hypergeometric {
            n: 70,
            k: 35,
            n_draws: 35,
        };
        assert!(h.pmf(17).is_finite());
        assert!(h.pmf(17) > 0.0);
    }

    #[test]
    fn beta_cdf_tail_is_well_behaved() {
        let b = Beta {
            alpha: 20.0,
            beta: 5.0,
        };
        let cdf = b.cdf(0.999);
        assert!(cdf.is_finite());
        assert!((0.0..=1.0).contains(&cdf));
    }

    #[test]
    fn sampling_empirics() {
        let mut rng = Rng::new(42);
        let n = Normal {
            mu: 2.0,
            sigma: 3.0,
        };
        let m: f64 = (0..50_000).map(|_| n.sample(&mut rng)).sum::<f64>() / 50_000.0;
        assert!((m - 2.0).abs() < 0.1, "empirical mean {m}");
        let u = rng.uniform();
        assert!((0.0..1.0).contains(&u));
    }
}
