//! Distributions: moments, pmf/pdf, cdf. Sampling lives in the parent module.

use crate::{rng::Rng, special::ln_gamma, F64Ext};
#[cfg(test)]
use crate::{markov_distribution, markov_step};

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

    /// Smallest value with non-negligible probability mass (inclusive).
    fn support_min(&self) -> i64 {
        0
    }

    /// Largest value with non-negligible probability mass (inclusive).
    fn support_max(&self) -> i64 {
        i64::MAX / 2
    }

    /// Inverse-CDF sampling: exponential search for an upper bound, then a
    /// binary search for the smallest `k` with `cdf(k) >= u`. O(log support)
    /// CDF evaluations instead of a linear walk.
    fn sample(&self, rng: &mut Rng) -> i64 {
        let u = rng.uniform();
        let max = self.support_max();

        let mut hi = self.support_min();
        let mut step = 1i64;
        while self.cdf(hi) < u && hi < max {
            hi = hi.saturating_add(step);
            step = step.saturating_mul(2);
        }
        if hi > max {
            hi = max;
        }

        let mut lo = self.support_min();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cdf(mid) < u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

pub trait ContinuousDist: Distribution {
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;

    /// Quantile function (inverse CDF). Default: binary search on `cdf`.
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        if q <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if q >= 1.0 {
            return f64::INFINITY;
        }
        let mut lo = -1e10;
        let mut hi = 1e10;
        for _ in 0..200 {
            let mid = f64::midpoint(lo, hi);
            if self.cdf(mid) < q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        f64::midpoint(lo, hi)
    }

    /// Survival function: `1 - cdf(x)`.
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    /// Inverse survival function: `ppf(1 - q)`.
    fn isf(&self, q: f64) -> f64 {
        self.ppf(1.0 - q)
    }
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
    fn support_min(&self) -> i64 {
        0
    }
    fn support_max(&self) -> i64 {
        1
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
impl Binomial {
    /// Log-space PMF (returns None on underflow).
    fn ln_pmf(&self, k: u64) -> Option<f64> {
        let k_eff = k.min(self.n - k);
        let ln_coeff = ln_gamma(self.n as f64 + 1.0)
            - ln_gamma(k_eff as f64 + 1.0)
            - ln_gamma((self.n - k_eff) as f64 + 1.0);
        let ln_pmf =
            ln_coeff + k as f64 * self.p.ln() + (self.n - k) as f64 * (1.0 - self.p).ln();
        if ln_pmf < -700.0 {
            None
        } else {
            Some(ln_pmf)
        }
    }

    /// Compute PMF via recurrence from the mode, avoiding underflow.
    fn pmf_from_mode(&self, k: u64) -> f64 {
        let mode = ((self.n as f64 + 1.0) * self.p).floor() as u64;
        let mode = mode.min(self.n);

        let ln_ref = self.ln_pmf_raw(mode);
        if ln_ref < -700.0 {
            return self.pmf_recurrent_full(k, mode);
        }
        let mode_pmf = ln_ref.exp();

        if k == mode {
            return mode_pmf;
        }

        let mut current = mode;
        let mut val = mode_pmf;
        let q = 1.0 - self.p;

        if k < mode {
            while current > k {
                val *= current as f64 * q / ((self.n - current + 1) as f64 * self.p);
                current -= 1;
            }
        } else {
            while current < k {
                val *= (self.n - current) as f64 * self.p / ((current + 1) as f64 * q);
                current += 1;
            }
        }
        val
    }

    /// Full recurrence between two arbitrary points (both in underflow region).
    fn pmf_recurrent_full(&self, k: u64, ref_point: u64) -> f64 {
        // Compute ln_pmf at ref_point without exp, then do recurrence in log-space
        let ln_ref = self.ln_pmf_raw(ref_point);
        let mut ln_val = ln_ref;
        let q = 1.0 - self.p;
        let mut current = ref_point;

        if k < ref_point {
            while current > k {
                ln_val += (current as f64).ln() + q.ln()
                    - ((self.n - current + 1) as f64).ln()
                    - self.p.ln();
                current -= 1;
            }
        } else {
            while current < k {
                ln_val += ((self.n - current) as f64).ln() + self.p.ln()
                    - ((current + 1) as f64).ln()
                    - q.ln();
                current += 1;
            }
        }
        ln_val.exp()
    }

    /// Raw `ln_pmf` without underflow guard.
    fn ln_pmf_raw(&self, k: u64) -> f64 {
        let k_eff = k.min(self.n - k);
        let ln_coeff = ln_gamma(self.n as f64 + 1.0)
            - ln_gamma(k_eff as f64 + 1.0)
            - ln_gamma((self.n - k_eff) as f64 + 1.0);
        ln_coeff + k as f64 * self.p.ln() + (self.n - k) as f64 * (1.0 - self.p).ln()
    }
}

impl DiscreteDist for Binomial {
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

        if let Some(ln) = self.ln_pmf(k_u) {
            ln.exp()
        } else {
            self.pmf_from_mode(k_u)
        }
    }
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        let n = self.n.min(i64::MAX as u64) as i64;
        if k >= n {
            return 1.0;
        }
        if self.p == 0.0 {
            return 1.0;
        }
        if self.p == 1.0 {
            return 0.0;
        }
        let k = k as f64;
        (self.n as f64 - k, k + 1.0).beta_inc(1.0 - self.p)
    }
    fn support_min(&self) -> i64 {
        0
    }
    fn support_max(&self) -> i64 {
        self.n.min(i64::MAX as u64) as i64
    }
}

/// `Poisson(Î»)`: count of events in a fixed interval.
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
impl Poisson {
    /// Log-space PMF (returns None on underflow).
    fn ln_pmf(&self, k: i64) -> Option<f64> {
        if k < 0 {
            return Some(f64::NEG_INFINITY);
        }
        let kf = k as f64;
        let ln_pmf = kf * self.lambda.ln() - self.lambda - ln_gamma(kf + 1.0);
        if ln_pmf < -700.0 {
            None
        } else {
            Some(ln_pmf)
        }
    }

    /// Compute PMF via recurrence from the mode, avoiding underflow.
    fn pmf_from_mode(&self, k: i64) -> f64 {
        let mode = self.lambda.floor() as i64;
        let mode_pmf = if let Some(ln) = self.ln_pmf(mode) {
            ln.exp()
        } else {
            return self.pmf_recurrent_full(k, mode);
        };

        if k == mode {
            return mode_pmf;
        }

        let mut current = mode;
        let mut val = mode_pmf;

        if k < mode {
            while current > k {
                val *= self.lambda / current as f64;
                current -= 1;
            }
        } else {
            while current < k {
                current += 1;
                val *= self.lambda / current as f64;
            }
        }
        val
    }

    /// Full recurrence in log-space between two arbitrary points.
    fn pmf_recurrent_full(&self, k: i64, ref_point: i64) -> f64 {
        let ln_ref = self.ln_pmf_raw(ref_point);
        let mut ln_val = ln_ref;
        let ln_lambda = self.lambda.ln();
        let mut current = ref_point;

        if k < ref_point {
            while current > k {
                ln_val += ln_lambda - (current as f64).ln();
                current -= 1;
            }
        } else {
            while current < k {
                current += 1;
                ln_val += ln_lambda - (current as f64).ln();
            }
        }
        ln_val.exp()
    }

    /// Raw `ln_pmf` without underflow guard.
    fn ln_pmf_raw(&self, k: i64) -> f64 {
        let kf = k as f64;
        kf * self.lambda.ln() - self.lambda - ln_gamma(kf + 1.0)
    }
}

impl DiscreteDist for Poisson {
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

        if let Some(ln) = self.ln_pmf(k) {
            ln.exp()
        } else {
            self.pmf_from_mode(k)
        }
    }
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        if self.lambda < 0.0 || !self.lambda.is_finite() {
            return f64::NAN;
        }
        if self.lambda == 0.0 {
            return 1.0;
        }
        // P(X <= k) = Q(k+1, lambda) = 1 - P(k+1, lambda), the regularized
        // upper incomplete gamma (DLMF 8.7.3). O(1) instead of summing k pmfs.
        1.0 - crate::special::reg_lower_gamma(k as f64 + 1.0, self.lambda)
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
        f64::midpoint(self.a, self.b)
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
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        self.a + q * (self.b - self.a)
    }
}

/// `Normal(Î¼, Ïƒ)`. CDF via Abramowitzâ€“Stegun erf approximation (|err| < 1.5e-7).
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
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        let z = norm_ppf(q);
        self.mu + self.sigma * z
    }
}
impl Normal {
    /// Boxâ€“Muller sampling.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        let u1 = rng.uniform().max(1e-300);
        let u2 = rng.uniform();
        self.mu + self.sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

pub fn erf(x: f64) -> f64 {
    crate::special::erf(x)
}

/// Inverse normal CDF (Beasley-Springer-Moro algorithm, ~1e-9 accuracy).
fn norm_ppf(q: f64) -> f64 {
    // Rational approximation constants (Beasley-Springer-Moro)
    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];

    debug_assert!((0.0..=1.0).contains(&q));
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    let q_val;
    let r;

    if q < p_low {
        q_val = q;
        r = (-2.0 * q_val.ln()).sqrt();
        (((((C[0] * r + C[1]) * r + C[2]) * r + C[3]) * r + C[4]) * r + C[5])
            / ((((D[0] * r + D[1]) * r + D[2]) * r + D[3]) * r + 1.0)
    } else if q <= p_high {
        q_val = q - 0.5;
        r = q_val * q_val;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q_val
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        q_val = 1.0 - q;
        r = (-2.0 * q_val.ln()).sqrt();
        -(((((C[0] * r + C[1]) * r + C[2]) * r + C[3]) * r + C[4]) * r + C[5])
            / ((((D[0] * r + D[1]) * r + D[2]) * r + D[3]) * r + 1.0)
    }
}

/// Exponential distribution with rate Î».
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
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        if q <= 0.0 {
            return 0.0;
        }
        -((1.0 - q).ln()) / self.lambda
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
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        let z = norm_ppf(q);
        (self.mu + self.sigma * z).exp()
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
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        if q <= 0.0 {
            return 0.0;
        }
        self.scale * (-((1.0 - q).ln())).powf(1.0 / self.shape)
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

/// Student's t-distribution with Î½ degrees of freedom.
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
        let coeff = f64::midpoint(nu, 1.0).gamma()
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
            let coeff = f64::midpoint(d1, d2).gamma() / ((d1 / 2.0).gamma() * (d2 / 2.0).gamma())
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
            let ln_coeff = ln_gamma(self.r + k)
                - ln_gamma(k + 1.0)
                - ln_gamma(self.r);
            (ln_coeff + self.r * self.p.ln() + k * (1.0 - self.p).ln()).exp()
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

        if self.k > self.n || self.n_draws > self.n {
            return f64::NAN;
        }

        let min_x = self.n_draws.saturating_sub(self.n - self.k);
        let max_x = self.k.min(self.n_draws);
        if x < min_x || x > max_x {
            return 0.0;
        }

        let n_minus_k = i128::from(self.n - self.k);
        let ln_num = ln_gamma(self.k as f64 + 1.0)
            - ln_gamma(x as f64 + 1.0)
            - ln_gamma((self.k - x) as f64 + 1.0)
            + ln_gamma(n_minus_k as f64 + 1.0)
            - ln_gamma((self.n_draws - x) as f64 + 1.0)
            - ln_gamma((n_minus_k - i128::from(self.n_draws) + i128::from(x)) as f64 + 1.0);
        let ln_den = ln_gamma(self.n as f64 + 1.0)
            - ln_gamma(self.n_draws as f64 + 1.0)
            - ln_gamma((self.n - self.n_draws) as f64 + 1.0);
        (ln_num - ln_den).exp()
    }
    fn cdf(&self, k: i64) -> f64 {
        (0..=k).map(|i| self.pmf(i)).sum()
    }
    fn support_min(&self) -> i64 {
        self.n_draws.saturating_sub(self.n - self.k) as i64
    }
    fn support_max(&self) -> i64 {
        (self.k.min(self.n_draws)) as i64
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
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        self.x0 + self.gamma * core::f64::consts::PI * (q - 0.5).tan()
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
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        if q < 0.5 {
            self.mu + self.b * (2.0 * q).ln()
        } else {
            self.mu - self.b * (2.0 * (1.0 - q)).ln()
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
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        self.mu - self.beta * (-q.ln()).ln()
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
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        self.xm / (1.0 - q).powf(1.0 / self.alpha)
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
        // Clippy false positive: formula is mathematically correct for triangular distribution variance
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
    fn ppf(&self, q: f64) -> f64 {
        assert!((0.0..=1.0).contains(&q), "q must be in [0, 1]");
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let fc = (c - a) / (b - a);
        if q <= fc {
            a + ((b - a) * (c - a) * q).sqrt()
        } else {
            b - ((b - a) * (b - c) * (1.0 - q)).sqrt()
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

/// Inverse Gaussian (Wald) distribution on `x > 0`.
#[must_use]
pub struct InverseGaussian {
    pub mu: f64,     // mean
    pub lambda: f64, // shape
}
impl Distribution for InverseGaussian {
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        self.mu.powi(3) / self.lambda
    }
}
impl ContinuousDist for InverseGaussian {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let diff = x - self.mu;
        (self.lambda / (2.0 * core::f64::consts::PI * x.powi(3))).sqrt()
            * (-self.lambda * diff * diff / (2.0 * self.mu * self.mu * x)).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let mu = self.mu;
        let lambda = self.lambda;
        let z1 = (lambda / x).sqrt() * (x / mu - 1.0);
        let z2 = -(lambda / x).sqrt() * (x / mu + 1.0);
        let std_normal = Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        std_normal.cdf(z1) + (2.0 * lambda / mu).exp() * std_normal.cdf(z2)
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
    fn pmf_pdf_cdf_edge_cases() {
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
        assert!((n.cdf(1.0) - 0.841_344_746).abs() < 1e-6);
        assert!((n.cdf(-1.0) - 0.158_655_254).abs() < 1e-6);
        assert!((n.pdf(0.0) - 1.0 / (2.0 * core::f64::consts::PI).sqrt()).abs() < 1e-12);
        let t = StudentsT { nu: 1.0 };
        assert!((t.pdf(0.0) - 1.0 / core::f64::consts::PI).abs() < 1e-12);
        assert!((t.cdf(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn inverse_gaussian_has_correct_mean_and_cdf() {
        let ig = InverseGaussian { mu: 2.0, lambda: 5.0 };
        assert!((ig.mean() - 2.0).abs() < 1e-12);
        assert!((ig.variance() - 8.0 / 5.0).abs() < 1e-12);
        assert_eq!(ig.pdf(0.0), 0.0);
        assert!(ig.pdf(2.0) > 0.0);
        assert!((ig.cdf(0.0) - 0.0).abs() < 1e-12);
        assert!(ig.cdf(2.0) > 0.5, "right-skewed: cdf at mean exceeds 0.5");
        assert!(ig.pdf(2.0) == ig.pdf(2.0), "pdf finite");
        assert!((ig.cdf(1e9) - 1.0).abs() < 1e-3);
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

    #[test]
    fn regression_binomial_large_n_avoids_underflow() {
        let b = Binomial { n: 10_000, p: 0.5 };
        assert!(b.pmf(5000).is_finite());
        assert!(b.pmf(5000) > 0.0);
    }

    #[test]
    fn regression_poisson_large_lambda_avoids_underflow() {
        let p = Poisson { lambda: 10_000.0 };
        assert!(p.pmf(10000).is_finite());
        assert!(p.pmf(10000) > 0.0);
    }

    #[test]
    fn regression_ppf_roundtrip_for_normal() {
        let n = Normal { mu: 0.0, sigma: 1.0 };
        for &x in &[0.1, 0.5, 0.9, 0.99] {
            let p = n.cdf(x);
            let x_back = n.ppf(p);
            assert!((x_back - x).abs() < 1e-4, "roundtrip failed: {x} -> {p} -> {x_back}");
        }
    }

    #[test]
    fn binomial_p_zero_and_one() {
        // Edge cases: p = 0 and p = 1
        let b0 = Binomial { n: 10, p: 0.0 };
        assert_eq!(b0.pmf(0), 1.0);
        assert_eq!(b0.pmf(10), 0.0);
        assert_eq!(b0.cdf(0), 1.0);
        assert_eq!(b0.cdf(-1), 0.0);

        let b1 = Binomial { n: 10, p: 1.0 };
        assert_eq!(b1.pmf(0), 0.0);
        assert_eq!(b1.pmf(10), 1.0);
        assert_eq!(b1.cdf(0), 0.0);
        assert_eq!(b1.cdf(10), 1.0);
    }

    #[test]
    fn poisson_zero_lambda() {
        let p = Poisson { lambda: 0.0 };
        assert_eq!(p.pmf(0), 1.0);
        assert_eq!(p.pmf(1), 0.0);
        assert_eq!(p.cdf(0), 1.0);
        assert_eq!(p.cdf(-1), 0.0);
    }

    #[test]
    fn geometric_p_edge_cases() {
        // Edge cases: p -> 0 and p -> 1
        // p=0 is degenerate, pmf should handle it
        let p1 = Geometric { p: 1.0 };
        assert_eq!(p1.pmf(1), 1.0);
        assert_eq!(p1.pmf(2), 0.0);
    }

    #[test]
    fn negative_binomial_r_edge_cases() {
        // Test with r = 1 (reduces to geometric) - use p=0.5 where pmf(0) = p^r = 0.5 > 0
        let nb1 = NegativeBinomial { r: 1.0, p: 0.5 };
        // PMF at k=0 = p^r = 0.5^1 = 0.5 > 0
        assert!(nb1.pmf(0) > 0.0);
        assert!(nb1.pmf(1) > 0.0);

        // Test with large r
        let nb2 = NegativeBinomial { r: 100.0, p: 0.5 };
        assert!(nb2.pmf(100).is_finite());
    }

    #[test]
    fn binomial_closed_form_cdf_matches_sum() {
        for (n, p) in [(20, 0.3), (50, 0.7), (5, 0.5)] {
            let b = Binomial { n, p };
            let mut running = 0.0;
            for k in 0..=n {
                running += b.pmf(k as i64);
                assert!(
                    (b.cdf(k as i64) - running).abs() < 1e-9,
                    "n={n}, p={p}, k={k}: cdf={} sum={}",
                    b.cdf(k as i64),
                    running
                );
            }
        }
    }

    #[test]
    fn poisson_closed_form_cdf_matches_sum() {
        for lambda in [0.5, 2.0, 10.0, 100.0] {
            let p = Poisson { lambda };
            let mut running = 0.0;
            for k in 0..=50 {
                running += p.pmf(k);
                assert!(
                    (p.cdf(k) - running).abs() < 1e-9,
                    "lambda={lambda}, k={k}: cdf={} sum={}",
                    p.cdf(k),
                    running
                );
            }
        }
    }

    #[test]
    fn poisson_cdf_large_k_stays_finite() {
        let p = Poisson { lambda: 5.0 };
        assert!((p.cdf(1000) - 1.0).abs() < 1e-12);
        let p_big = Poisson { lambda: 500.0 };
        assert!(p_big.cdf(600).is_finite());
        assert!(p_big.cdf(600) > 0.999);
        assert!(p_big.cdf(400).is_finite());
        assert!(p_big.cdf(400) < 1e-4);
    }

    #[test]
    fn hypergeometric_pmf_sums_to_one() {
        let h = Hypergeometric {
            n: 52,
            k: 13,
            n_draws: 5,
        };
        let total: f64 = (h.support_min()..=h.support_max()).map(|x| h.pmf(x)).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hypergeometric_edge_cases() {
        // When n_draws = 0
        let h0 = Hypergeometric {
            n: 10,
            k: 5,
            n_draws: 0,
        };
        assert!((h0.pmf(0) - 1.0).abs() < 1e-10);
        assert!((h0.pmf(1) - 0.0).abs() < 1e-10);

        // When n_draws = n
        let h1 = Hypergeometric {
            n: 10,
            k: 5,
            n_draws: 10,
        };
        assert!((h1.pmf(5) - 1.0).abs() < 1e-10);
        assert!((h1.pmf(6) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn cauchy_pdf_cdf_ppf() {
        let c = Cauchy { x0: 0.0, gamma: 1.0 };
        // PDF at 0 should be finite
        let pdf = c.pdf(0.0);
        assert!(pdf.is_finite());
        // CDF at 0 should be 0.5
        let cdf = c.cdf(0.0);
        assert!((cdf - 0.5).abs() < 1e-10);
        // PPF at 0.5 should be 0
        let ppf = c.ppf(0.5);
        assert!((ppf - 0.0).abs() < 1e-10);
    }

    #[test]
    fn laplace_pdf_cdf_ppf() {
        let l = Laplace { mu: 0.0, b: 1.0 };
        let pdf = l.pdf(0.0);
        assert!(pdf > 0.0);
        let cdf = l.cdf(0.0);
        assert!((cdf - 0.5).abs() < 1e-10);
        let ppf = l.ppf(0.5);
        assert!((ppf - 0.0).abs() < 1e-10);
    }

    #[test]
    fn gumbel_pdf_cdf_ppf() {
        let g = Gumbel { mu: 0.0, beta: 1.0 };
        let pdf = g.pdf(0.0);
        assert!(pdf.is_finite());
        let cdf = g.cdf(0.0);
        assert!(cdf.is_finite() && cdf > 0.0 && cdf < 1.0);
        let ppf = g.ppf(0.5);
        assert!(ppf.is_finite());
    }

    #[test]
    fn pareto_pdf_cdf_ppf() {
        let p = Pareto { xm: 1.0, alpha: 2.0 };
        let pdf = p.pdf(1.0);
        assert!(pdf > 0.0);
        let cdf = p.cdf(1.0);
        assert!((cdf - 0.0).abs() < 1e-10); // CDF at xm should be 0
        let ppf = p.ppf(0.5);
        assert!(ppf > 1.0);
    }

    #[test]
    fn triangular_distribution() {
        let t = Triangular { a: 0.0, b: 1.0, c: 1.0 };
        let mean = t.mean();
        assert!((mean - 2.0 / 3.0).abs() < 1e-10);
        let variance = t.variance();
        assert!(variance > 0.0);
    }

    #[test]
    fn beta_distribution_edge_cases() {
        // alpha = 1, beta = 1 (uniform)
        let b11 = Beta { alpha: 1.0, beta: 1.0 };
        assert!((b11.mean() - 0.5).abs() < 1e-10);
        assert!((b11.variance() - 1.0 / 12.0).abs() < 1e-10);

        // alpha -> infinity
        let b_inf = Beta { alpha: 1e6, beta: 1.0 };
        assert!((b_inf.mean() - 1.0).abs() < 1e-6);

        // beta -> infinity
        let b_inf2 = Beta { alpha: 1.0, beta: 1e6 };
        assert!((b_inf2.mean() - 1.0 / (1.0 + 1e6)).abs() < 1e-6);
    }

    #[test]
    fn gamma_distribution_edge_cases() {
        // shape = 1 (exponential)
        let g1 = Gamma { shape: 1.0, rate: 1.0 };
        assert!((g1.mean() - 1.0).abs() < 1e-10);
        assert!((g1.variance() - 1.0).abs() < 1e-10);

        // shape -> 0 (degenerate)
        let g0 = Gamma { shape: 0.1, rate: 1.0 };
        assert!(g0.mean() > 0.0);
        assert!(g0.variance() > 0.0);
    }

    #[test]
    fn exponential_distribution_edge_cases() {
        let e = Exponential { lambda: 1.0 };
        assert!((e.mean() - 1.0).abs() < 1e-10);
        assert!((e.variance() - 1.0).abs() < 1e-10);

        // PPF at 0.5 should be ln(2)
        let ppf = e.ppf(0.5);
        assert!((ppf - core::f64::consts::LN_2).abs() < 1e-10);
    }

    #[test]
    fn weibull_distribution_edge_cases() {
        let w = Weibull { shape: 1.0, scale: 1.0 };
        assert!((w.mean() - 1.0).abs() < 1e-10);
        assert!((w.variance() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn chi_squared_distribution_edge_cases() {
        let c = ChiSquared { k: 1.0 };
        assert!((c.mean() - 1.0).abs() < 1e-10);
        assert!((c.variance() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn students_t_distribution_edge_cases() {
        // nu = 1 (Cauchy-like)
        let t1 = StudentsT { nu: 1.0 };
        assert!(t1.mean().is_nan());
        assert!(t1.variance().is_nan());

        // nu = 2: variance is undefined (infinity)
        let t2 = StudentsT { nu: 2.0 };
        assert!((t2.mean() - 0.0).abs() < 1e-10);
        assert!(t2.variance().is_nan());

        // pdf at 0 should be finite
        assert!(t1.pdf(0.0).is_finite());
        assert!(t2.pdf(0.0).is_finite());
    }

    #[test]
    fn f_distribution_edge_cases() {
        let f = FDistribution { d1: 5.0, d2: 10.0 };
        assert!(f.mean().is_finite());
        assert!(f.variance().is_finite());
    }

    #[test]
    fn markov_step_edge_cases() {
        let mut rng = Rng::new(42);
        // Valid transition matrix
        let t: &[&[f64]] = &[&[0.5, 0.5], &[0.3, 0.7]];
        let state = markov_step(t, 0, &mut rng);
        assert!(state.is_ok());
        let s = state.unwrap();
        assert!(s == 0 || s == 1);

        // Invalid: row doesn't sum to 1
        let t_invalid: &[&[f64]] = &[&[0.2, 0.2], &[0.5, 0.5]];
        let result = markov_step(t_invalid, 0, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn markov_distribution_convergence() {
        let p = markov_distribution(
            &[&[0.9, 0.1], &[0.5, 0.5]],
            &[1.0, 0.0],
            500,
        );
        // Should be close to stationary distribution (5/6, 1/6)
        assert!((p[0] - 5.0 / 6.0).abs() < 0.1);
        assert!((p[1] - 1.0 / 6.0).abs() < 0.1);
    }
}
