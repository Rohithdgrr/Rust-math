//! Extended bivariate copulas: Clayton, Gumbel, Frank and Student-t, with
//! CDF, density, Marshall-Olkin / conditional sampling, and Kendall-tau
//! parameter recovery.

use crate::distributions::{ChiSquared, ContinuousDist, Normal};
use crate::rng::Rng;
use crate::special::{adaptive_simpson, ln_gamma};

/// Debye function of order 1: `D₁(x) = (1/x)∫₀^x t/(e^t - 1) dt`.
fn debye_1(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let f = |t: f64| {
        if t < 1e-9 {
            1.0
        } else {
            t / (t.exp() - 1.0)
        }
    };
    adaptive_simpson(&f, 0.0, x, 1e-11, 50_000) / x
}

/// Quantile of Student's t distribution with `nu` degrees of freedom,
/// by bisection on the CDF.
#[must_use]
pub fn students_t_quantile(nu: f64, p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    let t = crate::distributions::StudentsT { nu };
    let mut lo = -40.0;
    let mut hi = 40.0;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if t.cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Clayton copula `C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}`, θ > 0.
#[must_use]
#[derive(Clone, Debug)]
pub struct ClaytonCopula {
    pub theta: f64,
}

impl ClaytonCopula {
    /// CDF at `(u, v)`.
    #[must_use]
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        if self.theta <= 0.0 {
            return f64::NAN;
        }
        edge(u, v, |a, b| {
            (a.powf(-self.theta) + b.powf(-self.theta) - 1.0).powf(-1.0 / self.theta)
        })
    }

    /// Density `∂²C/∂u∂v`.
    #[must_use]
    pub fn density(&self, u: f64, v: f64) -> f64 {
        if !in_unit_square(u, v) || self.theta <= 0.0 {
            return 0.0;
        }
        let t = self.theta;
        (1.0 + t) * (u * v).powf(-1.0 - t)
            * (u.powf(-t) + v.powf(-t) - 1.0).powf(-2.0 - 1.0 / t)
    }

    /// Draw one pair by the Marshall-Olkin method (`V ~ Gamma(1/θ, 1)`).
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> (f64, f64) {
        let v = crate::distributions::Gamma {
            shape: 1.0 / self.theta,
            rate: 1.0,
        }
        .sample(rng);
        let transform = |u: f64| (1.0 - u.ln() / v).powf(-1.0 / self.theta);
        (transform(rng.uniform()), transform(rng.uniform()))
    }

    /// Kendall's tau `θ/(θ + 2)`.
    #[must_use]
    pub fn kendall_tau(&self) -> f64 {
        self.theta / (self.theta + 2.0)
    }

    /// Recover `θ` from an empirical Kendall's tau: `θ = 2τ/(1 - τ)`.
    #[must_use]
    pub fn theta_from_tau(tau: f64) -> f64 {
        if !(0.0..1.0).contains(&tau) {
            return f64::NAN;
        }
        2.0 * tau / (1.0 - tau)
    }
}

/// Gumbel copula `C(u,v) = exp(-[(-ln u)^θ + (-ln v)^θ]^{1/θ})`, θ ≥ 1.
#[must_use]
#[derive(Clone, Debug)]
pub struct GumbelCopula {
    pub theta: f64,
}

impl GumbelCopula {
    /// CDF at `(u, v)`.
    #[must_use]
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        if self.theta < 1.0 {
            return f64::NAN;
        }
        edge(u, v, |a, b| {
            let (la, lb) = (-a.ln(), -b.ln());
            (-(la.powf(self.theta) + lb.powf(self.theta)).powf(1.0 / self.theta)).exp()
        })
    }

    /// Density `∂²C/∂u∂v`.
    #[must_use]
    pub fn density(&self, u: f64, v: f64) -> f64 {
        if !in_unit_square(u, v) || self.theta < 1.0 {
            return 0.0;
        }
        let t = self.theta;
        let (la, lb) = (-u.ln(), -v.ln());
        let s = la.powf(t) + lb.powf(t);
        let inner = s.powf(1.0 / t);
        let c = (-inner).exp();
        c * (u * v).powf(-1.0) * (la * lb).powf(t - 1.0) * s.powf(1.0 / t - 2.0)
            * (inner + t - 1.0)
    }

    /// Draw one pair by the Marshall-Olkin method with a positive stable
    /// variate `V ~ S(1/θ, 1, cos(π/(2θ))^θ, 0)`. For `θ = 1` the copula is
    /// independence and the pair is returned directly.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> (f64, f64) {
        if (self.theta - 1.0).abs() < 1e-12 {
            return (rng.uniform(), rng.uniform());
        }
        let v = crate::stable::Stable {
            alpha: 1.0 / self.theta,
            beta: 1.0,
            gamma: (core::f64::consts::PI / (2.0 * self.theta)).cos().powf(self.theta),
            delta: 0.0,
        }
        .sample(rng);
        let transform = |u: f64| (-(-u.ln() / v).powf(1.0 / self.theta)).exp();
        (transform(rng.uniform()), transform(rng.uniform()))
    }

    /// Kendall's tau `1 - 1/θ`.
    #[must_use]
    pub fn kendall_tau(&self) -> f64 {
        1.0 - 1.0 / self.theta
    }

    /// Recover `θ` from Kendall's tau: `θ = 1/(1 - τ)`.
    #[must_use]
    pub fn theta_from_tau(tau: f64) -> f64 {
        if !(0.0..1.0).contains(&tau) {
            return f64::NAN;
        }
        1.0 / (1.0 - tau)
    }
}

/// Frank copula
/// `C(u,v) = -(1/θ)ln(1 + (e^{-θu}-1)(e^{-θv}-1)/(e^{-θ}-1))`, θ ≠ 0.
#[must_use]
#[derive(Clone, Debug)]
pub struct FrankCopula {
    pub theta: f64,
}

impl FrankCopula {
    /// CDF at `(u, v)`.
    #[must_use]
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        if self.theta == 0.0 {
            return f64::NAN;
        }
        if self.theta.abs() < 1e-6 {
            return u * v;
        }
        edge(u, v, |a, b| {
            let d = (-self.theta).exp() - 1.0;
            let ab = ((-self.theta * a).exp() - 1.0) * ((-self.theta * b).exp() - 1.0);
            -(1.0 / self.theta) * (1.0 + ab / d).ln()
        })
    }

    /// Density `∂²C/∂u∂v`.
    #[must_use]
    pub fn density(&self, u: f64, v: f64) -> f64 {
        if !in_unit_square(u, v) || self.theta == 0.0 {
            return 0.0;
        }
        let d = (-self.theta).exp() - 1.0;
        let a = (-self.theta * u).exp() - 1.0;
        let b = (-self.theta * v).exp() - 1.0;
        -self.theta * d * (-self.theta * (u + v)).exp() / (d + a * b).powi(2)
    }

    /// Draw one pair by inverting the conditional CDF.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> (f64, f64) {
        let u = rng.uniform();
        let w = rng.uniform();
        let d = (-self.theta).exp() - 1.0;
        let a = (-self.theta * u).exp() - 1.0;
        let num = w * d;
        let den = (-self.theta * u).exp() - w * a;
        let v = -(1.0 / self.theta) * (1.0 + num / den).ln();
        (u, v)
    }

    /// Kendall's tau `1 + 4(D₁(θ) - 1)/θ`.
    #[must_use]
    pub fn kendall_tau(&self) -> f64 {
        1.0 + 4.0 * (debye_1(self.theta) - 1.0) / self.theta
    }

    /// Recover `θ > 0` from a positive Kendall's tau by bisection.
    #[must_use]
    pub fn theta_from_tau(tau: f64) -> f64 {
        if !(0.0..1.0).contains(&tau) {
            return f64::NAN;
        }
        if tau.abs() < 1e-12 {
            return 0.0;
        }
        let target = tau;
        let mut lo = 1e-4;
        let mut hi = 100.0;
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let t = 1.0 + 4.0 * (debye_1(mid) - 1.0) / mid;
            if t < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

/// Student-t copula with `nu > 0` degrees of freedom and linear correlation
/// `rho ∈ (-1, 1)`.
#[must_use]
#[derive(Clone, Debug)]
pub struct StudentTCopula {
    pub nu: f64,
    pub rho: f64,
}

impl StudentTCopula {
    /// Density
    /// `c(u,v) = Γ((ν+2)/2)Γ(ν/2) / (Γ((ν+1)/2)²√(1-ρ²)) ·
    ///  (1+Q/ν)^{-(ν+2)/2} / ((1+t₁²/ν)(1+t₂²/ν))^{-(ν+1)/2}`
    /// where `t_i = T_ν^{-1}(u_i)` and `Q = (t₁² - 2ρt₁t₂ + t₂²)/(1-ρ²)`.
    #[must_use]
    pub fn density(&self, u: f64, v: f64) -> f64 {
        if !in_unit_square(u, v) || self.nu <= 0.0 || self.rho.abs() >= 1.0 {
            return 0.0;
        }
        let t1 = students_t_quantile(self.nu, u);
        let t2 = students_t_quantile(self.nu, v);
        let q = (t1 * t1 - 2.0 * self.rho * t1 * t2 + t2 * t2) / (1.0 - self.rho * self.rho);
        let num = (ln_gamma((self.nu + 2.0) / 2.0) + ln_gamma(self.nu / 2.0)
            - 2.0 * ln_gamma((self.nu + 1.0) / 2.0)
            - 0.5 * (1.0 - self.rho * self.rho).ln())
        .exp();
        let joint = (1.0 + q / self.nu).powf(-(self.nu + 2.0) / 2.0);
        let margin1 = (1.0 + t1 * t1 / self.nu).powf(-(self.nu + 1.0) / 2.0);
        let margin2 = (1.0 + t2 * t2 / self.nu).powf(-(self.nu + 1.0) / 2.0);
        num * joint / (margin1 * margin2)
    }

    /// Draw one pair: bivariate-t via `(Z₁, Z₂) ~ N(0, Σ)`,
    /// `T_i = Z_i / √(χ²_ν/ν)`, then probability integral transform.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> (f64, f64) {
        let z1 = Normal { mu: 0.0, sigma: 1.0 }.sample(rng);
        let z2 = self.rho * z1 + (1.0 - self.rho * self.rho).sqrt() * Normal {
            mu: 0.0,
            sigma: 1.0,
        }
        .sample(rng);
        let chi = ChiSquared { k: self.nu }.sample(rng);
        let scale = (self.nu / chi).sqrt();
        let t1 = z1 * scale;
        let t2 = z2 * scale;
        let tdist = crate::distributions::StudentsT { nu: self.nu };
        (tdist.cdf(t1), tdist.cdf(t2))
    }

    /// Kendall's tau `(2/π)·arcsin(ρ)`.
    #[must_use]
    pub fn kendall_tau(&self) -> f64 {
        2.0 * self.rho.asin() / core::f64::consts::PI
    }
}

fn in_unit_square(u: f64, v: f64) -> bool {
    u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0
}

/// Apply `f` on `(0,1)²` with the boundary values forced to the copula
/// axioms (`C(0, v) = C(u, 0) = 0`, `C(1, v) = v`, `C(u, 1) = u`).
fn edge(u: f64, v: f64, f: impl Fn(f64, f64) -> f64) -> f64 {
    if u <= 0.0 || v <= 0.0 {
        return 0.0;
    }
    if u >= 1.0 {
        return v;
    }
    if v >= 1.0 {
        return u;
    }
    f(u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multivariate::CorrelationMatrix;

    fn empirical_tau(pairs: &[(f64, f64)]) -> f64 {
        let xs: Vec<f64> = pairs.iter().map(|&(u, _)| u).collect();
        let ys: Vec<f64> = pairs.iter().map(|&(_, v)| v).collect();
        CorrelationMatrix::kendall_tau(&xs, &ys).unwrap()
    }

    #[test]
    fn clayton_cdf_density_hand_calc() {
        let c = ClaytonCopula { theta: 2.0 };
        let cdf = c.cdf(0.5, 0.5);
        let expected_cdf = 7.0f64.powf(-0.5);
        assert!((cdf - expected_cdf).abs() < 1e-12, "cdf {cdf}");
        let dens = c.density(0.5, 0.5);
        let expected_dens = 3.0 * 0.25f64.powi(-3) * 7.0f64.powf(-2.5);
        assert!((dens - expected_dens).abs() < 1e-9, "dens {dens} vs {expected_dens}");
        assert!((c.kendall_tau() - 0.5).abs() < 1e-12);
        assert!((ClaytonCopula::theta_from_tau(0.5) - 2.0).abs() < 1e-12);
        assert!((c.cdf(0.0, 0.5) - 0.0).abs() < 1e-15);
        assert!((c.cdf(1.0, 0.3) - 0.3).abs() < 1e-15);
    }

    #[test]
    fn gumbel_cdf_and_tau() {
        let g = GumbelCopula { theta: 2.0 };
        let cdf = g.cdf(0.5, 0.5);
        let expected = 2.0f64.powf(-2.0f64.sqrt());
        assert!((cdf - expected).abs() < 1e-12, "cdf {cdf}");
        assert!((g.kendall_tau() - 0.5).abs() < 1e-12);
        assert!((GumbelCopula::theta_from_tau(2.0 / 3.0) - 3.0).abs() < 1e-12);
        let ind = GumbelCopula { theta: 1.0 };
        assert!((ind.cdf(0.4, 0.7) - 0.28).abs() < 1e-12);
        assert!((ind.density(0.4, 0.7) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn frank_theta_recovery_round_trip() {
        for theta in [1.0, 3.0, 5.0, 10.0] {
            let f = FrankCopula { theta };
            let tau = f.kendall_tau();
            assert!(tau > 0.0 && tau < 1.0, "tau {tau} at theta {theta}");
            let back = FrankCopula::theta_from_tau(tau);
            assert!(
                (back - theta).abs() < 1e-3 * theta.max(1.0),
                "theta {theta} -> tau {tau} -> {back}"
            );
        }
        let small = FrankCopula { theta: 1e-4 };
        // At theta = 1e-4 the copula is close to (but not exactly)
        // independence: C(0.3, 0.6) ~ 0.180003, density ~ 0.99996.
        assert!((small.cdf(0.3, 0.6) - 0.18).abs() < 1e-4);
        assert!((small.density(0.3, 0.6) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn student_t_copula_properties() {
        // The t-copula with rho = 0 is NOT independence: the shared
        // chi-square induces tail dependence, so the density is > 1.
        let c = StudentTCopula { nu: 4.0, rho: 0.0 };
        let d = c.density(0.3, 0.7);
        assert!(d > 1.0, "density at rho=0 {d} should exceed 1");
        assert!((c.density(0.7, 0.3) - d).abs() < 1e-12);
        let c2 = StudentTCopula { nu: 4.0, rho: 0.5 };
        let d2 = c2.density(0.3, 0.7);
        assert!(d2 > 1.0, "positive dependence density {d2}");
        assert!((c2.density(0.7, 0.3) - d2).abs() < 1e-12);
        let tau = c2.kendall_tau();
        assert!((tau - 2.0 * 0.5f64.asin() / core::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn clayton_sampling_reproduces_kendall_tau() {
        let c = ClaytonCopula { theta: 2.0 };
        let mut rng = Rng::new(17);
        let pairs: Vec<(f64, f64)> = (0..30_000).map(|_| c.sample(&mut rng)).collect();
        let tau = empirical_tau(&pairs);
        assert!((tau - 0.5).abs() < 0.02, "tau {tau}");
    }

    #[test]
    fn gumbel_sampling_reproduces_kendall_tau() {
        let g = GumbelCopula { theta: 2.0 };
        let mut rng = Rng::new(19);
        let pairs: Vec<(f64, f64)> = (0..30_000).map(|_| g.sample(&mut rng)).collect();
        let tau = empirical_tau(&pairs);
        assert!((tau - 0.5).abs() < 0.03, "tau {tau}");
        let ind = GumbelCopula { theta: 1.0 };
        let pairs2: Vec<(f64, f64)> = (0..20_000).map(|_| ind.sample(&mut rng)).collect();
        let tau2 = empirical_tau(&pairs2);
        assert!(tau2.abs() < 0.03, "independent tau {tau2}");
    }

    #[test]
    fn frank_sampling_reproduces_kendall_tau() {
        let f = FrankCopula { theta: 5.0 };
        let mut rng = Rng::new(23);
        let pairs: Vec<(f64, f64)> = (0..30_000).map(|_| f.sample(&mut rng)).collect();
        let tau = empirical_tau(&pairs);
        let expected = f.kendall_tau();
        assert!((tau - expected).abs() < 0.03, "tau {tau} vs {expected}");
    }

    #[test]
    fn student_sampling_reproduces_kendall_tau() {
        let c = StudentTCopula { nu: 5.0, rho: 0.7 };
        let mut rng = Rng::new(29);
        let pairs: Vec<(f64, f64)> = (0..30_000).map(|_| c.sample(&mut rng)).collect();
        let tau = empirical_tau(&pairs);
        let expected = c.kendall_tau();
        assert!((tau - expected).abs() < 0.03, "tau {tau} vs {expected}");
        assert!(pairs.iter().all(|&(u, v)| u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0));
    }
}
