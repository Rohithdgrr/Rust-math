//! Stable (α-stable / Lévy) distributions in Nolan's S0 parameterization.
//!
//! PDF and CDF are computed by numerically inverting the characteristic
//! function (adaptive Simpson on the Gil-Pelaez integrals); sampling uses the
//! Chambers-Mallows-Stuck algorithm. The S0 parameterization is continuous in
//! all parameters, including α = 1.
//!
//! The inversion is most accurate for |x| within a few multiples of the scale
//! `gamma`; for far tails prefer sampling.

use crate::special::adaptive_simpson;

/// Stable distribution `S(α, β, γ, δ)` in Nolan's S0 parameterization.
///
/// - `alpha` ∈ (0, 2]: stability index (2 = Gaussian).
/// - `beta` ∈ [-1, 1]: skewness.
/// - `gamma` > 0: scale.
/// - `delta`: location.
#[must_use]
#[derive(Clone, Debug)]
pub struct Stable {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub delta: f64,
}

impl Stable {
    /// Characteristic function `E[exp(i·t·X)]` at real `t`, in S0 form:
    ///
    /// - α ≠ 1: `exp(-γ^α|t|^α (1 - i·β·sign(t)·tan(πα/2)) + i·δ·t)`
    /// - α = 1: `exp(-γ|t| (1 + i·β·sign(t)·(2/π)·ln|t|) + i·δ·t)`
    #[must_use]
    pub fn char_fn(&self, t: f64) -> (f64, f64) {
        let sign = t.signum();
        let at = t.abs();
        let (re_part, im_part) = if (self.alpha - 1.0).abs() > 1e-12 {
            let tail = self.gamma.powf(self.alpha) * at.powf(self.alpha);
            (
                -tail,
                self.delta * t - self.beta * tail * (core::f64::consts::PI * self.alpha / 2.0).tan() * sign,
            )
        } else {
            let tail = self.gamma * at;
            (
                -tail,
                self.delta * t - self.beta * tail * (2.0 / core::f64::consts::PI) * at.ln() * sign,
            )
        };
        let scale = re_part.exp();
        (scale * im_part.cos(), scale * im_part.sin())
    }

    /// Probability density at `x` by characteristic-function inversion:
    /// `f(x) = (1/π) ∫₀^∞ Re[e^{-itx}φ(t)] dt`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        let t_max = inversion_tail(self.gamma, self.alpha);
        let f = |t: f64| {
            let (re, im) = self.char_fn(t);
            // Re[e^{-itx} φ(t)] = re·cos(tx) + im·sin(tx)
            re * (t * x).cos() + im * (t * x).sin()
        };
        adaptive_simpson(&f, 0.0, t_max, 1e-9, 200_000) / core::f64::consts::PI
    }

    /// Cumulative distribution function at `x` by Gil-Pelaez inversion:
    /// `F(x) = 1/2 + (1/π) ∫₀^∞ Im[e^{itx}φ(t)]/t dt`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        let t_max = inversion_tail(self.gamma, self.alpha);
        let f = |t: f64| {
            let (re, im) = self.char_fn(t);
            // Im[e^{itx} φ(t)] / t
            (re * (t * x).sin() + im * (t * x).cos()) / t
        };
        0.5 + adaptive_simpson(&f, 1e-12, t_max, 1e-9, 200_000) / core::f64::consts::PI
    }

    /// Draw a sample (Chambers-Mallows-Stuck, S0 parameterization).
    ///
    /// Returns `γ·Z + δ` where `Z ~ S(α, β, 1, 0)`.
    #[must_use]
    pub fn sample(&self, rng: &mut crate::rng::Rng) -> f64 {
        let u = rng.uniform();
        let v = core::f64::consts::PI * (u - 0.5); // U(-π/2, π/2)
        let w = -rng.uniform().ln(); // Exp(1)
        let z = if (self.alpha - 1.0).abs() > 1e-12 {
            let theta = (-self.beta * (core::f64::consts::PI * self.alpha / 2.0).tan()).atan()
                / self.alpha;
            let inv_alpha = 1.0 / self.alpha;
            let term = (v - self.alpha * (v - theta)).cos() / w;
            (self.alpha * (v - theta)).sin() / v.cos().powf(inv_alpha) * term.powf(1.0 - inv_alpha)
        } else {
            let half_pi = core::f64::consts::FRAC_PI_2;
            (2.0 / core::f64::consts::PI)
                * ((half_pi + self.beta * v) * v.tan()
                    - self.beta * (w * v.cos() / (half_pi + self.beta * v)).ln())
        };
        self.gamma * z + self.delta
    }
}

/// Upper integration limit such that `exp(-γ^α T^α) / T < 1e-9`.
fn inversion_tail(gamma: f64, alpha: f64) -> f64 {
    if alpha <= 0.0 {
        return 100.0;
    }
    let inv = 1.0 / gamma.powf(alpha);
    let mut t = (40.0 * inv).powf(1.0 / alpha);
    for _ in 0..3 {
        t = ((t * 1e9).ln() * inv).powf(1.0 / alpha);
    }
    t.max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn alpha_two_is_gaussian() {
        // S(2, 0, 1, 0) = N(0, 2): pdf(0) = 1/(2·sqrt(π)), cdf(1) = Φ(1/√2).
        let s = Stable {
            alpha: 2.0,
            beta: 0.0,
            gamma: 1.0,
            delta: 0.0,
        };
        let pdf0 = s.pdf(0.0);
        assert!(
            (pdf0 - 1.0 / (2.0 * core::f64::consts::PI.sqrt())).abs() < 1e-6,
            "pdf0 {pdf0}"
        );
        let cdf1 = s.cdf(1.0);
        let expected = 0.5 + 0.5 * crate::special::erf(1.0 / 2.0f64.sqrt());
        assert!((cdf1 - expected).abs() < 1e-6, "cdf1 {cdf1} vs {expected}");
    }

    #[test]
    fn alpha_one_beta_zero_is_cauchy() {
        // S(1, 0, 1, 0) = Cauchy(0, 1): pdf(0) = 1/π, cdf(1) = 3/4.
        let s = Stable {
            alpha: 1.0,
            beta: 0.0,
            gamma: 1.0,
            delta: 0.0,
        };
        let pdf0 = s.pdf(0.0);
        assert!((pdf0 - 1.0 / core::f64::consts::PI).abs() < 1e-6, "pdf0 {pdf0}");
        let cdf1 = s.cdf(1.0);
        assert!((cdf1 - 0.75).abs() < 1e-6, "cdf1 {cdf1}");
    }

    #[test]
    fn pdf_and_cdf_agree() {
        let s = Stable {
            alpha: 1.5,
            beta: 0.2,
            gamma: 1.2,
            delta: 0.3,
        };
        let h = 1e-4;
        let diff = (s.cdf(0.6 + h) - s.cdf(0.6 - h)) / (2.0 * h);
        let pdf = s.pdf(0.6);
        assert!(
            (diff - pdf).abs() < 1e-3 * pdf.abs().max(1.0),
            "cdf slope {diff} vs pdf {pdf}"
        );
    }

    #[test]
    fn cdf_monotone_and_median_at_delta_when_symmetric() {
        let s = Stable {
            alpha: 1.5,
            beta: 0.0,
            gamma: 1.0,
            delta: 2.0,
        };
        let m = s.cdf(2.0);
        assert!((m - 0.5).abs() < 1e-6, "median cdf {m}");
        assert!(s.cdf(1.0) < m && m < s.cdf(3.0));
    }

    #[test]
    fn gaussian_sampling_has_finite_moments() {
        let s = Stable {
            alpha: 2.0,
            beta: 0.0,
            gamma: 1.0,
            delta: 0.0,
        };
        let mut rng = Rng::new(13);
        let n = 200_000;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        for i in 1..=n {
            let x = s.sample(&mut rng);
            let delta = x - mean;
            mean += delta / i as f64;
            m2 += delta * (x - mean);
        }
        assert!(mean.abs() < 0.02, "mean {mean}");
        let var = m2 / (n as f64 - 1.0);
        assert!((var - 2.0).abs() < 0.05, "var {var}");
    }

    #[test]
    fn cauchy_sampling_median_is_near_zero() {
        let s = Stable {
            alpha: 1.0,
            beta: 0.0,
            gamma: 1.0,
            delta: 0.0,
        };
        let mut rng = Rng::new(21);
        let mut xs: Vec<f64> = (0..50_000).map(|_| s.sample(&mut rng)).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = xs[xs.len() / 2];
        assert!(median.abs() < 0.05, "median {median}");
    }

    #[test]
    fn shifted_and_scaled_matches_location() {
        let s = Stable {
            alpha: 1.2,
            beta: -0.5,
            gamma: 0.7,
            delta: -1.5,
        };
        let mut rng = Rng::new(8);
        let n = 100_000;
        let mut xs = Vec::with_capacity(n);
        for _ in 0..n {
            xs.push(s.sample(&mut rng));
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = xs[n / 2];
        assert!((median + 1.5).abs() < 0.1, "median {median}");
    }
}
