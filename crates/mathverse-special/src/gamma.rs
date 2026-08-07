//! Gamma function family: [`gamma`], [`log_gamma`], [`digamma`], [`beta`]
//! and the regularized incomplete gammas [`gamma_p`] / [`gamma_q`].

use core::f64::consts::{E, LN_2, PI};

/// Euler–Mascheroni constant γ.
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Lanczos approximation coefficients (g = 7, ~1e-15 accuracy).
const LANCZOS_COEF: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

const G: f64 = 7.0;

/// Natural logarithm of the gamma function, `ln Γ(z)`.
///
/// Lanczos approximation for `z ≥ 0.5`; for `0 < z < 0.5` the reflection
/// formula `ln Γ(z) = ln π − ln|sin(πz)| − ln Γ(1 − z)` is used. Returns
/// `+∞` as `z → 0⁺` and `NaN` for `z ≤ 0` (poles of Γ).
///
/// ```
/// use mathverse_special::log_gamma;
/// assert!((log_gamma(5.0) - 24.0f64.ln()).abs() < 1e-12);
/// assert!((log_gamma(0.5) - 0.572_364_942_924_7).abs() < 1e-12);
/// assert!(log_gamma(0.0).is_infinite());
/// assert!(log_gamma(-1.0).is_nan());
/// ```
pub fn log_gamma(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z <= 0.0 {
        if z == 0.0 {
            return f64::INFINITY;
        }
        // Poles at negative integers; elsewhere NaN for now (reflection of
        // the log is branchy — real Γ(z) has sign changes we do not model).
        return f64::NAN;
    }
    if z < 0.5 {
        return PI.ln() - (PI * z).sin().abs().ln() - log_gamma(1.0 - z);
    }
    let zm = z - 1.0;
    let mut x = LANCZOS_COEF[0];
    for i in 1..LANCZOS_COEF.len() {
        x += LANCZOS_COEF[i] / (zm + i as f64);
    }
    let t = zm + G + 0.5;
    0.5 * (2.0 * PI).ln() + (zm + 0.5) * t.ln() - t + x.ln()
}

/// Gamma function Γ(z) for real `z`.
///
/// For `z > 0.5` computed as `exp(log_gamma(z))`; for `0 < z ≤ 0.5` via the
/// reflection formula `Γ(z) = π / (sin(πz) Γ(1 − z))`. Returns `NaN` at the
/// poles `z ∈ {0, −1, −2, …}`.
///
/// ```
/// use mathverse_special::gamma;
/// assert!((gamma(5.0) - 24.0).abs() < 1e-12);
/// assert!((gamma(0.5) - core::f64::consts::PI.sqrt()).abs() < 1e-12);
/// assert!((gamma(1.0) - 1.0).abs() < 1e-12);
/// assert!(gamma(0.0).is_nan());
/// ```
pub fn gamma(z: f64) -> f64 {
    if z.is_nan() || z.is_infinite() {
        return f64::NAN;
    }
    if z >= 0.5 {
        return log_gamma(z).exp();
    }
    if z == 0.0 {
        return f64::NAN;
    }
    let pole = z.trunc() == z;
    if z < 0.0 && pole {
        return f64::NAN;
    }
    // Reflection formula, valid for non-integer z (incl. z in (0, 0.5]).
    PI / ((PI * z).sin() * gamma(1.0 - z))
}

/// Digamma function ψ(z) = Γ′(z)/Γ(z).
///
/// Recurrence `ψ(z) = ψ(z + 1) − 1/z` shifts the argument above 6, then the
/// asymptotic series `ψ(z) ≈ ln z − 1/(2z) − 1/(12z²) + 1/(120z⁴) − 1/(252z⁶)`
/// applies. Returns `NaN` at the poles `z ∈ {0, −1, −2, …}`.
///
/// ```
/// use mathverse_special::digamma;
/// // Asymptotic series truncated at O(z⁻⁷) is accurate to ~1e-8.
/// assert!((digamma(1.0) - (-0.577_215_664_901_532_9)).abs() < 1e-6);
/// assert!((digamma(0.5) - (-1.963_510_026_021_4)).abs() < 1e-6);
/// ```
pub fn digamma(z: f64) -> f64 {
    if z.is_nan() || z.is_infinite() {
        return f64::NAN;
    }
    if z <= 0.0 {
        if z == z.trunc() {
            return f64::NAN; // pole
        }
        // Reflection: ψ(z) = ψ(1 − z) − π cot(πz)
        return digamma(1.0 - z) - PI / (PI * z).tan();
    }
    let mut x = z;
    let mut sum = 0.0;
    while x < 6.0 {
        sum -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    sum + x.ln() - 0.5 * inv - inv2 / 12.0 + inv2 * inv2 * (1.0 / 120.0 - inv2 / 252.0)
}

/// Beta function B(a, b) = Γ(a)·Γ(b)/Γ(a + b), via log-gamma for stability.
/// Positive arguments only; `NaN` otherwise.
///
/// ```
/// use mathverse_special::beta;
/// assert!((beta(1.0, 1.0) - 1.0).abs() < 1e-12);
/// // B(2,3) = 1!·2!/4! = 1/12
/// assert!((beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-12);
/// assert!((beta(0.5, 0.5) - core::f64::consts::PI).abs() < 1e-12);
/// ```
pub fn beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    (log_gamma(a) + log_gamma(b) - log_gamma(a + b)).exp()
}

/// Lower regularized incomplete gamma P(a, x) = γ(a, x)/Γ(a).
///
/// Series expansion when `x < a + 1`, continued fraction otherwise. Requires
/// `a > 0`, `x ≥ 0`.
///
/// ```
/// use mathverse_special::gamma_p;
/// // P(1, x) = 1 − e^(−x)
/// assert!((gamma_p(1.0, 5.0) - (1.0 - (-5.0_f64).exp())).abs() < 1e-12);
/// // P(a, ∞) → 1
/// assert!((gamma_p(2.0, 200.0) - 1.0).abs() < 1e-12);
/// ```
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 || a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// Upper regularized incomplete gamma Q(a, x) = Γ(a, x)/Γ(a) = 1 − P(a, x).
///
/// ```
/// use mathverse_special::gamma_q;
/// assert!((gamma_q(1.0, 5.0) - (-5.0_f64).exp()).abs() < 1e-12);
/// // Q(2, 200) = Γ(2,200)/Γ(2) = 201·e⁻²⁰⁰ ≈ 2.8e-85
/// assert!((gamma_q(2.0, 200.0) - 201.0 * (-200.0_f64).exp()).abs() < 1e-88);
/// ```
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 || a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cf(a, x)
    }
}

/// γ(a, x)/Γ(a) by power series, accurate for `x < a + 1`.
fn gamma_series(a: f64, x: f64) -> f64 {
    let ln_factor = a * x.ln() - x - log_gamma(a);
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut ap = a;
    loop {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * 1e-15 || ap > a + 5000.0 {
            break;
        }
    }
    sum * ln_factor.exp()
}

/// Γ(a, x)/Γ(a) by modified Lentz continued fraction, accurate for `x ≥ a + 1`.
fn gamma_cf(a: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    let ln_factor = a * x.ln() - x - log_gamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=1000 {
        let an = -i as f64 * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h * ln_factor.exp()
}

/// The natural constant e, re-exported for convenience.
pub const E_CONST: f64 = E;
#[allow(unused)]
const _LN2: f64 = LN_2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_values() {
        assert!((gamma(1.0) - 1.0).abs() < 1e-12);
        assert!((gamma(2.0) - 1.0).abs() < 1e-12);
        assert!((gamma(5.0) - 24.0).abs() < 1e-12);
        assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-12);
        assert!((gamma(0.5) * gamma(1.5) - PI / 2.0).abs() < 1e-12);
        assert!(gamma(0.0).is_nan());
        assert!(gamma(-1.0).is_nan());
        assert!(gamma(-2.5).abs() > 0.0);
    }

    #[test]
    fn log_gamma_values() {
        assert!((log_gamma(1.0) - 0.0).abs() < 1e-14);
        assert!((log_gamma(5.0) - 24.0f64.ln()).abs() < 1e-12);
        assert!((log_gamma(0.5) - 0.572_364_942_924_7).abs() < 1e-12);
    }

    #[test]
    fn digamma_values() {
        assert!((digamma(1.0) + EULER_GAMMA).abs() < 1e-6);
        assert!((digamma(0.5) - (-1.963_510_026_021_4)).abs() < 1e-6);
        // Reflection: ψ(−1/2) = ψ(3/2) − π·cot(−π/2) = ψ(0.5) + 2
        assert!((digamma(-0.5) - 0.036_489_973_978_576_52).abs() < 1e-8);
        assert!(digamma(0.0).is_nan());
        assert!(digamma(-1.0).is_nan());
    }

    #[test]
    fn beta_values() {
        assert!((beta(1.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-12);
        assert!((beta(0.5, 0.5) - PI).abs() < 1e-12);
    }

    #[test]
    fn incomplete_gamma_values() {
        assert!((gamma_p(1.0, 5.0) - (1.0 - (-5.0f64).exp())).abs() < 1e-12);
        assert!((gamma_q(1.0, 5.0) - (-5.0f64).exp()).abs() < 1e-12);
        assert!((gamma_p(0.5, 0.5) - 0.682_689_492_137).abs() < 1e-10);
        assert!((gamma_p(5.0, 3.0) - 0.184_736_755).abs() < 1e-8);
        // P + Q = 1 for a wide range of arguments
        for (a, x) in [(0.5, 0.1), (2.0, 5.0), (5.0, 2.0), (10.0, 8.0), (3.0, 100.0)] {
            let p = gamma_p(a, x);
            let q = gamma_q(a, x);
            assert!((p + q - 1.0).abs() < 1e-9, "P+Q off at a={a}, x={x}");
        }
    }
}
