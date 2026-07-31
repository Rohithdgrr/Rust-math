//! Special functions: gamma, beta, error function, and related functions.

use mathverse_core::error::{MathError, MathResult};

/// Gamma function Γ(z).
pub struct Gamma;

impl Gamma {
    /// Gamma function using Lanczos approximation.
    pub fn gamma(x: f64) -> f64 {
        if x < 0.0 {
            // Reflection formula: Γ(z) = π / (Γ(1-z) * sin(πz))
            let pi = core::f64::consts::PI;
            let sin_pi_x = (pi * x).sin();
            if sin_pi_x == 0.0 {
                return f64::INFINITY;
            }
            return pi / (Self::gamma(1.0 - x) * sin_pi_x);
        }
        
        if x < 0.5 {
            // Recurrence: Γ(z+1) = z*Γ(z)
            return Self::gamma(x + 1.0) / x;
        }
        
        // Lanczos approximation for x >= 0.5
        let g = 7.0;
        let c: [f64; 9] = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        
        let z = x - 1.0;
        let mut a = c[0];
        
        for i in 1..=8 {
            a += c[i] / (z + i as f64);
        }
        
        let t = z + g + 0.5;
        let sqrt_2pi = (2.0 * core::f64::consts::PI).sqrt();
        
        sqrt_2pi * t.powf(z + 0.5) * (-t).exp() * a
    }

    /// Log gamma function ln(Γ(z)).
    pub fn log_gamma(x: f64) -> f64 {
        if x < 0.0 {
            return f64::NAN;
        }
        
        if x < 1.0 {
            return Self::log_gamma(x + 1.0) - x.ln();
        }
        
        // Stirling's approximation with correction
        let n = x - 1.0;
        let stirling = n * n.ln() - n + 0.5 * (2.0 * core::f64::consts::PI * n).ln();
        
        // Bernoulli numbers correction
        let b2 = 1.0 / 12.0;
        let b4 = -1.0 / 360.0;
        let b6 = 1.0 / 1260.0;
        
        let correction = b2 / n + b4 / n.powi(3) + b6 / n.powi(5);
        
        stirling + correction
    }

    /// Incomplete gamma function γ(s, x).
    pub fn incomplete_gamma(s: f64, x: f64) -> f64 {
        if x < 0.0 || s <= 0.0 {
            return f64::NAN;
        }
        
        if x < s + 1.0 {
            // Series representation
            Self::incomplete_gamma_series(s, x)
        } else {
            // Continued fraction representation
            Self::gamma(s) - Self::incomplete_gamma_cf(s, x)
        }
    }

    /// Incomplete gamma using series.
    fn incomplete_gamma_series(s: f64, x: f64) -> f64 {
        let mut term = 1.0 / s;
        let mut sum = term;
        
        for n in 1..1000 {
            term *= x / (s + n as f64);
            let new_sum = sum + term;
            
            if (new_sum - sum).abs() < 1e-15 * sum.abs() {
                return new_sum * (-x).exp() * x.powf(s);
            }
            
            sum = new_sum;
        }
        
        sum * (-x).exp() * x.powf(s)
    }

    /// Incomplete gamma using continued fraction.
    fn incomplete_gamma_cf(s: f64, x: f64) -> f64 {
        let a = [1.0_f64];
        let b = [s, 1.0_f64];
        
        // Simplified continued fraction evaluation
        let mut f = b[0];
        let mut c = 1.0 / f64::EPSILON;
        let mut d = 0.0;
        
        for i in 1..100 {
            d = b[i % 2] + a[0] * d;
            if d.abs() < f64::EPSILON {
                d = f64::EPSILON;
            }
            d = 1.0 / d;
            
            c = b[i % 2] + a[0] / c;
            if c.abs() < f64::EPSILON {
                c = f64::EPSILON;
            }
            
            let delta = c * d;
            f *= delta;
            
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        
        f * (-x).exp() * x.powf(s)
    }

    /// Upper incomplete gamma function Γ(s, x).
    pub fn upper_incomplete_gamma(s: f64, x: f64) -> f64 {
        Self::gamma(s) - Self::incomplete_gamma(s, x)
    }

    /// Regularized gamma function P(s, x) = γ(s, x) / Γ(s).
    pub fn regularized_gamma(s: f64, x: f64) -> f64 {
        Self::incomplete_gamma(s, x) / Self::gamma(s)
    }

    /// Digamma function ψ(z) = Γ'(z)/Γ(z).
    pub fn digamma(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NAN;
        }
        
        if x < 1.0 {
            return Self::digamma(x + 1.0) - 1.0 / x;
        }
        
        // Asymptotic expansion
        let n = (x as i64) as f64;
        let harmonic = (1..=n as i64).map(|i| 1.0 / i as f64).sum::<f64>();
        
        let z = x - n;
        let bernoulli = [1.0 / 12.0, -1.0 / 120.0, 1.0 / 252.0, -1.0 / 240.0];
        
        let mut sum = harmonic;
        for (i, &b) in bernoulli.iter().enumerate() {
            sum -= b / z.powi(2 * i as i32 + 1);
        }
        
        sum
    }

    /// Polygamma function ψ^(n)(z).
    pub fn polygamma(n: u32, x: f64) -> f64 {
        if n == 0 {
            return Self::digamma(x);
        }
        
        if x <= 0.0 {
            return f64::NAN;
        }
        
        // Trigamma (n=1)
        if n == 1 {
            if x < 1.0 {
                return Self::polygamma(1, x + 1.0) + 1.0 / x.powi(2);
            }
            
            let z = x - (x as i64) as f64;
            let harmonic = (1..=100).map(|i| 1.0 / (z + i as f64).powi(2)).sum::<f64>();
            
            harmonic + 1.0 / z.powi(2)
        } else {
            // Higher polygamma (simplified)
            let mut result = 0.0;
            for i in 0..1000 {
                result += 1.0 / (x + i as f64).powi(n as i32 + 1);
            }
            (-1.0_f64).powi(n as i32 + 1) * Self::factorial(n) as f64 * result
        }
    }

    /// Factorial: n! = Γ(n+1).
    pub fn factorial(n: u64) -> MathResult<f64> {
        if n > 170 {
            return Err(MathError::InvalidArgument("factorial overflow for n > 170"));
        }
        Ok(Self::gamma((n + 1) as f64))
    }

    /// Double factorial: n!!.
    pub fn double_factorial(n: i64) -> MathResult<f64> {
        if n < -1 {
            return Err(MathError::InvalidArgument("double factorial undefined for n < -1"));
        }
        
        if n == 0 || n == -1 {
            return Ok(1.0);
        }
        
        let mut result = 1.0;
        let mut current = n as f64;
        
        while current > 0.0 {
            result *= current;
            current -= 2.0;
        }
        
        Ok(result)
    }
}

/// Beta function B(x, y).
pub struct Beta;

impl Beta {
    /// Beta function: B(x, y) = Γ(x)Γ(y) / Γ(x+y).
    pub fn beta(x: f64, y: f64) -> f64 {
        if x <= 0.0 || y <= 0.0 {
            return f64::NAN;
        }
        
        Gamma::gamma(x) * Gamma::gamma(y) / Gamma::gamma(x + y)
    }

    /// Log beta function: ln(B(x, y)).
    pub fn log_beta(x: f64, y: f64) -> f64 {
        Gamma::log_gamma(x) + Gamma::log_gamma(y) - Gamma::log_gamma(x + y)
    }

    /// Incomplete beta function B_x(a, b).
    pub fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
        if x < 0.0 || x > 1.0 || a <= 0.0 || b <= 0.0 {
            return f64::NAN;
        }
        
        if x == 0.0 {
            return 0.0;
        }
        
        if x == 1.0 {
            return Self::beta(a, b);
        }
        
        // Continued fraction representation
        Self::incomplete_beta_cf(x, a, b)
    }

    /// Incomplete beta using continued fraction.
    fn incomplete_beta_cf(x: f64, a: f64, b: f64) -> f64 {
        let max_iter = 100;
        let epsilon = 1e-15;
        
        let qab = a + b;
        let qap = a + 1.0;
        let qam = a - 1.0;
        
        let mut c = 1.0;
        let mut d = 1.0 - qab * x / qap;
        if d.abs() < epsilon {
            d = epsilon;
        }
        d = 1.0 / d;
        
        let mut h = d;
        
        for m in 1..max_iter {
            let m2 = 2 * m;
            let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if d.abs() < epsilon {
                d = epsilon;
            }
            c = 1.0 + aa / c;
            if c.abs() < epsilon {
                c = epsilon;
            }
            d = 1.0 / d;
            h *= d * c;
            
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if d.abs() < epsilon {
                d = epsilon;
            }
            c = 1.0 + aa / c;
            if c.abs() < epsilon {
                c = epsilon;
            }
            d = 1.0 / d;
            
            let delta = d * c;
            h *= delta;
            
            if (delta - 1.0).abs() < epsilon {
                break;
            }
        }
        
        h * a.powf(a) * (1.0 - x).powf(b) / a / Self::beta(a, b)
    }

    /// Regularized incomplete beta function I_x(a, b).
    pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
        Self::incomplete_beta(x, a, b) / Self::beta(a, b)
    }
}

/// Error function erf(x).
pub struct Erf;

impl Erf {
    /// Error function: erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt.
    pub fn erf(x: f64) -> f64 {
        if x.is_infinite() {
            return x.signum();
        }
        
        // Abramowitz and Stegun approximation
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        sign * y
    }

    /// Complementary error function: erfc(x) = 1 - erf(x).
    pub fn erfc(x: f64) -> f64 {
        if x.is_infinite() {
            return if x > 0.0 { 0.0 } else { 2.0 };
        }
        
        1.0 - Self::erf(x)
    }

    /// Inverse error function: erf^(-1)(y).
    pub fn inverse_erf(y: f64) -> f64 {
        if y <= -1.0 || y >= 1.0 {
            return f64::NAN;
        }
        
        // Approximation using rational function
        let a = [0.886226899, -1.645349621, 0.914624893, -0.140543331];
        let b = [1.0, -2.118377725, 1.442810227, -0.329097515, 0.012229801];
        
        let z = y.abs();
        let r = (z - 0.5) * 2.0;
        
        let mut num = 0.0;
        let mut den = 0.0;
        
        for i in 0..a.len() {
            num += a[i] * r.powi(i as i32);
        }
        
        for i in 0..b.len() {
            den += b[i] * r.powi(i as i32);
        }
        
        let result = num / den;
        
        if y < 0.0 {
            -result
        } else {
            result
        }
    }

    /// Inverse complementary error function: erfc^(-1)(y).
    pub fn inverse_erfc(y: f64) -> f64 {
        Self::inverse_erf(1.0 - y)
    }

    /// Error function for complex argument (simplified).
    pub fn complex_erf(z: crate::complex::Complex) -> crate::complex::Complex {
        // Faddeeva function approximation
        let z2 = z.mul(&z);
        let exp_neg_z2 = z2.scale(-1.0).exp();
        
        // Simplified: erf(z) ≈ 1 - exp(-z²) / (√π * z) for large z
        let sqrt_pi = core::f64::consts::PI.sqrt();
        let inv_sqrt_pi_z = crate::complex::Complex::new(1.0 / sqrt_pi, 0.0).div(&z).unwrap();
        let term = exp_neg_z2.mul(&inv_sqrt_pi_z);
        
        crate::complex::Complex::one().sub(&term)
    }
}

/// Bessel functions.
pub struct Bessel;

impl Bessel {
    /// Bessel function of the first kind J_n(x) (simplified for n=0,1).
    pub fn j0(x: f64) -> f64 {
        if x.abs() < 8.0 {
            // Polynomial approximation
            let y = x * x / 64.0;
            let r = (((((((-0.0000000118 * y + 0.0000001754) * y - 0.0000026037) * y
                + 0.0000348743) * y - 0.0004493937) * y + 0.0049448806) * y
                - 0.0444198139) * y + 0.2936635684) * y - 1.1352008234) * y + 0.7651976861;
            r
        } else {
            let z = 8.0 / x;
            let y = z * z;
            let r = (((((((-0.0000000247 * y + 0.0000003747) * y - 0.0000050395) * y
                + 0.0000611607) * y - 0.0006492926) * y + 0.0058390816) * y
                - 0.0418487486) * y + 0.2116520864) * y - 0.7847566544) * y + 0.7978845611;
            r * (x / 8.0).cos()
        }
    }

    /// Bessel function of the first kind J_1(x).
    pub fn j1(x: f64) -> f64 {
        if x.abs() < 8.0 {
            let y = x * x / 64.0;
            let r = (((((((((0.0000000118 * y - 0.0000002105) * y + 0.0000027267) * y
                - 0.0000308405) * y + 0.0003168151) * y - 0.0028490672) * y
                + 0.0212004268) * y - 0.1207822376) * y + 0.4975778768) * y - 1.1352008234) * y + 0.5;
            x * r
        } else {
            let z = 8.0 / x;
            let y = z * z;
            let r = ((((((((-0.0000000247 * y + 0.0000003747) * y - 0.0000050395) * y
                + 0.0000611607) * y - 0.0006492926) * y + 0.0058390816) * y
                - 0.0418487486) * y + 0.2116520864) * y - 0.7847566544) * y + 0.7978845611;
            r * (x / 8.0).sin()
        }
    }

    /// Bessel function of the second kind Y_0(x) (simplified).
    pub fn y0(x: f64) -> f64 {
        if x < 8.0 {
            let y = x * x / 64.0;
            let r = (((((((((0.0000000118 * y - 0.0000002105) * y + 0.0000027267) * y
                - 0.0000308405) * y + 0.0003168151) * y - 0.0028490672) * y
                + 0.0212004268) * y - 0.1207822376) * y + 0.4975778768) * y - 1.1352008234) * y + 0.5;
            r + Self::j0(x) * (2.0 / core::f64::consts::PI) * (x / 8.0).ln()
        } else {
            let z = 8.0 / x;
            let y = z * z;
            let r = (((((((-0.0000000247 * y + 0.0000003747) * y - 0.0000050395) * y
                + 0.0000611607) * y - 0.0006492926) * y + 0.0058390816) * y
                - 0.0418487486) * y + 0.2116520864) * y - 0.7847566544) * y + 0.7978845611;
            r * (x / 8.0).sin()
        }
    }

    /// Bessel function of the second kind Y_1(x) (simplified).
    pub fn y1(x: f64) -> f64 {
        if x < 8.0 {
            let y = x * x / 64.0;
            let r = (((((((-0.0000000118 * y + 0.0000001754) * y - 0.0000026037) * y
                + 0.0000348743) * y - 0.0004493937) * y + 0.0049448806) * y
                - 0.0444198139) * y + 0.2936635684) * y - 1.1352008234) * y + 0.7651976861;
            r * x + Self::j1(x) * (2.0 / core::f64::consts::PI) * ((x / 8.0).ln() - 1.0 / x)
        } else {
            let z = 8.0 / x;
            let y = z * z;
            let r = (((((((-0.0000000247 * y + 0.0000003747) * y - 0.0000050395) * y
                + 0.0000611607) * y - 0.0006492926) * y + 0.0058390816) * y
                - 0.0418487486) * y + 0.2116520864) * y - 0.7847566544) * y + 0.7978845611;
            -r * (x / 8.0).cos()
        }
    }
}

/// Airy functions.
pub struct Airy;

impl Airy {
    /// Airy function Ai(x) (simplified approximation).
    pub fn ai(x: f64) -> f64 {
        if x > 0.0 {
            // Exponential decay region
            let z = 2.0 * x.powf(1.5) / 3.0;
            0.35502805 * (-z).exp() / z.sqrt()
        } else {
            // Oscillatory region
            let z = 2.0 * (-x).powf(1.5) / 3.0;
            0.35502805 * (z / 3.0).cos() / z.powf(1.0 / 6.0)
        }
    }

    /// Airy function Bi(x) (simplified approximation).
    pub fn bi(x: f64) -> f64 {
        if x > 0.0 {
            let z = 2.0 * x.powf(1.5) / 3.0;
            0.61492667 * z.exp() / z.sqrt()
        } else {
            let z = 2.0 * (-x).powf(1.5) / 3.0;
            0.61492667 * (z / 3.0 + core::f64::consts::PI / 4.0).sin() / z.powf(1.0 / 6.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma() {
        assert!((Gamma::gamma(1.0) - 1.0).abs() < 1e-10);
        assert!((Gamma::gamma(5.0) - 24.0).abs() < 1e-10);
        assert!((Gamma::gamma(0.5) - 1.7724538509).abs() < 1e-8);
    }

    #[test]
    fn test_log_gamma() {
        let log_gamma = Gamma::log_gamma(5.0);
        let expected = 24.0_f64.ln();
        assert!((log_gamma - expected).abs() < 1e-10);
    }

    #[test]
    fn test_factorial() {
        assert!((Gamma::factorial(5).unwrap() - 120.0).abs() < 1e-10);
        assert!((Gamma::factorial(10).unwrap() - 3628800.0).abs() < 1e-10);
    }

    #[test]
    fn test_beta() {
        let b = Beta::beta(2.0, 3.0);
        assert!((b - 1.0 / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_erf() {
        assert!((Erf::erf(0.0) - 0.0).abs() < 1e-10);
        assert!((Erf::erf(1.0) - 0.8427007929).abs() < 1e-8);
        assert!((Erf::erf(f64::INFINITY) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_erfc() {
        assert!((Erf::erfc(0.0) - 1.0).abs() < 1e-10);
        assert!((Erf::erfc(1.0) - 0.1572992071).abs() < 1e-8);
    }

    #[test]
    fn test_inverse_erf() {
        let x = Erf::inverse_erf(0.5);
        let back = Erf::erf(x);
        assert!((back - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_bessel_j0() {
        assert!((Bessel::j0(0.0) - 1.0).abs() < 1e-10);
        assert!((Bessel::j0(2.4048) - 0.0).abs() < 0.01); // First zero
    }

    #[test]
    fn test_airy() {
        let ai_0 = Airy::ai(0.0);
        assert!((ai_0 - 0.35502805).abs() < 0.01);
    }
}
