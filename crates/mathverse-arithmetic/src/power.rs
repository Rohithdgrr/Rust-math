//! Advanced power operations: fractional exponents, logarithms, exponential functions.

use mathverse_core::error::{MathError, MathResult};

/// Advanced power and exponential operations.
pub struct Power;

impl Power {
    /// Power with fractional exponent: x^(a/b).
    pub fn rational_power(x: f64, numerator: i32, denominator: i32) -> MathResult<f64> {
        if denominator == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let exp = numerator as f64 / denominator as f64;
        
        if x >= 0.0 {
            Ok(x.powf(exp))
        } else {
            // Handle negative bases for rational exponents
            if denominator % 2 == 0 {
                return Err(MathError::InvalidArgument("even root of negative number"));
            }
            Ok((-x).powf(exp) * if numerator % 2 == 0 { 1.0 } else { -1.0 })
        }
    }

    /// Power with decimal exponent.
    pub fn decimal_power(x: f64, exponent: f64) -> f64 {
        x.powf(exponent)
    }

    /// Natural logarithm: ln(x).
    pub fn natural_log(x: f64) -> MathResult<f64> {
        if x <= 0.0 {
            return Err(MathError::InvalidArgument("logarithm of non-positive number"));
        }
        Ok(x.ln())
    }

    /// Base-10 logarithm: log10(x).
    pub fn log10(x: f64) -> MathResult<f64> {
        if x <= 0.0 {
            return Err(MathError::InvalidArgument("logarithm of non-positive number"));
        }
        Ok(x.log10())
    }

    /// Base-2 logarithm: log2(x).
    pub fn log2(x: f64) -> MathResult<f64> {
        if x <= 0.0 {
            return Err(MathError::InvalidArgument("logarithm of non-positive number"));
        }
        Ok(x.log2())
    }

    /// Logarithm with arbitrary base: log_b(x) = ln(x) / ln(b).
    pub fn log_base(x: f64, base: f64) -> MathResult<f64> {
        if x <= 0.0 || base <= 0.0 || base == 1.0 {
            return Err(MathError::InvalidArgument("invalid base or argument for logarithm"));
        }
        Ok(x.ln() / base.ln())
    }

    /// Natural exponential: e^x.
    pub fn exp(x: f64) -> f64 {
        x.exp()
    }

    /// Base-10 exponential: 10^x.
    pub fn exp10(x: f64) -> f64 {
        10.0_f64.powf(x)
    }

    /// Base-2 exponential: 2^x.
    pub fn exp2(x: f64) -> f64 {
        2.0_f64.powf(x)
    }

    /// Exponential with arbitrary base: b^x.
    pub fn exp_base(x: f64, base: f64) -> MathResult<f64> {
        if base <= 0.0 {
            return Err(MathError::InvalidArgument("base must be positive"));
        }
        Ok(base.powf(x))
    }

    /// Hyperbolic sine: sinh(x) = (e^x - e^(-x)) / 2.
    pub fn sinh(x: f64) -> f64 {
        x.sinh()
    }

    /// Hyperbolic cosine: cosh(x) = (e^x + e^(-x)) / 2.
    pub fn cosh(x: f64) -> f64 {
        x.cosh()
    }

    /// Hyperbolic tangent: tanh(x) = sinh(x) / cosh(x).
    pub fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    /// Inverse hyperbolic sine: asinh(x).
    pub fn asinh(x: f64) -> f64 {
        x.asinh()
    }

    /// Inverse hyperbolic cosine: acosh(x).
    pub fn asinh(x: f64) -> MathResult<f64> {
        if x < 1.0 {
            return Err(MathError::InvalidArgument("acosh requires x >= 1"));
        }
        Ok(x.acosh())
    }

    /// Inverse hyperbolic tangent: atanh(x).
    pub fn atanh(x: f64) -> MathResult<f64> {
        if x.abs() >= 1.0 {
            return Err(MathError::InvalidArgument("atanh requires |x| < 1"));
        }
        Ok(x.atanh())
    }

    /// Change of base formula: log_a(x) = log_b(x) / log_b(a).
    pub fn change_of_base(x: f64, from_base: f64, to_base: f64) -> MathResult<f64> {
        let log_x = Self::log_base(x, from_base)?;
        let log_a = Self::log_base(from_base, to_base)?;
        Ok(log_x / log_a)
    }

    /// Power rule for derivatives: d/dx(x^n) = n*x^(n-1).
    pub fn power_derivative(x: f64, n: f64) -> f64 {
        n * x.powf(n - 1.0)
    }

    /// Exponential rule for derivatives: d/dx(a^x) = a^x * ln(a).
    pub fn exp_derivative(x: f64, base: f64) -> MathResult<f64> {
        if base <= 0.0 {
            return Err(MathError::InvalidArgument("base must be positive"));
        }
        Ok(base.powf(x) * base.ln())
    }

    /// Logarithm rule for derivatives: d/dx(ln(x)) = 1/x.
    pub fn log_derivative(x: f64) -> MathResult<f64> {
        if x == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(1.0 / x)
    }

    /// Power tower: x^(x^(x^...)) with n levels.
    pub fn power_tower(x: f64, levels: u32) -> f64 {
        if levels == 0 {
            return 1.0;
        }
        
        let mut result = x;
        for _ in 1..levels {
            result = x.powf(result);
        }
        
        result
    }

    /// Lambert W function (principal branch) using Newton's method.
    pub fn lambert_w(x: f64) -> f64 {
        if x >= 0.0 {
            // Initial guess for x >= 0
            let mut w = (x + 1.0).ln();
            for _ in 0..50 {
                let ew = w.exp();
                let w_new = w - (w * ew - x) / (ew * (w + 1.0) - (w + 2.0) * (w * ew - x) / (2.0 * w + 2.0));
                if (w_new - w).abs() < 1e-15 {
                    return w_new;
                }
                w = w_new;
            }
            w
        } else {
            // Initial guess for x < 0
            let mut w = x;
            for _ in 0..50 {
                let ew = w.exp();
                let w_new = w - (w * ew - x) / (ew * (w + 1.0));
                if (w_new - w).abs() < 1e-15 {
                    return w_new;
                }
                w = w_new;
            }
            w
        }
    }
}

/// Logarithmic identities.
pub struct LogarithmicIdentities;

impl LogarithmicIdentities {
    /// Product rule: log(xy) = log(x) + log(y).
    pub fn product_rule(x: f64, y: f64) -> MathResult<f64> {
        let log_x = Power::natural_log(x)?;
        let log_y = Power::natural_log(y)?;
        Ok(log_x + log_y)
    }

    /// Quotient rule: log(x/y) = log(x) - log(y).
    pub fn quotient_rule(x: f64, y: f64) -> MathResult<f64> {
        let log_x = Power::natural_log(x)?;
        let log_y = Power::natural_log(y)?;
        Ok(log_x - log_y)
    }

    /// Power rule: log(x^n) = n * log(x).
    pub fn power_rule_log(x: f64, n: f64) -> MathResult<f64> {
        let log_x = Power::natural_log(x)?;
        Ok(n * log_x)
    }

    /// Change of base: log_a(x) = log(x) / log(a).
    pub fn change_of_base_log(x: f64, a: f64) -> MathResult<f64> {
        Power::log_base(x, a)
    }

    /// Verify logarithmic identity: log_a(x) * log_x(a) = 1.
    pub fn verify_reciprocal(x: f64, a: f64) -> MathResult<bool> {
        let log_a_x = Power::log_base(x, a)?;
        let log_x_a = Power::log_base(a, x)?;
        Ok((log_a_x * log_x_a - 1.0).abs() < 1e-10)
    }
}

/// Exponential identities.
pub struct ExponentialIdentities;

impl ExponentialIdentities {
    /// Product rule: a^(x+y) = a^x * a^y.
    pub fn product_rule_exp(base: f64, x: f64, y: f64) -> MathResult<f64> {
        let a_x = Power::exp_base(x, base)?;
        let a_y = Power::exp_base(y, base)?;
        Ok(a_x * a_y)
    }

    /// Quotient rule: a^(x-y) = a^x / a^y.
    pub fn quotient_rule_exp(base: f64, x: f64, y: f64) -> MathResult<f64> {
        let a_x = Power::exp_base(x, base)?;
        let a_y = Power::exp_base(y, base)?;
        Ok(a_x / a_y)
    }

    /// Power rule: (a^x)^y = a^(xy).
    pub fn power_rule_exp(base: f64, x: f64, y: f64) -> MathResult<f64> {
        Power::exp_base(x * y, base)
    }

    /// e^(ln(x)) = x.
    pub fn exp_log_identity(x: f64) -> MathResult<f64> {
        let log_x = Power::natural_log(x)?;
        Power::exp(log_x)
    }

    /// ln(e^x) = x.
    pub fn log_exp_identity(x: f64) -> f64 {
        Power::natural_log(Power::exp(x)).unwrap_or(f64::NAN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_power() {
        assert!((Power::rational_power(8.0, 1, 3).unwrap() - 2.0).abs() < 1e-10);
        assert!((Power::rational_power(-8.0, 1, 3).unwrap() + 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_logarithms() {
        assert!((Power::natural_log(core::f64::consts::E).unwrap() - 1.0).abs() < 1e-10);
        assert!((Power::log10(100.0).unwrap() - 2.0).abs() < 1e-10);
        assert!((Power::log2(8.0).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_base() {
        assert!((Power::log_base(8.0, 2.0).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponentials() {
        assert!((Power::exp(1.0) - core::f64::consts::E).abs() < 1e-10);
        assert!((Power::exp10(2.0) - 100.0).abs() < 1e-10);
        assert!((Power::exp2(3.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic() {
        let x = 1.0;
        let sinh = Power::sinh(x);
        let cosh = Power::cosh(x);
        let tanh = Power::tanh(x);
        
        assert!((tanh - sinh / cosh).abs() < 1e-10);
    }

    #[test]
    fn test_logarithmic_identities() {
        let result = LogarithmicIdentities::product_rule(2.0, 3.0).unwrap();
        let direct = Power::natural_log(6.0).unwrap();
        assert!((result - direct).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_identities() {
        let result = ExponentialIdentities::product_rule_exp(2.0, 1.0, 2.0).unwrap();
        let direct = Power::exp_base(3.0, 2.0).unwrap();
        assert!((result - direct).abs() < 1e-10);
    }

    #[test]
    fn test_lambert_w() {
        // W(0) = 0
        assert!((Power::lambert_w(0.0) - 0.0).abs() < 1e-10);
        // W(e) = 1
        assert!((Power::lambert_w(core::f64::consts::E) - 1.0).abs() < 1e-10);
    }
}
