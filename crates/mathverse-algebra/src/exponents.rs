//! Exponent, logarithm, and radical identity verifiers.

/// Verify `a^m · a^n = a^(m+n)` by computing both sides.
///
/// ```
/// # use mathverse_algebra::exponents::product_rule;
/// assert!(product_rule(2.0, 3.0, 4.0));
/// ```
pub fn product_rule(a: f64, m: f64, n: f64) -> bool {
    (a.powf(m) * a.powf(n) - a.powf(m + n)).abs() < 1e-9
}

/// Verify `(a^m)^n = a^(m·n)` by computing both sides.
///
/// ```
/// # use mathverse_algebra::exponents::power_rule;
/// assert!(power_rule(2.0, 3.0, 2.0));
/// ```
pub fn power_rule(a: f64, m: f64, n: f64) -> bool {
    ((a.powf(m)).powf(n) - a.powf(m * n)).abs() < 1e-9
}

/// Verify `a^0 = 1` (with `0^0` returning NaN as per convention).
///
/// ```
/// # use mathverse_algebra::exponents::zero_exponent;
/// assert!(zero_exponent(5.0));
/// assert!(zero_exponent(0.0).is_nan());
/// ```
pub fn zero_exponent(a: f64) -> f64 {
    if a == 0.0 {
        f64::NAN
    } else {
        1.0
    }
}

/// Verify `a^(-n) = 1/a^n`.
///
/// ```
/// # use mathverse_algebra::exponents::negative_exponent;
/// assert!(negative_exponent(2.0, 3.0));
/// ```
pub fn negative_exponent(a: f64, n: f64) -> bool {
    (a.powf(-n) - 1.0 / a.powf(n)).abs() < 1e-9
}

/// Verify `log_a(x·y) = log_a(x) + log_a(y)`.
///
/// ```
/// # use mathverse_algebra::exponents::log_product;
/// assert!(log_product(10.0, 100.0, 1000.0));
/// ```
pub fn log_product(base: f64, x: f64, y: f64) -> bool {
    (x.log(base) + y.log(base) - (x * y).log(base)).abs() < 1e-9
}

/// Verify `log_a(x/y) = log_a(x) - log_a(y)`.
///
/// ```
/// # use mathverse_algebra::exponents::log_quotient;
/// assert!(log_quotient(10.0, 1000.0, 10.0));
/// ```
pub fn log_quotient(base: f64, x: f64, y: f64) -> bool {
    (x.log(base) - y.log(base) - (x / y).log(base)).abs() < 1e-9
}

/// Verify `log_a(x^n) = n·log_a(x)`.
///
/// ```
/// # use mathverse_algebra::exponents::log_power;
/// assert!(log_power(10.0, 100.0, 2.0));
/// ```
pub fn log_power(base: f64, x: f64, n: f64) -> bool {
    ((x.powf(n)).log(base) - n * x.log(base)).abs() < 1e-9
}

/// Verify `log_a(b) = log_c(b) / log_c(a)` (change of base).
///
/// ```
/// # use mathverse_algebra::exponents::change_of_base;
/// assert!(change_of_base(2.0, 8.0, 10.0));
/// ```
pub fn change_of_base(a: f64, b: f64, c: f64) -> bool {
    (b.log(a) - b.log(c) / a.log(c)).abs() < 1e-9
}

/// Verify `ⁿ√(x·y) = ⁿ√x · ⁿ√y`.
///
/// ```
/// # use mathverse_algebra::exponents::radical_product;
/// assert!(radical_product(4.0, 9.0, 2.0));
/// ```
pub fn radical_product(x: f64, y: f64, n: f64) -> bool {
    ((x * y).powf(1.0 / n) - x.powf(1.0 / n) * y.powf(1.0 / n)).abs() < 1e-9
}

/// Verify `ⁿ√(x/y) = ⁿ√x / ⁿ√y`.
///
/// ```
/// # use mathverse_algebra::exponents::radical_quotient;
/// assert!(radical_quotient(8.0, 1.0, 3.0));
/// ```
pub fn radical_quotient(x: f64, y: f64, n: f64) -> bool {
    ((x / y).powf(1.0 / n) - x.powf(1.0 / n) / y.powf(1.0 / n)).abs() < 1e-9
}

/// Verify `ⁿ√(x^m) = x^(m/n)`.
///
/// ```
/// # use mathverse_algebra::exponents::radical_power;
/// assert!(radical_power(16.0, 4.0, 2.0));
/// ```
pub fn radical_power(x: f64, m: f64, n: f64) -> bool {
    ((x.powf(m)).powf(1.0 / n) - x.powf(m / n)).abs() < 1e-9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponent_identities() {
        assert!(product_rule(2.0, 3.0, 4.0));
        assert!(power_rule(2.0, 3.0, 2.0));
        assert_eq!(zero_exponent(5.0), 1.0);
        assert!(negative_exponent(2.0, 3.0));
    }

    #[test]
    fn logarithm_identities() {
        assert!(log_product(10.0, 100.0, 1000.0));
        assert!(log_quotient(10.0, 1000.0, 10.0));
        assert!(log_power(10.0, 100.0, 2.0));
        assert!(change_of_base(2.0, 8.0, 10.0));
    }

    #[test]
    fn radical_identities() {
        assert!(radical_product(4.0, 9.0, 2.0));
        assert!(radical_quotient(8.0, 1.0, 3.0));
        assert!(radical_power(16.0, 4.0, 2.0));
    }
}