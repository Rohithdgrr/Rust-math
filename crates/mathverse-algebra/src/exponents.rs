//! Exponent, logarithm, and radical identity helpers.
//!
//! These are convenience wrappers around the transcendental functions,
//! implementing the standard algebraic identities.

/// Product of powers: `aᵐ · aⁿ = aᵐ⁺ⁿ`.
pub fn product_of_powers(base: f64, m: f64, n: f64) -> f64 {
    base.powf(m + n)
}

/// Power of a power: `(aᵐ)ⁿ = aᵐⁿ`.
pub fn power_of_power(base: f64, m: f64, n: f64) -> f64 {
    base.powf(m * n)
}

/// Zero exponent: `a⁰ = 1` (for `a ≠ 0`).
pub fn zero_exponent(base: f64) -> f64 {
    if base.abs() < 1e-15 {
        f64::NAN
    } else {
        1.0
    }
}

/// Negative exponent: `a⁻ⁿ = 1/aⁿ`.
pub fn negative_exponent(base: f64, n: f64) -> f64 {
    1.0 / base.powf(n)
}

/// Power of a product: `(ab)ⁿ = aⁿbⁿ`.
pub fn power_of_product(a: f64, b: f64, n: f64) -> f64 {
    a.powf(n) * b.powf(n)
}

/// Power of a quotient: `(a/b)ⁿ = aⁿ/bⁿ`.
pub fn power_of_quotient(a: f64, b: f64, n: f64) -> f64 {
    a.powf(n) / b.powf(n)
}

/// Product rule for logs: `log_a(xy) = log_a(x) + log_a(y)`.
pub fn log_product(x: f64, y: f64, base: f64) -> f64 {
    x.log(base) + y.log(base)
}

/// Quotient rule for logs: `log_a(x/y) = log_a(x) − log_a(y)`.
pub fn log_quotient(x: f64, y: f64, base: f64) -> f64 {
    x.log(base) - y.log(base)
}

/// Power rule for logs: `log_a(xⁿ) = n·log_a(x)`.
pub fn log_power(x: f64, n: f64, base: f64) -> f64 {
    n * x.log(base)
}

/// Change of base: `log_a(x) = ln(x) / ln(a)`.
///
/// ```
/// # use mathverse_algebra::exponents::change_of_base;
/// // log₂(8) = ln(8)/ln(2) = 3
/// assert!((change_of_base(8.0, 2.0) - 3.0).abs() < 1e-12);
/// ```
pub fn change_of_base(x: f64, from_base: f64) -> f64 {
    x.ln() / from_base.ln()
}

/// Rational exponent: `a^(m/n) = ⁿ√(a^m)`.
///
/// ```
/// # use mathverse_algebra::exponents::rational_exponent;
/// assert!((rational_exponent(8.0, 1.0, 3.0) - 2.0).abs() < 1e-12); // ³√8
/// ```
pub fn rational_exponent(base: f64, m: f64, n: f64) -> f64 {
    base.powf(m / n)
}

/// Product under radical: `ⁿ√a · ⁿ√b = ⁿ√(ab)`.
pub fn product_under_radical(a: f64, b: f64, n: f64) -> f64 {
    (a * b).powf(1.0 / n)
}

/// Quotient under radical: `ⁿ√a / ⁿ√b = ⁿ√(a/b)`.
pub fn quotient_under_radical(a: f64, b: f64, n: f64) -> f64 {
    (a / b).powf(1.0 / n)
}

/// Power of a radical: `(ⁿ√a)ᵐ = ⁿ√(aᵐ)`.
pub fn power_of_radical(a: f64, m: f64, n: f64) -> f64 {
    a.powf(m / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn power_identities() {
        assert!(approx(product_of_powers(2.0, 3.0, 4.0), 128.0)); // 2³·2⁴ = 2⁷
        assert!(approx(power_of_power(2.0, 3.0, 2.0), 64.0)); // (2³)²
        assert!(approx(zero_exponent(5.0), 1.0));
        assert!(approx(negative_exponent(2.0, 3.0), 0.125)); // 2⁻³
        assert!(approx(power_of_product(2.0, 3.0, 2.0), 36.0)); // (2·3)²
        assert!(approx(power_of_quotient(4.0, 2.0, 2.0), 4.0)); // (4/2)²
    }

    #[test]
    fn log_identities() {
        assert!(approx(log_product(2.0, 3.0, 10.0), 2.0_f64.log10() + 3.0_f64.log10()));
        assert!(approx(log_quotient(6.0, 2.0, 10.0), 6.0_f64.log10() - 2.0_f64.log10()));
        assert!(approx(log_power(2.0, 3.0, 10.0), 3.0 * 2.0_f64.log10()));
    }

    #[test]
    fn change_base() {
        assert!(approx(change_of_base(8.0, 2.0), 3.0));
    }

    #[test]
    fn radicals() {
        assert!(approx(rational_exponent(8.0, 1.0, 3.0), 2.0));
        assert!(approx(product_under_radical(4.0, 9.0, 2.0), 6.0));
        assert!(approx(quotient_under_radical(9.0, 4.0, 2.0), 1.5));
        assert!(approx(power_of_radical(8.0, 2.0, 3.0), 4.0));
    }
}
