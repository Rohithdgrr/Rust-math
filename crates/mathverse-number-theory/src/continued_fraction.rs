//! Continued fractions: Euclidean algorithm, convergents, golden ratio, e, π.

/// Computes the simple continued fraction expansion of `n/d`.
///
/// ```
/// use mathverse_number_theory::continued_fraction;
/// assert_eq!(continued_fraction(7, 3), vec![2, 3]);
/// ```
#[must_use]
pub fn continued_fraction(n: u64, d: u64) -> Vec<u64> {
    if d == 0 {
        return if n == 0 { vec![0] } else { vec![n] };
    }
    let mut result = Vec::new();
    let (mut a, mut b) = (n, d);
    while b != 0 {
        result.push(a / b);
        let r = a % b;
        a = b;
        b = r;
    }
    result
}

/// Computes the convergents `p_k/q_k` of a continued fraction.
///
/// Returns a vector of `(numerator, denominator)` pairs as `u128`.
/// Returns an empty vector for empty input.
///
/// ```
/// use mathverse_number_theory::convergents;
/// let c = convergents(&[2, 3]);
/// assert_eq!(c.last(), Some(&(7, 3)));
/// ```
#[must_use]
pub fn convergents(cf: &[u64]) -> Vec<(u128, u128)> {
    let mut result = Vec::new();
    let (mut h0, mut h1) = (0i128, 1i128);
    let (mut k0, mut k1) = (1i128, 0i128);
    for &a in cf {
        let h = a as i128 * h1 + h0;
        let k = a as i128 * k1 + k0;
        result.push((h as u128, k as u128));
        h0 = h1;
        h1 = h;
        k0 = k1;
        k1 = k;
    }
    result
}

/// Computes the best rational approximation `n/d` using the first `terms`
/// convergents of its continued fraction.
///
/// ```
/// use mathverse_number_theory::approximant;
/// let v = approximant(22, 7, 2);
/// assert!((v - 22.0 / 7.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn approximant(n: u64, d: u64, terms: usize) -> f64 {
    let cf = continued_fraction(n, d);
    let convs = convergents(&cf);
    if convs.is_empty() {
        return 0.0;
    }
    let idx = (terms.saturating_sub(1)).min(convs.len() - 1);
    convs[idx].0 as f64 / convs[idx].1 as f64
}

/// Continued fraction expansion of the golden ratio φ = (1+√5)/2: `[1; 1, 1, 1, ...]`.
///
/// ```
/// use mathverse_number_theory::golden_ratio_cf;
/// assert_eq!(golden_ratio_cf(5), vec![1, 1, 1, 1, 1]);
/// ```
#[must_use]
pub fn golden_ratio_cf(terms: usize) -> Vec<u64> {
    vec![1; terms]
}

/// Continued fraction expansion of e (Euler's number).
///
/// Pattern: `[2; 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, ...]`
///
/// ```
/// use mathverse_number_theory::e_cf;
/// let cf = e_cf(6);
/// assert_eq!(cf, vec![2, 1, 2, 1, 1, 4]);
/// ```
#[must_use]
pub fn e_cf(terms: usize) -> Vec<u64> {
    if terms == 0 {
        return Vec::new();
    }
    let mut cf: Vec<u64> = vec![2];
    let mut k = 1u64;
    while cf.len() < terms {
        cf.push(1);
        if cf.len() >= terms {
            break;
        }
        cf.push(2 * k);
        k += 1;
        if cf.len() >= terms {
            break;
        }
        cf.push(1);
    }
    cf.truncate(terms);
    cf
}

/// First `terms` of the continued fraction expansion of π.
///
/// ```
/// use mathverse_number_theory::pi_cf;
/// let cf = pi_cf(5);
/// assert_eq!(cf, vec![3, 7, 15, 1, 292]);
/// ```
#[must_use]
pub fn pi_cf(terms: usize) -> Vec<u64> {
    const KNOWN: [u64; 40] = [
        3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2, 1, 84, 2, 1, 1, 15, 3, 13, 1,
        4, 2, 6, 6, 99, 1, 2, 2, 6, 3, 5,
    ];
    KNOWN.iter().copied().take(terms).collect()
}

/// Evaluates a simple continued fraction to `f64`.
///
/// Returns `NaN` for empty input. Returns `INFINITY` if a
/// zero partial quotient causes a division by zero (e.g. `[a, 0, ...]`).
///
/// ```
/// use mathverse_number_theory::cf_to_value;
/// assert!((cf_to_value(&[3, 7, 15, 1, 292]) - std::f64::consts::PI).abs() < 0.01);
/// assert!(cf_to_value(&[]).is_nan());
/// ```
#[must_use]
pub fn cf_to_value(cf: &[u64]) -> f64 {
    if cf.is_empty() {
        return f64::NAN;
    }
    let mut result = cf[cf.len() - 1] as f64;
    for &a in cf[..cf.len() - 1].iter().rev() {
        if result == 0.0 {
            return f64::INFINITY;
        }
        result = a as f64 + 1.0 / result;
    }
    result
}

/// Converts a continued fraction to its rational approximant as `(numerator, denominator)`.
///
/// Returns `(0, 1)` for empty input.
///
/// ```
/// use mathverse_number_theory::cf_to_fraction;
/// assert_eq!(cf_to_fraction(&[2, 3]), (7, 3));
/// ```
#[must_use]
pub fn cf_to_fraction(cf: &[u64]) -> (u128, u128) {
    convergents(cf).last().copied().unwrap_or((0, 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cf_basic() {
        assert_eq!(continued_fraction(7, 3), vec![2, 3]);
        assert_eq!(continued_fraction(0, 0), vec![0]);
        assert_eq!(continued_fraction(5, 0), vec![5]);
    }

    #[test]
    fn golden() {
        let phi = cf_to_value(&golden_ratio_cf(20));
        let exact_phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((phi - exact_phi).abs() < 1e-6);
    }

    #[test]
    fn convergents_test() {
        let c = convergents(&[2, 3]);
        assert_eq!(c.last(), Some(&(7, 3)));
        assert!(convergents(&[]).is_empty());
    }

    #[test]
    fn e_cf_test() {
        let cf = e_cf(6);
        assert_eq!(cf, vec![2, 1, 2, 1, 1, 4]);
        let val = cf_to_value(&e_cf(30));
        assert!((val - std::f64::consts::E).abs() < 1e-9);
    }

    #[test]
    fn pi_cf_test() {
        let cf = pi_cf(5);
        assert_eq!(cf, vec![3, 7, 15, 1, 292]);
        let val = cf_to_value(&cf);
        assert!((val - std::f64::consts::PI).abs() < 0.01);
    }

    #[test]
    fn cf_to_value_empty() {
        assert!(cf_to_value(&[]).is_nan());
    }

    #[test]
    fn cf_to_fraction_test() {
        assert_eq!(cf_to_fraction(&[2, 3]), (7, 3));
        assert_eq!(cf_to_fraction(&[]), (0, 1));
    }
}