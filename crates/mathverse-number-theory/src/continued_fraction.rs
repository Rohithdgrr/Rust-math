//! Continued fractions: Euclidean algorithm, convergents, golden ratio, e, π.

/// Computes the simple continued fraction of `n/d`.
pub fn continued_fraction(n: u64, d: u64) -> Vec<u64> {
    let mut result = Vec::new();
    let (mut a, mut b) = (n, d);
    while b != 0 { result.push(a / b); let r = a % b; a = b; b = r; }
    result
}

pub fn convergents(cf: &[u64]) -> Vec<(u128, u128)> {
    let mut result = Vec::new();
    let (mut h0, mut h1) = (0i128, 1i128);
    let (mut k0, mut k1) = (1i128, 0i128);
    for &a in cf {
        let h = a as i128 * h1 + h0;
        let k = a as i128 * k1 + k0;
        result.push((h as u128, k as u128));
        h0 = h1; h1 = h; k0 = k1; k1 = k;
    }
    result
}

pub fn approximant(n: u64, d: u64, terms: usize) -> f64 {
    let cf = continued_fraction(n, d);
    let convs = convergents(&cf);
    let idx = (terms - 1).min(convs.len() - 1);
    convs[idx].0 as f64 / convs[idx].1 as f64
}

pub fn golden_ratio_cf(terms: usize) -> Vec<u64> {
    vec![1; terms]
}

pub fn e_cf(terms: usize) -> Vec<u64> {
    let mut cf: Vec<u64> = vec![2];
    for k in 1..=terms / 3 { cf.push(1); cf.push(2 * k as u64); cf.push(1); }
    cf.truncate(terms);
    cf
}

pub fn pi_cf(terms: usize) -> Vec<u64> {
    // Known CF terms for π: [3; 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2, ...]
    #[rustfmt::skip]
    let known: Vec<u64> = vec![
        3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2, 1, 84, 2, 1, 1, 15,
        3, 13, 1, 4, 2, 6, 6, 99, 1, 2, 2, 6, 3, 5, 1, 1, 6, 8, 1, 7, 1, 2, 3, 7, 1, 2, 1, 1,
        12, 1, 1, 1, 3, 1, 1, 8, 1, 1, 2, 1, 6, 1, 1, 5, 2, 2, 3, 1, 2, 4, 4, 16, 1, 16, 2,
    ];
    known.into_iter().take(terms).collect()
}

pub fn cf_to_value(cf: &[u64]) -> f64 {
    let mut result = 0.0;
    for &a in cf.iter().rev() { result = a as f64 + 1.0 / result; }
    result
}

pub fn cf_to_fraction(cf: &[u64]) -> (u128, u128) {
    convergents(cf).last().copied().unwrap_or((0, 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cf_basic() {
        let cf = continued_fraction(7, 3);
        assert_eq!(cf, vec![2, 3]);
    }

    #[test]
    fn golden() {
        let phi = cf_to_value(&golden_ratio_cf(20));
        assert!((phi - (1.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn convergents_test() {
        let c = convergents(&[2, 3]);
        assert_eq!(c.last(), Some(&(7, 3)));
    }

    #[test]
    fn e_cf_test() {
        let cf = e_cf(30);
        // e = [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, ...]
        assert_eq!(cf[0], 2);
        assert_eq!(cf[1], 1);
        assert_eq!(cf[2], 2);
        assert_eq!(cf[3], 1);
        assert_eq!(cf[4], 1);
        assert_eq!(cf[5], 4);
        let val = cf_to_value(&cf);
        assert!((val - std::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn pi_cf_test() {
        let cf = pi_cf(5);
        // pi = [3; 7, 15, 1, 292, ...]
        assert_eq!(cf, vec![3, 7, 15, 1, 292]);
        let val = cf_to_value(&cf);
        assert!((val - std::f64::consts::PI).abs() < 0.01);
    }
}
