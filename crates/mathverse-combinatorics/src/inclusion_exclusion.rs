//! Inclusion-exclusion principle, birthday problem, coupon collector, pigeonhole.

/// Inclusion-exclusion for two sets: `|A ∪ B| = |A| + |B| - |A ∩ B|`.
pub fn inclusion_exclusion_2(a: u128, b: u128, ab: u128) -> u128 {
    a + b - ab
}

pub fn inclusion_exclusion_3(a: u128, b: u128, c: u128, ab: u128, ac: u128, bc: u128, abc: u128) -> u128 {
    a + b + c - ab - ac - bc + abc
}

pub fn union_count(set_sizes: &[u128], intersections: &[u128]) -> u128 {
    let n = set_sizes.len();
    if n == 0 { return 0; }
    if n == 1 { return set_sizes[0]; }
    let mut result: i128 = set_sizes.iter().map(|&x| x as i128).sum();
    for i in 0..intersections.len() {
        if i % 2 == 0 { result -= intersections[i] as i128; }
        else { result += intersections[i] as i128; }
    }
    result as u128
}

pub fn derangement_count(n: usize) -> u128 {
    if n == 0 { return 1; }
    let (mut a, mut b) = (1u128, 0u128);
    for i in 2..=n {
        let t = (i as u128) * (a + b);
        a = b;
        b = t;
    }
    b
}

pub fn birthday_probability(n_people: usize, n_days: usize) -> f64 {
    if n_people > n_days { return 1.0; }
    let mut prob = 1.0;
    for i in 0..n_people {
        prob *= (n_days - i) as f64 / n_days as f64;
    }
    1.0 - prob
}

pub fn coupon_collector_expected(n: usize) -> f64 {
    (1..=n).map(|i| n as f64 / i as f64).sum()
}

pub fn pigeonhole_min(n_items: usize, n_holes: usize) -> usize {
    if n_holes == 0 { return 0; }
    (n_items + n_holes - 1) / n_holes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ie2() {
        assert_eq!(inclusion_exclusion_2(10, 8, 3), 15);
    }

    #[test]
    fn ie3() {
        assert_eq!(inclusion_exclusion_3(10, 8, 5, 3, 2, 1, 1), 18);
    }

    #[test]
    fn birthday() {
        let p = birthday_probability(23, 365);
        assert!(p > 0.5 && p < 0.6);
    }

    #[test]
    fn coupon() {
        let e = coupon_collector_expected(6);
        assert!((e - 14.7).abs() < 1.0);
    }
}
