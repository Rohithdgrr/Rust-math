pub fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    while n.is_multiple_of(2) { out.push(2); n /= 2; }
    let mut d = 3u64;
    while (d as u128) * (d as u128) <= n as u128 {
        while n.is_multiple_of(d) { out.push(d); n /= d; }
        d += 2;
    }
    if n > 1 { out.push(n); }
    out
}

pub fn divisors(n: u64) -> Vec<u64> {
    let mut divs = Vec::new();
    let mut i = 1;
    while (i as u128) * (i as u128) <= n as u128 {
        if n.is_multiple_of(i) {
            divs.push(i);
            if i != n / i { divs.push(n / i); }
        }
        i += 1;
    }
    divs.sort();
    divs
}

pub fn divisor_count(n: u64) -> u64 {
    let factors = prime_factors(n);
    let mut result = 1u64;
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut count = 0;
        while i < factors.len() && factors[i] == p { count += 1; i += 1; }
        result *= count + 1;
    }
    result
}

pub fn divisor_sum(n: u64) -> u64 {
    divisors(n).iter().sum()
}

pub fn sigma_k(n: u64, k: u32) -> u64 {
    let factors = prime_factors(n);
    let mut result = 1u64;
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut count = 0;
        while i < factors.len() && factors[i] == p { count += 1; i += 1; }
        // sigma_k(p^a) = 1 + p^k + p^{2k} + ... + p^{a*k}
        //              = (p^{k*(a+1)} - 1) / (p^k - 1)
        // Use geometric series sum to avoid overflow where possible.
        let p_k = p.pow(k);
        let mut term = 1u64;
        let mut p_pow = 1u64; // p^{j*k} for j=0..count
        for _ in 0..count {
            p_pow = p_pow.saturating_mul(p_k);
            term = term.saturating_add(p_pow);
        }
        result = result.saturating_mul(term);
    }
    result
}

pub fn mobius(n: u64) -> i64 {
    if n == 1 { return 1; }
    let factors = prime_factors(n);
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut count = 0;
        while i < factors.len() && factors[i] == p { count += 1; i += 1; }
        if count > 1 { return 0; }
    }
    if factors.len() % 2 == 0 { 1 } else { -1 }
}

pub fn liouville(n: u64) -> i64 {
    let factors = prime_factors(n);
    if factors.len() % 2 == 0 { 1 } else { -1 }
}

pub fn is_perfect_number(n: u64) -> bool { divisor_sum(n) == 2 * n }

pub fn is_abundant(n: u64) -> bool { divisor_sum(n) > 2 * n }

pub fn is_deficient(n: u64) -> bool { divisor_sum(n) < 2 * n }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factors_test() {
        assert_eq!(prime_factors(84), vec![2, 2, 3, 7]);
        assert_eq!(divisors(12), vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(divisor_count(12), 6);
    }

    #[test]
    fn sigma_test() {
        assert_eq!(sigma_k(12, 1), 28);
        assert_eq!(sigma_k(6, 1), 12);
        assert_eq!(sigma_k(1, 1), 1);
        // sigma_2(12) = 1^2 + 2^2 + 3^2 + 4^2 + 6^2 + 12^2 = 210
        assert_eq!(sigma_k(12, 2), 210);
    }

    #[test]
    fn mobius_test() {
        assert_eq!(mobius(1), 1);
        assert_eq!(mobius(6), 1);
        assert_eq!(mobius(4), 0);
    }

    #[test]
    fn perfect() {
        assert!(is_perfect_number(6));
        assert!(is_perfect_number(28));
    }

    #[test]
    fn abundant_test() {
        assert!(is_abundant(12));   // sigma(12)=28 > 24
        assert!(is_abundant(18));   // sigma(18)=39 > 36
        assert!(!is_abundant(4));   // sigma(4)=7 < 8
        assert!(!is_abundant(6));   // sigma(6)=12 = 12 (perfect, not abundant)
        assert!(!is_abundant(1));   // sigma(1)=1 < 2
    }

    #[test]
    fn deficient_test() {
        assert!(is_deficient(4));   // sigma(4)=7 < 8
        assert!(is_deficient(1));   // sigma(1)=1 < 2
        assert!(!is_deficient(6));  // sigma(6)=12 = 12 (perfect, not deficient)
        assert!(!is_deficient(12)); // sigma(12)=28 > 24
    }
}
