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
        let pk = p.pow(count);
        result *= (pk.pow(k + 1) - 1) / (pk - 1);
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

pub fn is_abundant(n: u64) -> bool { divisor_sum(n) > n }

pub fn is_deficient(n: u64) -> bool { divisor_sum(n) < n }

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
}
