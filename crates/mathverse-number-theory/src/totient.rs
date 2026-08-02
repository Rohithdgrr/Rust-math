pub fn euler_totient(mut n: u64) -> u64 {
    let mut result = n;
    let mut d = 2u64;
    while (d as u128) * (d as u128) <= n as u128 {
        if n.is_multiple_of(d) {
            while n.is_multiple_of(d) { n /= d; }
            result -= result / d;
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if n > 1 { result -= result / n; }
    result
}

pub fn euler_totient_sum(limit: u64) -> Vec<u64> {
    (0..=limit).map(euler_totient).collect()
}

pub fn carmichael(n: u64) -> u64 {
    if n <= 2 { return n; }
    let factors = crate::factorization::prime_factors(n);
    let mut lambda = 1u64;
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut pk = 1u64;
        while i < factors.len() && factors[i] == p { pk *= p; i += 1; }
        let cp = if p == 2 && pk >= 8 { pk / 4 } else { pk - pk / p };
        lambda = lcm(lambda, cp);
    }
    lambda
}

fn lcm(a: u64, b: u64) -> u64 { a / gcd(a, b) * b }
fn gcd(a: u64, b: u64) -> u64 { if b == 0 { a } else { gcd(b, a % b) } }

pub fn is_coprime(a: u64, b: u64) -> bool { gcd(a, b) == 1 }

pub fn coprimes_up_to(n: u64) -> Vec<u64> {
    (1..=n).filter(|&k| is_coprime(k, n)).collect()
}

pub fn primitive_root(n: u64) -> Option<u64> {
    if n <= 1 { return None; }
    let phi = euler_totient(n);
    let factors = crate::factorization::prime_factors(phi);
    'outer: for g in 2..n {
        if !is_coprime(g, n) { continue; }
        let mut i = 0;
        while i < factors.len() {
            let p = factors[i];
            while i + 1 < factors.len() && factors[i + 1] == p { i += 1; }
            i += 1;
            if crate::modular::mod_pow(g, phi / p, n) == 1 { continue 'outer; }
        }
        return Some(g);
    }
    None
}

pub fn multiplicative_order(a: u64, n: u64) -> Option<u64> {
    if !is_coprime(a, n) { return None; }
    let phi = euler_totient(n);
    let factors = crate::factorization::prime_factors(phi);
    let mut order = phi;
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        while i + 1 < factors.len() && factors[i + 1] == p { i += 1; }
        i += 1;
        while order % p == 0 && crate::modular::mod_pow(a, order / p, n) == 1 { order /= p; }
    }
    Some(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn totient() {
        assert_eq!(euler_totient(10), 4);
        assert_eq!(euler_totient(97), 96);
        assert_eq!(euler_totient(36), 12);
    }

    #[test]
    fn carmichael_test() {
        assert_eq!(carmichael(1), 1);
        assert_eq!(carmichael(8), 2);
    }

    #[test]
    fn primitive_test() {
        assert!(primitive_root(7).is_some());
    }

    #[test]
    fn order_test() {
        assert_eq!(multiplicative_order(2, 7), Some(3));
    }
}
