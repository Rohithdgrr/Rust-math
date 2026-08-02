pub fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    let mut i = 5;
    while i * i <= n { if n % i == 0 || n % (i + 2) == 0 { return false; } i += 6; }
    true
}

pub fn sieve(limit: usize) -> Vec<u64> {
    if limit < 2 { return Vec::new(); }
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut i = 2;
    while i * i <= limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= limit { is_prime[j] = false; j += i; }
        }
        i += 1;
    }
    (2..=limit as u64).filter(|&p| is_prime[p as usize]).collect()
}

pub fn nth_prime(n: usize) -> u64 {
    let mut count = 0;
    let mut num = 1;
    while count < n { num += 1; if is_prime(num) { count += 1; } }
    num
}

pub fn prime_between(a: u64, b: u64) -> Vec<u64> {
    (a..=b).filter(|&p| is_prime(p)).collect()
}

pub fn twin_primes(limit: u64) -> Vec<(u64, u64)> {
    (3..limit).filter(|&p| is_prime(p) && is_prime(p + 2)).map(|p| (p, p + 2)).collect()
}

pub fn goldbach(n: u64) -> Option<(u64, u64)> {
    if n < 4 || n % 2 != 0 { return None; }
    (2..=n/2).find(|&p| is_prime(p) && is_prime(n - p)).map(|p| (p, n - p))
}

pub fn mersenne_prime(p: u64) -> Option<u64> {
    if !is_prime(p) { return None; }
    if p >= 64 { return None; } // 2^p - 1 overflows u64
    let m = (1u64 << p) - 1;
    if is_prime(m) { Some(m) } else { None }
}

pub fn prime_gap(n: u64) -> u64 {
    let mut p = n;
    while !is_prime(p) { p += 1; }
    let mut q = p + 1;
    while !is_prime(q) { q += 1; }
    q - p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes() {
        assert!(is_prime(2));
        assert!(is_prime(97));
        assert!(!is_prime(15));
        let p = sieve(50);
        assert_eq!(p.len(), 15);
    }

    #[test]
    fn nth() {
        assert_eq!(nth_prime(1), 2);
        assert_eq!(nth_prime(10), 29);
    }

    #[test]
    fn goldbach_test() {
        assert_eq!(goldbach(10), Some((3, 7)));
        assert!(goldbach(11).is_none());
    }

    #[test]
    fn twins() {
        let t = twin_primes(20);
        assert!(t.contains(&(3, 5)));
        assert!(t.contains(&(5, 7)));
    }

    #[test]
    fn mersenne_test() {
        assert_eq!(mersenne_prime(2), Some(3));       // 2^2-1=3
        assert_eq!(mersenne_prime(3), Some(7));       // 2^3-1=7
        assert_eq!(mersenne_prime(5), Some(31));      // 2^5-1=31
        assert_eq!(mersenne_prime(7), Some(127));     // 2^7-1=127
        assert_eq!(mersenne_prime(11), None);         // 2047=23*89
        assert_eq!(mersenne_prime(64), None);         // overflow guard
        assert_eq!(mersenne_prime(1), None);          // 1 is not prime
    }
}
