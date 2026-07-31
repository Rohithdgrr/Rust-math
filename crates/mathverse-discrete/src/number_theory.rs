//! Number theory: modular arithmetic, primes, GCD, LCM, and related functions.

/// Number theory operations.
pub struct NumberTheory;

impl NumberTheory {
    /// Greatest common divisor using Euclidean algorithm.
    pub fn gcd(a: i64, b: i64) -> i64 {
        let mut a = a.abs();
        let mut b = b.abs();
        
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        
        a
    }

    /// Extended Euclidean algorithm: returns (gcd, x, y) such that ax + by = gcd.
    pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
        if b == 0 {
            (a.abs(), if a < 0 { -1 } else { 1 }, 0)
        } else {
            let (gcd, x1, y1) = Self::extended_gcd(b, a % b);
            let x = y1;
            let y = x1 - (a / b) * y1;
            (gcd, x, y)
        }
    }

    /// Least common multiple.
    pub fn lcm(a: i64, b: i64) -> i64 {
        if a == 0 || b == 0 {
            0
        } else {
            (a.abs() / Self::gcd(a, b)) * b.abs()
        }
    }

    /// Modular exponentiation: a^b mod m.
    pub fn mod_pow(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
        if modulus == 1 {
            return 0;
        }
        
        let mut result = 1;
        base = base % modulus;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base) % modulus;
            }
            exp /= 2;
            base = (base * base) % modulus;
        }
        
        result
    }

    /// Modular inverse using extended Euclidean algorithm.
    pub fn mod_inverse(a: i64, modulus: i64) -> Option<i64> {
        let (gcd, x, _) = Self::extended_gcd(a, modulus);
        
        if gcd != 1 {
            None
        } else {
            Some(((x % modulus) + modulus) % modulus)
        }
    }

    /// Check if number is prime (deterministic for small numbers, probabilistic for large).
    pub fn is_prime(n: i64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 || n % 3 == 0 {
            return false;
        }
        
        let mut i = 5;
        while i * i <= n {
            if n % i == 0 || n % (i + 2) == 0 {
                return false;
            }
            i += 6;
        }
        
        true
    }

    /// Miller-Rabin primality test (probabilistic).
    pub fn miller_rabin(n: i64, k: usize) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        // Write n-1 as 2^r * d
        let mut d = n - 1;
        let mut r = 0;
        
        while d % 2 == 0 {
            d /= 2;
            r += 1;
        }
        
        // Witness loop
        for _ in 0..k {
            let a = 2 + (rand::random::<i64>() % (n - 4));
            let mut x = Self::mod_pow(a, d, n);
            
            if x == 1 || x == n - 1 {
                continue;
            }
            
            let mut composite = true;
            for _ in 0..r - 1 {
                x = Self::mod_pow(x, 2, n);
                if x == n - 1 {
                    composite = false;
                    break;
                }
            }
            
            if composite {
                return false;
            }
        }
        
        true
    }

    /// Sieve of Eratosthenes: generate all primes up to n.
    pub fn sieve_of_eratosthenes(n: usize) -> Vec<usize> {
        if n < 2 {
            return Vec::new();
        }
        
        let mut is_prime = vec![true; n + 1];
        is_prime[0] = false;
        is_prime[1] = false;
        
        let mut p = 2;
        while p * p <= n {
            if is_prime[p] {
                let mut i = p * p;
                while i <= n {
                    is_prime[i] = false;
                    i += p;
                }
            }
            p += 1;
        }
        
        (2..=n).filter(|&i| is_prime[i]).collect()
    }

    /// Prime factorization.
    pub fn prime_factorization(mut n: i64) -> Vec<(i64, usize)> {
        let mut factors = Vec::new();
        
        // Handle 2 separately
        let mut count = 0;
        while n % 2 == 0 {
            count += 1;
            n /= 2;
        }
        if count > 0 {
            factors.push((2, count));
        }
        
        // Check odd factors
        let mut i = 3;
        while i * i <= n {
            count = 0;
            while n % i == 0 {
                count += 1;
                n /= i;
            }
            if count > 0 {
                factors.push((i, count));
            }
            i += 2;
        }
        
        // If n is still > 1, it's prime
        if n > 1 {
            factors.push((n, 1));
        }
        
        factors
    }

    /// Euler's totient function φ(n): count of numbers ≤ n coprime to n.
    pub fn euler_totient(n: i64) -> i64 {
        if n == 0 {
            return 0;
        }
        
        let mut result = n;
        let mut n_mut = n;
        let mut p = 2;
        
        while p * p <= n_mut {
            if n_mut % p == 0 {
                while n_mut % p == 0 {
                    n_mut /= p;
                }
                result -= result / p;
            }
            p += 1;
        }
        
        if n_mut > 1 {
            result -= result / n_mut;
        }
        
        result
    }

    /// Carmichael function λ(n): smallest m such that a^m ≡ 1 (mod n) for all a coprime to n.
    pub fn carmichael(n: i64) -> i64 {
        if n == 1 {
            return 1;
        }
        
        let factors = Self::prime_factorization(n);
        let mut result = 1;
        
        for (p, exp) in factors {
            let p = p as i64;
            let exp = exp as i64;
            
            let lambda_p = if p == 2 {
                if exp == 1 {
                    1
                } else if exp == 2 {
                    2
                } else {
                    2_i64.pow((exp - 2) as u32)
                }
            } else {
                Self::euler_totient(p.pow(exp as u32))
            };
            
            result = Self::lcm(result, lambda_p);
        }
        
        result
    }

    /// Chinese Remainder Theorem: solve x ≡ a_i (mod n_i).
    pub fn chinese_remainder(remainders: &[i64], moduli: &[i64]) -> Option<i64> {
        if remainders.len() != moduli.len() || moduli.is_empty() {
            return None;
        }
        
        // Check that moduli are pairwise coprime
        for i in 0..moduli.len() {
            for j in (i + 1)..moduli.len() {
                if Self::gcd(moduli[i], moduli[j]) != 1 {
                    return None;
                }
            }
        }
        
        let mut result = 0;
        let mut product = 1;
        
        for &m in moduli {
            product *= m;
        }
        
        for (i, (&a, &m)) in remainders.iter().zip(moduli.iter()).enumerate() {
            let p = product / m;
            let inv = Self::mod_inverse(p, m)?;
            result += a * p * inv;
        }
        
        Some(((result % product) + product) % product)
    }

    /// Check if two numbers are coprime.
    pub fn are_coprime(a: i64, b: i64) -> bool {
        Self::gcd(a, b) == 1
    }

    /// Legendre symbol (a/p) for odd prime p.
    pub fn legendre_symbol(a: i64, p: i64) -> i64 {
        if p <= 2 || !Self::is_prime(p) {
            return 0;
        }
        
        let a = a % p;
        
        if a == 0 {
            return 0;
        }
        if a == 1 {
            return 1;
        }
        
        // Quadratic reciprocity
        let mut a = a;
        let mut p = p;
        let mut symbol = 1;
        
        loop {
            a = a % p;
            if a == 0 {
                return 0;
            }
            if a == 1 {
                return symbol;
            }
            
            // Factor out powers of 2
            let mut count = 0;
            while a % 2 == 0 {
                a /= 2;
                count += 1;
            }
            
            if count % 2 == 1 {
                if p % 8 == 3 || p % 8 == 5 {
                    symbol = -symbol;
                }
            }
            
            // Apply quadratic reciprocity
            if (a % 4 == 3) && (p % 4 == 3) {
                symbol = -symbol;
            }
            
            std::mem::swap(&mut a, &mut p);
        }
    }

    /// Jacobi symbol (a/n) for odd positive n.
    pub fn jacobi_symbol(a: i64, n: i64) -> i64 {
        if n <= 0 || n % 2 == 0 {
            return 0;
        }
        
        let mut a = a;
        let mut n = n;
        let mut symbol = 1;
        
        a = a % n;
        
        loop {
            while a % 2 == 0 {
                a /= 2;
                if n % 8 == 3 || n % 8 == 5 {
                    symbol = -symbol;
                }
            }
            
            if a == 0 {
                return 0;
            }
            if a == 1 {
                return symbol;
            }
            
            if (a % 4 == 3) && (n % 4 == 3) {
                symbol = -symbol;
            }
            
            let temp = a;
            a = n % temp;
            n = temp;
        }
    }

    /// Solve linear Diophantine equation ax + by = c.
    pub fn diophantine(a: i64, b: i64, c: i64) -> Option<(i64, i64, i64)> {
        let (gcd, x0, y0) = Self::extended_gcd(a, b);
        
        if c % gcd != 0 {
            return None;
        }
        
        let factor = c / gcd;
        Some((gcd, x0 * factor, y0 * factor))
    }

    /// Count divisors of n.
    pub fn divisor_count(n: i64) -> usize {
        let factors = Self::prime_factorization(n.abs());
        
        factors.iter().map(|(_, exp)| exp + 1).product()
    }

    /// Sum of divisors of n.
    pub fn divisor_sum(n: i64) -> i64 {
        let factors = Self::prime_factorization(n.abs());
        
        factors.iter().map(|(p, exp)| {
            let p = *p as i64;
            let exp = *exp as i64;
            (p.pow((exp + 1) as u32) - 1) / (p - 1)
        }).product()
    }

    /// Möbius function μ(n).
    pub fn mobius(n: i64) -> i64 {
        if n == 0 {
            return 0;
        }
        
        let factors = Self::prime_factorization(n.abs());
        
        // If any factor has exponent > 1, μ(n) = 0
        for (_, exp) in &factors {
            if *exp > 1 {
                return 0;
            }
        }
        
        // μ(n) = (-1)^k where k is number of prime factors
        if factors.len() % 2 == 0 {
            1
        } else {
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(NumberTheory::gcd(48, 18), 6);
        assert_eq!(NumberTheory::gcd(17, 5), 1);
        assert_eq!(NumberTheory::gcd(0, 5), 5);
    }

    #[test]
    fn test_extended_gcd() {
        let (gcd, x, y) = NumberTheory::extended_gcd(48, 18);
        assert_eq!(gcd, 6);
        assert_eq!(48 * x + 18 * y, gcd);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(NumberTheory::lcm(12, 18), 36);
        assert_eq!(NumberTheory::lcm(4, 5), 20);
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(NumberTheory::mod_pow(2, 10, 1000), 24);
        assert_eq!(NumberTheory::mod_pow(3, 5, 7), 5);
    }

    #[test]
    fn test_mod_inverse() {
        assert_eq!(NumberTheory::mod_inverse(3, 7), Some(5));
        assert_eq!(NumberTheory::mod_inverse(4, 9), Some(7));
        assert_eq!(NumberTheory::mod_inverse(2, 4), None);
    }

    #[test]
    fn test_is_prime() {
        assert!(NumberTheory::is_prime(2));
        assert!(NumberTheory::is_prime(17));
        assert!(NumberTheory::is_prime(97));
        assert!(!NumberTheory::is_prime(1));
        assert!(!NumberTheory::is_prime(15));
    }

    #[test]
    fn test_miller_rabin() {
        assert!(NumberTheory::miller_rabin(2, 5));
        assert!(NumberTheory::miller_rabin(17, 5));
        assert!(!NumberTheory::miller_rabin(15, 5));
    }

    #[test]
    fn test_sieve() {
        let primes = NumberTheory::sieve_of_eratosthenes(20);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_prime_factorization() {
        let factors = NumberTheory::prime_factorization(60);
        assert!(factors.contains(&(2, 2)));
        assert!(factors.contains(&(3, 1)));
        assert!(factors.contains(&(5, 1)));
    }

    #[test]
    fn test_euler_totient() {
        assert_eq!(NumberTheory::euler_totient(1), 1);
        assert_eq!(NumberTheory::euler_totient(10), 4);
        assert_eq!(NumberTheory::euler_totient(13), 12);
    }

    #[test]
    fn test_chinese_remainder() {
        let result = NumberTheory::chinese_remainder(&[2, 3], &[3, 5]);
        assert_eq!(result, Some(8));
    }

    #[test]
    fn test_legendre_symbol() {
        assert_eq!(NumberTheory::legendre_symbol(2, 7), 1);
        assert_eq!(NumberTheory::legendre_symbol(3, 7), -1);
    }

    #[test]
    fn test_divisor_count() {
        assert_eq!(NumberTheory::divisor_count(12), 6); // 1, 2, 3, 4, 6, 12
        assert_eq!(NumberTheory::divisor_count(28), 6); // 1, 2, 4, 7, 14, 28
    }

    #[test]
    fn test_divisor_sum() {
        assert_eq!(NumberTheory::divisor_sum(6), 12); // 1 + 2 + 3 + 6 = 12
        assert_eq!(NumberTheory::divisor_sum(28), 56); // 1 + 2 + 4 + 7 + 14 + 28 = 56
    }

    #[test]
    fn test_mobius() {
        assert_eq!(NumberTheory::mobius(1), 1);
        assert_eq!(NumberTheory::mobius(6), 1); // 2*3, two distinct primes
        assert_eq!(NumberTheory::mobius(4), 0); // 2^2, repeated prime
    }
}
