//! Advanced modulus operations: Euclidean, floor division, modular arithmetic.

use mathverse_core::error::{MathError, MathResult};

/// Advanced modulus operations.
pub struct Modulus;

impl Modulus {
    /// Euclidean modulus: always non-negative remainder.
    pub fn euclidean(x: i64, m: i64) -> MathResult<i64> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let r = x % m;
        if r < 0 {
            Ok(r + m.abs())
        } else {
            Ok(r)
        }
    }

    /// Truncated modulus (C-style): remainder has same sign as dividend.
    pub fn truncated(x: i64, m: i64) -> MathResult<i64> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(x % m)
    }

    /// Floored modulus: remainder has same sign as divisor.
    pub fn floored(x: i64, m: i64) -> MathResult<i64> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let r = x % m;
        if (r < 0 && m > 0) || (r > 0 && m < 0) {
            Ok(r + m)
        } else {
            Ok(r)
        }
    }

    /// Floating-point modulus.
    pub fn fmod(x: f64, m: f64) -> MathResult<f64> {
        if m == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(x % m)
    }

    /// Floating-point Euclidean modulus.
    pub fn fmod_euclidean(x: f64, m: f64) -> MathResult<f64> {
        if m == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        let r = x % m;
        if r < 0.0 {
            Ok(r + m.abs())
        } else {
            Ok(r)
        }
    }

    /// Floor division: x // m.
    pub fn floor_div(x: i64, m: i64) -> MathResult<i64> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let q = x / m;
        if x % m != 0 && ((x < 0) ^ (m < 0)) {
            Ok(q - 1)
        } else {
            Ok(q)
        }
    }

    /// Ceiling division: ceil(x / m).
    pub fn ceil_div(x: i64, m: i64) -> MathResult<i64> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let q = x / m;
        if x % m != 0 && !((x < 0) ^ (m < 0)) {
            Ok(q + 1)
        } else {
            Ok(q)
        }
    }

    /// Floating-point floor division.
    pub fn floor_div_f(x: f64, m: f64) -> MathResult<f64> {
        if m == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok((x / m).floor())
    }

    /// Check if two numbers are congruent modulo m.
    pub fn is_congruent(a: i64, b: i64, m: i64) -> MathResult<bool> {
        let a_mod = Self::euclidean(a, m)?;
        let b_mod = Self::euclidean(b, m)?;
        Ok(a_mod == b_mod)
    }

    /// Modular addition: (a + b) mod m.
    pub fn mod_add(a: i64, b: i64, m: i64) -> MathResult<i64> {
        if m <= 0 {
            return Err(MathError::InvalidArgument("modulus must be positive"));
        }
        Self::euclidean(a + b, m)
    }

    /// Modular subtraction: (a - b) mod m.
    pub fn mod_sub(a: i64, b: i64, m: i64) -> MathResult<i64> {
        if m <= 0 {
            return Err(MathError::InvalidArgument("modulus must be positive"));
        }
        Self::euclidean(a - b, m)
    }

    /// Modular multiplication: (a * b) mod m.
    pub fn mod_mul(a: i64, b: i64, m: i64) -> MathResult<i64> {
        if m <= 0 {
            return Err(MathError::InvalidArgument("modulus must be positive"));
        }
        
        // Handle overflow by using modular reduction during multiplication
        let a_mod = Self::euclidean(a, m)?;
        let b_mod = Self::euclidean(b, m)?;
        
        let result = ((a_mod as i128) * (b_mod as i128) % (m as i128)) as i64;
        Self::euclidean(result, m)
    }

    /// Modular exponentiation: a^b mod m (using binary exponentiation).
    pub fn mod_pow(a: i64, mut exp: i64, m: i64) -> MathResult<i64> {
        if m <= 0 {
            return Err(MathError::InvalidArgument("modulus must be positive"));
        }
        
        if exp < 0 {
            return Err(MathError::InvalidArgument("exponent must be non-negative"));
        }
        
        let mut result = 1i64;
        let mut base = Self::euclidean(a, m)?;
        
        while exp > 0 {
            if exp & 1 == 1 {
                result = Self::mod_mul(result, base, m)?;
            }
            base = Self::mod_mul(base, base, m)?;
            exp >>= 1;
        }
        
        Ok(result)
    }

    /// Modular inverse using extended Euclidean algorithm.
    pub fn mod_inverse(a: i64, m: i64) -> MathResult<i64> {
        if m <= 0 {
            return Err(MathError::InvalidArgument("modulus must be positive"));
        }
        
        let (gcd, x, _) = Self::extended_gcd(a, m);
        
        if gcd != 1 {
            return Err(MathError::InvalidArgument("no modular inverse exists (not coprime)"));
        }
        
        Self::euclidean(x, m)
    }

    /// Extended Euclidean algorithm: returns (gcd, x, y) where ax + by = gcd(a,b).
    pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
        if a == 0 {
            return (b, 0, 1);
        }
        
        let (gcd, x1, y1) = Self::extended_gcd(b % a, a);
        let x = y1 - (b / a) * x1;
        let y = x1;
        
        (gcd, x, y)
    }

    /// Chinese Remainder Theorem: find x such that x ≡ a (mod m) and x ≡ b (mod n).
    pub fn chinese_remainder(a: i64, m: i64, b: i64, n: i64) -> MathResult<i64> {
        if m <= 0 || n <= 0 {
            return Err(MathError::InvalidArgument("moduli must be positive"));
        }
        
        let (gcd, p, q) = Self::extended_gcd(m, n);
        
        if (a - b) % gcd != 0 {
            return Err(MathError::InvalidArgument("no solution exists (moduli not coprime)"));
        }
        
        let lcm = m / gcd * n;
        let x = a + m * ((b - a) / gcd * p % (n / gcd));
        
        Self::euclidean(x, lcm)
    }
}

/// Modular arithmetic properties.
pub struct ModularArithmetic;

impl ModularArithmetic {
    /// Check if modulus is prime.
    pub fn is_prime_modulus(m: i64) -> bool {
        if m < 2 {
            return false;
        }
        
        for i in 2..=(m as f64).sqrt() as i64 {
            if m % i == 0 {
                return false;
            }
        }
        
        true
    }

    /// Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p.
    pub fn fermat_little_theorem(a: i64, p: i64) -> MathResult<bool> {
        if !Self::is_prime_modulus(p) {
            return Err(MathError::InvalidArgument("modulus must be prime"));
        }
        
        let a_mod = Modulus::euclidean(a, p)?;
        let result = Modulus::mod_pow(a_mod, p - 1, p)?;
        
        Ok(result == 1)
    }

    /// Euler's theorem: a^φ(n) ≡ 1 (mod n) for coprime a, n.
    pub fn euler_theorem(a: i64, n: i64) -> MathResult<bool> {
        if n <= 0 {
            return Err(MathError::InvalidArgument("modulus must be positive"));
        }
        
        let phi = Self::euler_totient(n);
        let a_mod = Modulus::euclidean(a, n)?;
        let result = Modulus::mod_pow(a_mod, phi, n)?;
        
        Ok(result == 1)
    }

    /// Euler's totient function φ(n): count of numbers ≤ n coprime to n.
    pub fn euler_totient(n: i64) -> i64 {
        if n <= 0 {
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

    /// Check if two numbers are coprime.
    pub fn are_coprime(a: i64, b: i64) -> bool {
        let (gcd, _, _) = Modulus::extended_gcd(a.abs(), b.abs());
        gcd == 1
    }

    /// Carmichael function λ(n): smallest m such that a^m ≡ 1 (mod n) for all a coprime to n.
    pub fn carmichael_function(n: i64) -> i64 {
        if n <= 0 {
            return 0;
        }
        
        if n == 1 {
            return 1;
        }
        
        let mut result = 1;
        let mut n_mut = n;
        let mut p = 2;
        
        while p * p <= n_mut {
            if n_mut % p == 0 {
                let mut power = 0;
                while n_mut % p == 0 {
                    n_mut /= p;
                    power += 1;
                }
                
                if p == 2 && power >= 3 {
                    result = Self::lcm(result, Self::euler_totient(p.pow(power)));
                } else if p == 2 {
                    result = Self::lcm(result, Self::euler_totient(p.pow(power)));
                } else {
                    result = Self::lcm(result, p.pow(power - 1) * (p - 1));
                }
            }
            p += 1;
        }
        
        if n_mut > 1 {
            result = Self::lcm(result, n_mut - 1);
        }
        
        result
    }

    /// Least Common Multiple.
    pub fn lcm(a: i64, b: i64) -> i64 {
        let (gcd, _, _) = Modulus::extended_gcd(a.abs(), b.abs());
        (a.abs() / gcd) * b.abs()
    }
}

/// Division algorithms.
pub struct Division;

impl Division {
    /// Integer division with remainder: returns (quotient, remainder).
    pub fn div_rem(x: i64, m: i64) -> MathResult<(i64, i64)> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let q = x / m;
        let r = x % m;
        Ok((q, r))
    }

    /// Euclidean division: always non-negative remainder.
    pub fn euclidean_division(x: i64, m: i64) -> MathResult<(i64, i64)> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let q = (x as f64 / m as f64).floor() as i64;
        let r = x - q * m;
        Ok((q, r))
    }

    /// Division with rounding to nearest.
    pub fn div_round(x: i64, m: i64) -> MathResult<i64> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let q = (x as f64 / m as f64).round() as i64;
        Ok(q)
    }

    /// Division with ceiling.
    pub fn div_ceil(x: i64, m: i64) -> MathResult<i64> {
        Modulus::ceil_div(x, m)
    }

    /// Division with floor.
    pub fn div_floor(x: i64, m: i64) -> MathResult<i64> {
        Modulus::floor_div(x, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_modulus() {
        assert_eq!(Modulus::euclidean(17, 5).unwrap(), 2);
        assert_eq!(Modulus::euclidean(-17, 5).unwrap(), 3);
        assert_eq!(Modulus::euclidean(17, -5).unwrap(), 2);
    }

    #[test]
    fn test_floored_modulus() {
        assert_eq!(Modulus::floored(17, 5).unwrap(), 2);
        assert_eq!(Modulus::floored(-17, 5).unwrap(), 3);
        assert_eq!(Modulus::floored(17, -5).unwrap(), -3);
    }

    #[test]
    fn test_floor_div() {
        assert_eq!(Modulus::floor_div(17, 5).unwrap(), 3);
        assert_eq!(Modulus::floor_div(-17, 5).unwrap(), -4);
        assert_eq!(Modulus::floor_div(17, -5).unwrap(), -4);
    }

    #[test]
    fn test_ceil_div() {
        assert_eq!(Modulus::ceil_div(17, 5).unwrap(), 4);
        assert_eq!(Modulus::ceil_div(-17, 5).unwrap(), -3);
    }

    #[test]
    fn test_mod_arithmetic() {
        assert_eq!(Modulus::mod_add(7, 5, 10).unwrap(), 2);
        assert_eq!(Modulus::mod_sub(7, 5, 10).unwrap(), 2);
        assert_eq!(Modulus::mod_mul(7, 5, 10).unwrap(), 5);
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(Modulus::mod_pow(2, 10, 1000).unwrap(), 24);
    }

    #[test]
    fn test_mod_inverse() {
        assert_eq!(Modulus::mod_inverse(3, 7).unwrap(), 5); // 3*5 = 15 ≡ 1 (mod 7)
    }

    #[test]
    fn test_chinese_remainder() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5) => x = 8
        let result = Modulus::chinese_remainder(2, 3, 3, 5).unwrap();
        assert_eq!(result, 8);
    }

    #[test]
    fn test_euler_totient() {
        assert_eq!(ModularArithmetic::euler_totient(10), 4); // 1,3,7,9
        assert_eq!(ModularArithmetic::euler_totient(7), 6);
    }

    #[test]
    fn test_are_coprime() {
        assert!(ModularArithmetic::are_coprime(8, 15));
        assert!(!ModularArithmetic::are_coprime(8, 12));
    }

    #[test]
    fn test_division() {
        assert_eq!(Division::div_rem(17, 5).unwrap(), (3, 2));
        assert_eq!(Division::euclidean_division(17, 5).unwrap(), (3, 2));
        assert_eq!(Division::div_round(18, 5).unwrap(), 4);
    }
}
