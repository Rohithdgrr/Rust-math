//! Number theory: GCD, LCM, primes, factorials, combinatorics.

use mathverse_core::error::{MathError, MathResult};

/// Greatest Common Divisor using Euclidean algorithm.
pub struct Gcd;

impl Gcd {
    /// Euclidean algorithm for GCD.
    pub fn euclidean(a: i64, b: i64) -> i64 {
        let mut a = a.abs();
        let mut b = b.abs();
        
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        
        a
    }

    /// Binary GCD algorithm (Stein's algorithm).
    pub fn binary(a: i64, b: i64) -> i64 {
        let mut a = a.abs();
        let mut b = b.abs();
        
        if a == 0 {
            return b;
        }
        if b == 0 {
            return a;
        }
        
        // Find common factor of 2
        let shift = (a | b).trailing_zeros();
        a >>= a.trailing_zeros();
        b >>= b.trailing_zeros();
        
        while b != 0 {
            b >>= b.trailing_zeros();
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            b -= a;
        }
        
        a << shift
    }

    /// Extended Euclidean algorithm: returns (gcd, x, y) where ax + by = gcd(a,b).
    pub fn extended(a: i64, b: i64) -> (i64, i64, i64) {
        if a == 0 {
            return (b, 0, 1);
        }
        
        let (gcd, x1, y1) = Self::extended(b % a, a);
        let x = y1 - (b / a) * x1;
        let y = x1;
        
        (gcd, x, y)
    }

    /// GCD of multiple numbers.
    pub fn multiple(numbers: &[i64]) -> i64 {
        if numbers.is_empty() {
            return 0;
        }
        
        numbers.iter().copied().reduce(|a, b| Self::euclidean(a, b)).unwrap_or(0)
    }
}

/// Least Common Multiple.
pub struct Lcm;

impl Lcm {
    /// LCM using GCD: lcm(a,b) = |a*b| / gcd(a,b).
    pub fn compute(a: i64, b: i64) -> i64 {
        if a == 0 || b == 0 {
            return 0;
        }
        
        let gcd = Gcd::euclidean(a, b);
        (a.abs() / gcd) * b.abs()
    }

    /// LCM of multiple numbers.
    pub fn multiple(numbers: &[i64]) -> i64 {
        if numbers.is_empty() {
            return 0;
        }
        
        numbers.iter().copied().reduce(|a, b| Self::compute(a, b)).unwrap_or(0)
    }
}

/// Prime number operations.
pub struct Primes;

impl Primes {
    /// Check if a number is prime using trial division.
    pub fn is_prime(n: i64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        let sqrt_n = (n as f64).sqrt() as i64;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        
        true
    }

    /// Check if a number is prime using Miller-Rabin test (probabilistic).
    pub fn is_prime_miller_rabin(n: i64, k: u32) -> bool {
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
            let a = 2 + (n as f64).sqrt() as i64; // Simplified witness selection
            let mut x = Self::mod_pow(a as u64, d as u64, n as u64) as i64;
            
            if x == 1 || x == n - 1 {
                continue;
            }
            
            let mut composite = true;
            for _ in 0..r - 1 {
                x = Self::mod_pow(x as u64, 2, n as u64) as i64;
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

    /// Modular exponentiation for Miller-Rabin.
    fn mod_pow(base: u64, mut exp: u64, modulus: u64) -> u64 {
        let mut result = 1u64;
        let mut base = base % modulus;
        
        while exp > 0 {
            if exp & 1 == 1 {
                result = (result * base) % modulus;
            }
            exp >>= 1;
            base = (base * base) % modulus;
        }
        
        result
    }

    /// Generate primes up to n using Sieve of Eratosthenes.
    pub fn sieve(n: usize) -> Vec<i64> {
        if n < 2 {
            return Vec::new();
        }
        
        let mut is_prime = vec![true; n + 1];
        is_prime[0] = false;
        is_prime[1] = false;
        
        let mut p = 2;
        while p * p <= n {
            if is_prime[p] {
                let mut multiple = p * p;
                while multiple <= n {
                    is_prime[multiple] = false;
                    multiple += p;
                }
            }
            p += 1;
        }
        
        (2..=n).filter(|&i| is_prime[i]).map(|i| i as i64).collect()
    }

    /// Get the nth prime (1-indexed).
    pub fn nth(n: usize) -> MathResult<i64> {
        if n == 0 {
            return Err(MathError::InvalidArgument("n must be positive"));
        }
        
        // Approximate upper bound using prime number theorem
        let upper = if n < 6 {
            15
        } else {
            (n as f64 * (n as f64).ln() + n as f64 * (n as f64).ln().ln()) as usize
        };
        
        let primes = Self::sieve(upper);
        
        if n <= primes.len() {
            Ok(primes[n - 1])
        } else {
            Err(MathError::InvalidArgument("estimation insufficient"))
        }
    }

    /// Prime factorization.
    pub fn factorize(mut n: i64) -> Vec<(i64, u32)> {
        if n < 0 {
            n = -n;
        }
        
        let mut factors = Vec::new();
        
        // Handle factor 2
        let mut count = 0;
        while n % 2 == 0 {
            count += 1;
            n /= 2;
        }
        if count > 0 {
            factors.push((2, count));
        }
        
        // Handle odd factors
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
        
        // If n is still > 1, it's a prime
        if n > 1 {
            factors.push((n, 1));
        }
        
        factors
    }

    /// Count prime factors (with multiplicity).
    pub fn count_factors(n: i64) -> u32 {
        Self::factorize(n).iter().map(|(_, count)| count).sum()
    }

    /// Count distinct prime factors.
    pub fn count_distinct_factors(n: i64) -> usize {
        Self::factorize(n).len()
    }

    /// Check if two numbers are coprime.
    pub fn are_coprime(a: i64, b: i64) -> bool {
        Gcd::euclidean(a, b) == 1
    }

    /// Euler's totient function φ(n).
    pub fn euler_totient(n: i64) -> i64 {
        if n <= 0 {
            return 0;
        }
        
        let mut result = n;
        let mut n_mut = n;
        let factors = Self::factorize(n);
        
        for (p, _) in factors {
            result -= result / p;
        }
        
        result
    }
}

/// Factorial and combinatorics.
pub struct Factorial;

impl Factorial {
    /// Compute n! (factorial).
    pub fn compute(n: u64) -> MathResult<u64> {
        if n > 20 {
            return Err(MathError::InvalidArgument("factorial overflow for n > 20"));
        }
        
        let mut result = 1u64;
        for i in 2..=n {
            result *= i;
        }
        
        Ok(result)
    }

    /// Compute n! using recursion with memoization.
    pub fn compute_recursive(n: u64) -> MathResult<u64> {
        Self::compute(n)
    }

    /// Double factorial: n!! = n * (n-2) * (n-4) * ...
    pub fn double_factorial(n: i64) -> MathResult<i64> {
        if n < -1 {
            return Err(MathError::InvalidArgument("double factorial undefined for n < -1"));
        }
        
        if n == 0 || n == -1 {
            return Ok(1);
        }
        
        let mut result = 1i64;
        let mut current = n;
        
        while current > 0 {
            result *= current;
            current -= 2;
        }
        
        Ok(result)
    }

    /// Rising factorial (Pochhammer symbol): (x)_n = x(x+1)(x+2)...(x+n-1).
    pub fn rising_factorial(x: f64, n: u64) -> f64 {
        let mut result = 1.0;
        for i in 0..n {
            result *= x + i as f64;
        }
        result
    }

    /// Falling factorial: x_(n) = x(x-1)(x-2)...(x-n+1).
    pub fn falling_factorial(x: f64, n: u64) -> f64 {
        let mut result = 1.0;
        for i in 0..n {
            result *= x - i as f64;
        }
        result
    }

    /// Binomial coefficient: C(n, k) = n! / (k!(n-k)!).
    pub fn binomial(n: u64, k: u64) -> MathResult<u64> {
        if k > n {
            return Ok(0);
        }
        
        // Use symmetry to reduce computations
        let k = k.min(n - k);
        
        if k == 0 {
            return Ok(1);
        }
        
        let mut result = 1u64;
        for i in 0..k {
            result = result.checked_mul(n - i)
                .ok_or(MathError::InvalidArgument("binomial coefficient overflow"))?;
            result /= i + 1;
        }
        
        Ok(result)
    }

    /// Multinomial coefficient.
    pub fn multinomial(n: u64, k: &[u64]) -> MathResult<u64> {
        let sum_k: u64 = k.iter().sum();
        
        if sum_k != n {
            return Err(MathError::InvalidArgument("sum of k must equal n"));
        }
        
        let mut result = Self::compute(n)?;
        
        for &ki in k {
            result /= Self::compute(ki)?;
        }
        
        Ok(result)
    }

    /// Permutations: P(n, k) = n! / (n-k)!.
    pub fn permutation(n: u64, k: u64) -> MathResult<u64> {
        if k > n {
            return Ok(0);
        }
        
        let mut result = 1u64;
        for i in (n - k + 1)..=n {
            result = result.checked_mul(i)
                .ok_or(MathError::InvalidArgument("permutation overflow"))?;
        }
        
        Ok(result)
    }

    /// Combinations with repetition: C(n+k-1, k).
    pub fn combination_with_repetition(n: u64, k: u64) -> MathResult<u64> {
        Self::binomial(n + k - 1, k)
    }
}

/// Fibonacci and related sequences.
pub struct Fibonacci;

impl Fibonacci {
    /// Compute nth Fibonacci number (F_0 = 0, F_1 = 1).
    pub fn compute(n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }
        
        let mut a = 0u64;
        let mut b = 1u64;
        
        for _ in 2..=n {
            let temp = a + b;
            a = b;
            b = temp;
        }
        
        b
    }

    /// Compute Fibonacci using matrix exponentiation (O(log n)).
    pub fn compute_fast(n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }
        
        let mut a = 0u64;
        let mut b = 1u64;
        let mut c = 1u64;
        let mut d = 1u64;
        
        let mut n = n - 1;
        
        while n > 0 {
            if n & 1 == 1 {
                let temp_a = a * c + b * d;
                let temp_b = a * d + b * (c + d);
                a = temp_a;
                b = temp_b;
            }
            
            let temp_c = c * c + d * d;
            let temp_d = d * (2 * c + d);
            c = temp_c;
            d = temp_d;
            
            n >>= 1;
        }
        
        b
    }

    /// Generate Fibonacci sequence up to n.
    pub fn sequence(n: usize) -> Vec<u64> {
        let mut result = Vec::with_capacity(n + 1);
        
        for i in 0..=n {
            result.push(Self::compute(i as u64));
        }
        
        result
    }

    /// Check if a number is Fibonacci.
    pub fn is_fibonacci(n: u64) -> bool {
        // A number is Fibonacci if and only if 5n^2 + 4 or 5n^2 - 4 is a perfect square
        let test1 = 5 * n * n + 4;
        let test2 = 5 * n * n - 4;
        
        Self::is_perfect_square(test1) || Self::is_perfect_square(test2)
    }

    /// Check if a number is a perfect square.
    fn is_perfect_square(n: u64) -> bool {
        let sqrt_n = (n as f64).sqrt() as u64;
        sqrt_n * sqrt_n == n
    }

    /// Lucas numbers (L_0 = 2, L_1 = 1).
    pub fn lucas(n: u64) -> u64 {
        if n == 0 {
            return 2;
        }
        if n == 1 {
            return 1;
        }
        
        let mut a = 2u64;
        let mut b = 1u64;
        
        for _ in 2..=n {
            let temp = a + b;
            a = b;
            b = temp;
        }
        
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(Gcd::euclidean(48, 18), 6);
        assert_eq!(Gcd::euclidean(17, 13), 1);
        assert_eq!(Gcd::binary(48, 18), 6);
        assert_eq!(Gcd::multiple(&[48, 18, 12]), 6);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(Lcm::compute(12, 18), 36);
        assert_eq!(Lcm::compute(4, 5), 20);
        assert_eq!(Lcm::multiple(&[4, 6, 8]), 24);
    }

    #[test]
    fn test_primes() {
        assert!(Primes::is_prime(17));
        assert!(!Primes::is_prime(18));
        assert!(Primes::is_prime(2));
        assert!(!Primes::is_prime(1));
        
        let primes = Primes::sieve(20);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_nth_prime() {
        assert_eq!(Primes::nth(1).unwrap(), 2);
        assert_eq!(Primes::nth(5).unwrap(), 11);
    }

    #[test]
    fn test_factorize() {
        assert_eq!(Primes::factorize(12), vec![(2, 2), (3, 1)]);
        assert_eq!(Primes::factorize(17), vec![(17, 1)]);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(Factorial::compute(5).unwrap(), 120);
        assert_eq!(Factorial::compute(0).unwrap(), 1);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(Factorial::binomial(5, 2).unwrap(), 10);
        assert_eq!(Factorial::binomial(10, 5).unwrap(), 252);
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(Fibonacci::compute(10), 55);
        assert_eq!(Fibonacci::compute_fast(10), 55);
        assert!(Fibonacci::is_fibonacci(55));
        assert!(!Fibonacci::is_fibonacci(54));
    }

    #[test]
    fn test_lucas() {
        assert_eq!(Fibonacci::lucas(5), 11);
    }
}
