//! Sequence operations: arithmetic, geometric, and other sequences.

use mathverse_core::error::{MathError, MathResult};

/// Arithmetic sequence: a, a+d, a+2d, a+3d, ...
pub struct ArithmeticSequence;

impl ArithmeticSequence {
    /// Get nth term: a_n = a + (n-1)d.
    pub fn nth_term(a: f64, d: f64, n: usize) -> f64 {
        a + (n - 1) as f64 * d
    }

    /// Sum of first n terms: S_n = n/2 * (2a + (n-1)d).
    pub fn sum(a: f64, d: f64, n: usize) -> f64 {
        n as f64 / 2.0 * (2.0 * a + (n - 1) as f64 * d)
    }

    /// Sum from term m to term n.
    pub fn sum_range(a: f64, d: f64, m: usize, n: usize) -> f64 {
        if n < m {
            return 0.0;
        }
        Self::sum(a, d, n) - Self::sum(a, d, m - 1)
    }

    /// Get common difference from two terms.
    pub fn common_difference(term1: f64, term2: f64, n1: usize, n2: usize) -> f64 {
        (term2 - term1) / (n2 - n1) as f64
    }

    /// Get first term from nth term and common difference.
    pub fn first_term(nth_term: f64, d: f64, n: usize) -> f64 {
        nth_term - (n - 1) as f64 * d
    }

    /// Check if sequence is arithmetic.
    pub fn is_arithmetic(terms: &[f64]) -> bool {
        if terms.len() < 2 {
            return true;
        }
        
        let d = terms[1] - terms[0];
        terms.windows(2).all(|w| (w[1] - w[0] - d).abs() < 1e-10)
    }

    /// Generate sequence.
    pub fn generate(a: f64, d: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| a + i as f64 * d).collect()
    }
}

/// Geometric sequence: a, ar, ar², ar³, ...
pub struct GeometricSequence;

impl GeometricSequence {
    /// Get nth term: a_n = a * r^(n-1).
    pub fn nth_term(a: f64, r: f64, n: usize) -> f64 {
        a * r.powi((n - 1) as i32)
    }

    /// Sum of first n terms: S_n = a(1-r^n) / (1-r) for r ≠ 1.
    pub fn sum(a: f64, r: f64, n: usize) -> f64 {
        if r == 1.0 {
            a * n as f64
        } else {
            a * (1.0 - r.powi(n as i32)) / (1.0 - r)
        }
    }

    /// Infinite sum: S = a / (1-r) for |r| < 1.
    pub fn infinite_sum(a: f64, r: f64) -> MathResult<f64> {
        if r.abs() >= 1.0 {
            return Err(MathError::InvalidArgument("|r| must be < 1 for convergence"));
        }
        Ok(a / (1.0 - r))
    }

    /// Sum from term m to term n.
    pub fn sum_range(a: f64, r: f64, m: usize, n: usize) -> f64 {
        if n < m {
            return 0.0;
        }
        Self::sum(a, r, n) - Self::sum(a, r, m - 1)
    }

    /// Get common ratio from two terms.
    pub fn common_ratio(term1: f64, term2: f64, n1: usize, n2: usize) -> f64 {
        (term2 / term1).powf(1.0 / (n2 - n1) as f64)
    }

    /// Get first term from nth term and common ratio.
    pub fn first_term(nth_term: f64, r: f64, n: usize) -> f64 {
        nth_term / r.powi((n - 1) as i32)
    }

    /// Check if sequence is geometric.
    pub fn is_geometric(terms: &[f64]) -> bool {
        if terms.len() < 2 {
            return true;
        }
        
        if terms[0] == 0.0 {
            return terms.iter().all(|&t| t == 0.0);
        }
        
        let r = terms[1] / terms[0];
        terms.windows(2).all(|w| (w[1] / w[0] - r).abs() < 1e-10)
    }

    /// Generate sequence.
    pub fn generate(a: f64, r: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| a * r.powi(i as i32)).collect()
    }
}

/// Harmonic sequence: 1, 1/2, 1/3, 1/4, ...
pub struct HarmonicSequence;

impl HarmonicSequence {
    /// Get nth term: H_n = 1/n.
    pub fn nth_term(n: usize) -> f64 {
        1.0 / n as f64
    }

    /// Harmonic number H_n = 1 + 1/2 + 1/3 + ... + 1/n.
    pub fn harmonic_number(n: usize) -> f64 {
        (1..=n).map(|i| 1.0 / i as f64).sum()
    }

    /// Approximate harmonic number using ln(n) + γ + 1/(2n).
    pub fn harmonic_number_approx(n: usize) -> f64 {
        let n_f = n as f64;
        let gamma = 0.57721566490153286060651209008240243104215933593992;
        n_f.ln() + gamma + 1.0 / (2.0 * n_f)
    }

    /// Generate harmonic sequence.
    pub fn generate(n: usize) -> Vec<f64> {
        (1..=n).map(|i| 1.0 / i as f64).collect()
    }
}

/// Fibonacci sequence: F_0=0, F_1=1, F_n=F_{n-1}+F_{n-2}.
pub struct FibonacciSequence;

impl FibonacciSequence {
    /// Get nth Fibonacci number using Binet's formula.
    pub fn nth_term(n: usize) -> u64 {
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }
        
        let sqrt5 = 5.0_f64.sqrt();
        let phi = (1.0 + sqrt5) / 2.0;
        let psi = (1.0 - sqrt5) / 2.0;
        
        ((phi.powi(n as i32) - psi.powi(n as i32)) / sqrt5).round() as u64
    }

    /// Generate Fibonacci sequence.
    pub fn generate(n: usize) -> Vec<u64> {
        (0..n).map(|i| Self::nth_term(i)).collect()
    }

    /// Check if number is Fibonacci.
    pub fn is_fibonacci(n: u64) -> bool {
        // A number is Fibonacci if and only if 5n² ± 4 is a perfect square
        let test1 = 5 * n * n + 4;
        let test2 = 5 * n * n - 4;
        
        Self::is_perfect_square(test1) || Self::is_perfect_square(test2)
    }

    fn is_perfect_square(n: u64) -> bool {
        let sqrt_n = (n as f64).sqrt() as u64;
        sqrt_n * sqrt_n == n
    }

    /// Golden ratio approximation from consecutive Fibonacci numbers.
    pub fn golden_ratio_approximation(n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }
        
        let fn_1 = Self::nth_term(n);
        let fn_2 = Self::nth_term(n - 1);
        
        fn_1 as f64 / fn_2 as f64
    }
}

/// Lucas sequence: L_0=2, L_1=1, L_n=L_{n-1}+L_{n-2}.
pub struct LucasSequence;

impl LucasSequence {
    /// Get nth Lucas number recurrence.
    pub fn nth_term(n: usize) -> u64 {
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

    /// Generate Lucas sequence.
    pub fn generate(n: usize) -> Vec<u64> {
        (0..n).map(|i| Self::nth_term(i)).collect()
    }
}

/// Triangular numbers: T_n = n(n+1)/2.
pub struct TriangularNumbers;

impl TriangularNumbers {
    /// Get nth triangular number.
    pub fn nth_term(n: usize) -> usize {
        n * (n + 1) / 2
    }

    /// Check if number is triangular.
    pub fn is_triangular(n: usize) -> bool {
        let discriminant = 8 * n + 1;
        let sqrt_disc = (discriminant as f64).sqrt() as usize;
        
        sqrt_disc * sqrt_disc == discriminant && (sqrt_disc - 1) % 2 == 0
    }

    /// Generate triangular numbers.
    pub fn generate(n: usize) -> Vec<usize> {
        (1..=n).map(|i| Self::nth_term(i)).collect()
    }

    /// Sum of first n triangular numbers.
    pub fn sum(n: usize) -> usize {
        n * (n + 1) * (n + 2) / 6
    }
}

/// Square numbers: S_n = n².
pub struct SquareNumbers;

impl SquareNumbers {
    /// Get nth square number.
    pub fn nth_term(n: usize) -> usize {
        n * n
    }

    /// Check if number is square.
    pub fn is_square(n: usize) -> bool {
        let sqrt_n = (n as f64).sqrt() as usize;
        sqrt_n * sqrt_n == n
    }

    /// Generate square numbers.
    pub fn generate(n: usize) -> Vec<usize> {
        (1..=n).map(|i| Self::nth_term(i)).collect()
    }

    /// Sum of first n square numbers.
    pub fn sum(n: usize) -> usize {
        n * (n + 1) * (2 * n + 1) / 6
    }
}

/// Pentagonal numbers: P_n = n(3n-1)/2.
pub struct PentagonalNumbers;

impl PentagonalNumbers {
    /// Get nth pentagonal number.
    pub fn nth_term(n: usize) -> usize {
        n * (3 * n - 1) / 2
    }

    /// Check if number is pentagonal.
    pub fn is_pentagonal(n: usize) -> bool {
        let discriminant = 24 * n + 1;
        let sqrt_disc = (discriminant as f64).sqrt() as usize;
        
        sqrt_disc * sqrt_disc == discriminant && (sqrt_disc + 1) % 6 == 0
    }

    /// Generate pentagonal numbers.
    pub fn generate(n: usize) -> Vec<usize> {
        (1..=n).map(|i| Self::nth_term(i)).collect()
    }

    /// Sum of first n pentagonal numbers.
    pub fn sum(n: usize) -> usize {
        n * (n + 1) * (3 * n + 2) / 6
    }
}

/// Hexagonal numbers: H_n = n(2n-1).
pub struct HexagonalNumbers;

impl HexagonalNumbers {
    /// Get nth hexagonal number.
    pub fn nth_term(n: usize) -> usize {
        n * (2 * n - 1)
    }

    /// Check if number is hexagonal.
    pub fn is_hexagonal(n: usize) -> bool {
        let discriminant = 8 * n + 1;
        let sqrt_disc = (discriminant as f64).sqrt() as usize;
        
        sqrt_disc * sqrt_disc == discriminant && (sqrt_disc + 1) % 4 == 0
    }

    /// Generate hexagonal numbers.
    pub fn generate(n: usize) -> Vec<usize> {
        (1..=n).map(|i| Self::nth_term(i)).collect()
    }

    /// Sum of first n hexagonal numbers.
    pub fn sum(n: usize) -> usize {
        n * (2 * n - 1) * (2 * n + 1) / 3
    }
}

/// Catalan numbers: C_n = (2n)! / ((n+1)!n!).
pub struct CatalanNumbers;

impl CatalanNumbers {
    /// Get nth Catalan number using recurrence.
    pub fn nth_term(n: usize) -> u64 {
        if n == 0 {
            return 1;
        }
        
        let mut catalan = vec![1u64; n + 1];
        
        for i in 1..=n {
            catalan[i] = 0;
            for j in 0..i {
                catalan[i] += catalan[j] * catalan[i - 1 - j];
            }
        }
        
        catalan[n]
    }

    /// Get nth Catalan number using binomial coefficient.
    pub fn nth_term_binomial(n: usize) -> u64 {
        // C_n = (2n choose n) / (n+1)
        let binom = crate::number_theory::Factorial::binomial(2 * n as u64, n as u64).unwrap();
        binom / (n + 1) as u64
    }

    /// Generate Catalan numbers.
    pub fn generate(n: usize) -> Vec<u64> {
        (0..n).map(|i| Self::nth_term(i)).collect()
    }

    /// Check if number is Catalan.
    pub fn is_catalan(n: u64) -> bool {
        for i in 0..=20 {
            if Self::nth_term(i) == n {
                return true;
            }
        }
        false
    }
}

/// Prime numbers sequence.
pub struct PrimeSequence;

impl PrimeSequence {
    /// Get nth prime.
    pub fn nth_prime(n: usize) -> MathResult<usize> {
        if n == 0 {
            return Err(MathError::InvalidArgument("n must be positive"));
        }
        
        let mut count = 0;
        let mut candidate = 2;
        
        loop {
            if Self::is_prime(candidate) {
                count += 1;
                if count == n {
                    return Ok(candidate);
                }
            }
            candidate += 1;
        }
    }

    /// Check if number is prime.
    pub fn is_prime(n: usize) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        let sqrt_n = (n as f64).sqrt() as usize;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        
        true
    }

    /// Generate first n primes.
    pub fn generate(n: usize) -> Vec<usize> {
        let mut primes = Vec::new();
        let mut candidate = 2;
        
        while primes.len() < n {
            if Self::is_prime(candidate) {
                primes.push(candidate);
            }
            candidate += 1;
        }
        
        primes
    }

    /// Prime gap between consecutive primes.
    pub fn prime_gap(n: usize) -> MathResult<usize> {
        let p1 = Self::nth_prime(n)?;
        let p2 = Self::nth_prime(n + 1)?;
        Ok(p2 - p1)
    }
}

/// Sequence operations and utilities.
pub struct SequenceOps;

impl SequenceOps {
    /// Check if sequence is monotonic increasing.
    pub fn is_monotonic_increasing(seq: &[f64]) -> bool {
        seq.windows(2).all(|w| w[0] <= w[1])
    }

    /// Check if sequence is monotonic decreasing.
    pub fn is_monotonic_decreasing(seq: &[f64]) -> bool {
        seq.windows(2).all(|w| w[0] >= w[1])
    }

    /// Check if sequence is strictly increasing.
    pub fn is_strictly_increasing(seq: &[f64]) -> bool {
        seq.windows(2).all(|w| w[0] < w[1])
    }

    /// Check if sequence is strictly decreasing.
    pub fn is_strictly_decreasing(seq: &[f64]) -> bool {
        seq.windows(2).all(|w| w[0] > w[1])
    }

    /// Check if sequence is periodic.
    pub fn is_periodic(seq: &[f64], max_period: usize) -> bool {
        if seq.len() < 2 {
            return false;
        }
        
        for period in 1..=max_period.min(seq.len() / 2) {
            let is_period = (period..seq.len())
                .all(|i| (seq[i] - seq[i - period]).abs() < 1e-10);
            
            if is_period {
                return true;
            }
        }
        
        false
    }

    /// Find period of sequence.
    pub fn find_period(seq: &[f64], max_period: usize) -> Option<usize> {
        for period in 1..=max_period.min(seq.len() / 2) {
            let is_period = (period..seq.len())
                .all(|i| (seq[i] - seq[i - period]).abs() < 1e-10);
            
            if is_period {
                return Some(period);
            }
        }
        
        None
    }

    /// Compute differences between consective terms.
    pub fn differences(seq: &[f64]) -> Vec<f64> {
        seq.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Compute nth differences.
    pub fn nth_differences(seq: &[f64], n: usize) -> Vec<f64> {
        let mut result = seq.to_vec();
        
        for _ in 0..n {
            result = Self::differences(&result);
            if result.len() < 2 {
                break;
            }
        }
        
        result
    }

    /// Check if sequence is arithmetic using differences.
    pub fn is_arithmetic_by_differences(seq: &[f64]) -> bool {
        let diffs = Self::differences(seq);
        diffs.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10))
    }

    /// Check if sequence is geometric using ratios.
    pub fn is_geometric_by_ratios(seq: &[f64]) -> bool {
        if seq.iter().any(|&x| x == 0.0) {
            return false;
        }
        
        let ratios: Vec<f64> = seq.windows(2).map(|w| w[1] / w[0]).collect();
        ratios.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10))
    }

    /// Partial sum of sequence.
    pub fn partial_sums(seq: &[f64]) -> Vec<f64> {
        let mut sums = Vec::with_capacity(seq.len());
        let mut running_sum = 0.0;
        
        for &value in seq {
            running_sum += value;
            sums.push(running_sum);
        }
        
        sums
    }

    /// Partial products of sequence.
    pub fn partial_products(seq: &[f64]) -> Vec<f64> {
        let mut products = Vec::with_capacity(seq.len());
        let mut running_product = 1.0;
        
        for &value in seq {
            running_product *= value;
            products.push(running_product);
        }
        
        products
    }

    /// Moving average of sequence.
    pub fn moving_average(seq: &[f64], window: usize) -> Vec<f64> {
        if window == 0 || window > seq.len() {
            return Vec::new();
        }
        
        let mut averages = Vec::new();
        
        for i in 0..=seq.len() - window {
            let sum: f64 = seq[i..i + window].iter().sum();
            averages.push(sum / window as f64);
        }
        
        averages
    }

    /// Cumulative moving average.
    pub fn cumulative_moving_average(seq: &[f64]) -> Vec<f64> {
        let mut averages = Vec::with_capacity(seq.len());
        let mut running_sum = 0.0;
        
        for (i, &value) in seq.iter().enumerate() {
            running_sum += value;
            averages.push(running_sum / (i + 1) as f64);
        }
        
        averages
    }

    /// Find limit of sequence (if convergent).
    pub fn limit(seq: &[f64], tolerance: f64) -> Option<f64> {
        if seq.len() < 2 {
            return None;
        }
        
        let last = seq[seq.len() - 1];
        let second_last = seq[seq.len() - 2];
        
        if (last - second_last).abs() < tolerance {
            Some(last)
        } else {
            None
        }
    }

    /// Check if sequence converges.
    pub fn converges(seq: &[f64], tolerance: f64) -> bool {
        Self::limit(seq, tolerance).is_some()
    }

    /// Rate of convergence.
    pub fn convergence_rate(seq: &[f64]) -> f64 {
        if seq.len() < 3 {
            return 0.0;
        }
        
        let limit = seq[seq.len() - 1];
        let error1 = (seq[seq.len() - 2] - limit).abs();
        let error2 = (seq[seq.len() - 3] - limit).abs();
        
        if error1 == 0.0 {
            return f64::INFINITY;
        }
        
        error1 / error2
    }
}

/// Recurrence relations.
pub struct RecurrenceRelations;

impl RecurrenceRelations {
    /// Solve linear homogeneous recurrence: a_n = c1*a_{n-1} + c2*a_{n-2}.
    pub fn linear_homogeneous(c1: f64, c2: f64, a0: f64, a1: f64, n: usize) -> f64 {
        if n == 0 {
            return a0;
        }
        if n == 1 {
            return a1;
        }
        
        let mut a_prev2 = a0;
        let mut a_prev1 = a1;
        
        for _ in 2..=n {
            let a_current = c1 * a_prev1 + c2 * a_prev2;
            a_prev2 = a_prev1;
            a_prev1 = a_current;
        }
        
        a_prev1
    }

    /// Solve linear non-homogeneous recurrence: a_n = c1*a_{n-1} + c2*a_{n-2} + f(n).
    pub fn linear_nonhomogeneous(
        c1: f64,
        c2: f64,
        f: impl Fn(usize) -> f64,
        a0: f64,
        a1: f64,
        n: usize,
    ) -> f64 {
        if n == 0 {
            return a0;
        }
        if n == 1 {
            return a1;
        }
        
        let mut a_prev2 = a0;
        let mut a_prev1 = a1;
        
        for i in 2..=n {
            let a_current = c1 * a_prev1 + c2 * a_prev2 + f(i);
            a_prev2 = a_prev1;
            a_prev1 = a_current;
        }
        
        a_prev1
    }

    /// General recurrence: a_n = f(a_{n-1}, a_{n-2}, ..., a_{n-k}).
    pub fn general(
        k: usize,
        f: impl Fn(&[f64]) -> f64,
        initial: &[f64],
        n: usize,
    ) -> f64 {
        if n < initial.len() {
            return initial[n];
        }
        
        let mut seq = initial.to_vec();
        
        for i in initial.len()..=n {
            let window = &seq[seq.len() - k..];
            let next = f(window);
            seq.push(next);
        }
        
        seq[n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_sequence() {
        assert_eq!(ArithmeticSequence::nth_term(1.0, 2.0, 5), 9.0);
        assert_eq!(ArithmeticSequence::sum(1.0, 2.0, 5), 25.0);
        assert!(ArithmeticSequence::is_arithmetic(&[1.0, 3.0, 5.0, 7.0]));
    }

    #[test]
    fn test_geometric_sequence() {
        assert_eq!(GeometricSequence::nth_term(2.0, 3.0, 4), 54.0);
        assert_eq!(GeometricSequence::sum(2.0, 3.0, 4), 80.0);
        assert!(GeometricSequence::is_geometric(&[2.0, 6.0, 18.0, 54.0]));
    }

    #[test]
    fn test_harmonic_sequence() {
        assert_eq!(HarmonicSequence::nth_term(5), 0.2);
        assert!((HarmonicSequence::harmonic_number(10) - 2.928968254).abs() < 1e-8);
    }

    #[test]
    fn test_fibonacci_sequence() {
        assert_eq!(FibonacciSequence::nth_term(10), 55);
        assert!(FibonacciSequence::is_fibonacci(55));
        assert!(!FibonacciSequence::is_fibonacci(54));
    }

    #[test]
    fn test_lucas_sequence() {
        assert_eq!(LucasSequence::nth_term(5), 11);
    }

    #[test]
    fn test_triangular_numbers() {
        assert_eq!(TriangularNumbers::nth_term(5), 15);
        assert!(TriangularNumbers::is_triangular(15));
        assert!(!TriangularNumbers::is_triangular(14));
    }

    #[test]
    fn test_square_numbers() {
        assert_eq!(SquareNumbers::nth_term(5), 25);
        assert!(SquareNumbers::is_square(25));
        assert!(!SquareNumbers::is_square(26));
    }

    #[test]
    fn test_pentagonal_numbers() {
        assert_eq!(PentagonalNumbers::nth_term(5), 35);
        assert!(PentagonalNumbers::is_pentagonal(35));
    }

    #[test]
    fn test_hexagonal_numbers() {
        assert_eq!(HexagonalNumbers::nth_term(5), 45);
        assert!(HexagonalNumbers::is_hexagonal(45));
    }

    #[test]
    fn test_catalan_numbers() {
        assert_eq!(CatalanNumbers::nth_term(5), 42);
        assert_eq!(CatalanNumbers::nth_term_binomial(5), 42);
    }

    #[test]
    fn test_prime_sequence() {
        assert_eq!(PrimeSequence::nth_prime(1).unwrap(), 2);
        assert_eq!(PrimeSequence::nth_prime(5).unwrap(), 11);
        assert!(PrimeSequence::is_prime(17));
        assert!(!PrimeSequence::is_prime(18));
    }

    #[test]
    fn test_sequence_ops() {
        assert!(SequenceOps::is_monotonic_increasing(&[1.0, 2.0, 3.0, 4.0]));
        assert!(SequenceOps::is_periodic(&[1.0, 2.0, 1.0, 2.0], 2));
        assert_eq!(SequenceOps::differences(&[1.0, 3.0, 6.0, 10.0]), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_recurrence_relations() {
        let result = RecurrenceRelations::linear_homogeneous(1.0, 1.0, 0.0, 1.0, 10);
        assert_eq!(result, 55.0); // Fibonacci
    }
}
