//! Combinatorics: permutations, combinations, partitions, and related functions.

/// Combinatorial operations.
pub struct Combinatorics;

impl Combinatorics {
    /// Factorial n!.
    pub fn factorial(n: usize) -> usize {
        if n == 0 || n == 1 {
            1
        } else {
            (1..=n).product()
        }
    }

    /// Binomial coefficient C(n, k) = n! / (k! * (n-k)!).
    pub fn binomial(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        
        // Use multiplicative formula for efficiency
        let k = k.min(n - k);
        let mut result = 1;
        
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        
        result
    }

    /// Permutations P(n, k) = n! / (n-k)!.
    pub fn permutations(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 {
            return 1;
        }
        
        (n - k + 1..=n).product()
    }

    /// Multinomial coefficient: n! / (k1! * k2! * ... * km!).
    pub fn multinomial(n: usize, ks: &[usize]) -> usize {
        let sum: usize = ks.iter().sum();
        if sum != n {
            return 0;
        }
        
        let mut result = Self::factorial(n);
        for &k in ks {
            result /= Self::factorial(k);
        }
        
        result
    }

    /// Stirling numbers of the first kind (unsigned): s(n, k).
    /// Number of permutations of n elements with exactly k cycles.
    pub fn stirling_first(n: usize, k: usize) -> usize {
        if k > n || k == 0 {
            return if n == 0 && k == 0 { 1 } else { 0 };
        }
        if n == k {
            return 1;
        }
        
        Self::stirling_first(n - 1, k - 1) + (n - 1) * Self::stirling_first(n - 1, k)
    }

    /// Stirling numbers of the second kind: S(n, k).
    /// Number of ways to partition n elements into k non-empty subsets.
    pub fn stirling_second(n: usize, k: usize) -> usize {
        if k > n || k == 0 {
            return if n == 0 && k == 0 { 1 } else { 0 };
        }
        if n == k {
            return 1;
        }
        if k == 1 {
            return 1;
        }
        
        k * Self::stirling_second(n - 1, k) + Self::stirling_second(n - 1, k - 1)
    }

    /// Bell numbers: B(n) = sum_{k=0}^n S(n, k).
    /// Number of partitions of a set of n elements.
    pub fn bell(n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        
        (0..=n).map(|k| Self::stirling_second(n, k)).sum()
    }

    /// Catalan numbers: C_n = (1/(n+1)) * C(2n, n).
    pub fn catalan(n: usize) -> usize {
        Self::binomial(2 * n, n) / (n + 1)
    }

    /// Fibonacci numbers: F_n = F_{n-1} + F_{n-2}.
    pub fn fibonacci(n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }
        
        let mut a = 0;
        let mut b = 1;
        
        for _ in 2..=n {
            let temp = a + b;
            a = b;
            b = temp;
        }
        
        b
    }

    /// Lucas numbers: L_n = L_{n-1} + L_{n-2}, L_0 = 2, L_1 = 1.
    pub fn lucas(n: usize) -> usize {
        if n == 0 {
            return 2;
        }
        if n == 1 {
            return 1;
        }
        
        let mut a = 2;
        let mut b = 1;
        
        for _ in 2..=n {
            let temp = a + b;
            a = b;
            b = temp;
        }
        
        b
    }

    /// Generate all permutations of n elements.
    pub fn generate_permutations(n: usize) -> Vec<Vec<usize>> {
        let mut elements: Vec<usize> = (0..n).collect();
        let mut result = Vec::new();
        Self::permute(&mut elements, 0, &mut result);
        result
    }

    fn permute(elements: &mut [usize], start: usize, result: &mut Vec<Vec<usize>>) {
        if start == elements.len() {
            result.push(elements.to_vec());
            return;
        }
        
        for i in start..elements.len() {
            elements.swap(start, i);
            Self::permute(elements, start + 1, result);
            elements.swap(start, i);
        }
    }

    /// Generate all combinations of k elements from n.
    pub fn generate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut current = Vec::new();
        Self::combine(n, k, 0, &mut current, &mut result);
        result
    }

    fn combine(n: usize, k: usize, start: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        
        for i in start..n {
            current.push(i);
            Self::combine(n, k, i + 1, current, result);
            current.pop();
        }
    }

    /// Integer partition: number of ways to write n as sum of positive integers.
    pub fn partition(n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        
        let mut p = vec![0; n + 1];
        p[0] = 1;
        
        for k in 1..=n {
            for i in k..=n {
                p[i] += p[i - k];
            }
        }
        
        p[n]
    }

    /// Partition function with restricted parts (at most k parts).
    pub fn partition_restricted(n: usize, k: usize) -> usize {
        if n == 0 {
            return 1;
        }
        
        let mut p = vec![vec![0; k + 1]; n + 1];
        for i in 0..=k {
            p[0][i] = 1;
        }
        
        for i in 1..=n {
            for j in 1..=k {
                if j > i {
                    p[i][j] = p[i][j - 1];
                } else {
                    p[i][j] = p[i][j - 1] + p[i - j][j];
                }
            }
        }
        
        p[n][k]
    }

    /// Derangements: number of permutations with no fixed points.
    pub fn derangement(n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        if n == 1 {
            return 0;
        }
        
        let mut d = vec![0; n + 1];
        d[0] = 1;
        d[1] = 0;
        
        for i in 2..=n {
            d[i] = (i - 1) * (d[i - 1] + d[i - 2]);
        }
        
        d[n]
    }

    /// Inclusion-exclusion principle for counting.
    /// Given sizes of sets and their intersections, count union.
    pub fn inclusion_exclusion(
        single_sizes: &[usize],
        pair_intersections: &[usize],
        triple_intersections: &[usize],
    ) -> usize {
        let mut result = single_sizes.iter().sum::<usize>();
        
        // Subtract pair intersections
        for &size in pair_intersections {
            result -= size;
        }
        
        // Add triple intersections
        for &size in triple_intersections {
            result += size;
        }
        
        result
    }

    /// Pigeonhole principle: minimum number of items to guarantee at least k in one box.
    pub fn pigeonhole(n_boxes: usize, k_items: usize) -> usize {
        (n_boxes - 1) * k_items + 1
    }

    /// Stars and bars: number of ways to distribute n identical items into k distinct boxes.
    pub fn stars_and_bars(n: usize, k: usize) -> usize {
        Self::binomial(n + k - 1, k - 1)
    }

    /// Number of ways to choose k items from n with repetition allowed.
    pub fn combinations_with_repetition(n: usize, k: usize) -> usize {
        Self::binomial(n + k - 1, k)
    }

    /// Eulerian numbers: number of permutations with exactly k ascents.
    pub fn eulerian(n: usize, k: usize) -> usize {
        if k >= n {
            return 0;
        }
        if n == 0 {
            return 0;
        }
        if k == 0 {
            return 1;
        }
        
        (k + 1) * Self::eulerian(n - 1, k) + (n - k) * Self::eulerian(n - 1, k - 1)
    }

    /// Bernoulli numbers (simplified computation).
    pub fn bernoulli(n: usize) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return -0.5;
        }
        if n % 2 == 1 {
            return 0.0;
        }
        
        // Use recursive formula
        let mut b = vec![0.0; n + 1];
        b[0] = 1.0;
        
        for m in 1..=n {
            b[m] = 0.0;
            for k in 0..m {
                b[m] -= Self::binomial(m + 1, k) as f64 * b[k] / (m + 1) as f64;
            }
        }
        
        b[n]
    }

    /// Harmonic numbers: H_n = 1 + 1/2 + 1/3 + ... + 1/n.
    pub fn harmonic(n: usize) -> f64 {
        (1..=n).map(|i| 1.0 / i as f64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(Combinatorics::factorial(0), 1);
        assert_eq!(Combinatorics::factorial(5), 120);
        assert_eq!(Combinatorics::factorial(10), 3628800);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(Combinatorics::binomial(5, 2), 10);
        assert_eq!(Combinatorics::binomial(10, 5), 252);
        assert_eq!(Combinatorics::binomial(5, 0), 1);
        assert_eq!(Combinatorics::binomial(5, 5), 1);
    }

    #[test]
    fn test_permutations() {
        assert_eq!(Combinatorics::permutations(5, 2), 20);
        assert_eq!(Combinatorics::permutations(5, 3), 60);
    }

    #[test]
    fn test_stirling_second() {
        assert_eq!(Combinatorics::stirling_second(4, 2), 7);
        assert_eq!(Combinatorics::stirling_second(5, 3), 25);
    }

    #[test]
    fn test_bell() {
        assert_eq!(Combinatorics::bell(0), 1);
        assert_eq!(Combinatorics::bell(1), 1);
        assert_eq!(Combinatorics::bell(2), 2);
        assert_eq!(Combinatorics::bell(3), 5);
        assert_eq!(Combinatorics::bell(4), 15);
    }

    #[test]
    fn test_catalan() {
        assert_eq!(Combinatorics::catalan(0), 1);
        assert_eq!(Combinatorics::catalan(1), 1);
        assert_eq!(Combinatorics::catalan(2), 2);
        assert_eq!(Combinatorics::catalan(3), 5);
        assert_eq!(Combinatorics::catalan(4), 14);
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(Combinatorics::fibonacci(0), 0);
        assert_eq!(Combinatorics::fibonacci(1), 1);
        assert_eq!(Combinatorics::fibonacci(10), 55);
    }

    #[test]
    fn test_lucas() {
        assert_eq!(Combinatorics::lucas(0), 2);
        assert_eq!(Combinatorics::lucas(1), 1);
        assert_eq!(Combinatorics::lucas(5), 11);
    }

    #[test]
    fn test_partition() {
        assert_eq!(Combinatorics::partition(0), 1);
        assert_eq!(Combinatorics::partition(1), 1);
        assert_eq!(Combinatorics::partition(4), 5);
        assert_eq!(Combinatorics::partition(5), 7);
    }

    #[test]
    fn test_derangement() {
        assert_eq!(Combinatorics::derangement(0), 1);
        assert_eq!(Combinatorics::derangement(1), 0);
        assert_eq!(Combinatorics::derangement(4), 9);
    }

    #[test]
    fn test_generate_combinations() {
        let combos = Combinatorics::generate_combinations(4, 2);
        assert_eq!(combos.len(), 6);
    }

    #[test]
    fn test_stars_and_bars() {
        assert_eq!(Combinatorics::stars_and_bars(5, 3), 21);
    }

    #[test]
    fn test_harmonic() {
        let h = Combinatorics::harmonic(10);
        assert!((h - 2.928968254).abs() < 1e-8);
    }
}
