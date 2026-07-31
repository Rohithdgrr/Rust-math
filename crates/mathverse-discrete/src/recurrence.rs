//! Recurrence relations: solving linear recurrences, generating sequences.

/// Recurrence relation operations.
pub struct RecurrenceRelations;

impl RecurrenceRelations {
    /// Solve linear homogeneous recurrence: a_n = c1*a_{n-1} + c2*a_{n-2} + ... + ck*a_{n-k}.
    /// Returns sequence of n terms given initial conditions.
    pub fn linear_homogeneous(coeffs: &[f64], initial: &[f64], n: usize) -> Vec<f64> {
        if coeffs.is_empty() || initial.is_empty() {
            return Vec::new();
        }
        
        let k = coeffs.len();
        let mut sequence = initial.to_vec();
        
        if sequence.len() >= n {
            return sequence[..n].to_vec();
        }
        
        while sequence.len() < n {
            let mut next = 0.0;
            for (i, &coeff) in coeffs.iter().enumerate() {
                if i < sequence.len() {
                    next += coeff * sequence[sequence.len() - 1 - i];
                }
            }
            sequence.push(next);
        }
        
        sequence
    }

    /// Solve linear non-homogeneous recurrence: a_n = c1*a_{n-1} + ... + ck*a_{n-k} + f(n).
    pub fn linear_nonhomogeneous(
        coeffs: &[f64],
        f: impl Fn(usize) -> f64,
        initial: &[f64],
        n: usize,
    ) -> Vec<f64> {
        if coeffs.is_empty() || initial.is_empty() {
            return Vec::new();
        }
        
        let k = coeffs.len();
        let mut sequence = initial.to_vec();
        
        if sequence.len() >= n {
            return sequence[..n].to_vec();
        }
        
        while sequence.len() < n {
            let mut next = f(sequence.len());
            for (i, &coeff) in coeffs.iter().enumerate() {
                if i < sequence.len() {
                    next += coeff * sequence[sequence.len() - 1 - i];
                }
            }
            sequence.push(next);
        }
        
        sequence
    }

    /// Fibonacci sequence: F_n = F_{n-1} + F_{n-2}, F_0 = 0, F_1 = 1.
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

    /// Lucas sequence: L_n = L_{n-1} + L_{n-2}, L_0 = 2, L_1 = 1.
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

    /// Tribonacci sequence: T_n = T_{n-1} + T_{n-2} + T_{n-3}.
    pub fn tribonacci(n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        if n == 1 || n == 2 {
            return 1;
        }
        
        let mut a = 0;
        let mut b = 1;
        let mut c = 1;
        
        for _ in 3..=n {
            let temp = a + b + c;
            a = b;
            b = c;
            c = temp;
        }
        
        c
    }

    /// Padovan sequence: P_n = P_{n-2} + P_{n-3}.
    pub fn padovan(n: usize) -> usize {
        if n == 0 || n == 1 || n == 2 {
            return 1;
        }
        
        let mut a = 1;
        let mut b = 1;
        let mut c = 1;
        
        for _ in 3..=n {
            let temp = a + b;
            a = b;
            b = c;
            c = temp;
        }
        
        c
    }

    /// Perrin sequence: P_n = P_{n-2} + P_{n-3}.
    pub fn perrin(n: usize) -> usize {
        if n == 0 {
            return 3;
        }
        if n == 1 || n == 2 {
            return 0;
        }
        
        let mut a = 3;
        let mut b = 0;
        let mut c = 2;
        
        for _ in 3..=n {
            let temp = a + b;
            a = b;
            b = c;
            c = temp;
        }
        
        c
    }

    /// Arithmetic sequence: a_n = a_0 + n*d.
    pub fn arithmetic(a0: f64, d: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| a0 + i as f64 * d).collect()
    }

    /// Geometric sequence: a_n = a_0 * r^n.
    pub fn geometric(a0: f64, r: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| a0 * r.powi(i as i32)).collect()
    }

    /// Harmonic sequence: H_n = 1/n.
    pub fn harmonic(n: usize) -> Vec<f64> {
        (1..=n).map(|i| 1.0 / i as f64).collect()
    }

    /// Solve recurrence using characteristic equation (for linear homogeneous).
    /// Returns closed-form coefficients for solution.
    pub fn characteristic_solution(coeffs: &[f64]) -> Option<Vec<f64>> {
        let k = coeffs.len();
        
        match k {
            1 => {
                // a_n = c * a_{n-1}: solution is a_n = a_0 * c^n
                Some(vec![coeffs[0]])
            }
            2 => {
                // a_n = c1 * a_{n-1} + c2 * a_{n-2}
                // Characteristic equation: r^2 - c1*r - c2 = 0
                let c1 = coeffs[0];
                let c2 = coeffs[1];
                
                let discriminant = c1 * c1 + 4.0 * c2;
                
                if discriminant >= 0.0 {
                    let r1 = (c1 + discriminant.sqrt()) / 2.0;
                    let r2 = (c1 - discriminant.sqrt()) / 2.0;
                    Some(vec![r1, r2])
                } else {
                    // Complex roots
                    let real_part = c1 / 2.0;
                    let imag_part = (-discriminant).sqrt() / 2.0;
                    Some(vec![real_part, imag_part])
                }
            }
            _ => None, // Higher order not implemented
        }
    }

    /// Generate terms using closed-form solution.
    pub fn closed_form(r1: f64, r2: f64, c1: f64, c2: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| {
            let i_f = i as f64;
            c1 * r1.powi(i_f as i32) + c2 * r2.powi(i_f as i32)
        }).collect()
    }

    /// Z-transform of sequence (simplified).
    pub fn z_transform(sequence: &[f64]) -> Vec<f64> {
        let n = sequence.len();
        (0..n).map(|k| {
            sequence.iter().enumerate().map(|(i, &a)| a / (k as f64 + 1.0_f64).powi(i as i32)).sum()
        }).collect()
    }

    /// Inverse Z-transform (simplified using power series).
    pub fn inverse_z_transform(coeffs: &[f64], n: usize) -> Vec<f64> {
        let mut sequence = Vec::new();
        
        for i in 0..n {
            let mut sum = 0.0;
            for (k, &coeff) in coeffs.iter().enumerate() {
                if k <= i {
                    sum += coeff * (i as f64).powi(k as i32);
                }
            }
            sequence.push(sum);
        }
        
        sequence
    }

    /// Convolution of two sequences.
    pub fn convolution(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let m = b.len();
        let mut result = vec![0.0; n + m - 1];
        
        for i in 0..n {
            for j in 0..m {
                result[i + j] += a[i] * b[j];
            }
        }
        
        result
    }

    /// Check if sequence satisfies given recurrence.
    pub fn satisfies_recurrence(sequence: &[f64], coeffs: &[f64]) -> bool {
        let k = coeffs.len();
        
        for i in k..sequence.len() {
            let mut expected = 0.0;
            for (j, &coeff) in coeffs.iter().enumerate() {
                expected += coeff * sequence[i - 1 - j];
            }
            
            if (sequence[i] - expected).abs() > 1e-10 {
                return false;
            }
        }
        
        true
    }

    /// Find recurrence coefficients from sequence (using least squares).
    pub fn find_recurrence(sequence: &[f64], order: usize) -> Option<Vec<f64>> {
        if sequence.len() < 2 * order {
            return None;
        }
        
        let n = sequence.len() - order;
        let mut matrix = vec![vec![0.0; order]; n];
        let mut rhs = vec![0.0; n];
        
        for i in 0..n {
            for j in 0..order {
                matrix[i][j] = sequence[i + order - 1 - j];
            }
            rhs[i] = sequence[i + order];
        }
        
        // Simple Gaussian elimination
        Self::solve_linear_system(&matrix, &rhs)
    }

    fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let n = a.len();
        if n == 0 {
            return Some(Vec::new());
        }
        
        let m = a[0].len();
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        
        // Forward elimination
        for i in 0..n {
            let mut pivot = i;
            for j in (i + 1)..n {
                if a[j][i].abs() > a[pivot][i].abs() {
                    pivot = j;
                }
            }
            
            if a[pivot][i].abs() < 1e-15 {
                return None;
            }
            
            a.swap(i, pivot);
            b.swap(i, pivot);
            
            for j in (i + 1)..n {
                let factor = a[j][i] / a[i][i];
                for k in i..m {
                    a[j][k] -= factor * a[i][k];
                }
                b[j] -= factor * b[i];
            }
        }
        
        // Back substitution
        let mut x = vec![0.0; m];
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..m {
                sum -= a[i][j] * x[j];
            }
            x[i] = sum / a[i][i];
        }
        
        Some(x)
    }

    /// Generating function coefficients for recurrence.
    pub fn generating_function(coeffs: &[f64], initial: &[f64]) -> Vec<f64> {
        let k = coeffs.len();
        let mut result = initial.to_vec();
        
        // G(x) = (a_0 + (a_1 - c_1*a_0)x + ...) / (1 - c_1*x - c_2*x^2 - ...)
        let mut denominator_coeffs = vec![1.0];
        for &c in coeffs {
            denominator_coeffs.push(-c);
        }
        
        // Numerator coefficients
        let mut numerator_coeffs = vec![0.0; k];
        for i in 0..k.min(initial.len()) {
            numerator_coeffs[i] = initial[i];
            for j in 0..i {
                numerator_coeffs[i] -= coeffs[j] * numerator_coeffs[i - 1 - j];
            }
        }
        
        // Expand as power series (simplified)
        let mut series = Vec::new();
        for n in 0..20 {
            let mut term = 0.0;
            for i in 0..numerator_coeffs.len().min(n + 1) {
                if i < numerator_coeffs.len() {
                    term += numerator_coeffs[i];
                }
            }
            series.push(term);
        }
        
        series
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_homogeneous() {
        let coeffs = vec![1.0, 1.0]; // Fibonacci
        let initial = vec![0.0, 1.0];
        let result = RecurrenceRelations::linear_homogeneous(&coeffs, &initial, 10);
        
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[5], 5.0);
        assert_eq!(result[9], 34.0);
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(RecurrenceRelations::fibonacci(0), 0);
        assert_eq!(RecurrenceRelations::fibonacci(1), 1);
        assert_eq!(RecurrenceRelations::fibonacci(10), 55);
    }

    #[test]
    fn test_lucas() {
        assert_eq!(RecurrenceRelations::lucas(0), 2);
        assert_eq!(RecurrenceRelations::lucas(1), 1);
        assert_eq!(RecurrenceRelations::lucas(5), 11);
    }

    #[test]
    fn test_tribonacci() {
        assert_eq!(RecurrenceRelations::tribonacci(0), 0);
        assert_eq!(RecurrenceRelations::tribonacci(1), 1);
        assert_eq!(RecurrenceRelations::tribonacci(2), 1);
        assert_eq!(RecurrenceRelations::tribonacci(5), 7);
    }

    #[test]
    fn test_arithmetic() {
        let result = RecurrenceRelations::arithmetic(1.0, 2.0, 5);
        assert_eq!(result, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_geometric() {
        let result = RecurrenceRelations::geometric(1.0, 2.0, 5);
        assert_eq!(result, vec![1.0, 2.0, 4.0, 8.0, 16.0]);
    }

    #[test]
    fn test_convolution() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0];
        let result = RecurrenceRelations::convolution(&a, &b);
        
        assert_eq!(result, vec![1.0, 3.0, 5.0, 3.0]);
    }

    #[test]
    fn test_satisfies_recurrence() {
        let sequence = vec![0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0];
        let coeffs = vec![1.0, 1.0];
        
        assert!(RecurrenceRelations::satisfies_recurrence(&sequence, &coeffs));
    }

    #[test]
    fn test_find_recurrence() {
        let sequence = vec![0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
        let coeffs = RecurrenceRelations::find_recurrence(&sequence, 2);
        
        assert!(coeffs.is_some());
        let found = coeffs.unwrap();
        assert!((found[0] - 1.0).abs() < 0.1);
        assert!((found[1] - 1.0).abs() < 0.1);
    }
}
