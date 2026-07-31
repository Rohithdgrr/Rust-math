//! Continued fractions: simple and generalized continued fractions, convergents.

use mathverse_core::error::{MathError, MathResult};

/// Simple continued fraction: [a0; a1, a2, ...]
pub struct SimpleContinuedFraction {
    pub coefficients: Vec<i64>,
}

impl SimpleContinuedFraction {
    /// Create from coefficients.
    pub fn new(coefficients: Vec<i64>) -> Self {
        SimpleContinuedFraction { coefficients }
    }

    /// Create from f64.
    pub fn from_f64(x: f64, max_terms: usize) -> Self {
        let mut coefficients = Vec::new();
        let mut x = x;
        
        for _ in 0..max_terms {
            let a = x.floor() as i64;
            coefficients.push(a);
            
            let frac = x - a as f64;
            if frac.abs() < 1e-15 {
                break;
            }
            
            x = 1.0 / frac;
        }
        
        SimpleContinuedFraction { coefficients }
    }

    /// Convert to f64.
    pub fn to_f64(&self) -> f64 {
        if self.coefficients.is_empty() {
            return 0.0;
        }
        
        let mut result = self.coefficients[self.coefficients.len() - 1] as f64;
        
        for i in (0..self.coefficients.len() - 1).rev() {
            result = self.coefficients[i] as f64 + 1.0 / result;
        }
        
        result
    }

    /// Get convergents (successive approximations).
    pub fn convergents(&self) -> Vec<(i64, i64)> {
        if self.coefficients.is_empty() {
            return Vec::new();
        }
        
        let mut convergents = Vec::new();
        let mut p_prev2 = 1i64;
        let mut p_prev1 = self.coefficients[0];
        let mut q_prev2 = 0i64;
        let mut q_prev1 = 1i64;
        
        convergents.push((p_prev1, q_prev1));
        
        for i in 1..self.coefficients.len() {
            let a = self.coefficients[i];
            let p = a * p_prev1 + p_prev2;
            let q = a * q_prev1 + q_prev2;
            
            convergents.push((p, q));
            
            p_prev2 = p_prev1;
            p_prev1 = p;
            q_prev2 = q_prev1;
            q_prev1 = q;
        }
        
        convergents
    }

    /// Get nth convergent.
    pub fn nth_convergent(&self, n: usize) -> MathResult<(i64, i64)> {
        let convergents = self.convergents();
        if n >= convergents.len() {
            return Err(MathError::InvalidArgument("convergent index out of bounds"));
        }
        Ok(convergents[n])
    }

    /// Check if periodic (for quadratic irrationals).
    pub fn is_periodic(&self, max_check: usize) -> bool {
        if self.coefficients.len() < 2 {
            return false;
        }
        
        for period in 1..=max_check.min(self.coefficients.len() / 2) {
            let is_period = (period..self.coefficients.len())
                .all(|i| self.coefficients[i] == self.coefficients[i - period]);
            
            if is_period {
                return true;
            }
        }
        
        false
    }

    /// Get period length if periodic.
    pub fn period_length(&self, max_check: usize) -> Option<usize> {
        if self.coefficients.len() < 2 {
            return None;
        }
        
        for period in 1..=max_check.min(self.coefficients.len() / 2) {
            let is_period = (period..self.coefficients.len())
                .all(|i| self.coefficients[i] == self.coefficients[i - period]);
            
            if is_period {
                return Some(period);
            }
        }
        
        None
    }
}

/// Generalized continued fraction: b0 + a1/(b1 + a2/(b2 + a3/(b3 + ...)))
pub struct GeneralizedContinuedFraction {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

impl GeneralizedContinuedFraction {
    /// Create from a and b coefficients.
    pub fn new(a: Vec<f64>, b: Vec<f64>) -> Self {
        GeneralizedContinuedFraction { a, b }
    }

    /// Convert to f64 using backward recurrence.
    pub fn to_f64(&self) -> f64 {
        if self.b.is_empty() {
            return 0.0;
        }
        
        let mut result = self.b[self.b.len() - 1];
        
        for i in (0..self.b.len() - 1).rev() {
            result = self.b[i] + self.a[i + 1] / result;
        }
        
        result
    }

    /// Evaluate using forward recurrence (Lentz's algorithm).
    pub fn evaluate_lentz(&self, tolerance: f64) -> f64 {
        if self.b.is_empty() {
            return 0.0;
        }
        
        let tiny = 1e-30;
        let mut f = self.b[0];
        if f.abs() < tiny {
            f = tiny;
        }
        
        let mut c = f;
        let mut d = 0.0;
        
        for i in 1..self.b.len() {
            d = self.b[i] + self.a[i] * d;
            if d.abs() < tiny {
                d = tiny;
            }
            d = 1.0 / d;
            
            c = self.b[i] + self.a[i] / c;
            if c.abs() < tiny {
                c = tiny;
            }
            
            let delta = c * d;
            f *= delta;
            
            if (delta - 1.0).abs() < tolerance {
                break;
            }
        }
        
        f
    }

    /// Create from square root: sqrt(n) = [a0; (a1, a2, ..., ak)] where the part in parentheses repeats.
    pub fn from_sqrt(n: f64, max_terms: usize) -> Self {
        let a0 = n.sqrt().floor();
        let mut a = Vec::new();
        let mut b = Vec::new();
        
        a.push(0.0);
        b.push(a0);
        
        let mut m = 0.0;
        let mut d = 1.0;
        let mut a0_f = a0;
        
        for _ in 0..max_terms {
            m = d * a0_f - m;
            d = (n - m * m) / d;
            a0_f = ((a0 + m) / d).floor();
            
            a.push(1.0);
            b.push(a0_f);
            
            if a0_f == 2.0 * a0 {
                break;
            }
        }
        
        GeneralizedContinuedFraction { a, b }
    }
}

/// Continued fraction operations.
pub struct ContinuedFractionOps;

impl ContinuedFractionOps {
    /// Convert rational to simple continued fraction.
    pub fn from_rational(numerator: i64, denominator: i64) -> MathResult<SimpleContinuedFraction> {
        if denominator == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let mut coefficients = Vec::new();
        let mut num = numerator.abs();
        let mut den = denominator.abs();
        
        while den != 0 {
            coefficients.push(num / den);
            let temp = num % den;
            num = den;
            den = temp;
        }
        
        Ok(SimpleContinuedFraction::new(coefficients))
    }

    /// Get best rational approximation with denominator <= max_den.
    pub fn best_approximation(x: f64, max_den: i64) -> (i64, i64) {
        let cf = SimpleContinuedFraction::from_f64(x, 20);
        let convergents = cf.convergents();
        
        for (num, den) in convergents {
            if den <= max_den {
                return (num, den);
            }
        }
        
        // Fallback to simple rounding
        let num = (x * max_den as f64).round() as i64;
        (num, max_den)
    }

    /// Mediant of two convergents.
    pub fn mediant(p1: i64, q1: i64, p2: i64, q2: i64) -> (i64, i64) {
        (p1 + p2, q1 + q2)
    }

    /// Check if convergent is best approximation.
    pub fn is_best_approximation(p: i64, q: i64, x: f64) -> bool {
        let value = p as f64 / q as f64;
        let error = (value - x).abs();
        
        // Check if any fraction with smaller denominator has smaller error
        for d in 1..q {
            let n = (x * d as f64).round() as i64;
            let test_value = n as f64 / d as f64;
            if (test_value - x).abs() < error {
                return false;
            }
        }
        
        true
    }

    /// Euler's continued fraction for e.
    pub fn euler_e(terms: usize) -> GeneralizedContinuedFraction {
        let mut a = Vec::new();
        let mut b = Vec::new();
        
        a.push(0.0);
        b.push(2.0);
        
        for i in 1..terms {
            a.push(1.0);
            if i % 3 == 0 {
                b.push((i as f64 / 3.0 * 2.0) as f64);
            } else {
                b.push(1.0);
            }
        }
        
        GeneralizedContinuedFraction { a, b }
    }

    /// Continued fraction for π.
    pub fn pi(terms: usize) -> GeneralizedContinuedFraction {
        let mut a = Vec::new();
        let mut b = Vec::new();
        
        a.push(0.0);
        b.push(3.0);
        
        for k in 1..terms {
            let k_f = k as f64;
            a.push((2.0 * k - 1.0).powi(2));
            b.push(6.0);
        }
        
        GeneralizedContinuedFraction { a, b }
    }

    /// Continued fraction for golden ratio φ.
    pub fn golden_ratio(terms: usize) -> SimpleContinuedFraction {
        let coefficients = vec![1; terms];
        SimpleContinuedFraction::new(coefficients)
    }

    /// Continued fraction for sqrt(2).
    pub fn sqrt_2(terms: usize) -> SimpleContinuedFraction {
        let mut coefficients = vec![1];
        coefficients.extend(vec![2; terms - 1]);
        SimpleContinuedFraction::new(coefficients)
    }

    /// Gauss's continued fraction for hypergeometric functions.
    pub fn gauss_hypergeometric(a: f64, b: f64, c: f64, z: f64, terms: usize) -> GeneralizedContinuedFraction {
        let mut a_coeffs = Vec::new();
        let mut b_coeffs = Vec::new();
        
        a_coeffs.push(0.0);
        b_coeffs.push(1.0);
        
        for n in 1..terms {
            let n_f = n as f64;
            a_coeffs.push((a + n_f - 1.0) * (b + n_f - 1.0) * z / (n_f * (c + n_f - 1.0)));
            b_coeffs.push(1.0);
        }
        
        GeneralizedContinuedFraction {
            a: a_coeffs,
            b: b_coeffs,
        }
    }

    /// J-fraction representation (Jacobi-type continued fraction).
    pub fn j_fraction(alpha: Vec<f64>, beta: Vec<f64>) -> GeneralizedContinuedFraction {
        let mut a = Vec::new();
        let mut b = Vec::new();
        
        a.push(0.0);
        b.push(alpha[0]);
        
        for i in 1..alpha.len() {
            a.push(beta[i - 1]);
            b.push(alpha[i]);
        }
        
        GeneralizedContinuedFraction { a, b }
    }

    /// Stieltjes continued fraction.
    pub fn stieltjes(mu: Vec<f64>) -> GeneralizedContinuedFraction {
        let mut a = Vec::new();
        let mut b = Vec::new();
        
        a.push(0.0);
        b.push(mu[0]);
        
        for i in 1..mu.len() {
            a.push(1.0);
            b.push(mu[i]);
        }
        
        GeneralizedContinuedFraction { a, b }
    }
}

/// Convergent properties.
pub struct ConvergentProperties;

impl ConvergentProperties {
    /// Check if convergents alternate around the value.
    pub fn alternates(cf: &SimpleContinuedFraction) -> bool {
        let convergents = cf.convergents();
        if convergents.len() < 2 {
            return false;
        }
        
        let target = cf.to_f64();
        
        for i in 0..convergents.len() - 1 {
            let val1 = convergents[i].0 as f64 / convergents[i].1 as f64;
            let val2 = convergents[i + 1].0 as f64 / convergents[i + 1].1 as f64;
            
            if (val1 > target && val2 > target) || (val1 < target && val2 < target) {
                return false;
            }
        }
        
        true
    }

    /// Error bound for nth convergent.
    pub fn error_bound(cf: &SimpleContinuedFraction, n: usize) -> f64 {
        let convergents = cf.convergents();
        if n >= convergents.len() {
            return f64::INFINITY;
        }
        
        let (p_n, q_n) = convergents[n];
        let (p_next, q_next) = if n + 1 < convergents.len() {
            convergents[n + 1]
        } else {
            return f64::INFINITY;
        };
        
        1.0 / (q_n as f64 * q_next as f64)
    }

    /// Rate of convergence.
    pub fn convergence_rate(cf: &SimpleContinuedFraction) -> f64 {
        let convergents = cf.convergents();
        if convergents.len() < 3 {
            return 0.0;
        }
        
        let target = cf.to_f64();
        let error1 = (convergents[convergents.len() - 2].0 as f64 / convergents[convergents.len() - 2].1 as f64 - target).abs();
        let error2 = (convergents[convergents.len() - 1].0 as f64 / convergents[convergents.len() - 1].1 as f64 - target).abs();
        
        if error1 == 0.0 {
            return f64::INFINITY;
        }
        
        error1 / error2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_continued_fraction() {
        let cf = SimpleContinuedFraction::from_f64(3.14159, 10);
        let value = cf.to_f64();
        
        assert!((value - 3.14159).abs() < 0.01);
    }

    #[test]
    fn test_convergents() {
        let cf = SimpleContinuedFraction::new(vec![3, 7, 15, 1, 292]);
        let convergents = cf.convergents();
        
        assert_eq!(convergents[0], (3, 1));
        assert_eq!(convergents[1], (22, 7));
        assert_eq!(convergents[2], (333, 106));
    }

    #[test]
    fn test_from_rational() {
        let cf = ContinuedFractionOps::from_rational(22, 7).unwrap();
        assert_eq!(cf.coefficients, vec![3, 7]);
    }

    #[test]
    fn test_best_approximation() {
        let (num, den) = ContinuedFractionOps::best_approximation(3.14159, 100);
        
        let value = num as f64 / den as f64;
        assert!((value - 3.14159).abs() < 0.01);
    }

    #[test]
    fn test_golden_ratio() {
        let cf = ContinuedFractionOps::golden_ratio(10);
        let value = cf.to_f64();
        
        assert!((value - 1.618033988).abs() < 0.01);
    }

    #[test]
    fn test_sqrt_2() {
        let cf = ContinuedFractionOps::sqrt_2(10);
        let value = cf.to_f64();
        
        assert!((value - 1.414213562).abs() < 0.01);
    }

    #[test]
    fn test_euler_e() {
        let cf = ContinuedFractionOps::euler_e(10);
        let value = cf.evaluate_lentz(1e-15);
        
        assert!((value - 2.718281828).abs() < 0.01);
    }

    #[test]
    fn test_generalized_continued_fraction() {
        let gcf = GeneralizedContinuedFraction::new(vec![0.0, 1.0, 1.0], vec![2.0, 1.0, 2.0]);
        let value = gcf.to_f64();
        
        assert!((value - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_alternates() {
        let cf = SimpleContinuedFraction::new(vec![3, 7, 15, 1, 292]);
        assert!(ConvergentProperties::alternates(&cf));
    }
}
