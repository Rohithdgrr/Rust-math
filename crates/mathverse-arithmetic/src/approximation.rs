//! Approximation algorithms: numerical methods for function approximation.

use mathverse_core::error::{MathError, MathResult};

/// Taylor series approximation.
pub struct TaylorSeries;

impl TaylorSeries {
    /// Approximate sin(x) using Taylor series.
    pub fn sin(x: f64, terms: usize) -> f64 {
        let mut result = 0.0;
        let mut sign = 1.0;
        let mut power = 1;
        let mut factorial = 1u64;
        
        for n in 0..terms {
            let term = sign * x.powi(power) as f64 / factorial as f64;
            result += term;
            
            sign *= -1.0;
            power += 2;
            factorial *= (power) as u64 * (power - 1) as u64;
        }
        
        result
    }

    /// Approximate cos(x) using Taylor series.
    pub fn cos(x: f64, terms: usize) -> f64 {
        let mut result = 0.0;
        let mut sign = 1.0;
        let mut power = 0;
        let mut factorial = 1u64;
        
        for n in 0..terms {
            let term = sign * x.powi(power) as f64 / factorial as f64;
            result += term;
            
            sign *= -1.0;
            power += 2;
            if power > 0 {
                factorial *= (power) as u64 * (power - 1) as u64;
            }
        }
        
        result
    }

    /// Approximate exp(x) using Taylor series.
    pub fn exp(x: f64, terms: usize) -> f64 {
        let mut result = 0.0;
        let mut power = 0;
        let mut factorial = 1u64;
        
        for n in 0..terms {
            let term = x.powi(power) as f64 / factorial as f64;
            result += term;
            
            power += 1;
            factorial *= power as u64;
        }
        
        result
    }

    /// Approximate ln(1+x) using Taylor series (valid for |x| < 1).
    pub fn ln_one_plus_x(x: f64, terms: usize) -> f64 {
        let mut result = 0.0;
        let mut sign = 1.0;
        
        for n in 1..=terms {
            let term = sign * x.powi(n as i32) / n as f64;
            result += term;
            sign *= -1.0;
        }
        
        result
    }

    /// General Taylor series approximation.
    pub fn approximate(
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64) -> f64,
        ddf: impl Fn(f64) -> f64,
        x0: f64,
        x: f64,
        terms: usize,
    ) -> f64 {
        let h = x - x0;
        let mut result = f(x0);
        
        if terms >= 2 {
            result += df(x0) * h;
        }
        if terms >= 3 {
            result += ddf(x0) * h * h / 2.0;
        }
        
        result
    }
}

/// Padé approximation.
pub struct PadeApproximation;

impl PadeApproximation {
    /// Compute Padé approximant [m/n] for a function given by its Taylor coefficients.
    pub fn compute(taylor_coeffs: &[f64], m: usize, n: usize) -> MathResult<(Vec<f64>, Vec<f64>)> {
        if m + n > taylor_coeffs.len() {
            return Err(MathError::InvalidArgument("not enough Taylor coefficients"));
        }
        
        // Simplified implementation for [1/1] approximant
        if m == 1 && n == 1 {
            let c0 = taylor_coeffs[0];
            let c1 = taylor_coeffs[1];
            let c2 = taylor_coeffs[2];
            
            // Solve for a and b in (c0 + c1*x) / (1 + b*x) ≈ c0 + c1*x + c2*x^2
            let b = -c2 / c1;
            let a = c1 + c0 * b;
            
            Ok(vec![c0, a], vec![1.0, b])
        } else {
            Err(MathError::InvalidArgument("only [1/1] approximant implemented"))
        }
    }

    /// Evaluate Padé approximant.
    pub fn evaluate(numerator: &[f64], denominator: &[f64], x: f64) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        
        for (i, &coeff) in numerator.iter().enumerate() {
            num += coeff * x.powi(i as i32);
        }
        
        for (i, &coeff) in denominator.iter().enumerate() {
            den += coeff * x.powi(i as i32);
        }
        
        num / den
    }
}

/// Chebyshev approximation.
pub struct ChebyshevApproximation;

impl ChebyshevApproximation {
    /// Chebyshev polynomial of first kind T_n(x).
    pub fn t_n(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return x;
        }
        
        let mut t_prev2 = 1.0;
        let mut t_prev1 = x;
        
        for _ in 2..=n {
            let t_current = 2.0 * x * t_prev1 - t_prev2;
            t_prev2 = t_prev1;
            t_prev1 = t_current;
        }
        
        t_prev1
    }

    /// Chebyshev polynomial of second kind U_n(x).
    pub fn u_n(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return 2.0 * x;
        }
        
        let mut u_prev2 = 1.0;
        let mut u_prev1 = 2.0 * x;
        
        for _ in 2..=n {
            let u_current = 2.0 * x * u_prev1 - u_prev2;
            u_prev2 = u_prev1;
            u_prev1 = u_current;
        }
        
        u_prev1
    }

    /// Chebyshev nodes for approximation on [-1, 1].
    pub fn nodes(n: usize) -> Vec<f64> {
        (0..n).map(|i| {
            core::f64::consts::PI * (i as f64 + 0.5) / n as f64
        }).map(|theta| theta.cos()).collect()
    }

    /// Chebyshev coefficients for function approximation.
    pub fn coefficients(f: impl Fn(f64) -> f64, n: usize) -> Vec<f64> {
        let nodes = Self::nodes(n);
        let mut coeffs = vec![0.0; n];
        
        for (i, &x) in nodes.iter().enumerate() {
            let y = f(x);
            for j in 0..n {
                coeffs[j] += y * Self::t_n(j, x);
            }
        }
        
        for coeff in &mut coeffs {
            *coeff /= n as f64;
        }
        
        coeffs[0] /= 2.0;
        coeffs
    }

    /// Evaluate Chebyshev approximation.
    pub fn evaluate(coeffs: &[f64], x: f64) -> f64 {
        let mut result = 0.0;
        
        for (i, &coeff) in coeffs.iter().enumerate() {
            result += coeff * Self::t_n(i, x);
        }
        
        result
    }
}

/// Fourier series approximation.
pub struct FourierSeries;

impl FourierSeries {
    /// Compute Fourier coefficients for periodic function.
    pub fn coefficients(f: impl Fn(f64) -> f64, period: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut a_coeffs = vec![0.0; n + 1];
        let mut b_coeffs = vec![0.0; n + 1];
        
        let samples = 1000;
        let dx = period / samples as f64;
        
        for i in 0..=n {
            let mut a_sum = 0.0;
            let mut b_sum = 0.0;
            
            for j in 0..samples {
                let x = j as f64 * dx;
                let y = f(x);
                let k = 2.0 * core::f64::consts::PI * i as f64 / period;
                
                a_sum += y * (k * x).cos();
                b_sum += y * (k * x).sin();
            }
            
            a_coeffs[i] = 2.0 * a_sum / samples as f64;
            b_coeffs[i] = 2.0 * b_sum / samples as f64;
        }
        
        a_coeffs[0] /= 2.0;
        (a_coeffs, b_coeffs)
    }

    /// Evaluate Fourier series.
    pub fn evaluate(a: &[f64], b: &[f64], x: f64, period: f64) -> f64 {
        let mut result = a[0];
        
        for i in 1..a.len().min(b.len()) {
            let k = 2.0 * core::f64::consts::PI * i as f64 / period;
            result += a[i] * (k * x).cos() + b[i] * (k * x).sin();
        }
        
        result
    }

    /// Discrete Fourier Transform (simplified).
    pub fn dft(values: &[f64]) -> Vec<Complex> {
        use super::complex::Complex;
        
        let n = values.len();
        let mut result = Vec::with_capacity(n);
        
        for k in 0..n {
            let mut sum = Complex::zero();
            
            for (t, &value) in values.iter().enumerate() {
                let angle = -2.0 * core::f64::consts::PI * k as f64 * t as f64 / n as f64;
                let term = Complex::from_polar(value, angle);
                sum = sum.add(&term);
            }
            
            result.push(sum.scale(1.0 / n as f64));
        }
        
        result
    }
}

/// Polynomial approximation.
pub struct PolynomialApproximation;

impl PolynomialApproximation {
    /// Least squares polynomial fit.
    pub fn least_squares(points: &[(f64, f64)], degree: usize) -> MathResult<Vec<f64>> {
        if points.is_empty() || degree == 0 {
            return Err(MathError::InvalidArgument("invalid input"));
        }
        
        let n = degree + 1;
        let mut matrix = vec![vec![0.0; n]; n];
        let mut rhs = vec![0.0; n];
        
        // Build normal equations
        for &(x, y) in points {
            let mut x_powers = vec![1.0; n];
            for i in 1..n {
                x_powers[i] = x_powers[i - 1] * x;
            }
            
            for i in 0..n {
                for j in 0..n {
                    matrix[i][j] += x_powers[i] * x_powers[j];
                }
                rhs[i] += y * x_powers[i];
            }
        }
        
        // Solve using Gaussian elimination (simplified)
        Self::solve_linear(&mut matrix, &mut rhs)
    }

    /// Solve linear system using Gaussian elimination.
    fn solve_linear(matrix: &mut [Vec<f64>], rhs: &mut [f64]) -> MathResult<Vec<f64>> {
        let n = matrix.len();
        
        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut pivot = i;
            for j in (i + 1)..n {
                if matrix[j][i].abs() > matrix[pivot][i].abs() {
                    pivot = j;
                }
            }
            
            if matrix[pivot][i].abs() < 1e-15 {
                return Err(MathError::InvalidArgument("singular matrix"));
            }
            
            // Swap rows
            matrix.swap(i, pivot);
            rhs.swap(i, pivot);
            
            // Eliminate
            for j in (i + 1)..n {
                let factor = matrix[j][i] / matrix[i][i];
                for k in i..n {
                    matrix[j][k] -= factor * matrix[i][k];
                }
                rhs[j] -= factor * rhs[i];
            }
        }
        
        // Back substitution
        let mut solution = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = rhs[i];
            for j in (i + 1)..n {
                sum -= matrix[i][j] * solution[j];
            }
            solution[i] = sum / matrix[i][i];
        }
        
        Ok(solution)
    }

    /// Evaluate polynomial.
    pub fn evaluate(coeffs: &[f64], x: f64) -> f64 {
        let mut result = 0.0;
        
        for (i, &coeff) in coeffs.iter().enumerate() {
            result += coeff * x.powi(i as i32);
        }
        
        result
    }

    /// Lagrange interpolation polynomial.
    pub fn lagrange(points: &[(f64, f64)], x: f64) -> f64 {
        let mut result = 0.0;
        
        for (i, &(xi, yi)) in points.iter().enumerate() {
            let mut term = yi;
            
            for (j, &(xj, _)) in points.iter().enumerate() {
                if i != j {
                    term *= (x - xj) / (xi - xj);
                }
            }
            
            result += term;
        }
        
        result
    }

    /// Neville's algorithm for polynomial interpolation.
    pub fn neville(points: &[(f64, f64)], x: f64) -> f64 {
        let n = points.len();
        let mut table = vec![vec![0.0; n]; n];
        
        for (i, &(_, yi)) in points.iter().enumerate() {
            table[i][0] = yi;
        }
        
        for j in 1..n {
            for i in 0..n - j {
                let xi = points[i].0;
                let xij = points[i + j].0;
                
                table[i][j] = ((x - xij) * table[i + 1][j - 1] + (xi - x) * table[i][j - 1]) / (xi - xij);
            }
        }
        
        table[0][n - 1]
    }
}

/// Rational approximation.
pub struct RationalApproximation;

impl RationalApproximation {
    /// Minimax approximation (simplified using Remez exchange algorithm).
    pub fn minimax(f: impl Fn(f64) -> f64, a: f64, b: f64, m: usize, n: usize, iterations: usize) -> MathResult<(Vec<f64>, Vec<f64>)> {
        // Simplified implementation: use Chebyshev nodes as initial guess
        let num_points = m + n + 2;
        let mut nodes = ChebyshevApproximation::nodes(num_points);
        
        // Scale nodes to [a, b]
        for node in &mut nodes {
            *node = (*node + 1.0) / 2.0 * (b - a) + a;
        }
        
        // Build linear system for rational approximation
        let mut matrix = vec![vec![0.0; m + n + 1]; num_points];
        let mut rhs = vec![0.0; num_points];
        
        for (i, &x) in nodes.iter().enumerate() {
            let fx = f(x);
            
            // Numerator terms
            for j in 0..=m {
                matrix[i][j] = x.powi(j as i32);
            }
            
            // Denominator terms (with error term)
            for j in 0..=n {
                matrix[i][m + 1 + j] = -fx * x.powi(j as i32);
            }
            
            rhs[i] = fx;
        }
        
        let solution = PolynomialApproximation::solve_linear(&mut matrix, &mut rhs)?;
        
        let mut numerator = solution[..=m].to_vec();
        let denominator = vec![1.0].into_iter().chain(solution[m + 1..].iter().cloned()).collect();
        
        // Refine using simple iteration
        for _ in 0..iterations {
            let error = Self::compute_error(&f, &numerator, &denominator, a, b);
            // Could add refinement logic here
        }
        
        Ok((numerator, denominator))
    }

    /// Compute maximum error of rational approximation.
    fn compute_error(f: impl Fn(f64) -> f64, num: &[f64], den: &[f64], a: f64, b: f64) -> f64 {
        let samples = 100;
        let dx = (b - a) / samples as f64;
        
        let mut max_error = 0.0;
        
        for i in 0..=samples {
            let x = a + i as f64 * dx;
            let approx = PadeApproximation::evaluate(num, den, x);
            let error = (f(x) - approx).abs();
            max_error = max_error.max(error);
        }
        
        max_error
    }

    /// Evaluate rational approximation.
    pub fn evaluate(numerator: &[f64], denominator: &[f64], x: f64) -> f64 {
        PadeApproximation::evaluate(numerator, denominator, x)
    }
}

/// Complex type for Fourier DFT.
struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    fn zero() -> Self {
        Complex { real: 0.0, imag: 0.0 }
    }
    
    fn add(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
    
    fn scale(&self, s: f64) -> Complex {
        Complex {
            real: self.real * s,
            imag: self.imag * s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_taylor_sin() {
        let approx = TaylorSeries::sin(1.0, 10);
        let actual = 1.0.sin();
        
        assert!((approx - actual).abs() < 1e-10);
    }

    #[test]
    fn test_taylor_exp() {
        let approx = TaylorSeries::exp(1.0, 20);
        let actual = 1.0.exp();
        
        assert!((approx - actual).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev() {
        let t3 = ChebyshevApproximation::t_n(3, 0.5);
        // T_3(x) = 4x^3 - 3x
        let expected = 4.0 * 0.5.powi(3) - 3.0 * 0.5;
        
        assert!((t3 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_nodes() {
        let nodes = ChebyshevApproximation::nodes(5);
        assert_eq!(nodes.len(), 5);
        
        // All nodes should be in [-1, 1]
        for &node in &nodes {
            assert!(node >= -1.0 && node <= 1.0);
        }
    }

    #[test]
    fn test_lagrange() {
        let points = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)];
        let result = PolynomialApproximation::lagrange(&points, 1.5);
        
        // Should interpolate between points
        assert!(result > 2.0 && result < 5.0);
    }

    #[test]
    fn test_neville() {
        let points = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)];
        let result = PolynomialApproximation::neville(&points, 1.5);
        
        let lagrange = PolynomialApproximation::lagrange(&points, 1.5);
        assert!((result - lagrange).abs() < 1e-10);
    }

    #[test]
    fn test_least_squares() {
        let points = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)];
        let coeffs = PolynomialApproximation::least_squares(&points, 1).unwrap();
        
        // Should approximate y = x + 1
        assert!((coeffs[0] - 1.0).abs() < 0.1);
        assert!((coeffs[1] - 1.0).abs() < 0.1);
    }
}
