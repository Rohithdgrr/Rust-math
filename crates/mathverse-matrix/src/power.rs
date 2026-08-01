//! Matrix power and related operations.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Matrix power operations.
pub struct MatrixPower;

impl MatrixPower {
    /// Compute A^n for integer n (positive, zero, or negative).
    pub fn compute(m: &Matrix, n: i32) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        if n == 0 {
            return Ok(Matrix::identity(m.rows));
        }
        
        if n < 0 {
            let inv = m.inverse()?;
            return Self::compute(&inv, -n);
        }
        
        // Binary exponentiation
        let mut result = Matrix::identity(m.rows);
        let mut base = m.clone();
        let mut exp = n;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.mul(&base)?;
            }
            base = base.mul(&base)?;
            exp /= 2;
        }
        
        Ok(result)
    }

    /// Compute A^p for rational p = a/b using eigenvalue decomposition.
    pub fn rational(m: &Matrix, a: i32, b: i32) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        if b == 0 {
            return Err(MathError::InvalidArgument("division by zero in rational power"));
        }
        
        let (vals, vecs) = m.eigen_symmetric()?;
        let n = m.rows;
        
        // Check for non-negative eigenvalues if b is even
        if b % 2 == 0 {
            for &v in &vals {
                if v < 0.0 {
                    return Err(MathError::InvalidArgument("negative eigenvalue for even root"));
                }
            }
        }
        
        // Compute D^(a/b)
        let mut powered = Matrix::zeros(n, n);
        for i in 0..n {
            let val = if vals[i] >= 0.0 {
                vals[i].powf(a as f64 / b as f64)
            } else {
                -((-vals[i]).powf(a as f64 / b as f64))
            };
            powered.set(i, i, val);
        }
        
        // A^(a/b) = V D^(a/b) V^T
        let v_powered = vecs.mul(&powered)?;
        v_powered.mul(&vecs.transpose())
    }

    /// Matrix square root (principal).
    pub fn sqrt(m: &Matrix) -> MathResult<Matrix> {
        Self::rational(m, 1, 2)
    }

    /// Matrix cube root.
    pub fn cbrt(m: &Matrix) -> MathResult<Matrix> {
        Self::rational(m, 1, 3)
    }

    /// Matrix inverse (A^(-1)).
    pub fn inverse(m: &Matrix) -> MathResult<Matrix> {
        m.inverse()
    }

    /// Matrix transpose power: (A^T)^n = (A^n)^T.
    pub fn transpose_power(m: &Matrix, n: i32) -> MathResult<Matrix> {
        let power = Self::compute(m, n)?;
        Ok(power.transpose())
    }

    /// Commutator: [A, B] = AB - BA.
    pub fn commutator(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if !a.is_square() || !b.is_square() || a.rows != b.rows {
            return Err(MathError::DimensionMismatch);
        }
        
        let ab = a.mul(b)?;
        let ba = b.mul(a)?;
        ab.sub(&ba)
    }

    /// Anti-commutator: {A, B} = AB + BA.
    pub fn anti_commutator(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if !a.is_square() || !b.is_square() || a.rows != b.rows {
            return Err(MathError::DimensionMismatch);
        }
        
        let ab = a.mul(b)?;
        let ba = b.mul(a)?;
        ab.add(&ba)
    }

    /// Check if matrices commute: AB = BA.
    pub fn commutes(a: &Matrix, b: &Matrix, tolerance: f64) -> MathResult<bool> {
        let commutator = Self::commutator(a, b)?;
        let norm = crate::norms::MatrixNorms::frobenius(&commutator);
        Ok(norm < tolerance)
    }

    /// Matrix exponential series: exp(A) = Σ A^k / k!.
    pub fn exp_series(m: &Matrix, terms: usize) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        let mut result = Matrix::identity(n);
        let mut term = Matrix::identity(n);
        let mut factorial = 1.0;
        
        for k in 1..=terms {
            term = term.mul(m)?;
            factorial *= k as f64;
            let term_scaled = term.scale(1.0 / factorial);
            result = result.add(&term_scaled)?;
            
            let term_norm = crate::norms::MatrixNorms::linf(&term_scaled);
            if term_norm < 1e-15 {
                break;
            }
        }
        
        Ok(result)
    }

    /// Matrix logarithm series: log(I + X) for ||X|| < 1.
    pub fn log_series(m: &Matrix, terms: usize) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let identity = Matrix::identity(m.rows);
        let x = m.sub(&identity)?;
        
        let norm = crate::norms::MatrixNorms::linf(&x);
        if norm >= 1.0 {
            return Err(MathError::InvalidArgument("matrix norm >= 1, series may not converge"));
        }
        
        let mut result = Matrix::zeros(m.rows, m.cols);
        let mut term = x.clone();
        
        for k in 1..=terms {
            let coeff = if k % 2 == 0 { -1.0 } else { 1.0 } / k as f64;
            result = result.add(&term.scale(coeff))?;
            term = term.mul(&x)?;
            
            let term_norm = crate::norms::MatrixNorms::linf(&term);
            if term_norm < 1e-15 {
                break;
            }
        }
        
        Ok(result)
    }
}

/// Matrix polynomial functions.
pub struct MatrixPolynomial;

impl MatrixPolynomial {
    /// Evaluate polynomial p(A) = c₀I + c₁A + c₂A² + ... + cₙAⁿ.
    pub fn evaluate(m: &Matrix, coefficients: &[f64]) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        if coefficients.is_empty() {
            return Ok(Matrix::zeros(m.rows, m.cols));
        }
        
        let n = m.rows;
        let mut result = Matrix::identity(n).scale(coefficients[0]);
        let mut power = Matrix::identity(n);
        
        for (i, &coeff) in coefficients.iter().enumerate().skip(1) {
            power = power.mul(m)?;
            result = result.add(&power.scale(coeff))?;
        }
        
        Ok(result)
    }

    /// Characteristic polynomial coefficients (via eigenvalues).
    pub fn characteristic(m: &Matrix) -> MathResult<Vec<f64>> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let (vals, _) = m.eigen_symmetric()?;
        
        // Build polynomial from roots: (λ - λ₁)(λ - λ₂)...(λ - λₙ)
        let mut coeffs = vec![1.0];
        
        for &val in &vals {
            let mut new_coeffs = vec![0.0; coeffs.len() + 1];
            for (i, &c) in coeffs.iter().enumerate() {
                new_coeffs[i] += c;
                new_coeffs[i + 1] -= c * val;
            }
            coeffs = new_coeffs;
        }
        
        Ok(coeffs)
    }

    /// Minimal polynomial (simplified - returns characteristic for now).
    pub fn minimal(m: &Matrix) -> MathResult<Vec<f64>> {
        Self::characteristic(m)
    }

    /// Cayley-Hamilton theorem: p(A) = 0 where p is characteristic polynomial.
    pub fn cayley_hamilton(m: &Matrix) -> MathResult<bool> {
        let coeffs = Self::characteristic(m)?;
        let p_a = Self::evaluate(m, &coeffs)?;
        let norm = crate::norms::MatrixNorms::frobenius(&p_a);
        Ok(norm < 1e-10)
    }
}

/// Matrix functions via power series.
pub struct MatrixSeriesFunctions;

impl MatrixSeriesFunctions {
    /// Compute factorial for Taylor series coefficients.
    fn factorial(n: u64) -> f64 {
        if n == 0 || n == 1 {
            1.0
        } else {
            (2..=n).fold(1.0_f64, |acc, i| acc * i as f64)
        }
    }

    /// Matrix sine via Taylor series.
    pub fn sin(m: &Matrix, terms: usize) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        let mut result = Matrix::zeros(n, n);
        let mut term = m.clone();
        let mut sign = 1.0;
        
        for k in 1..=terms {
            let factorial = Self::factorial((2 * k - 1) as u64);
            let coeff = sign / factorial;
            result = result.add(&term.scale(coeff))?;
            term = term.mul(m)?.mul(m)?;
            sign = -sign;
            
            let term_norm = crate::norms::MatrixNorms::linf(&term);
            if term_norm < 1e-15 {
                break;
            }
        }
        
        Ok(result)
    }

    /// Matrix cosine via Taylor series.
    pub fn cos(m: &Matrix, terms: usize) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        let mut result = Matrix::identity(n);
        let mut term = Matrix::identity(n);
        let mut sign = -1.0;
        
        for k in 1..=terms {
            term = term.mul(m)?.mul(m)?;
            let factorial = Self::factorial((2 * k) as u64);
            let coeff = sign / factorial;
            result = result.add(&term.scale(coeff))?;
            sign = -sign;
            
            let term_norm = crate::norms::MatrixNorms::linf(&term);
            if term_norm < 1e-15 {
                break;
            }
        }
        
        Ok(result)
    }

    /// Matrix hyperbolic sine.
    pub fn sinh(m: &Matrix, terms: usize) -> MathResult<Matrix> {
        let sin = Self::sin(m, terms)?;
        let sin_neg = Self::sin(&m.scale(-1.0), terms)?;
        Ok(sin.sub(&sin_neg)?.scale(0.5))
    }

    /// Matrix hyperbolic cosine.
    pub fn cosh(m: &Matrix, terms: usize) -> MathResult<Matrix> {
        let cos = Self::cos(m, terms)?;
        let cos_neg = Self::cos(&m.scale(-1.0), terms)?;
        Ok(cos.add(&cos_neg)?.scale(0.5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_power() {
        let m = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 3.0]]).unwrap();
        let m_squared = MatrixPower::compute(&m, 2).unwrap();
        
        assert!((m_squared.get(0, 0) - 4.0).abs() < 1e-10);
        assert!((m_squared.get(1, 1) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_power_zero() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let m_zero = MatrixPower::compute(&m, 0).unwrap();
        
        assert!(m_zero.get(0, 0) - 1.0 < 1e-10);
        assert!(m_zero.get(1, 1) - 1.0 < 1e-10);
    }

    #[test]
    fn test_matrix_sqrt() {
        let m = Matrix::from_rows(&[&[4.0, 0.0], &[0.0, 9.0]]).unwrap();
        let sqrt_m = MatrixPower::sqrt(&m).unwrap();
        
        assert!((sqrt_m.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((sqrt_m.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_commutator() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let b = Matrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]).unwrap();
        
        let comm = MatrixPower::commutator(&a, &b).unwrap();
        assert!(comm.get(0, 0).abs() < 1e-10); // Should be zero for 2x2
    }

    #[test]
    fn test_polynomial_evaluate() {
        let m = Matrix::identity(2);
        let coeffs = vec![1.0, 2.0, 3.0]; // I + 2I + 3I = 6I
        let result = MatrixPolynomial::evaluate(&m, &coeffs).unwrap();
        
        assert!((result.get(0, 0) - 6.0).abs() < 1e-10);
    }
}
