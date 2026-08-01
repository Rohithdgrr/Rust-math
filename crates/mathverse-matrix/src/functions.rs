//! Matrix functions: exponential, logarithm, square root, and other matrix functions.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Matrix exponential: exp(A) = Σ A^k / k!.
pub struct MatrixExponential;

impl MatrixExponential {
    /// Compute matrix exponential using Taylor series (scaling and squaring).
    pub fn compute(m: &Matrix) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        
        // Scaling and squaring method
        let norm = crate::norms::MatrixNorms::linf(m);
        let s = if norm > 1.0 {
            (norm.log2().ceil() as usize).max(1)
        } else {
            0
        };
        
        let a_scaled = if s > 0 {
            m.scale(1.0 / (1 << s) as f64)
        } else {
            m.clone()
        };
        
        // Padé approximation
        let exp_a = Self::pade_approx(&a_scaled)?;
        
        // Squaring
        let mut result = exp_a;
        for _ in 0..s {
            result = result.mul(&result)?;
        }
        
        Ok(result)
    }

    /// Padé approximation for matrix exponential.
    fn pade_approx(m: &Matrix) -> MathResult<Matrix> {
        let n = m.rows;
        let identity = Matrix::identity(n);
        
        // [6/6] Padé approximation coefficients
        let b = [
            720.0, 720.0, 360.0, 120.0, 30.0, 6.0, 1.0
        ];
        
        let mut u = Matrix::identity(n);
        let mut v = Matrix::identity(n);
        let mut a = m.clone();
        
        for i in 1..=6 {
            let coeff = b[6 - i];
            if i % 2 == 0 {
                u = u.add(&a.scale(coeff))?;
            } else {
                v = v.add(&a.scale(coeff))?;
            }
            a = a.mul(m)?;
        }
        
        u.add(&a.scale(1.0))?;
        v.sub(&a.scale(1.0))?;
        
        v.inverse()?.mul(&u)
    }

    /// Matrix exponential via eigenvalue decomposition (for diagonalizable matrices).
    pub fn via_eigen(m: &Matrix) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let (vals, vecs) = m.eigen_symmetric()?;
        let n = m.rows;
        
        // exp(D) where D is diagonal
        let mut exp_d = Matrix::zeros(n, n);
        for i in 0..n {
            exp_d.set(i, i, vals[i].exp());
        }
        
        // exp(A) = V exp(D) V^T
        let v_exp_d = vecs.mul(&exp_d)?;
        v_exp_d.mul(&vecs.transpose())
    }
}

/// Matrix logarithm: log(A) such that exp(log(A)) = A.
pub struct MatrixLogarithm;

impl MatrixLogarithm {
    /// Compute matrix logarithm using inverse scaling and squaring.
    pub fn compute(m: &Matrix) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        // Check if matrix is close to identity
        let identity = Matrix::identity(m.rows);
        let diff = m.sub(&identity)?;
        let norm = crate::norms::MatrixNorms::linf(&diff);
        
        if norm < 0.5 {
            // Use Taylor series for matrices close to I
            Self::taylor_series(m)
        } else {
            // Use eigenvalue decomposition for symmetric matrices
            Self::via_eigen(m)
        }
    }

    /// Taylor series for log(I + X) where ||X|| < 1.
    fn taylor_series(m: &Matrix) -> MathResult<Matrix> {
        let identity = Matrix::identity(m.rows);
        let x = m.sub(&identity)?;
        
        let mut result = Matrix::zeros(m.rows, m.cols);
        let mut term = x.clone();
        let mut sign = 1.0;
        
        for k in 1..=20 {
            let coeff = sign / k as f64;
            result = result.add(&term.scale(coeff))?;
            term = term.mul(&x)?;
            sign = -sign;
            
            let term_norm = crate::norms::MatrixNorms::linf(&term);
            if term_norm < 1e-15 {
                break;
            }
        }
        
        Ok(result)
    }

    /// Matrix logarithm via eigenvalue decomposition.
    pub fn via_eigen(m: &Matrix) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let (vals, vecs) = m.eigen_symmetric()?;
        let n = m.rows;
        
        // Check for positive eigenvalues
        for &v in &vals {
            if v <= 0.0 {
                return Err(MathError::InvalidArgument("matrix has non-positive eigenvalues"));
            }
        }
        
        // sqrt(D) where D is diagonal
        let mut log_d = Matrix::zeros(n, n);
        for i in 0..n {
            log_d.set(i, i, vals[i].ln());
        }
        
        // log(A) = V log(D) V^T
        let v_log_d = vecs.mul(&log_d)?;
        v_log_d.mul(&vecs.transpose())
    }
}

/// Matrix square root: sqrt(A) such that sqrt(A) * sqrt(A) = A.
pub struct MatrixSquareRoot;

impl MatrixSquareRoot {
    /// Compute matrix square root using Denman-Beavers iteration.
    pub fn compute(m: &Matrix, tolerance: f64) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        let mut y = m.clone();
        let mut z = Matrix::identity(n);
        
        for _ in 0..100 {
            let y_inv = y.inverse()?;
            let y_next = y.add(&y_inv)?;
            let y_next = y_next.scale(0.5);
            
            let z_next = z.add(&z.mul(&y_inv)?)?;
            let z_next = z_next.scale(0.5);
            
            let diff = y_next.sub(&y)?;
            let norm = crate::norms::MatrixNorms::linf(&diff);
            
            y = y_next;
            z = z_next;
            
            if norm < tolerance {
                break;
            }
        }
        
        Ok(y)
    }

    /// Matrix square root via eigenvalue decomposition.
    pub fn via_eigen(m: &Matrix) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let (vals, vecs) = m.eigen_symmetric()?;
        let n = m.rows;
        
        // Check for positive eigenvalues
        for &v in &vals {
            if v < 0.0 {
                return Err(MathError::InvalidArgument("matrix has negative eigenvalues"));
            }
        }
        
        // sqrt(D) where D is diagonal
        let mut sqrt_d = Matrix::zeros(n, n);
        for i in 0..n {
            sqrt_d.set(i, i, vals[i].sqrt());
        }
        
        // sqrt(A) = V sqrt(D) V^T
        let v_sqrt_d = vecs.mul(&sqrt_d)?;
        v_sqrt_d.mul(&vecs.transpose())
    }

    /// Principal square root (unique positive definite square root).
    pub fn principal(m: &Matrix) -> MathResult<Matrix> {
        Self::via_eigen(m)
    }
}

/// General matrix functions.
pub struct MatrixFunctions;

impl MatrixFunctions {
    /// Compute factorial for Taylor series coefficients.
    fn factorial(n: u64) -> f64 {
        if n == 0 || n == 1 {
            1.0
        } else {
            (2..=n).fold(1.0_f64, |acc, i| acc * i as f64)
        }
    }

    /// Apply scalar function to matrix element-wise.
    pub fn elementwise(m: &Matrix, f: impl Fn(f64) -> f64) -> Matrix {
        Matrix {
            rows: m.rows,
            cols: m.cols,
            data: m.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Matrix power: A^n for integer n.
    pub fn power(m: &Matrix, n: i32) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        if n == 0 {
            return Ok(Matrix::identity(m.rows));
        }
        
        if n < 0 {
            let inv = m.inverse()?;
            return Self::power(&inv, -n);
        }
        
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

    /// Matrix sine using Taylor series.
    pub fn sin(m: &Matrix) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let identity = Matrix::identity(m.rows);
        let mut result = Matrix::zeros(m.rows, m.cols);
        let mut term = m.clone();
        let mut sign = 1.0;
        
        for k in 1..=20 {
            let coeff = sign / Self::factorial(k as u64);
            result = result.add(&term.scale(coeff))?;
            term = term.mul(m)?;
            sign = -sign;
            
            let term_norm = crate::norms::MatrixNorms::linf(&term);
            if term_norm < 1e-15 {
                break;
            }
        }
        
        Ok(result)
    }

    /// Matrix cosine using Taylor series.
    pub fn cos(m: &Matrix) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let identity = Matrix::identity(m.rows);
        let mut result = identity.clone();
        let mut term = m.clone();
        let mut sign = -1.0;
        
        for k in 2..=20 {
            let coeff = sign / Self::factorial(k as u64);
            result = result.add(&term.scale(coeff))?;
            term = term.mul(m)?;
            sign = -sign;
            
            let term_norm = crate::norms::MatrixNorms::linf(&term);
            if term_norm < 1e-15 {
                break;
            }
        }
        
        Ok(result)
    }

    /// Matrix hyperbolic sine.
    pub fn sinh(m: &Matrix) -> MathResult<Matrix> {
        let exp_m = MatrixExponential::compute(m)?;
        let exp_neg_m = MatrixExponential::compute(&m.scale(-1.0))?;
        Ok(exp_m.sub(&exp_neg_m)?.scale(0.5))
    }

    /// Matrix hyperbolic cosine.
    pub fn cosh(m: &Matrix) -> MathResult<Matrix> {
        let exp_m = MatrixExponential::compute(m)?;
        let exp_neg_m = MatrixExponential::compute(&m.scale(-1.0))?;
        Ok(exp_m.add(&exp_neg_m)?.scale(0.5))
    }

    /// Matrix absolute value (element-wise).
    pub fn abs(m: &Matrix) -> Matrix {
        Self::elementwise(m, f64::abs)
    }

    /// Matrix sign function (element-wise).
    pub fn sign(m: &Matrix) -> Matrix {
        Self::elementwise(m, |x| x.signum())
    }

    /// Matrix floor (element-wise).
    pub fn floor(m: &Matrix) -> Matrix {
        Self::elementwise(m, f64::floor)
    }

    /// Matrix ceil (element-wise).
    pub fn ceil(m: &Matrix) -> Matrix {
        Self::elementwise(m, f64::ceil)
    }

    /// Matrix round (element-wise).
    pub fn round(m: &Matrix) -> Matrix {
        Self::elementwise(m, f64::round)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_exp_identity() {
        let m = Matrix::identity(2);
        let exp_m = MatrixExponential::compute(&m).unwrap();
        let e = core::f64::consts::E;
        
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { e } else { 0.0 };
                assert!((exp_m.get(i, j) - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_sqrt_identity() {
        let m = Matrix::identity(3);
        let sqrt_m = MatrixSquareRoot::compute(&m, 1e-10).unwrap();
        
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((sqrt_m.get(i, j) - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_power() {
        let m = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 3.0]]).unwrap();
        let m_squared = MatrixFunctions::power(&m, 2).unwrap();
        
        assert!((m_squared.get(0, 0) - 4.0).abs() < 1e-10);
        assert!((m_squared.get(1, 1) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_elementwise_abs() {
        let m = Matrix::from_rows(&[&[-1.0, 2.0], &[3.0, -4.0]]).unwrap();
        let abs_m = MatrixFunctions::abs(&m);
        
        assert!((abs_m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((abs_m.get(1, 1) - 4.0).abs() < 1e-10);
    }
}
