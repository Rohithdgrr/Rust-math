//! Positive definiteness tests and related matrix properties.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Positive definiteness testing.
pub struct PositiveDefinite;

impl PositiveDefinite {
    /// Check if matrix is positive definite using Cholesky decomposition.
    pub fn is_positive_definite(m: &Matrix) -> bool {
        m.cholesky().is_ok()
    }

    /// Check if matrix is positive semi-definite.
    pub fn is_positive_semi_definite(m: &Matrix, tolerance: f64) -> bool {
        if !m.is_square() || !m.is_symmetric(1e-10) {
            return false;
        }
        
        let (vals, _) = m.eigen_symmetric().unwrap_or((vec![], Matrix::zeros(1, 1)));
        vals.iter().all(|&v| v >= -tolerance)
    }

    /// Check if matrix is negative definite.
    pub fn is_negative_definite(m: &Matrix) -> bool {
        if !m.is_square() || !m.is_symmetric(1e-10) {
            return false;
        }
        
        let neg_m = m.scale(-1.0);
        Self::is_positive_definite(&neg_m)
    }

    /// Check if matrix is negative semi-definite.
    pub fn is_negative_semi_definite(m: &Matrix, tolerance: f64) -> bool {
        if !m.is_square() || !m.is_symmetric(1e-10) {
            return false;
        }
        
        let neg_m = m.scale(-1.0);
        Self::is_positive_semi_definite(&neg_m, tolerance)
    }

    /// Check if matrix is indefinite.
    pub fn is_indefinite(m: &Matrix, tolerance: f64) -> bool {
        if !m.is_square() || !m.is_symmetric(1e-10) {
            return false;
        }
        
        let (vals, _) = m.eigen_symmetric().unwrap_or((vec![], Matrix::zeros(1, 1)));
        let has_positive = vals.iter().any(|&v| v > tolerance);
        let has_negative = vals.iter().any(|&v| v < -tolerance);
        
        has_positive && has_negative
    }

    /// Check Sylvester's criterion: all leading principal minors positive.
    pub fn sylvester_criterion(m: &Matrix) -> bool {
        if !m.is_square() {
            return false;
        }
        
        for k in 1..=m.rows {
            let leading = Self::leading_principal_minor(m, k);
            let det = leading.det();
            if det.is_err() || det.unwrap() <= 0.0 {
                return false;
            }
        }
        
        true
    }

    /// Extract leading principal minor of order k.
    fn leading_principal_minor(m: &Matrix, k: usize) -> Matrix {
        let mut minor = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                minor.set(i, j, m.get(i, j));
            }
        }
        minor
    }

    /// Check if matrix is symmetric positive definite via eigenvalues.
    pub fn eigenvalue_check(m: &Matrix, tolerance: f64) -> bool {
        if !m.is_square() || !m.is_symmetric(1e-10) {
            return false;
        }
        
        let (vals, _) = match m.eigen_symmetric() {
            Ok(result) => result,
            Err(_) => return false,
        };
        
        vals.iter().all(|&v| v > tolerance)
    }

    /// Make matrix positive definite by adding diagonal shift.
    pub fn make_positive_definite(m: &Matrix, min_eigenvalue: f64) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let (vals, _) = m.eigen_symmetric()?;
        let min_val = vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let shift = if min_val < min_eigenvalue {
            min_eigenvalue - min_val
        } else {
            0.0
        };
        
        let mut result = m.clone();
        for i in 0..m.rows {
            result.set(i, i, result.get(i, i) + shift);
        }
        
        Ok(result)
    }

    /// Nearest positive definite matrix (Higham's algorithm simplified).
    pub fn nearest_positive_definite(m: &Matrix, tolerance: f64) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        // Symmetrize
        let mut a = m.clone();
        for i in 0..m.rows {
            for j in (i + 1)..m.cols {
                let avg = (m.get(i, j) + m.get(j, i)) / 2.0;
                a.set(i, j, avg);
                a.set(j, i, avg);
            }
        }
        
        // Ensure positive eigenvalues
        let (vals, vecs) = a.eigen_symmetric()?;
        let mut vals_clamped = Vec::new();
        for &v in &vals {
            vals_clamped.push(v.max(tolerance));
        }
        
        // Reconstruct
        let d = Matrix::diagonal(&vals_clamped);
        vecs.mul(&d)?.mul(&vecs.transpose())
    }
}

/// Matrix definiteness classification.
pub struct DefinitenessClassification;

impl DefinitenessClassification {
    /// Classify matrix definiteness.
    pub fn classify(m: &Matrix, tolerance: f64) -> &'static str {
        if !m.is_square() {
            return "not square";
        }
        
        if !m.is_symmetric(1e-10) {
            return "not symmetric";
        }
        
        if PositiveDefinite::is_positive_definite(m) {
            "positive definite"
        } else if PositiveDefinite::is_positive_semi_definite(m, tolerance) {
            "positive semi-definite"
        } else if PositiveDefinite::is_negative_definite(m) {
            "negative definite"
        } else if PositiveDefinite::is_negative_semi_definite(m, tolerance) {
            "negative semi-definite"
        } else if PositiveDefinite::is_indefinite(m, tolerance) {
            "indefinite"
        } else {
            "unknown"
        }
    }

    /// Check if matrix forms an inner product.
    pub fn is_inner_product(m: &Matrix) -> bool {
        PositiveDefinite::is_positive_definite(m)
    }

    /// Check if matrix defines a valid covariance matrix.
    pub fn is_valid_covariance(m: &Matrix) -> bool {
        PositiveDefinite::is_positive_semi_definite(m, 1e-10) && m.is_symmetric(1e-10)
    }
}

/// Cholesky-based definiteness tests.
pub struct CholeskyTests;

impl CholeskyTests {
    /// Attempt Cholesky and check for success.
    pub fn test(m: &Matrix) -> bool {
        m.cholesky().is_ok()
    }

    /// Cholesky with pivoting for indefinite matrices.
    pub fn pivoted_cholesky(m: &Matrix, tolerance: f64) -> MathResult<(Matrix, Vec<usize>)> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        let mut l = Matrix::zeros(n, n);
        let mut pivots: Vec<usize> = (0..n).collect();
        
        for k in 0..n {
            // Find pivot
           let mut max_diag = m.get(pivots[k], pivots[k]);
            let mut max_idx = k;
            
            for i in (k + 1)..n {
                let diag = m.get(pivots[i], pivots[i]);
                if diag > max_diag {
                    max_diag = diag;
                    max_idx = i;
                }
            }
            
            pivots.swap(k, max_idx);
            
            let d = m.get(pivots[k], pivots[k]);
            if d <= tolerance {
                return Err(MathError::InvalidArgument("matrix not positive definite"));
            }
            
            l.set(k, k, d.sqrt());
            
            for j in (k + 1)..n {
                let mut sum = 0.0;
                for i in 0..k {
                    sum += l.get(k, i) * l.get(j, i);
                }
                l.set(j, k, (m.get(pivots[j], pivots[k]) - sum) / l.get(k, k));
            }
        }
        
        Ok((l, pivots))
    }
}

/// Quadratic form analysis.
pub struct QuadraticForm;

impl QuadraticForm {
    /// Evaluate quadratic form: x^T A x.
    pub fn evaluate(m: &Matrix, x: &[f64]) -> MathResult<f64> {
        if m.rows != x.len() || m.cols != x.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        let x_vec = mathverse_vector::Vector::new(x.to_vec());
        let ax = m.mul_vec(&x_vec)?;
        Ok(x_vec.dot(&ax))
    }

    /// Check if quadratic form is positive for all non-zero x.
    pub fn is_positive(m: &Matrix) -> bool {
        PositiveDefinite::is_positive_definite(m)
    }

    /// Rayleigh quotient: (x^T A x) / (x^T x).
    pub fn rayleigh_quotient(m: &Matrix, x: &[f64]) -> MathResult<f64> {
        let quadratic = Self::evaluate(m, x)?;
        let norm_sq: f64 = x.iter().map(|v| v * v).sum();
        
        if norm_sq > 0.0 {
            Ok(quadratic / norm_sq)
        } else {
            Err(MathError::InvalidArgument("zero vector"))
        }
    }

    /// Minimize Rayleigh quotient (smallest eigenvalue).
    pub fn minimize_rayleigh(m: &Matrix) -> MathResult<(f64, Vec<f64>)> {
        let (vals, vecs) = m.eigen_symmetric()?;
        let min_idx = vals.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        let min_val = vals[min_idx];
        let min_vec = vecs.col(min_idx);
        
        Ok((min_val, min_vec))
    }

    /// Maximize Rayleigh quotient (largest eigenvalue).
    pub fn maximize_rayleigh(m: &Matrix) -> MathResult<(f64, Vec<f64>)> {
        let (vals, vecs) = m.eigen_symmetric()?;
        let max_idx = vals.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        let max_val = vals[max_idx];
        let max_vec = vecs.col(max_idx);
        
        Ok((max_val, max_vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_definite() {
        let m = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        assert!(PositiveDefinite::is_positive_definite(&m));
    }

    #[test]
    fn test_not_positive_definite() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]).unwrap();
        assert!(!PositiveDefinite::is_positive_definite(&m));
    }

    #[test]
    fn test_classification() {
        let pd = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        assert_eq!(DefinitenessClassification::classify(&pd, 1e-10), "positive definite");
        
        let ind = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, -1.0]]).unwrap();
        assert_eq!(DefinitenessClassification::classify(&ind, 1e-10), "indefinite");
    }

    #[test]
    fn test_quadratic_form() {
        let m = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 3.0]]).unwrap();
        let x = vec![1.0, 2.0];
        let qf = QuadraticForm::evaluate(&m, &x).unwrap();
        assert!((qf - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_rayleigh_quotient() {
        let m = Matrix::identity(2);
        let x = vec![1.0, 0.0];
        let rq = QuadraticForm::rayleigh_quotient(&m, &x).unwrap();
        assert!((rq - 1.0).abs() < 1e-10);
    }
}
