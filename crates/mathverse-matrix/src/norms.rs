//! Matrix norms: Frobenius, L1, L∞, spectral, induced norms, nuclear norm.

use crate::Matrix;
use mathverse_core::error::MathResult;

/// Matrix norms.
pub struct MatrixNorms;

impl MatrixNorms {
    /// Frobenius norm: ||A||_F = sqrt(Σ |a_ij|²).
    pub fn frobenius(m: &Matrix) -> f64 {
        m.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// L1 norm (maximum column sum): ||A||_1 = max_j Σ_i |a_ij|.
    pub fn l1(m: &Matrix) -> f64 {
        (0..m.cols)
            .map(|j| (0..m.rows).map(|i| m.get(i, j).abs()).sum::<f64>())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// L∞ norm (maximum row sum): ||A||_∞ = max_i Σ_j |a_ij|.
    pub fn linf(m: &Matrix) -> f64 {
        (0..m.rows)
            .map(|i| (0..m.cols).map(|j| m.get(i, j).abs()).sum::<f64>())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Max norm (maximum absolute entry): ||A||_max = max_ij |a_ij|.
    pub fn max(m: &Matrix) -> f64 {
        m.data.iter().map(|&x| x.abs()).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Spectral norm (largest singular value): ||A||_2 = σ_max.
    pub fn spectral(m: &Matrix) -> MathResult<f64> {
        let svd = m.svd()?;
        Ok(svd.s[0])
    }

    /// Nuclear norm (sum of singular values): ||A||_* = Σ σ_i.
    pub fn nuclear(m: &Matrix) -> MathResult<f64> {
        let svd = m.svd()?;
        Ok(svd.s.iter().sum())
    }

    /// P-norm for vectors (used in induced matrix norms).
    pub fn vector_p_norm(v: &[f64], p: f64) -> f64 {
        if p == f64::INFINITY {
            v.iter().map(|&x| x.abs()).fold(f64::NEG_INFINITY, f64::max)
        } else {
            v.iter().map(|&x| x.abs().powf(p)).sum::<f64>().powf(1.0 / p)
        }
    }

    /// Induced p-norm: ||A||_p = max_{||x||_p=1} ||Ax||_p.
    /// For p=1, this equals L1 norm; for p=∞, equals L∞ norm.
    pub fn induced_p_norm(m: &Matrix, p: f64) -> MathResult<f64> {
        if p == 1.0 {
            Ok(Self::l1(m))
        } else if p == f64::INFINITY {
            Ok(Self::linf(m))
        } else if p == 2.0 {
            Self::spectral(m)
        } else {
            // General case: approximate via power iteration
            Self::approximate_induced_norm(m, p)
        }
    }

    /// Approximate induced p-norm using power iteration.
    fn approximate_induced_norm(m: &Matrix, p: f64) -> MathResult<f64> {
        let n = m.cols;
        let mut x = vec![1.0 / (n as f64).sqrt(); n];
        
        for _ in 0..100 {
            let ax = m.mul_vec(&mathverse_vector::Vector::new(x.clone()))?;
            let norm = Self::vector_p_norm(&ax.data, p);
            if norm > 0.0 {
                x = ax.data.iter().map(|&v| v / norm).collect();
            }
        }
        
        let ax = m.mul_vec(&mathverse_vector::Vector::new(x.clone()))?;
        Ok(Self::vector_p_norm(&ax.data, p))
    }

    /// Schatten p-norm: (Σ σ_i^p)^(1/p).
    pub fn schatten_p_norm(m: &Matrix, p: f64) -> MathResult<f64> {
        let svd = m.svd()?;
        let sum: f64 = svd.s.iter().map(|&s| s.powf(p)).sum();
        Ok(sum.powf(1.0 / p))
    }

    /// Trace norm (same as nuclear norm).
    pub fn trace(m: &Matrix) -> MathResult<f64> {
        Self::nuclear(m)
    }

    /// Relative error: ||A - B|| / ||B||.
    pub fn relative_error(a: &Matrix, b: &Matrix, norm: fn(&Matrix) -> f64) -> f64 {
        let diff = a.sub(b).unwrap_or_else(|_| Matrix::zeros(1, 1));
        let num = norm(&diff);
        let denom = norm(b);
        if denom > 0.0 {
            num / denom
        } else {
            num
        }
    }

    /// Distance between two matrices using specified norm.
    pub fn distance(a: &Matrix, b: &Matrix, norm: fn(&Matrix) -> f64) -> MathResult<f64> {
        let diff = a.sub(b)?;
        Ok(norm(&diff))
    }
}

/// Norm properties and comparisons.
pub struct NormProperties;

impl NormProperties {
    /// Check if a norm satisfies triangle inequality: ||A + B|| ≤ ||A|| + ||B||.
    pub fn triangle_inequality(
        a: &Matrix,
        b: &Matrix,
        norm: fn(&Matrix) -> f64,
        tolerance: f64,
    ) -> bool {
        let sum = a.add(b).unwrap_or_else(|_| Matrix::zeros(1, 1));
        let lhs = norm(&sum);
        let rhs = norm(a) + norm(b);
        lhs <= rhs + tolerance
    }

    /// Check homogeneity: ||αA|| = |α| ||A||.
    pub fn homogeneity(
        m: &Matrix,
        alpha: f64,
        norm: fn(&Matrix) -> f64,
        tolerance: f64,
    ) -> bool {
        let scaled = m.scale(alpha);
        let lhs = norm(&scaled);
        let rhs = alpha.abs() * norm(m);
        (lhs - rhs).abs() < tolerance
    }

    /// Check if norm is sub-multiplicative: ||AB|| ≤ ||A|| ||B||.
    pub fn sub_multiplicative(
        a: &Matrix,
        b: &Matrix,
        norm: fn(&Matrix) -> f64,
        tolerance: f64,
    ) -> bool {
        let prod = a.mul(b).unwrap_or_else(|_| Matrix::zeros(1, 1));
        let lhs = norm(&prod);
        let rhs = norm(a) * norm(b);
        lhs <= rhs + tolerance
    }

    /// Consistency between matrix and vector norms.
    pub fn consistency(
        m: &Matrix,
        v: &[f64],
        matrix_norm: fn(&Matrix) -> f64,
        vector_norm: fn(&[f64]) -> f64,
        tolerance: f64,
    ) -> bool {
        let mv = m.mul_vec(&mathverse_vector::Vector::new(v.to_vec())).unwrap_or_else(|_| mathverse_vector::Vector::new(vec![0.0]));
        let lhs = vector_norm(&mv.data);
        let rhs = matrix_norm(m) * vector_norm(v);
        lhs <= rhs + tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_matrix() -> Matrix {
        Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap()
    }

    #[test]
    fn test_frobenius_norm() {
        let m = test_matrix();
        let norm = MatrixNorms::frobenius(&m);
        let expected = (1.0_f64 + 4.0 + 9.0 + 16.0).sqrt();
        assert!((norm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_l1_norm() {
        let m = test_matrix();
        let norm = MatrixNorms::l1(&m);
        let expected = (1.0_f64 + 3.0).max(2.0 + 4.0);
        assert!((norm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_linf_norm() {
        let m = test_matrix();
        let norm = MatrixNorms::linf(&m);
        let expected = (1.0_f64 + 2.0).max(3.0 + 4.0);
        assert!((norm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_max_norm() {
        let m = test_matrix();
        let norm = MatrixNorms::max(&m);
        assert!((norm - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_norm() {
        let m = Matrix::identity(2);
        let norm = MatrixNorms::spectral(&m).unwrap();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_inequality() {
        let a = test_matrix();
        let b = Matrix::identity(2);
        assert!(NormProperties::triangle_inequality(&a, &b, MatrixNorms::frobenius, 1e-10));
    }

    #[test]
    fn test_homogeneity() {
        let m = test_matrix();
        assert!(NormProperties::homogeneity(&m, 2.0, MatrixNorms::frobenius, 1e-10));
    }
}
