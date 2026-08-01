//! Condition number: computation, analysis, and sensitivity to perturbations.

use crate::Matrix;
use crate::norms::MatrixNorms;
use mathverse_core::error::{MathError, MathResult};

/// Condition number analysis.
pub struct ConditionNumber;

impl ConditionNumber {
    /// Condition number in 2-norm: κ(A) = ||A||_2 * ||A⁻¹||_2 = σ_max / σ_min.
    pub fn spectral(m: &Matrix) -> MathResult<f64> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let svd = m.svd()?;
        let sigma_max = svd.s[0];
        let sigma_min = svd.s.last().copied().unwrap_or(0.0);
        
        if sigma_min < 1e-14 {
            Ok(f64::INFINITY)
        } else {
            Ok(sigma_max / sigma_min)
        }
    }

    /// Condition number in Frobenius norm: κ_F(A) = ||A||_F * ||A⁻¹||_F.
    pub fn frobenius(m: &Matrix) -> MathResult<f64> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let norm_a = MatrixNorms::frobenius(m);
        let inv = m.inverse()?;
        let norm_inv = MatrixNorms::frobenius(&inv);
        
        Ok(norm_a * norm_inv)
    }

    /// Condition number in 1-norm: κ_1(A) = ||A||_1 * ||A⁻¹||_1.
    pub fn l1(m: &Matrix) -> MathResult<f64> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let norm_a = MatrixNorms::l1(m);
        let inv = m.inverse()?;
        let norm_inv = MatrixNorms::l1(&inv);
        
        Ok(norm_a * norm_inv)
    }

    /// Condition number in ∞-norm: κ_∞(A) = ||A||_∞ * ||A⁻¹||_∞.
    pub fn linf(m: &Matrix) -> MathResult<f64> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let norm_a = MatrixNorms::linf(m);
        let inv = m.inverse()?;
        let norm_inv = MatrixNorms::linf(&inv);
        
        Ok(norm_a * norm_inv)
    }

    /// General condition number: κ_p(A) = ||A||_p * ||A⁻¹||_p.
    pub fn induced_p(m: &Matrix, p: f64) -> MathResult<f64> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let norm_a = MatrixNorms::induced_p_norm(m, p)?;
        let inv = m.inverse()?;
        let norm_inv = MatrixNorms::induced_p_norm(&inv, p)?;
        
        Ok(norm_a * norm_inv)
    }

    /// Effective condition number for solving Ax = b.
    pub fn effective(m: &Matrix, b: &mathverse_vector::Vector) -> MathResult<f64> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let x = m.solve(b)?;
        let norm_x = MatrixNorms::vector_p_norm(&x.data, 2.0);
        let norm_b = MatrixNorms::vector_p_norm(&b.data, 2.0);
        
        let cond = Self::spectral(m)?;
        Ok(cond * norm_b / norm_x)
    }

    /// Relative error bound for solution perturbation.
    /// Δx/x ≤ κ(A) * (ΔA/A + Δb/b).
    pub fn error_bound(
        m: &Matrix,
        delta_a: f64,
        delta_b: f64,
    ) -> MathResult<f64> {
        let cond = Self::spectral(m)?;
        Ok(cond * (delta_a + delta_b))
    }

    /// Condition number of eigenvalue problem.
    pub fn eigenvalue(m: &Matrix) -> MathResult<f64> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let (vals, vecs) = m.eigen_symmetric()?;
        let cond = vals.iter()
            .zip(vecs.data.chunks(m.rows))
            .map(|(&lam, v)| {
                let v_norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if v_norm > 0.0 {
                    v_norm / lam.abs()
                } else {
                    f64::INFINITY
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        
        Ok(cond)
    }

    /// Condition number for least squares problem.
    pub fn least_squares(m: &Matrix) -> MathResult<f64> {
        let svd = m.svd()?;
        let sigma_max = svd.s[0];
        let sigma_min = svd.s.last().copied().unwrap_or(0.0);
        
        if sigma_min < 1e-14 {
            Ok(f64::INFINITY)
        } else {
            Ok((sigma_max / sigma_min).powi(2))
        }
    }

    /// Reciprocal condition number: 1/κ(A).
    pub fn reciprocal(m: &Matrix) -> MathResult<f64> {
        let cond = Self::spectral(m)?;
        if cond.is_infinite() || cond == 0.0 {
            Ok(0.0)
        } else {
            Ok(1.0 / cond)
        }
    }

    /// Estimate condition number without computing inverse (using SVD).
    pub fn estimate(m: &Matrix) -> MathResult<f64> {
        Self::spectral(m)
    }
}

/// Sensitivity analysis.
pub struct SensitivityAnalysis;

impl SensitivityAnalysis {
    /// Perturbation analysis for linear system Ax = b.
    pub fn linear_system(
        m: &Matrix,
        b: &mathverse_vector::Vector,
        delta_a: f64,
        delta_b: f64,
    ) -> MathResult<(f64, f64)> {
        let cond = ConditionNumber::spectral(m)?;
        let x = m.solve(b)?;
        
        let norm_x = MatrixNorms::vector_p_norm(&x.data, 2.0);
        let norm_b = MatrixNorms::vector_p_norm(&b.data, 2.0);
        
        let relative_error_x = cond * (delta_a + delta_b);
        let absolute_error_x = relative_error_x * norm_x;
        
        Ok((relative_error_x, absolute_error_x))
    }

    /// Backward error analysis.
    pub fn backward_error(
        m: &Matrix,
        x: &mathverse_vector::Vector,
        b: &mathverse_vector::Vector,
    ) -> f64 {
        let residual = m.mul_vec(x).unwrap();
        let diff = residual.sub(&b);
        let norm_residual = MatrixNorms::vector_p_norm(&diff.data, 2.0);
        let norm_b = MatrixNorms::vector_p_norm(&b.data, 2.0);
        
        if norm_b > 0.0 {
            norm_residual / norm_b
        } else {
            norm_residual
        }
    }

    /// Forward error analysis.
    pub fn forward_error(
        x_computed: &mathverse_vector::Vector,
        x_exact: &mathverse_vector::Vector,
    ) -> f64 {
        let diff = x_computed.sub(&x_exact);
        let norm_diff = MatrixNorms::vector_p_norm(&diff.data, 2.0);
        let norm_exact = MatrixNorms::vector_p_norm(&x_exact.data, 2.0);
        
        if norm_exact > 0.0 {
            norm_diff / norm_exact
        } else {
            norm_diff
        }
    }

    /// Componentwise relative error.
    pub fn componentwise_error(
        x_computed: &mathverse_vector::Vector,
        x_exact: &mathverse_vector::Vector,
    ) -> f64 {
        let max_error = x_computed.data.iter()
            .zip(x_exact.data.iter())
            .map(|(&c, &e)| {
                if e.abs() > 0.0 {
                    (c - e).abs() / e.abs()
                } else {
                    (c - e).abs()
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        
        max_error
    }
}

/// Conditioning classification.
pub struct ConditioningClassification;

impl ConditioningClassification {
    /// Classify matrix based on condition number.
    pub fn classify(cond: f64) -> &'static str {
        if cond < 10.0 {
            "well-conditioned"
        } else if cond < 100.0 {
            "moderately conditioned"
        } else if cond < 1000.0 {
            "ill-conditioned"
        } else if cond < 1e10 {
            "severely ill-conditioned"
        } else {
            "singular or nearly singular"
        }
    }

    /// Check if matrix is numerically singular.
    pub fn is_singular(m: &Matrix, tolerance: f64) -> MathResult<bool> {
        let cond = ConditionNumber::spectral(m)?;
        Ok(cond > 1.0 / tolerance)
    }

    /// Distance to nearest singular matrix.
    pub fn distance_to_singular(m: &Matrix) -> MathResult<f64> {
        let svd = m.svd()?;
        Ok(svd.s.last().copied().unwrap_or(0.0))
    }

    /// Effective rank based on singular values.
    pub fn effective_rank(m: &Matrix, tolerance: f64) -> MathResult<usize> {
        let svd = m.svd()?;
        let sigma_max = svd.s[0];
        
        let rank = svd.s.iter()
            .filter(|&&s| s > tolerance * sigma_max)
            .count();
        
        Ok(rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn well_conditioned() -> Matrix {
        Matrix::identity(3)
    }

    fn ill_conditioned() -> Matrix {
        Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 1.000001]]).unwrap()
    }

    #[test]
    fn test_spectral_condition() {
        let m = well_conditioned();
        let cond = ConditionNumber::spectral(&m).unwrap();
        assert!((cond - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ill_conditioned() {
        let m = ill_conditioned();
        let cond = ConditionNumber::spectral(&m).unwrap();
        assert!(cond > 1e6);
    }

    #[test]
    fn test_reciprocal_condition() {
        let m = well_conditioned();
        let rec = ConditionNumber::reciprocal(&m).unwrap();
        assert!((rec - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_classification() {
        assert_eq!(ConditioningClassification::classify(5.0), "well-conditioned");
        assert_eq!(ConditioningClassification::classify(50.0), "moderately conditioned");
        assert_eq!(ConditioningClassification::classify(500.0), "ill-conditioned");
    }

    #[test]
    fn test_backward_error() {
        let m = Matrix::identity(2);
        let x = mathverse_vector::Vector::new(vec![1.0, 2.0]);
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0]);
        let error = SensitivityAnalysis::backward_error(&m, &x, &b);
        assert!(error < 1e-10);
    }
}
