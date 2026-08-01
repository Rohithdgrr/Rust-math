//! Hadamard product and element-wise matrix operations.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Hadamard product (element-wise multiplication).
pub struct HadamardProduct;

impl HadamardProduct {
    /// Compute Hadamard product: (A ∘ B)_{i,j} = A_{i,j} * B_{i,j}.
    pub fn compute(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = Matrix::zeros(a.rows, a.cols);
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                result.set(i, j, a.get(i, j) * b.get(i, j));
            }
        }
        
        Ok(result)
    }

    /// Hadamard product with scalar: A ∘ c = c * A (element-wise).
    pub fn with_scalar(m: &Matrix, scalar: f64) -> Matrix {
        m.scale(scalar)
    }

    /// Hadamard power: A^{∘n} = A ∘ A ∘ ... ∘ A (n times).
    pub fn power(m: &Matrix, n: u32) -> MathResult<Matrix> {
        if n == 0 {
            return Ok(Matrix::ones(m.rows, m.cols));
        }
        
        let mut result = m.clone();
        for _ in 1..n {
            result = Self::compute(&result, m)?;
        }
        
        Ok(result)
    }

    /// Hadamard division: (A ∘ B) / C (element-wise).
    pub fn divide(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = Matrix::zeros(a.rows, a.cols);
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                let denom = b.get(i, j);
                if denom.abs() < 1e-15 {
                    return Err(MathError::InvalidArgument("division by zero in Hadamard division"));
                }
                result.set(i, j, a.get(i, j) / denom);
            }
        }
        
        Ok(result)
    }

    /// Hadamard square root: √A (element-wise).
    pub fn sqrt(m: &Matrix) -> MathResult<Matrix> {
        let mut result = Matrix::zeros(m.rows, m.cols);
        
        for i in 0..m.rows {
            for j in 0..m.cols {
                let val = m.get(i, j);
                if val < 0.0 {
                    return Err(MathError::InvalidArgument("negative value in Hadamard sqrt"));
                }
                result.set(i, j, val.sqrt());
            }
        }
        
        Ok(result)
    }

    /// Hadamard comparison: A > B (element-wise, returns boolean matrix).
    pub fn greater_than(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = Matrix::zeros(a.rows, a.cols);
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                result.set(i, j, if a.get(i, j) > b.get(i, j) { 1.0 } else { 0.0 });
            }
        }
        
        Ok(result)
    }

    /// Hadamard less than: A < B (element-wise).
    pub fn less_than(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = Matrix::zeros(a.rows, a.cols);
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                result.set(i, j, if a.get(i, j) < b.get(i, j) { 1.0 } else { 0.0 });
            }
        }
        
        Ok(result)
    }

    /// Hadamard equality: A == B (element-wise).
    pub fn equal(a: &Matrix, b: &Matrix, tolerance: f64) -> MathResult<Matrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = Matrix::zeros(a.rows, a.cols);
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                result.set(i, j, if (a.get(i, j) - b.get(i, j)).abs() < tolerance { 1.0 } else { 0.0 });
            }
        }
        
        Ok(result)
    }
}

/// Element-wise operations.
pub struct ElementWiseOps;

impl ElementWiseOps {
    /// Element-wise addition: A + B.
    pub fn add(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        a.add(b)
    }

    /// Element-wise subtraction: A - B.
    pub fn sub(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        a.sub(b)
    }

    /// Element-wise maximum: max(A, B).
    pub fn max(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = Matrix::zeros(a.rows, a.cols);
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                result.set(i, j, a.get(i, j).max(b.get(i, j)));
            }
        }
        
        Ok(result)
    }

    /// Element-wise minimum: min(A, B).
    pub fn min(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = Matrix::zeros(a.rows, a.cols);
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                result.set(i, j, a.get(i, j).min(b.get(i, j)));
            }
        }
        
        Ok(result)
    }

    /// Element-wise power: A.^p.
    pub fn pow(m: &Matrix, p: f64) -> Matrix {
        Matrix {
            rows: m.rows,
            cols: m.cols,
            data: m.data.iter().map(|&x| x.powf(p)).collect(),
        }
    }

    /// Element-wise modulo: A mod B.
    pub fn mod_op(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = Matrix::zeros(a.rows, a.cols);
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                let denom = b.get(i, j);
                if denom.abs() < 1e-15 {
                    return Err(MathError::InvalidArgument("division by zero in modulo"));
                }
                result.set(i, j, a.get(i, j) % denom);
            }
        }
        
        Ok(result)
    }

    /// Clip values to range [min, max].
    pub fn clip(m: &Matrix, min_val: f64, max_val: f64) -> Matrix {
        Matrix {
            rows: m.rows,
            cols: m.cols,
            data: m.data.iter().map(|&x| x.max(min_val).min(max_val)).collect(),
        }
    }

    /// Apply function element-wise.
    pub fn apply(m: &Matrix, f: impl Fn(f64) -> f64) -> Matrix {
        Matrix {
            rows: m.rows,
            cols: m.cols,
            data: m.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Element-wise reciprocal: 1./A.
    pub fn reciprocal(m: &Matrix) -> MathResult<Matrix> {
        let mut result = Matrix::zeros(m.rows, m.cols);
        
        for i in 0..m.rows {
            for j in 0..m.cols {
                let val = m.get(i, j);
                if val.abs() < 1e-15 {
                    return Err(MathError::InvalidArgument("zero value in reciprocal"));
                }
                result.set(i, j, 1.0 / val);
            }
        }
        
        Ok(result)
    }

    /// Element-wise square: A.^2.
    pub fn square(m: &Matrix) -> Matrix {
        Self::pow(m, 2.0)
    }

    /// Element-wise cube: A.^3.
    pub fn cube(m: &Matrix) -> Matrix {
        Self::pow(m, 3.0)
    }

    /// Element-wise inverse square root: 1./sqrt(A).
    pub fn inv_sqrt(m: &Matrix) -> MathResult<Matrix> {
        let sqrt_m = HadamardProduct::sqrt(m)?;
        Self::reciprocal(&sqrt_m)
    }
}

/// Matrix of ones.
pub struct MatrixOnes;

impl MatrixOnes {
    /// Create matrix of ones.
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![1.0; rows * cols],
        }
    }

    /// Hadamard identity: A ∘ I = A.
    pub fn hadamard_identity(m: &Matrix) -> MathResult<Matrix> {
        let ones = Self::new(m.rows, m.cols);
        HadamardProduct::compute(m, &ones)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_product() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let b = Matrix::from_rows(&[&[2.0, 3.0], &[4.0, 5.0]]).unwrap();
        
        let hadamard = HadamardProduct::compute(&a, &b).unwrap();
        
        assert!((hadamard.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((hadamard.get(0, 1) - 6.0).abs() < 1e-10);
        assert!((hadamard.get(1, 0) - 12.0).abs() < 1e-10);
        assert!((hadamard.get(1, 1) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_power() {
        let m = Matrix::from_rows(&[&[2.0, 3.0]]).unwrap();
        let power = HadamardProduct::power(&m, 3).unwrap();
        
        assert!((power.get(0, 0) - 8.0).abs() < 1e-10);
        assert!((power.get(0, 1) - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_elementwise_max() {
        let a = Matrix::from_rows(&[&[1.0, 5.0], &[3.0, 2.0]]).unwrap();
        let b = Matrix::from_rows(&[&[2.0, 3.0], &[4.0, 5.0]]).unwrap();
        
        let max = ElementWiseOps::max(&a, &b).unwrap();
        
        assert!((max.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((max.get(0, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_elementwise_pow() {
        let m = Matrix::from_rows(&[&[2.0, 3.0]]).unwrap();
        let pow = ElementWiseOps::pow(&m, 2.0);
        
        assert!((pow.get(0, 0) - 4.0).abs() < 1e-10);
        assert!((pow.get(0, 1) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip() {
        let m = Matrix::from_rows(&[&[-5.0, 10.0], &[3.0, 15.0]]).unwrap();
        let clipped = ElementWiseOps::clip(&m, 0.0, 10.0);
        
        assert!((clipped.get(0, 0) - 0.0).abs() < 1e-10);
        assert!((clipped.get(0, 1) - 10.0).abs() < 1e-10);
        assert!((clipped.get(1, 1) - 10.0).abs() < 1e-10);
    }
}
