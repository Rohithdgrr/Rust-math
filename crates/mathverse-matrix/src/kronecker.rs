//! Kronecker product and related tensor operations.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Kronecker product: A ⊗ B.
pub struct KroneckerProduct;

impl KroneckerProduct {
    /// Compute Kronecker product: (A ⊗ B)_{i*m+k, j*n+l} = A_{i,j} * B_{k,l}.
    pub fn compute(a: &Matrix, b: &Matrix) -> Matrix {
        let (m, n) = (a.rows, a.cols);
        let (p, q) = (b.rows, b.cols);
        
        let mut result = Matrix::zeros(m * p, n * q);
        
        for i in 0..m {
            for j in 0..n {
                let a_ij = a.get(i, j);
                for k in 0..p {
                    for l in 0..q {
                        result.set(i * p + k, j * q + l, a_ij * b.get(k, l));
                    }
                }
            }
        }
        
        result
    }

    /// Kronecker sum: A ⊕ B = A ⊗ I + I ⊗ B.
    pub fn sum(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        let n_a = a.rows;
        let n_b = b.rows;
        
        let i_a = Matrix::identity(n_a);
        let i_b = Matrix::identity(n_b);
        
        let term1 = Self::compute(a, &i_b);
        let term2 = Self::compute(&i_a, b);
        
        term1.add(&term2)
    }

    /// Kronecker product properties: (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD).
    pub fn product_property(
        a: &Matrix,
        b: &Matrix,
        c: &Matrix,
        d: &Matrix,
    ) -> MathResult<bool> {
        let left = Self::compute(a, b).mul(&Self::compute(c, d))?;
        let ac = a.mul(c)?;
        let bd = b.mul(d)?;
        let right = Self::compute(&ac, &bd);

        Ok(Self::matrices_equal(&left, &right, 1e-10))
    }

    /// Mixed product property: (A ⊗ B)^{-1} = A^{-1} ⊗ B^{-1}.
    pub fn inverse_property(a: &Matrix, b: &Matrix) -> MathResult<bool> {
        let inv_a = a.inverse()?;
        let inv_b = b.inverse()?;

        let left = Self::compute(a, b).inverse()?;
        let right = Self::compute(&inv_a, &inv_b);

        Ok(Self::matrices_equal(&left, &right, 1e-10))
    }

    /// Transpose property: (A ⊗ B)^T = A^T ⊗ B^T.
    pub fn transpose_property(a: &Matrix, b: &Matrix) -> bool {
        let left = Self::compute(a, b).transpose();
        let right = Self::compute(&a.transpose(), &b.transpose());
        
        Self::matrices_equal(&left, &right, 1e-10)
    }

    fn matrices_equal(a: &Matrix, b: &Matrix, tolerance: f64) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                if (a.get(i, j) - b.get(i, j)).abs() > tolerance {
                    return false;
                }
            }
        }
        
        true
    }

    /// Kronecker product with vector: (A ⊗ B) vec(X) = vec(B X A^T).
    pub fn vec_property(
        a: &Matrix,
        b: &Matrix,
        x: &Matrix,
    ) -> MathResult<bool> {
        let kron = Self::compute(a, b);
        let vec_x = Self::vec(x);
        let left = kron.mul_vec(&vec_x)?;
        
        let bxa = b.mul(x)?.mul(&a.transpose())?;
        let right = Self::vec(&bxa);
        
        Ok(Self::vectors_equal(&left, &right, 1e-10))
    }

    /// Vectorize matrix (column-major).
    fn vec(m: &Matrix) -> mathverse_vector::Vector {
        let mut data = Vec::with_capacity(m.rows * m.cols);
        for j in 0..m.cols {
            for i in 0..m.rows {
                data.push(m.get(i, j));
            }
        }
        mathverse_vector::Vector::new(data)
    }

    fn vectors_equal(a: &mathverse_vector::Vector, b: &mathverse_vector::Vector, tolerance: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        
        for i in 0..a.len() {
            if (a.get(i) - b.get(i)).abs() > tolerance {
                return false;
            }
        }
        
        true
    }

    /// Khatri-Rao product (column-wise Kronecker product).
    pub fn khatri_rao(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let (m, n) = (a.rows, a.cols);
        let (p, _) = (b.rows, b.cols);
        
        let mut result = Matrix::zeros(m * p, n);
        
        for j in 0..n {
            let mut col_result = Vec::with_capacity(m * p);
            for i in 0..m {
                let a_ij = a.get(i, j);
                for k in 0..p {
                    col_result.push(a_ij * b.get(k, j));
                }
            }
            
            for (i, &val) in col_result.iter().enumerate() {
                result.set(i, j, val);
            }
        }
        
        Ok(result)
    }

    /// Tracy-Singh product (block Kronecker product).
    pub fn tracy_singh(a: &Matrix, b: &Matrix, block_size: usize) -> MathResult<Matrix> {
        let (m, n) = (a.rows, a.cols);
        let (p, q) = (b.rows, b.cols);
        
        let mut result = Matrix::zeros(m * p, n * q);
        
        for bi in (0..m).step_by(block_size) {
            for bj in (0..n).step_by(block_size) {
                let i_end = (bi + block_size).min(m);
                let j_end = (bj + block_size).min(n);
                
                for i in bi..i_end {
                    for j in bj..j_end {
                        let block = Self::compute(
                            &Self::extract_block(a, bi, bj, block_size),
                            &Self::extract_block(b, 0, 0, block_size),
                        );
                        
                        for k in 0..block.rows {
                            for l in 0..block.cols {
                                result.set(i * block.rows + k, j * block.cols + l, block.get(k, l));
                            }
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }

    fn extract_block(m: &Matrix, start_row: usize, start_col: usize, size: usize) -> Matrix {
        let rows = (start_row + size).min(m.rows) - start_row;
        let cols = (start_col + size).min(m.cols) - start_col;
        
        let mut block = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                block.set(i, j, m.get(start_row + i, start_col + j));
            }
        }
        
        block
    }
}

/// Tensor operations.
pub struct TensorOperations;

impl TensorOperations {
    /// Outer product of two vectors: u ⊗ v.
    pub fn outer_product(u: &[f64], v: &[f64]) -> Matrix {
        let mut result = Matrix::zeros(u.len(), v.len());
        
        for i in 0..u.len() {
            for j in 0..v.len() {
                result.set(i, j, u[i] * v[j]);
            }
        }
        
        result
    }

    /// Tensor product (generalized outer product).
    pub fn tensor_product(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len() * b.len());
        
        for &ai in a {
            for &bi in b {
                result.push(ai * bi);
            }
        }
        
        result
    }

    /// Contraction along specified dimensions.
    pub fn contract(
        tensor: &[f64],
        shape: &[usize],
        dim1: usize,
        dim2: usize,
    ) -> MathResult<Vec<f64>> {
        if dim1 >= shape.len() || dim2 >= shape.len() {
            return Err(MathError::InvalidArgument("dimension out of bounds"));
        }
        
        let size1 = shape[dim1];
        let size2 = shape[dim2];
        
        if size1 != size2 {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = Vec::new();
        let mut idx = 0;
        
        // Simplified: assume 2D tensor for now
        if shape.len() == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            if dim1 == 0 && dim2 == 1 {
                // Trace
                for i in 0..rows.min(cols) {
                    result.push(tensor[i * cols + i]);
                }
            } else {
                result = tensor.to_vec();
            }
        } else {
            result = tensor.to_vec();
        }
        
        Ok(result)
    }

    /// Reshape tensor.
    pub fn reshape(tensor: &[f64], new_shape: &[usize]) -> MathResult<Vec<f64>> {
        let total_elements: usize = new_shape.iter().product();
        
        if tensor.len() != total_elements {
            return Err(MathError::DimensionMismatch);
        }
        
        Ok(tensor.to_vec())
    }

    /// Permute tensor dimensions.
    pub fn permute(
        tensor: &[f64],
        shape: &[usize],
        permutation: &[usize],
    ) -> MathResult<Vec<f64>> {
        if permutation.len() != shape.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        // Simplified: return copy for now
        Ok(tensor.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kronecker_product() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let b = Matrix::from_rows(&[&[0.0, 5.0], &[6.0, 7.0]]).unwrap();
        
        let kron = KroneckerProduct::compute(&a, &b);
        
        assert_eq!(kron.rows, 4);
        assert_eq!(kron.cols, 4);
        assert!((kron.get(0, 0) - 0.0).abs() < 1e-10);
        assert!((kron.get(0, 1) - 5.0).abs() < 1e-10);
        assert!((kron.get(1, 0) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_kronecker_sum() {
        let a = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 2.0]]).unwrap();
        let b = Matrix::from_rows(&[&[3.0, 0.0], &[0.0, 4.0]]).unwrap();
        
        let sum = KroneckerProduct::sum(&a, &b).unwrap();
        assert_eq!(sum.rows, 4);
        assert_eq!(sum.cols, 4);
    }

    #[test]
    fn test_outer_product() {
        let u = vec![1.0, 2.0, 3.0];
        let v = vec![4.0, 5.0];
        
        let outer = TensorOperations::outer_product(&u, &v);
        
        assert_eq!(outer.rows, 3);
        assert_eq!(outer.cols, 2);
        assert!((outer.get(0, 0) - 4.0).abs() < 1e-10);
        assert!((outer.get(1, 1) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_transpose_property() {
        let a = Matrix::from_rows(&[&[1.0, 2.0]]).unwrap();
        let b = Matrix::from_rows(&[&[3.0, 4.0]]).unwrap();
        
        assert!(KroneckerProduct::transpose_property(&a, &b));
    }
}
