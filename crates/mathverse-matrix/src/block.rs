//! Block matrices: operations on matrices partitioned into blocks.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Block matrix structure.
#[derive(Debug, Clone)]
pub struct BlockMatrix {
    pub rows: usize,
    pub cols: usize,
    pub block_rows: usize,
    pub block_cols: usize,
    pub blocks: Vec<Matrix>,
}

impl BlockMatrix {
    /// Create block matrix from list of blocks.
    pub fn new(blocks: Vec<Matrix>, block_rows: usize, block_cols: usize) -> MathResult<Self> {
        if blocks.len() != block_rows * block_cols {
            return Err(MathError::InvalidArgument("block count doesn't match block dimensions"));
        }
        
        // Check block dimensions are consistent
        let mut row_sizes = Vec::new();
        let mut col_sizes = Vec::new();
        
        for i in 0..block_rows {
            let first_block = &blocks[i * block_cols];
            row_sizes.push(first_block.rows);
        }
        
        for j in 0..block_cols {
            let first_block = &blocks[j];
            col_sizes.push(first_block.cols);
        }
        
        for (idx, block) in blocks.iter().enumerate() {
            let i = idx / block_cols;
            let j = idx % block_cols;
            
            if block.rows != row_sizes[i] {
                return Err(MathError::InvalidArgument("inconsistent block row sizes"));
            }
            if block.cols != col_sizes[j] {
                return Err(MathError::InvalidArgument("inconsistent block column sizes"));
            }
        }
        
        let total_rows: usize = row_sizes.iter().sum();
        let total_cols: usize = col_sizes.iter().sum();
        
        Ok(BlockMatrix {
            rows: total_rows,
            cols: total_cols,
            block_rows,
            block_cols,
            blocks,
        })
    }

    /// Get block at (block_i, block_j).
    pub fn get_block(&self, block_i: usize, block_j: usize) -> &Matrix {
        &self.blocks[block_i * self.block_cols + block_j]
    }

    /// Set block at (block_i, block_j).
    pub fn set_block(&mut self, block_i: usize, block_j: usize, block: Matrix) -> MathResult<()> {
        if block_i >= self.block_rows || block_j >= self.block_cols {
            return Err(MathError::InvalidArgument("block index out of bounds"));
        }
        
        let idx = block_i * self.block_cols + block_j;
        self.blocks[idx] = block;
        Ok(())
    }

    /// Convert to full matrix.
    pub fn to_full(&self) -> Matrix {
        let mut result = Matrix::zeros(self.rows, self.cols);
        
        let mut row_offset = 0;
        for i in 0..self.block_rows {
            let mut col_offset = 0;
            for j in 0..self.block_cols {
                let block = self.get_block(i, j);
                
                for bi in 0..block.rows {
                    for bj in 0..block.cols {
                        result.set(row_offset + bi, col_offset + bj, block.get(bi, bj));
                    }
                }
                
                col_offset += block.cols;
            }
            row_offset += self.get_block(i, 0).rows;
        }
        
        result
    }

    /// Create block matrix from full matrix with given block sizes.
    pub fn from_full(m: &Matrix, row_sizes: &[usize], col_sizes: &[usize]) -> MathResult<Self> {
        let block_rows = row_sizes.len();
        let block_cols = col_sizes.len();
        let mut blocks = Vec::new();
        
        let mut row_offset = 0;
        for &row_size in row_sizes {
            let mut col_offset = 0;
            for &col_size in col_sizes {
                let mut block = Matrix::zeros(row_size, col_size);
                
                for bi in 0..row_size {
                    for bj in 0..col_size {
                        block.set(bi, bj, m.get(row_offset + bi, col_offset + bj));
                    }
                }
                
                blocks.push(block);
                col_offset += col_size;
            }
            row_offset += row_size;
        }
        
        Self::new(blocks, block_rows, block_cols)
    }

    /// Block matrix addition.
    pub fn add(&self, other: &BlockMatrix) -> MathResult<BlockMatrix> {
        if self.block_rows != other.block_rows || self.block_cols != other.block_cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut new_blocks = Vec::new();
        
        for i in 0..self.block_rows {
            for j in 0..self.block_cols {
                let sum = self.get_block(i, j).add(other.get_block(i, j))?;
                new_blocks.push(sum);
            }
        }
        
        Self::new(new_blocks, self.block_rows, self.block_cols)
    }

    /// Block matrix multiplication.
    pub fn mul(&self, other: &BlockMatrix) -> MathResult<BlockMatrix> {
        if self.block_cols != other.block_rows {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut new_blocks = Vec::new();
        
        for i in 0..self.block_rows {
            for j in 0..other.block_cols {
                let mut sum = Matrix::zeros(
                    self.get_block(i, 0).rows,
                    other.get_block(0, j).cols,
                );
                
                for k in 0..self.block_cols {
                    let product = self.get_block(i, k).mul(other.get_block(k, j))?;
                    sum = sum.add(&product)?;
                }
                
                new_blocks.push(sum);
            }
        }
        
        Self::new(new_blocks, self.block_rows, other.block_cols)
    }

    /// Block matrix transpose.
    pub fn transpose(&self) -> BlockMatrix {
        let mut new_blocks = Vec::new();
        
        for j in 0..self.block_cols {
            for i in 0..self.block_rows {
                new_blocks.push(self.get_block(i, j).transpose());
            }
        }
        
        BlockMatrix {
            rows: self.cols,
            cols: self.rows,
            block_rows: self.block_cols,
            block_cols: self.block_rows,
            blocks: new_blocks,
        }
    }
}

/// Block matrix operations.
pub struct BlockOperations;

impl BlockOperations {
    /// Create block diagonal matrix.
    pub fn diagonal(blocks: Vec<Matrix>) -> MathResult<BlockMatrix> {
        let n = blocks.len();
        let mut full_blocks = Vec::new();
        
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    full_blocks.push(blocks[i].clone());
                } else {
                    let rows = blocks[i].rows;
                    let cols = blocks[j].cols;
                    full_blocks.push(Matrix::zeros(rows, cols));
                }
            }
        }
        
        BlockMatrix::new(full_blocks, n, n)
    }

    /// Create block triangular matrix.
    pub fn triangular(blocks: Vec<Matrix>, upper: bool) -> MathResult<BlockMatrix> {
        let n = blocks.len();
        let mut full_blocks = Vec::new();
        
        for i in 0..n {
            for j in 0..n {
                if (upper && j >= i) || (!upper && j <= i) {
                    full_blocks.push(blocks[i * n + j].clone());
                } else {
                    let rows = blocks[i * n].rows;
                    let cols = blocks[j].cols;
                    full_blocks.push(Matrix::zeros(rows, cols));
                }
            }
        }
        
        BlockMatrix::new(full_blocks, n, n)
    }

    /// Extract block from full matrix.
    pub fn extract_block(m: &Matrix, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Matrix {
        let mut block = Matrix::zeros(rows, cols);
        
        for i in 0..rows {
            for j in 0..cols {
                block.set(i, j, m.get(row_start + i, col_start + j));
            }
        }
        
        block
    }

    /// Insert block into full matrix.
    pub fn insert_block(m: &mut Matrix, block: &Matrix, row_start: usize, col_start: usize) {
        for i in 0..block.rows {
            for j in 0..block.cols {
                m.set(row_start + i, col_start + j, block.get(i, j));
            }
        }
    }

    /// Block LU decomposition.
    pub fn block_lu(m: &BlockMatrix) -> MathResult<(BlockMatrix, BlockMatrix)> {
        let n = m.block_rows;

        let mut l = BlockOperations::diagonal(vec![Matrix::identity(m.get_block(0, 0).rows); n])?;
        let mut u = m.clone();
        
        for k in 0..n {
            let u_kk = u.get_block(k, k);
            let inv_u_kk = u_kk.inverse()?;
            
            for i in (k + 1)..n {
                let u_ik = u.get_block(i, k);
                let l_ik = u_ik.mul(&inv_u_kk)?;
                l.set_block(i, k, l_ik.clone())?;

                for j in k..n {
                    let u_ij = u.get_block(i, j);
                    let u_kj = u.get_block(k, j);
                    let update = l_ik.mul(&u_kj)?;
                    let new_u_ij = u_ij.sub(&update)?;
                    u.set_block(i, j, new_u_ij)?;
                }
            }
        }
        
        Ok((l, u))
    }

    /// Block Cholesky decomposition.
    pub fn block_cholesky(m: &BlockMatrix) -> MathResult<BlockMatrix> {
        let n = m.block_rows;
        let mut l = BlockOperations::diagonal(vec![Matrix::zeros(m.get_block(0, 0).rows, m.get_block(0, 0).cols); n])?;
        
        for j in 0..n {
            // L_jj = Cholesky(A_jj - sum(L_jk L_jk^T))
            let mut sum = m.get_block(j, j).clone();
            for k in 0..j {
                let l_jk = l.get_block(j, k);
                let l_jk_t = l_jk.transpose();
                let product = l_jk.mul(&l_jk_t)?;
                sum = sum.sub(&product)?;
            }
            
            let l_jj = sum.cholesky()?;
            l.set_block(j, j, l_jj.clone())?;

            // L_ij = (A_ij - sum(L_ik L_jk^T)) L_jj^{-T}
            for i in (j + 1)..n {
                let mut sum_ij = m.get_block(i, j).clone();
                for k in 0..j {
                    let l_ik = l.get_block(i, k);
                    let l_jk = l.get_block(j, k);
                    let l_jk_t = l_jk.transpose();
                    let product = l_ik.mul(&l_jk_t)?;
                    sum_ij = sum_ij.sub(&product)?;
                }

                let l_jj_inv = l_jj.inverse()?;
                let l_jj_inv_t = l_jj_inv.transpose();
                let l_ij = sum_ij.mul(&l_jj_inv_t)?;
                l.set_block(i, j, l_ij)?;
            }
        }
        
        Ok(l)
    }

    /// Schur complement.
    pub fn schur_complement(m: &BlockMatrix, block_i: usize, block_j: usize) -> MathResult<Matrix> {
        let a = m.get_block(block_i, block_i);
        let b = m.get_block(block_i, block_j);
        let c = m.get_block(block_j, block_i);
        let d = m.get_block(block_j, block_j);
        
        let a_inv = a.inverse()?;
        let a_inv_b = a_inv.mul(&b)?;
        let c_a_inv_b = c.mul(&a_inv_b)?;
        d.sub(&c_a_inv_b)
    }
}

/// Block matrix utilities.
pub struct BlockUtils;

impl BlockUtils {
    /// Check if block matrix is block diagonal.
    pub fn is_block_diagonal(m: &BlockMatrix, tolerance: f64) -> bool {
        for i in 0..m.block_rows {
            for j in 0..m.block_cols {
                if i != j {
                    let block = m.get_block(i, j);
                    let norm = crate::norms::MatrixNorms::frobenius(block);
                    if norm > tolerance {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check if block matrix is block symmetric.
    pub fn is_block_symmetric(m: &BlockMatrix, tolerance: f64) -> bool {
        if m.block_rows != m.block_cols {
            return false;
        }
        
        for i in 0..m.block_rows {
            for j in 0..m.block_cols {
                let block_ij = m.get_block(i, j);
                let block_ji = m.get_block(j, i);
                let block_ji_t = block_ji.transpose();
                
                let diff = block_ij.sub(&block_ji_t).unwrap_or_else(|_| Matrix::zeros(1, 1));
                let norm = crate::norms::MatrixNorms::frobenius(&diff);
                
                if norm > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Permute blocks.
    pub fn permute_blocks(m: &BlockMatrix, row_perm: &[usize], col_perm: &[usize]) -> MathResult<BlockMatrix> {
        let mut new_blocks = Vec::new();
        
        for &i in row_perm {
            for &j in col_perm {
                new_blocks.push(m.get_block(i, j).clone());
            }
        }
        
        BlockMatrix::new(new_blocks, row_perm.len(), col_perm.len())
    }

    /// Merge blocks.
    pub fn merge_blocks(blocks: &[Matrix], direction: &str) -> MathResult<Matrix> {
        if direction == "horizontal" {
            let rows = blocks[0].rows;
            let mut cols = 0;
            for block in blocks {
                if block.rows != rows {
                    return Err(MathError::DimensionMismatch);
                }
                cols += block.cols;
            }
            
            let mut result = Matrix::zeros(rows, cols);
            let mut col_offset = 0;
            
            for block in blocks {
                for i in 0..rows {
                    for j in 0..block.cols {
                        result.set(i, col_offset + j, block.get(i, j));
                    }
                }
                col_offset += block.cols;
            }
            
            Ok(result)
        } else if direction == "vertical" {
            let cols = blocks[0].cols;
            let mut rows = 0;
            for block in blocks {
                if block.cols != cols {
                    return Err(MathError::DimensionMismatch);
                }
                rows += block.rows;
            }
            
            let mut result = Matrix::zeros(rows, cols);
            let mut row_offset = 0;
            
            for block in blocks {
                for i in 0..block.rows {
                    for j in 0..cols {
                        result.set(row_offset + i, j, block.get(i, j));
                    }
                }
                row_offset += block.rows;
            }
            
            Ok(result)
        } else {
            Err(MathError::InvalidArgument("invalid direction, use 'horizontal' or 'vertical'"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_matrix_creation() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let b = Matrix::from_rows(&[&[5.0], &[6.0]]).unwrap();
        let c = Matrix::from_rows(&[&[7.0, 8.0]]).unwrap();
        let d = Matrix::from_rows(&[&[9.0]]).unwrap();
        
        let blocks = vec![a.clone(), b.clone(), c.clone(), d.clone()];
        let block_mat = BlockMatrix::new(blocks, 2, 2).unwrap();
        
        assert_eq!(block_mat.rows, 3);
        assert_eq!(block_mat.cols, 3);
    }

    #[test]
    fn test_block_to_full() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let b = Matrix::from_rows(&[&[5.0], &[6.0]]).unwrap();
        let c = Matrix::from_rows(&[&[7.0, 8.0]]).unwrap();
        let d = Matrix::from_rows(&[&[9.0]]).unwrap();
        
        let blocks = vec![a, b, c, d];
        let block_mat = BlockMatrix::new(blocks, 2, 2).unwrap();
        let full = block_mat.to_full();
        
        assert_eq!(full.rows, 3);
        assert_eq!(full.cols, 3);
        assert!((full.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((full.get(2, 2) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_block_diagonal() {
        let a = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]).unwrap();
        let b = Matrix::from_rows(&[&[2.0]]).unwrap();
        
        let block_diag = BlockOperations::diagonal(vec![a, b]).unwrap();
        let full = block_diag.to_full();
        
        assert!((full.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((full.get(2, 2) - 2.0).abs() < 1e-10);
        assert!((full.get(0, 2) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_block() {
        let m = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]).unwrap();
        let block = BlockOperations::extract_block(&m, 0, 0, 2, 2);
        
        assert_eq!(block.rows, 2);
        assert_eq!(block.cols, 2);
        assert!((block.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((block.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_merge_blocks_horizontal() {
        let a = Matrix::from_rows(&[&[1.0], &[2.0]]).unwrap();
        let b = Matrix::from_rows(&[&[3.0], &[4.0]]).unwrap();
        
        let merged = BlockUtils::merge_blocks(&[a, b], "horizontal").unwrap();
        
        assert_eq!(merged.rows, 2);
        assert_eq!(merged.cols, 2);
        assert!((merged.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((merged.get(0, 1) - 3.0).abs() < 1e-10);
    }
}
