//! Sparse matrix formats: CSR, CSC, COO, and conversions.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Compressed Sparse Row (CSR) format.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
    pub col_indices: Vec<usize>,
    pub row_ptr: Vec<usize>,
}

impl CsrMatrix {
    /// Create CSR matrix from full matrix.
    pub fn from_full(m: &Matrix) -> Self {
        let (rows, cols) = (m.rows, m.cols);
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0; rows + 1];
        
        for i in 0..rows {
            for j in 0..cols {
                let val = m.get(i, j);
                if val.abs() > 1e-15 {
                    values.push(val);
                    col_indices.push(j);
                }
            }
            row_ptr[i + 1] = values.len();
        }
        
        CsrMatrix {
            rows,
            cols,
            values,
            col_indices,
            row_ptr,
        }
    }

    /// Get value at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i >= self.rows || j >= self.cols {
            return 0.0;
        }
        
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        
        for idx in start..end {
            if self.col_indices[idx] == j {
                return self.values[idx];
            }
        }
        
        0.0
    }

    /// Set value at (i, j).
    pub fn set(&mut self, i: usize, j: usize, value: f64) -> MathResult<()> {
        if i >= self.rows || j >= self.cols {
            return Err(MathError::InvalidArgument("index out of bounds"));
        }
        
        // Find existing entry
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        
        for idx in start..end {
            if self.col_indices[idx] == j {
                self.values[idx] = value;
                return Ok(());
            }
        }
        
        // Insert new entry (inefficient - rebuild recommended)
        let mut full = self.to_full();
        full.set(i, j, value);
        *self = Self::from_full(&full);
        Ok(())
    }

    /// Convert to full matrix.
    pub fn to_full(&self) -> Matrix {
        let mut m = Matrix::zeros(self.rows, self.cols);
        
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            
            for idx in start..end {
                let j = self.col_indices[idx];
                m.set(i, j, self.values[idx]);
            }
        }
        
        m
    }

    /// Matrix-vector product (optimized for CSR).
    pub fn mul_vec(&self, v: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        if self.cols != v.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = vec![0.0; self.rows];
        
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            
            for idx in start..end {
                let j = self.col_indices[idx];
                result[i] += self.values[idx] * v.get(j);
            }
        }
        
        Ok(mathverse_vector::Vector::new(result))
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Transpose to CSC.
    pub fn transpose(&self) -> CscMatrix {
        CscMatrix::from_csr(self)
    }
}

/// Compressed Sparse Column (CSC) format.
#[derive(Debug, Clone)]
pub struct CscMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
    pub row_indices: Vec<usize>,
    pub col_ptr: Vec<usize>,
}

impl CscMatrix {
    /// Create CSC matrix from full matrix.
    pub fn from_full(m: &Matrix) -> Self {
        let (rows, cols) = (m.rows, m.cols);
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptr = vec![0; cols + 1];
        
        for j in 0..cols {
            for i in 0..rows {
                let val = m.get(i, j);
                if val.abs() > 1e-15 {
                    values.push(val);
                    row_indices.push(i);
                }
            }
            col_ptr[j + 1] = values.len();
        }
        
        CscMatrix {
            rows,
            cols,
            values,
            row_indices,
            col_ptr,
        }
    }

    /// Create CSC from CSR.
    pub fn from_csr(csr: &CsrMatrix) -> Self {
        let full = csr.to_full();
        Self::from_full(&full)
    }

    /// Get value at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i >= self.rows || j >= self.cols {
            return 0.0;
        }
        
        let start = self.col_ptr[j];
        let end = self.col_ptr[j + 1];
        
        for idx in start..end {
            if self.row_indices[idx] == i {
                return self.values[idx];
            }
        }
        
        0.0
    }

    /// Convert to full matrix.
    pub fn to_full(&self) -> Matrix {
        let mut m = Matrix::zeros(self.rows, self.cols);
        
        for j in 0..self.cols {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            
            for idx in start..end {
                let i = self.row_indices[idx];
                m.set(i, j, self.values[idx]);
            }
        }
        
        m
    }

    /// Matrix-vector product (optimized for CSC).
    pub fn mul_vec(&self, v: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        if self.cols != v.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = vec![0.0; self.rows];
        
        for j in 0..self.cols {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            let vj = v.get(j);
            
            for idx in start..end {
                let i = self.row_indices[idx];
                result[i] += self.values[idx] * vj;
            }
        }
        
        Ok(mathverse_vector::Vector::new(result))
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Transpose to CSR.
    pub fn transpose(&self) -> CsrMatrix {
        CsrMatrix::from_full(&self.to_full())
    }
}

/// Coordinate list (COO) format (extended from sparse.rs).
#[derive(Debug, Clone)]
pub struct CooMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl CooMatrix {
    /// Create COO matrix from full matrix.
    pub fn from_full(m: &Matrix) -> Self {
        let mut entries = Vec::new();
        
        for i in 0..m.rows {
            for j in 0..m.cols {
                let val = m.get(i, j);
                if val.abs() > 1e-15 {
                    entries.push((i, j, val));
                }
            }
        }
        
        CooMatrix {
            rows: m.rows,
            cols: m.cols,
            entries,
        }
    }

    /// Add entry.
    pub fn add(&mut self, i: usize, j: usize, value: f64) -> MathResult<()> {
        if i >= self.rows || j >= self.cols {
            return Err(MathError::InvalidArgument("index out of bounds"));
        }
        
        self.entries.push((i, j, value));
        Ok(())
    }

    /// Convert to CSR.
    pub fn to_csr(&self) -> CsrMatrix {
        let full = self.to_full();
        CsrMatrix::from_full(&full)
    }

    /// Convert to CSC.
    pub fn to_csc(&self) -> CscMatrix {
        let full = self.to_full();
        CscMatrix::from_full(&full)
    }

    /// Convert to full matrix.
    pub fn to_full(&self) -> Matrix {
        let mut m = Matrix::zeros(self.rows, self.cols);
        
        for &(i, j, v) in &self.entries {
            m.set(i, j, v);
        }
        
        m
    }

    /// Sort entries by row then column.
    pub fn sort(&mut self) {
        self.entries.sort_by(|a, b| {
            if a.0 != b.0 {
                a.0.cmp(&b.0)
            } else {
                a.1.cmp(&b.1)
            }
        });
    }

    /// Remove duplicate entries (summing values).
    pub fn dedup(&mut self) {
        self.sort();
        
        let mut deduped = Vec::new();
        let mut i = 0;
        
        while i < self.entries.len() {
            let (row, col, mut val) = self.entries[i];
            
            while i + 1 < self.entries.len() && self.entries[i + 1].0 == row && self.entries[i + 1].1 == col {
                val += self.entries[i + 1].2;
                i += 1;
            }
            
            if val.abs() > 1e-15 {
                deduped.push((row, col, val));
            }
            
            i += 1;
        }
        
        self.entries = deduped;
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }
}

/// Diagonal sparse format.
#[derive(Debug, Clone)]
pub struct DiagonalSparse {
    pub n: usize,
    pub diag: Vec<f64>,
}

impl DiagonalSparse {
    /// Create from full matrix.
    pub fn from_full(m: &Matrix) -> Self {
        let n = m.rows.min(m.cols);
        let mut diag = Vec::with_capacity(n);
        
        for i in 0..n {
            diag.push(m.get(i, i));
        }
        
        DiagonalSparse { n, diag }
    }

    /// Get value at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i == j && i < self.n {
            self.diag[i]
        } else {
            0.0
        }
    }

    /// Matrix-vector product.
    pub fn mul_vec(&self, v: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        let n = self.n.min(v.len());
        let result: Vec<f64> = self.diag[..n]
            .iter()
            .zip(v.data.iter())
            .map(|(&d, &v)| d * v)
            .collect();
        
        Ok(mathverse_vector::Vector::new(result))
    }

    /// Convert to full matrix.
    pub fn to_full(&self) -> Matrix {
        Matrix::diagonal(&self.diag)
    }
}

/// Sparse matrix operations.
pub struct SparseOperations;

impl SparseOperations {
    /// Add two CSR matrices.
    pub fn csr_add(a: &CsrMatrix, b: &CsrMatrix) -> MathResult<CsrMatrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let full_a = a.to_full();
        let full_b = b.to_full();
        let sum = full_a.add(&full_b)?;
        
        Ok(CsrMatrix::from_full(&sum))
    }

    /// Multiply two CSR matrices.
    pub fn csr_mul(a: &CsrMatrix, b: &CsrMatrix) -> MathResult<CsrMatrix> {
        if a.cols != b.rows {
            return Err(MathError::DimensionMismatch);
        }
        
        let full_a = a.to_full();
        let full_b = b.to_full();
        let product = full_a.mul(&full_b)?;
        
        Ok(CsrMatrix::from_full(&product))
    }

    /// Convert between formats.
    pub fn coo_to_csr(coo: &CooMatrix) -> CsrMatrix {
        coo.to_csr()
    }

    pub fn coo_to_csc(coo: &CooMatrix) -> CscMatrix {
        coo.to_csc()
    }

    pub fn csr_to_csc(csr: &CsrMatrix) -> CscMatrix {
        csr.transpose()
    }

    pub fn csc_to_csr(csc: &CscMatrix) -> CsrMatrix {
        csc.transpose()
    }

    /// Sparsity ratio.
    pub fn sparsity_ratio(m: &Matrix) -> f64 {
        let total = m.rows * m.cols;
        let mut nnz = 0;
        
        for i in 0..m.rows {
            for j in 0..m.cols {
                if m.get(i, j).abs() > 1e-15 {
                    nnz += 1;
                }
            }
        }
        
        if total > 0 {
            nnz as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Recommend format based on sparsity pattern.
    pub fn recommend_format(m: &Matrix) -> &'static str {
        let sparsity = Self::sparsity_ratio(m);
        
        if sparsity < 0.1 {
            "CSR or CSC"
        } else if sparsity < 0.3 {
            "COO"
        } else {
            "Dense"
        }
    }
}

/// Sparse matrix utilities.
pub struct SparseUtils;

impl SparseUtils {
    /// Extract diagonal from sparse matrix.
    pub fn extract_diagonal(m: &Matrix) -> Vec<f64> {
        let n = m.rows.min(m.cols);
        (0..n).map(|i| m.get(i, i)).collect()
    }

    /// Create sparse identity matrix.
    pub fn sparse_identity(n: usize) -> CsrMatrix {
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0; n + 1];
        
        for i in 0..n {
            values.push(1.0);
            col_indices.push(i);
            row_ptr[i + 1] = i + 1;
        }
        
        CsrMatrix {
            rows: n,
            cols: n,
            values,
            col_indices,
            row_ptr,
        }
    }

    /// Permute rows of CSR matrix.
    pub fn permute_rows(csr: &CsrMatrix, perm: &[usize]) -> MathResult<CsrMatrix> {
        if perm.len() != csr.rows {
            return Err(MathError::DimensionMismatch);
        }
        
        let full = csr.to_full();
        let mut permuted = Matrix::zeros(csr.rows, csr.cols);
        
        for (new_i, &old_i) in perm.iter().enumerate() {
            for j in 0..csr.cols {
                permuted.set(new_i, j, full.get(old_i, j));
            }
        }
        
        Ok(CsrMatrix::from_full(&permuted))
    }

    /// Permute columns of CSC matrix.
    pub fn permute_cols(csc: &CscMatrix, perm: &[usize]) -> MathResult<CscMatrix> {
        if perm.len() != csc.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let full = csc.to_full();
        let mut permuted = Matrix::zeros(csc.rows, csc.cols);
        
        for (new_j, &old_j) in perm.iter().enumerate() {
            for i in 0..csc.rows {
                permuted.set(i, new_j, full.get(i, old_j));
            }
        }
        
        Ok(CscMatrix::from_full(&permuted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_from_full() {
        let m = Matrix::from_rows(&[&[1.0, 0.0, 3.0], &[0.0, 5.0, 0.0]]).unwrap();
        let csr = CsrMatrix::from_full(&m);
        
        assert_eq!(csr.rows, 2);
        assert_eq!(csr.cols, 3);
        assert_eq!(csr.nnz(), 3);
        assert!((csr.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((csr.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_mul_vec() {
        let m = Matrix::from_rows(&[&[1.0, 0.0, 3.0], &[0.0, 5.0, 0.0]]).unwrap();
        let csr = CsrMatrix::from_full(&m);
        let v = mathverse_vector::Vector::new(vec![1.0, 2.0, 3.0]);
        
        let result = csr.mul_vec(&v).unwrap();
        
        assert!((result.get(0) - 10.0).abs() < 1e-10);  // 1*1 + 3*3
        assert!((result.get(1) - 10.0).abs() < 1e-10);  // 5*2
    }

    #[test]
    fn test_csc_from_full() {
        let m = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 5.0], &[3.0, 0.0]]).unwrap();
        let csc = CscMatrix::from_full(&m);
        
        assert_eq!(csc.rows, 3);
        assert_eq!(csc.cols, 2);
        assert_eq!(csc.nnz(), 3);
    }

    #[test]
    fn test_coo_operations() {
        let m = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 5.0]]).unwrap();
        let mut coo = CooMatrix::from_full(&m);
        
        coo.add(0, 1, 2.0).unwrap();
        coo.dedup();
        
        assert_eq!(coo.nnz(), 3);
    }

    #[test]
    fn test_sparsity_ratio() {
        let dense = Matrix::identity(10);
        let sparse = Matrix::zeros(10, 10);
        
        assert!(SparseOperations::sparsity_ratio(&dense) > 0.1);
        assert!(SparseOperations::sparsity_ratio(&sparse) == 0.0);
    }

    #[test]
    fn test_sparse_identity() {
        let csr = SparseUtils::sparse_identity(3);
        
        assert!((csr.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((csr.get(1, 1) - 1.0).abs() < 1e-10);
        assert!((csr.get(2, 2) - 1.0).abs() < 1e-10);
        assert!(csr.get(0, 1) == 0.0);
    }
}
