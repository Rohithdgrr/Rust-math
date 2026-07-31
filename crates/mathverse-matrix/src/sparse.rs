//! Coordinate-list (COO) sparse matrix: triplet storage, sparse matvec.
//!
//! # ponytail: COO with linear-scan `get`; switch to CSR (sorted rows) when
//! matrices grow — solves and decompositions then go through dense anyway.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};
use mathverse_vector::Vector;

/// Sparse matrix in coordinate format: `(row, col, value)` triplets.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseMatrix {
    rows: usize,
    cols: usize,
    entries: Vec<(usize, usize, f64)>,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> SparseMatrix {
        SparseMatrix { rows, cols, entries: Vec::new() }
    }

    /// Insert `value` at `(r, c)`; duplicates accumulate on matvec.
    /// Errors if `(r, c)` is out of bounds.
    pub fn add(&mut self, r: usize, c: usize, value: f64) -> MathResult<()> {
        if r >= self.rows || c >= self.cols {
            return Err(MathError::InvalidArgument("entry out of bounds"));
        }
        self.entries.push((r, c, value));
        Ok(())
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    pub fn triplets(&self) -> &[(usize, usize, f64)] {
        &self.entries
    }

    /// Value at `(r, c)`, or 0.0 if absent.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.entries
            .iter()
            .find(|&&(rr, cc, _)| rr == r && cc == c)
            .map(|&(_, _, v)| v)
            .unwrap_or(0.0)
    }

    /// Sparse matvec; error on `cols != v.len()`.
    pub fn mul_vec(&self, v: &Vector) -> MathResult<Vector> {
        if self.cols != v.len() {
            return Err(MathError::DimensionMismatch);
        }
        let mut out = vec![0.0; self.rows];
        for &(r, c, val) in &self.entries {
            out[r] += val * v.get(c);
        }
        Ok(Vector::new(out))
    }

    pub fn transpose(&self) -> SparseMatrix {
        let mut t = SparseMatrix::new(self.cols, self.rows);
        for &(r, c, v) in &self.entries {
            t.entries.push((c, r, v));
        }
        t
    }

    pub fn to_dense(&self) -> Matrix {
        let mut m = Matrix::zeros(self.rows, self.cols);
        for &(r, c, v) in &self.entries {
            let cur = m.get(r, c);
            m.set(r, c, cur + v);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_matches_dense() {
        // 0 1 0
        // 2 0 3
        let mut s = SparseMatrix::new(2, 3);
        s.add(0, 1, 1.0).unwrap();
        s.add(1, 0, 2.0).unwrap();
        s.add(1, 2, 3.0).unwrap();
        assert_eq!(s.nnz(), 3);
        assert_eq!(s.get(0, 1), 1.0);
        assert_eq!(s.get(1, 1), 0.0);
        assert!(s.add(5, 0, 1.0).is_err());

        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        let dense = s.to_dense().mul_vec(&v).unwrap();
        let sparse = s.mul_vec(&v).unwrap();
        assert_eq!(dense, sparse);
        assert_eq!(sparse, Vector::new(vec![2.0, 11.0]));
        assert!(s.mul_vec(&Vector::new(vec![1.0])).is_err());

        let t = s.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.get(1, 0), 1.0);
        assert_eq!(t.mul_vec(&Vector::new(vec![1.0, 1.0])).unwrap(), Vector::new(vec![2.0, 1.0, 3.0]));
    }
}
