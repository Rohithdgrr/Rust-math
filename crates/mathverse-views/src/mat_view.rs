//! Zero-copy borrowed matrix view.

use alloc::vec::Vec;
use mathverse_core::error::{MathError, MathResult};

/// A borrowed, zero-copy view into row-major `f64` data.
///
/// `MatView` wraps `&[f64]` with row/column dimensions, providing
/// zero-copy row, column, and submatrix extraction.
#[derive(Debug, Clone, Copy)]
pub struct MatView<'a> {
    data: &'a [f64],
    rows: usize,
    cols: usize,
}

impl<'a> MatView<'a> {
    /// Create a matrix view from row-major data.
    pub fn new(data: &'a [f64], rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "data length {} does not match dimensions {rows}x{cols}",
            data.len()
        );
        Self { data, rows, cols }
    }

    /// Try to create a view; returns error on dimension mismatch.
    pub fn try_new(data: &'a [f64], rows: usize, cols: usize) -> MathResult<Self> {
        if data.len() != rows * cols {
            return Err(MathError::DimensionMismatch);
        }
        Ok(Self { data, rows, cols })
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element at (row, col).
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Get the underlying row-major slice.
    pub fn as_slice(&self) -> &'a [f64] {
        self.data
    }

    /// Extract row `i` as a `Vec<f64>` (copies).
    pub fn row(&self, i: usize) -> Vec<f64> {
        self.data[i * self.cols..(i + 1) * self.cols].to_vec()
    }

    /// Extract column `j` as a `Vec<f64>` (copies).
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows)
            .map(|i| self.data[i * self.cols + j])
            .collect()
    }

    /// Submatrix view: rows `r_start..r_end`, cols `c_start..c_end`.
    ///
    /// Returns an owned `MatViewData` since the rows may not be contiguous.
    pub fn submatrix(
        &self,
        r_start: usize,
        r_end: usize,
        c_start: usize,
        c_end: usize,
    ) -> MatViewData {
        let new_rows = r_end - r_start;
        let new_cols = c_end - c_start;
        let mut data = Vec::with_capacity(new_rows * new_cols);
        for r in r_start..r_end {
            for c in c_start..c_end {
                data.push(self.get(r, c));
            }
        }
        MatViewData {
            data,
            rows: new_rows,
            cols: new_cols,
        }
    }

    /// Diagonal elements (for square or non-square matrices).
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }

    /// Transpose (returns owned).
    pub fn transpose(&self) -> MatViewData {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                data.push(self.get(r, c));
            }
        }
        MatViewData {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Row-major slice for row `i`.
    pub fn row_slice(&self, i: usize) -> &'a [f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Trace (sum of diagonal); errors if not square.
    pub fn trace(&self) -> MathResult<f64> {
        if self.rows != self.cols {
            return Err(MathError::DimensionMismatch);
        }
        Ok((0..self.rows).map(|i| self.get(i, i)).sum())
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Iterate over rows.
    pub fn rows(&self) -> impl Iterator<Item = &[f64]> {
        self.data.chunks(self.cols)
    }

    /// Convert to owned matrix data.
    pub fn to_owned_data(&self) -> MatViewData {
        MatViewData {
            data: self.data.to_vec(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

/// Owned matrix data (from submatrix/transpose operations).
#[derive(Debug, Clone)]
pub struct MatViewData {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MatViewData {
    /// Get element at (row, col).
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Convert to a view.
    pub fn as_view(&self) -> MatView<'_> {
        MatView::new(&self.data, self.rows, self.cols)
    }

    /// Consume and return the raw data.
    pub fn into_inner(self) -> Vec<f64> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_matrix() -> MatView<'static> {
        // 1.0 2.0 3.0
        // 4.0 5.0 6.0
        static DATA: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        MatView::new(&DATA, 2, 3)
    }

    #[test]
    fn basic_view() {
        let m = test_matrix();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    fn row_col() {
        let m = test_matrix();
        assert_eq!(m.row(0), vec![1.0, 2.0, 3.0]);
        assert_eq!(m.col(1), vec![2.0, 5.0]);
    }

    #[test]
    fn submatrix() {
        let m = test_matrix();
        let sub = m.submatrix(0, 2, 1, 3);
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 2);
        assert_eq!(sub.get(0, 0), 2.0);
        assert_eq!(sub.get(1, 1), 6.0);
    }

    #[test]
    fn transpose() {
        let m = test_matrix();
        let t = m.transpose();
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn diagonal() {
        let m = test_matrix();
        assert_eq!(m.diagonal(), vec![1.0, 5.0]);
    }

    #[test]
    fn trace() {
        let m = test_matrix();
        assert!(m.trace().is_err()); // not square

        static SQ: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        let sq = MatView::new(&SQ, 2, 2);
        assert!((sq.trace().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn frobenius() {
        let m = test_matrix();
        let expected = (1.0 + 4.0 + 9.0 + 16.0 + 25.0 + 36.0_f64).sqrt();
        assert!((m.frobenius_norm() - expected).abs() < 1e-12);
    }

    #[test]
    fn rows_iter() {
        let m = test_matrix();
        let rows: Vec<&[f64]> = m.rows().collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], &[1.0, 2.0, 3.0]);
        assert_eq!(rows[1], &[4.0, 5.0, 6.0]);
    }
}
