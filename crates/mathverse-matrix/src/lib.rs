//! Dense matrices over `f64`, row-major storage.
//!
//! Dimension mismatches and singular matrices return [`MathError`]; nothing
//! panics on user input.

use core::ops::Index;
use mathverse_core::error::{MathError, MathResult};
use mathverse_vector::Vector;

pub mod rng;

pub mod decompositions;
pub mod sparse;

pub mod banded;
pub mod block;
pub mod calculus;
pub mod condition;
pub mod eigen_general;
pub mod eigen_generalized;
pub mod equations;
pub mod functions;
pub mod hadamard;
pub mod iterative;
pub mod kronecker;
pub mod ldl;
pub mod leastsquares;
pub mod lowrank;
pub mod norms;
pub mod polar;
pub mod positivedefinite;
pub mod power;
pub mod pseudoinverse;
pub mod rank;
pub mod schur;
pub mod sparse_formats;

pub use decompositions::{Lu, Qr, Svd};
pub use sparse::SparseMatrix;

/// Dense row-major matrix: `data[row * cols + col]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Create a matrix from raw row-major data.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::InvalidArgument`] if `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> MathResult<Self> {
        if data.len() != rows * cols {
            return Err(MathError::InvalidArgument(
                "data length must equal rows * cols",
            ));
        }
        Ok(Self { rows, cols, data })
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Dimensions as `(rows, cols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the flat row-major data buffer.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Mutably borrow the flat row-major data buffer.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Borrow the flat row-major data buffer (alias of [`Matrix::data`]).
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Consume the matrix, returning the flat row-major data buffer.
    #[must_use]
    pub fn into_data(self) -> Vec<f64> {
        self.data
    }

    /// Get the element at `(r, c)`, returning an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::OutOfRange`] if the index is out of bounds.
    pub fn try_get(&self, r: usize, c: usize) -> MathResult<f64> {
        if r >= self.rows || c >= self.cols {
            return Err(MathError::OutOfRange);
        }
        Ok(self.data[r * self.cols + c])
    }

    /// Set the element at `(r, c)`, returning an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::OutOfRange`] if the index is out of bounds.
    pub fn try_set(&mut self, r: usize, c: usize, v: f64) -> MathResult<()> {
        if r >= self.rows || c >= self.cols {
            return Err(MathError::OutOfRange);
        }
        self.data[r * self.cols + c] = v;
        Ok(())
    }

    /// Convert vector to column matrix.
    pub fn from_vector(v: &mathverse_vector::Vector) -> Self {
        Self {
            rows: v.data.len(),
            cols: 1,
            data: v.data.clone(),
        }
    }

    /// Ones matrix.
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![1.0; rows * cols],
        }
    }

    /// Zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Diagonal matrix.
    pub fn diagonal(diag: &[f64]) -> Self {
        let n = diag.len();
        let mut m = Self::zeros(n, n);
        for (i, &v) in diag.iter().enumerate() {
            m.set(i, i, v);
        }
        m
    }

    /// From rows; errors on empty input or ragged rows.
    pub fn from_rows(rows: &[&[f64]]) -> MathResult<Self> {
        if rows.is_empty() || rows[0].is_empty() {
            return Err(MathError::InvalidArgument(
                "matrix must have at least one row and column",
            ));
        }
        let cols = rows[0].len();
        let mut data = Vec::with_capacity(rows.len() * cols);
        for r in rows {
            if r.len() != cols {
                return Err(MathError::InvalidArgument(
                    "ragged rows: all rows must have equal length",
                ));
            }
            data.extend_from_slice(r);
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data,
        })
    }

    /// Get the element at `(r, c)` without bounds checking.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `(r, c)` is out of bounds.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        debug_assert!(
            r < self.rows && c < self.cols,
            "matrix index ({r}, {c}) out of bounds"
        );
        self.data[r * self.cols + c]
    }

    /// Set the element at `(r, c)` without bounds checking.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `(r, c)` is out of bounds.
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        debug_assert!(
            r < self.rows && c < self.cols,
            "matrix index ({r}, {c}) out of bounds"
        );
        self.data[r * self.cols + c] = v;
    }

    /// Whether the matrix has equally many rows and columns.
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Column `j` as a vector (copy).
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    /// Row `i` as a vector (copy).
    pub fn row(&self, i: usize) -> Vec<f64> {
        (0..self.cols).map(|j| self.get(i, j)).collect()
    }

    /// `a[i][j] == a[j][i]` within `tol * max(|a|)`, and square.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if !self.is_square() {
            return false;
        }
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Matrix addition.
    pub fn add(&self, other: &Self) -> MathResult<Self> {
        self.binary(other, |a, b| a + b)
    }

    /// Matrix subtraction.
    pub fn sub(&self, other: &Self) -> MathResult<Self> {
        self.binary(other, |a, b| a - b)
    }

    fn binary(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> MathResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MathError::DimensionMismatch);
        }
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| f(*a, *b))
                .collect(),
        })
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|v| v * s).collect(),
        }
    }

    /// Matrix product; error on `self.cols != other.rows`.
    pub fn mul(&self, other: &Self) -> MathResult<Self> {
        if self.cols != other.rows {
            return Err(MathError::DimensionMismatch);
        }
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                for j in 0..other.cols {
                    out.data[i * other.cols + j] += a * other.get(k, j);
                }
            }
        }
        Ok(out)
    }

    /// Matrix-vector product; error on `self.cols != v.len()`.
    pub fn mul_vec(&self, v: &Vector) -> MathResult<Vector> {
        if self.cols != v.len() {
            return Err(MathError::DimensionMismatch);
        }
        let mut out = vec![0.0; self.rows];
        for (i, o) in out.iter_mut().enumerate() {
            *o = (0..self.cols).map(|j| self.get(i, j) * v.get(j)).sum();
        }
        Ok(Vector::new(out))
    }

    /// Transpose: element `(i, j)` moves to `(j, i)`.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.get(i, j));
            }
        }
        t
    }

    /// Trace (sum of diagonal); error if not square.
    pub fn trace(&self) -> MathResult<f64> {
        if !self.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        Ok((0..self.rows).map(|i| self.get(i, i)).sum())
    }

    /// Determinant via LU with partial pivoting.
    /// [`MathError::Singular`] for singular matrices.
    pub fn det(&self) -> MathResult<f64> {
        let lu = self.lu()?;
        let mut d = lu.sign;
        for i in 0..self.rows {
            d *= lu.u.get(i, i);
        }
        Ok(d)
    }

    /// Solve `A x = b` via LU; error on singular or dimension mismatch.
    #[allow(clippy::needless_range_loop)] // index arithmetic clearer in substitution loops
    pub fn solve(&self, b: &Vector) -> MathResult<Vector> {
        let lu = self.lu()?;
        if b.len() != self.rows {
            return Err(MathError::DimensionMismatch);
        }
        // Pb
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows {
            y[i] = b.get(lu.pivots[i]);
        }
        // L y = Pb (unit diagonal)
        for i in 0..self.rows {
            let mut s = y[i];
            for j in 0..i {
                s -= lu.l.get(i, j) * y[j];
            }
            y[i] = s;
        }
        // U x = y
        let mut x = vec![0.0; self.rows];
        for i in (0..self.rows).rev() {
            let mut s = y[i];
            for j in (i + 1)..self.rows {
                s -= lu.u.get(i, j) * x[j];
            }
            x[i] = s / lu.u.get(i, i);
        }
        Ok(Vector::new(x))
    }

    /// Inverse via LU; error on singular matrices.
    pub fn inverse(&self) -> MathResult<Self> {
        if !self.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        let mut inv = Self::zeros(self.rows, self.cols);
        for k in 0..self.rows {
            let mut e = Vector::zeros(self.rows);
            e.set(k, 1.0);
            let col = self.solve(&e)?;
            for i in 0..self.rows {
                inv.set(i, k, col.get(i));
            }
        }
        Ok(inv)
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        debug_assert!(
            r < self.rows && c < self.cols,
            "matrix index ({r}, {c}) out of bounds"
        );
        &self.data[r * self.cols + c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a() -> Matrix {
        Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap()
    }

    #[test]
    fn construction() {
        assert!(Matrix::from_rows(&[]).is_err());
        assert!(Matrix::from_rows(&[&[1.0, 2.0], &[3.0]]).is_err());
        let m = Matrix::identity(3);
        assert_eq!(m.trace().unwrap(), 3.0);
        assert_eq!(Matrix::diagonal(&[1.0, 2.0]).get(1, 1), 2.0);
    }

    #[test]
    fn arithmetic() {
        let s = a().add(&a()).unwrap();
        assert_eq!(s, a().scale(2.0));
        assert!(a().add(&Matrix::identity(3)).is_err());
        let p = a()
            .mul(&Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]).unwrap())
            .unwrap();
        assert_eq!(p, a());
        let v = a().mul_vec(&Vector::new(vec![1.0, 1.0])).unwrap();
        assert_eq!(v, Vector::new(vec![3.0, 7.0]));
        assert!(a().mul_vec(&Vector::new(vec![1.0])).is_err());
        let t = a().transpose();
        assert_eq!(t.get(0, 1), 3.0);
    }

    #[test]
    fn det_inverse_solve() {
        assert_eq!(a().det().unwrap(), -2.0);
        assert_eq!(Matrix::identity(3).det().unwrap(), 1.0);
        let sing = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]).unwrap();
        assert_eq!(sing.det(), Err(MathError::Singular));

        let inv = a().inverse().unwrap();
        let prod = a().mul(&inv).unwrap();
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((prod.get(1, 1) - 1.0).abs() < 1e-12);
        assert!(prod.get(0, 1).abs() < 1e-12);

        let b = Vector::new(vec![1.0, 2.0]);
        let x = a().solve(&b).unwrap();
        let back = a().mul_vec(&x).unwrap();
        assert!((back.get(0) - 1.0).abs() < 1e-12);
        assert!((back.get(1) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ill_conditioned() {
        // Hilbert 4x4: known determinant 1/6048000, extreme conditioning.
        let h = Matrix::from_rows(&[
            &[1.0, 0.5, 1.0 / 3.0, 0.25],
            &[0.5, 1.0 / 3.0, 0.25, 0.2],
            &[1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0],
            &[0.25, 0.2, 1.0 / 6.0, 1.0 / 7.0],
        ])
        .unwrap();
        let det = h.det().unwrap();
        assert!((det - 1.0 / 6_048_000.0).abs() < 1e-13, "det = {det}");
        let prod = h.mul(&h.inverse().unwrap()).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((prod.get(i, j) - want).abs() < 1e-10, "({i},{j})");
            }
        }
    }
}
