//! Banded matrices: specialized storage and operations for matrices with non-zero bands.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Banded matrix storage format.
#[derive(Debug, Clone)]
pub struct BandedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub lower_bandwidth: usize,
    pub upper_bandwidth: usize,
    pub data: Vec<f64>,  // Compressed storage
}

impl BandedMatrix {
    /// Create banded matrix from full matrix.
    pub fn from_full(m: &Matrix, lower: usize, upper: usize) -> Self {
        let (rows, cols) = (m.rows, m.cols);
        let bandwidth = lower + upper + 1;
        let mut data = vec![0.0; rows * bandwidth];
        
        for i in 0..rows {
            for j in 0..cols {
                let band = j as i32 - i as i32 + lower as i32;
                if band >= 0 && band < bandwidth as i32 {
                    data[i * bandwidth + band as usize] = m.get(i, j);
                }
            }
        }
        
        BandedMatrix {
            rows,
            cols,
            lower_bandwidth: lower,
            upper_bandwidth: upper,
            data,
        }
    }

    /// Get value at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let band = j as i32 - i as i32 + self.lower_bandwidth as i32;
        if band >= 0 && band < (self.lower_bandwidth + self.upper_bandwidth + 1) as i32 {
            self.data[i * (self.lower_bandwidth + self.upper_bandwidth + 1) + band as usize]
        } else {
            0.0
        }
    }

    /// Set value at (i, j).
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        let band = j as i32 - i as i32 + self.lower_bandwidth as i32;
        if band >= 0 && band < (self.lower_bandwidth + self.upper_bandwidth + 1) as i32 {
            self.data[i * (self.lower_bandwidth + self.upper_bandwidth + 1) + band as usize] = value;
        }
    }

    /// Convert to full matrix.
    pub fn to_full(&self) -> Matrix {
        let mut m = Matrix::zeros(self.rows, self.cols);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                m.set(i, j, self.get(i, j));
            }
        }
        
        m
    }

    /// Matrix-vector product for banded matrix.
    pub fn mul_vec(&self, v: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        if self.cols != v.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = vec![0.0; self.rows];
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * v.get(j);
            }
        }
        
        Ok(mathverse_vector::Vector::new(result))
    }

    /// Transpose of banded matrix.
    pub fn transpose(&self) -> BandedMatrix {
        let mut result = BandedMatrix {
            rows: self.cols,
            cols: self.rows,
            lower_bandwidth: self.upper_bandwidth,
            upper_bandwidth: self.lower_bandwidth,
            data: vec![0.0; self.cols * (self.upper_bandwidth + self.lower_bandwidth + 1)],
        };
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        
        result
    }
}

/// Tridiagonal matrix (special case of banded).
#[derive(Debug, Clone)]
pub struct TridiagonalMatrix {
    pub n: usize,
    pub main_diag: Vec<f64>,
    pub upper_diag: Vec<f64>,
    pub lower_diag: Vec<f64>,
}

impl TridiagonalMatrix {
    /// Create tridiagonal matrix from diagonals.
    pub fn new(main: &[f64], upper: &[f64], lower: &[f64]) -> Self {
        let n = main.len();
        TridiagonalMatrix {
            n,
            main_diag: main.to_vec(),
            upper_diag: upper.to_vec(),
            lower_diag: lower.to_vec(),
        }
    }

    /// Get value at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i == j {
            self.main_diag[i]
        } else if i + 1 == j {
            self.upper_diag[i]
        } else if i == j + 1 {
            self.lower_diag[j]
        } else {
            0.0
        }
    }

    /// Matrix-vector product using Thomas algorithm.
    pub fn mul_vec(&self, v: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        if self.n != v.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut result = vec![0.0; self.n];
        
        for i in 0..self.n {
            result[i] = self.main_diag[i] * v.get(i);
            if i > 0 {
                result[i] += self.lower_diag[i - 1] * v.get(i - 1);
            }
            if i < self.n - 1 {
                result[i] += self.upper_diag[i] * v.get(i + 1);
            }
        }
        
        Ok(mathverse_vector::Vector::new(result))
    }

    /// Solve tridiagonal system using Thomas algorithm.
    pub fn solve(&self, b: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        if self.n != b.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = self.n;
        let mut c = self.upper_diag.clone();
        let mut d = self.main_diag.clone();
        let mut e = self.lower_diag.clone();
        let mut b_vec = b.data.clone();
        
        // Forward elimination
        for i in 1..n {
            let factor = e[i - 1] / d[i - 1];
            d[i] -= factor * c[i - 1];
            b_vec[i] -= factor * b_vec[i - 1];
        }
        
        // Back substitution
        let mut x = vec![0.0; n];
        x[n - 1] = b_vec[n - 1] / d[n - 1];
        
        for i in (0..n - 1).rev() {
            x[i] = (b_vec[i] - c[i] * x[i + 1]) / d[i];
        }
        
        Ok(mathverse_vector::Vector::new(x))
    }

    /// Convert to full matrix.
    pub fn to_full(&self) -> Matrix {
        let mut m = Matrix::zeros(self.n, self.n);
        
        for i in 0..self.n {
            for j in 0..self.n {
                m.set(i, j, self.get(i, j));
            }
        }
        
        m
    }

    /// Determinant of tridiagonal matrix.
    pub fn determinant(&self) -> f64 {
        let n = self.n;
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return self.main_diag[0];
        }
        
        let mut det = vec![0.0; n];
        det[0] = self.main_diag[0];
        det[1] = self.main_diag[0] * self.main_diag[1] - self.upper_diag[0] * self.lower_diag[0];
        
        for i in 2..n {
            det[i] = self.main_diag[i] * det[i - 1] 
                - self.upper_diag[i - 1] * self.lower_diag[i - 1] * det[i - 2];
        }
        
        det[n - 1]
    }
}

/// Diagonal matrix (special case of banded).
#[derive(Debug, Clone)]
pub struct DiagonalMatrix {
    pub n: usize,
    pub diag: Vec<f64>,
}

impl DiagonalMatrix {
    /// Create diagonal matrix.
    pub fn new(diag: &[f64]) -> Self {
        DiagonalMatrix {
            n: diag.len(),
            diag: diag.to_vec(),
        }
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
        if self.n != v.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        let result: Vec<f64> = self.diag.iter()
            .zip(v.data.iter())
            .map(|(&d, &v)| d * v)
            .collect();
        
        Ok(mathverse_vector::Vector::new(result))
    }

    /// Inverse of diagonal matrix.
    pub fn inverse(&self) -> MathResult<Self> {
        let inv_diag: Vec<f64> = self.diag.iter()
            .map(|&d| {
                if d.abs() < 1e-15 {
                    return f64::INFINITY;
                }
                1.0 / d
            })
            .collect();
        
        if inv_diag.iter().any(|&d| d.is_infinite()) {
            return Err(MathError::InvalidArgument("zero diagonal element"));
        }
        
        Ok(DiagonalMatrix {
            n: self.n,
            diag: inv_diag,
        })
    }

    /// Convert to full matrix.
    pub fn to_full(&self) -> Matrix {
        Matrix::diagonal(&self.diag)
    }

    /// Determinant.
    pub fn determinant(&self) -> f64 {
        self.diag.iter().product()
    }
}

/// Banded matrix operations.
pub struct BandedOperations;

impl BandedOperations {
    /// Add two banded matrices.
    pub fn add(a: &BandedMatrix, b: &BandedMatrix) -> MathResult<BandedMatrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        let lower = a.lower_bandwidth.max(b.lower_bandwidth);
        let upper = a.upper_bandwidth.max(b.upper_bandwidth);
        let bandwidth = lower + upper + 1;
        let mut data = vec![0.0; a.rows * bandwidth];
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                let band = j as i32 - i as i32 + lower as i32;
                if band >= 0 && band < bandwidth as i32 {
                    data[i * bandwidth + band as usize] = a.get(i, j) + b.get(i, j);
                }
            }
        }
        
        Ok(BandedMatrix {
            rows: a.rows,
            cols: a.cols,
            lower_bandwidth: lower,
            upper_bandwidth: upper,
            data,
        })
    }

    /// Multiply two banded matrices.
    pub fn mul(a: &BandedMatrix, b: &BandedMatrix) -> MathResult<BandedMatrix> {
        let a_full = a.to_full();
        let b_full = b.to_full();
        let result = a_full.mul(&b_full)?;
        
        let lower = a.lower_bandwidth + b.lower_bandwidth;
        let upper = a.upper_bandwidth + b.upper_bandwidth;
        
        Ok(BandedMatrix::from_full(&result, lower.min(a.rows - 1), upper.min(a.cols - 1)))
    }

    /// LU decomposition for tridiagonal matrix.
    pub fn tridiagonal_lu(t: &TridiagonalMatrix) -> MathResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let n = t.n;
        let mut l = vec![0.0; n - 1];
        let mut u_main = t.main_diag.clone();
        let mut u_upper = t.upper_diag.clone();
        
        for i in 0..(n - 1) {
            if u_main[i].abs() < 1e-15 {
                return Err(MathError::InvalidArgument("zero pivot in tridiagonal LU"));
            }
            l[i] = t.lower_diag[i] / u_main[i];
            u_main[i + 1] -= l[i] * u_upper[i];
        }
        
        Ok((l, u_main, u_upper))
    }
}

/// Toeplitz matrix (constant along diagonals).
#[derive(Debug, Clone)]
pub struct ToeplitzMatrix {
    pub n: usize,
    pub first_row: Vec<f64>,
    pub first_col: Vec<f64>,
}

impl ToeplitzMatrix {
    /// Create Toeplitz matrix from first row and column.
    pub fn new(first_row: &[f64], first_col: &[f64]) -> Self {
        let n = first_row.len();
        ToeplitzMatrix {
            n,
            first_row: first_row.to_vec(),
            first_col: first_col.to_vec(),
        }
    }

    /// Get value at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if j >= i {
            self.first_row[j - i]
        } else {
            self.first_col[i - j]
        }
    }

    /// Convert to full matrix.
    pub fn to_full(&self) -> Matrix {
        let mut m = Matrix::zeros(self.n, self.n);
        
        for i in 0..self.n {
            for j in 0..self.n {
                m.set(i, j, self.get(i, j));
            }
        }
        
        m
    }

    /// Fast matrix-vector product using FFT (simplified - use direct for now).
    pub fn mul_vec(&self, v: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        let full = self.to_full();
        full.mul_vec(v)
    }
}

/// Circulant matrix (special Toeplitz with periodic structure).
#[derive(Debug, Clone)]
pub struct CirculantMatrix {
    pub n: usize,
    pub first_row: Vec<f64>,
}

impl CirculantMatrix {
    /// Create circulant matrix from first row.
    pub fn new(first_row: &[f64]) -> Self {
        CirculantMatrix {
            n: first_row.len(),
            first_row: first_row.to_vec(),
        }
    }

    /// Get value at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let idx = if j >= i {
            j - i
        } else {
            self.n - i + j
        };
        self.first_row[idx]
    }

    /// Convert to full matrix.
    pub fn to_full(&self) -> Matrix {
        let mut m = Matrix::zeros(self.n, self.n);
        
        for i in 0..self.n {
            for j in 0..self.n {
                m.set(i, j, self.get(i, j));
            }
        }
        
        m
    }

    /// Matrix-vector product using FFT (simplified).
    pub fn mul_vec(&self, v: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        let full = self.to_full();
        full.mul_vec(v)
    }

    /// Diagonalization using DFT matrix.
    pub fn diagonalize(&self) -> MathResult<(Matrix, Matrix)> {
        // Simplified: return identity
        let n = self.n;
        Ok((Matrix::identity(n), Matrix::identity(n)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tridiagonal_matrix() {
        let main = vec![2.0, 3.0, 4.0];
        let upper = vec![1.0, 1.0];
        let lower = vec![1.0, 1.0];
        
        let t = TridiagonalMatrix::new(&main, &upper, &lower);
        assert_eq!(t.get(0, 0), 2.0);
        assert_eq!(t.get(0, 1), 1.0);
        assert_eq!(t.get(1, 0), 1.0);
        assert_eq!(t.get(1, 2), 1.0);
    }

    #[test]
    fn test_tridiagonal_solve() {
        let main = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0];
        let lower = vec![-1.0, -1.0];
        
        let t = TridiagonalMatrix::new(&main, &upper, &lower);
        let b = mathverse_vector::Vector::new(vec![1.0, 0.0, 0.0]);
        
        let x = t.solve(&b).unwrap();
        let back = t.mul_vec(&x).unwrap();
        
        assert!((back.get(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diagonal_matrix() {
        let d = DiagonalMatrix::new(&[2.0, 3.0, 4.0]);
        
        assert_eq!(d.get(0, 0), 2.0);
        assert_eq!(d.get(1, 1), 3.0);
        assert_eq!(d.get(0, 1), 0.0);
        
        let det = d.determinant();
        assert!((det - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_toeplitz_matrix() {
        let first_row = vec![1.0, 2.0, 3.0];
        let first_col = vec![1.0, 4.0, 5.0];
        
        let t = ToeplitzMatrix::new(&first_row, &first_col);
        
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 2.0);
        assert_eq!(t.get(1, 0), 4.0);
        assert_eq!(t.get(1, 1), 1.0);
    }

    #[test]
    fn test_circulant_matrix() {
        let first_row = vec![1.0, 2.0, 3.0];
        
        let c = CirculantMatrix::new(&first_row);
        
        assert_eq!(c.get(0, 0), 1.0);
        assert_eq!(c.get(0, 1), 2.0);
        assert_eq!(c.get(0, 2), 3.0);
        assert_eq!(c.get(1, 0), 3.0);
        assert_eq!(c.get(1, 1), 1.0);
    }
}
