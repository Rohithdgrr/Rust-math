//! Complex matrix operations: arithmetic, decomposition, eigenvalues.

use crate::Complex;

/// Complex matrix with row-major storage.
#[derive(Debug, Clone)]
pub struct ComplexMatrix {
    pub data: Vec<Complex>,
    pub rows: usize,
    pub cols: usize,
}

impl ComplexMatrix {
    /// Create a new complex matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        ComplexMatrix {
            data: vec![Complex::zero(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create matrix from data.
    pub fn from_data(data: Vec<Complex>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        ComplexMatrix { data, rows, cols }
    }

    /// Get element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> Complex {
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, value: Complex) {
        self.data[row * self.cols + col] = value;
    }

    /// Create zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        ComplexMatrix::new(rows, cols)
    }

    /// Create identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut matrix = ComplexMatrix::new(n, n);
        for i in 0..n {
            matrix.set(i, i, Complex::one());
        }
        matrix
    }

    /// Matrix addition.
    pub fn add(&self, other: &ComplexMatrix) -> ComplexMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        
        let mut result = ComplexMatrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    /// Matrix subtraction.
    pub fn sub(&self, other: &ComplexMatrix) -> ComplexMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        
        let mut result = ComplexMatrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: Complex) -> ComplexMatrix {
        let mut result = ComplexMatrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] * scalar;
        }
        result
    }

    /// Matrix multiplication.
    pub fn mul(&self, other: &ComplexMatrix) -> ComplexMatrix {
        assert_eq!(self.cols, other.rows);
        
        let mut result = ComplexMatrix::new(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = Complex::zero();
                for k in 0..self.cols {
                    sum = sum + self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        
        result
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> ComplexMatrix {
        let mut result = ComplexMatrix::new(self.cols, self.rows);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        
        result
    }

    /// Conjugate transpose (Hermitian transpose).
    pub fn hermitian(&self) -> ComplexMatrix {
        let mut result = ComplexMatrix::new(self.cols, self.rows);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j).conjugate());
            }
        }
        
        result
    }

    /// Matrix trace.
    pub fn trace(&self) -> Complex {
        assert_eq!(self.rows, self.cols);
        
        let mut sum = Complex::zero();
        for i in 0..self.rows {
            sum = sum + self.get(i, i);
        }
        sum
    }

    /// Matrix determinant (for small matrices).
    pub fn determinant(&self) -> Complex {
        assert_eq!(self.rows, self.cols);
        
        match self.rows {
            1 => self.get(0, 0),
            2 => {
                let a = self.get(0, 0);
                let b = self.get(0, 1);
                let c = self.get(1, 0);
                let d = self.get(1, 1);
                a * d - b * c
            }
            3 => {
                let a = self.get(0, 0);
                let b = self.get(0, 1);
                let c = self.get(0, 2);
                let d = self.get(1, 0);
                let e = self.get(1, 1);
                let f = self.get(1, 2);
                let g = self.get(2, 0);
                let h = self.get(2, 1);
                let i = self.get(2, 2);
                
                a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
            }
            _ => {
                // Use LU decomposition for larger matrices
                let (l, u, _) = self.lu_decomposition().unwrap();
                let mut det = Complex::one();
                
                for i in 0..self.rows {
                    det = det * l.get(i, i) * u.get(i, i);
                }
                
                det
            }
        }
    }

    /// LU decomposition with partial pivoting.
    pub fn lu_decomposition(&self) -> Option<(ComplexMatrix, ComplexMatrix, Vec<usize>)> {
        if self.rows != self.cols {
            return None;
        }
        
        let n = self.rows;
        let mut l = ComplexMatrix::identity(n);
        let mut u = self.clone();
        let mut pivot = (0..n).collect::<Vec<_>>();
        
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            let mut max_val = u.get(k, k).norm();
            
            for i in (k + 1)..n {
                let val = u.get(i, k).norm();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }
            
            if max_val < 1e-15 {
                return None; // Singular matrix
            }
            
            // Swap rows
            if max_row != k {
                pivot.swap(k, max_row);
                for j in 0..n {
                    let temp = u.get(k, j);
                    u.set(k, j, u.get(max_row, j));
                    u.set(max_row, j, temp);
                }
            }
            
            // Elimination
            for i in (k + 1)..n {
                let factor = u.get(i, k) / u.get(k, k);
                l.set(i, k, factor);
                
                for j in k..n {
                    let new_val = u.get(i, j) - factor * u.get(k, j);
                    u.set(i, j, new_val);
                }
            }
        }
        
        Some((l, u, pivot))
    }

    /// Solve linear system Ax = b using LU decomposition.
    pub fn solve(&self, b: &[Complex]) -> Option<Vec<Complex>> {
        if self.rows != self.cols || b.len() != self.rows {
            return None;
        }
        
        let (l, u, pivot) = self.lu_decomposition()?;
        let n = self.rows;
        
        // Apply pivot to b
        let mut pb = b.to_vec();
        for i in 0..n {
            pb[i] = b[pivot[i]];
        }
        
        // Forward substitution: Ly = pb
        let mut y = vec![Complex::zero(); n];
        for i in 0..n {
            let mut sum = pb[i];
            for j in 0..i {
                sum = sum - l.get(i, j) * y[j];
            }
            y[i] = sum / l.get(i, i);
        }
        
        // Back substitution: Ux = y
        let mut x = vec![Complex::zero(); n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum = sum - u.get(i, j) * x[j];
            }
            x[i] = sum / u.get(i, i);
        }
        
        Some(x)
    }

    /// Matrix inverse.
    pub fn inverse(&self) -> Option<ComplexMatrix> {
        if self.rows != self.cols {
            return None;
        }
        
        let n = self.rows;
        let mut result = ComplexMatrix::new(n, n);
        
        // Solve for each column of identity
        for j in 0..n {
            let mut b = vec![Complex::zero(); n];
            b[j] = Complex::one();
            
            if let Some(col) = self.solve(&b) {
                for i in 0..n {
                    result.set(i, j, col[i]);
                }
            } else {
                return None;
            }
        }
        
        Some(result)
    }

    /// Check if matrix is Hermitian (equal to its conjugate transpose).
    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        if self.rows != self.cols {
            return false;
        }
        
        let hermitian = self.hermitian();
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                let diff = (self.get(i, j) - hermitian.get(i, j)).norm();
                if diff > tolerance {
                    return false;
                }
            }
        }
        
        true
    }

    /// Check if matrix is unitary (A * A^H = I).
    pub fn is_unitary(&self, tolerance: f64) -> bool {
        if self.rows != self.cols {
            return false;
        }
        
        let hermitian = self.hermitian();
        let product = self.mul(&hermitian);
        let identity = ComplexMatrix::identity(self.rows);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                let diff = (product.get(i, j) - identity.get(i, j)).norm();
                if diff > tolerance {
                    return false;
                }
            }
        }
        
        true
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|c| c.norm_sq()).sum::<f64>().sqrt()
    }

    /// Matrix power A^n.
    pub fn power(&self, n: usize) -> ComplexMatrix {
        if self.rows != self.cols {
            panic!("Matrix must be square");
        }
        
        if n == 0 {
            return ComplexMatrix::identity(self.rows);
        }
        
        let mut result = ComplexMatrix::identity(self.rows);
        let mut base = self.clone();
        let mut exp = n;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            exp /= 2;
        }
        
        result
    }

    /// Matrix exponential e^A using Taylor series.
    pub fn exp(&self, iterations: usize) -> ComplexMatrix {
        if self.rows != self.cols {
            panic!("Matrix must be square");
        }
        
        let mut result = ComplexMatrix::identity(self.rows);
        let mut term = ComplexMatrix::identity(self.rows);
        let mut factorial: f64 = 1.0;
        
        for n in 1..=iterations {
            factorial *= n as f64;
            term = term.mul(self);
            
            for i in 0..term.data.len() {
                result.data[i] = result.data[i] + term.data[i] / Complex::real(factorial);
            }
        }
        
        result
    }

    /// Matrix logarithm ln(A) using Taylor series.
    pub fn ln(&self, iterations: usize) -> ComplexMatrix {
        if self.rows != self.cols {
            panic!("Matrix must be square");
        }
        
        let identity = ComplexMatrix::identity(self.rows);
        let a_minus_i = self.sub(&identity);
        
        let mut result = ComplexMatrix::zeros(self.rows, self.cols);
        let mut term = a_minus_i.clone();
        
        for n in 1..=iterations {
            let sign = if n % 2 == 0 { Complex::one() } else { -Complex::one() };
            let scalar = sign / Complex::real(n as f64);
            
            for i in 0..result.data.len() {
                result.data[i] = result.data[i] + term.data[i] * scalar;
            }
            
            term = term.mul(&a_minus_i);
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = ComplexMatrix::new(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data.len(), 6);
    }

    #[test]
    fn test_identity() {
        let i = ComplexMatrix::identity(3);
        assert_eq!(i.get(0, 0), Complex::one());
        assert_eq!(i.get(1, 1), Complex::one());
        assert_eq!(i.get(2, 2), Complex::one());
        assert_eq!(i.get(0, 1), Complex::zero());
    }

    #[test]
    fn test_matrix_addition() {
        let mut m1 = ComplexMatrix::new(2, 2);
        m1.set(0, 0, Complex::real(1.0));
        m1.set(0, 1, Complex::real(2.0));
        
        let mut m2 = ComplexMatrix::new(2, 2);
        m2.set(0, 0, Complex::real(3.0));
        m2.set(0, 1, Complex::real(4.0));
        
        let result = m1.add(&m2);
        assert_eq!(result.get(0, 0), Complex::real(4.0));
        assert_eq!(result.get(0, 1), Complex::real(6.0));
    }

    #[test]
    fn test_matrix_multiplication() {
        let mut m1 = ComplexMatrix::new(2, 2);
        m1.set(0, 0, Complex::real(1.0));
        m1.set(0, 1, Complex::real(2.0));
        m1.set(1, 0, Complex::real(3.0));
        m1.set(1, 1, Complex::real(4.0));
        
        let mut m2 = ComplexMatrix::new(2, 2);
        m2.set(0, 0, Complex::real(5.0));
        m2.set(0, 1, Complex::real(6.0));
        m2.set(1, 0, Complex::real(7.0));
        m2.set(1, 1, Complex::real(8.0));
        
        let result = m1.mul(&m2);
        
        assert_eq!(result.get(0, 0), Complex::real(19.0)); // 1*5 + 2*7
        assert_eq!(result.get(0, 1), Complex::real(22.0)); // 1*6 + 2*8
        assert_eq!(result.get(1, 0), Complex::real(43.0)); // 3*5 + 4*7
        assert_eq!(result.get(1, 1), Complex::real(50.0)); // 3*6 + 4*8
    }

    #[test]
    fn test_determinant_2x2() {
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, Complex::real(1.0));
        m.set(0, 1, Complex::real(2.0));
        m.set(1, 0, Complex::real(3.0));
        m.set(1, 1, Complex::real(4.0));
        
        let det = m.determinant();
        assert_eq!(det, Complex::real(-2.0)); // 1*4 - 2*3
    }

    #[test]
    fn test_transpose() {
        let mut m = ComplexMatrix::new(2, 3);
        m.set(0, 0, Complex::real(1.0));
        m.set(0, 1, Complex::real(2.0));
        m.set(0, 2, Complex::real(3.0));
        m.set(1, 0, Complex::real(4.0));
        m.set(1, 1, Complex::real(5.0));
        m.set(1, 2, Complex::real(6.0));
        
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), Complex::real(1.0));
        assert_eq!(t.get(1, 0), Complex::real(2.0));
        assert_eq!(t.get(0, 1), Complex::real(4.0));
    }

    #[test]
    fn test_hermitian() {
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, Complex::real(1.0));
        m.set(0, 1, Complex::new(1.0, 2.0));
        m.set(1, 0, Complex::new(1.0, -2.0));
        m.set(1, 1, Complex::real(3.0));
        
        assert!(m.is_hermitian(1e-10));
    }

    #[test]
    fn test_solve() {
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, Complex::real(2.0));
        m.set(0, 1, Complex::real(1.0));
        m.set(1, 0, Complex::real(1.0));
        m.set(1, 1, Complex::real(1.0));
        
        let b = vec![Complex::real(3.0), Complex::real(2.0)];
        let x = m.solve(&b).unwrap();
        
        // Solution should be [1, 1]
        assert!((x[0].re - 1.0).abs() < 1e-10);
        assert!((x[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse() {
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, Complex::real(2.0));
        m.set(0, 1, Complex::real(1.0));
        m.set(1, 0, Complex::real(1.0));
        m.set(1, 1, Complex::real(1.0));
        
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv);
        
        // Should be close to identity
        assert!((product.get(0, 0) - Complex::one()).norm() < 1e-10);
        assert!((product.get(1, 1) - Complex::one()).norm() < 1e-10);
    }

    #[test]
    fn test_power() {
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, Complex::real(1.0));
        m.set(0, 1, Complex::real(1.0));
        m.set(1, 0, Complex::real(0.0));
        m.set(1, 1, Complex::real(1.0));
        
        let m2 = m.power(2);
        
        assert_eq!(m2.get(0, 0), Complex::real(1.0));
        assert_eq!(m2.get(0, 1), Complex::real(2.0));
        assert_eq!(m2.get(1, 0), Complex::zero());
        assert_eq!(m2.get(1, 1), Complex::real(1.0));
    }
}
