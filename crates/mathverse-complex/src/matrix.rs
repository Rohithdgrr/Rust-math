//! Complex matrix operations: arithmetic, decomposition, eigenvalues.

use crate::Complex;
use mathverse_core::error::{MathError, MathResult};
use std::ops::{Index, IndexMut};

/// Complex matrix with row-major storage.
#[derive(Debug, Clone)]
pub struct ComplexMatrix {
    /// Row-major flat storage.
    pub data: Vec<Complex>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl ComplexMatrix {
    /// Create a new complex matrix filled with zeros.
    pub fn new(rows: usize, cols: usize) -> Self {
        ComplexMatrix {
            data: vec![Complex::zero(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create matrix from data.
    ///
    /// # Panics
    /// If `data.len() != rows * cols`.
    pub fn from_data(data: Vec<Complex>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "data length must equal rows * cols"
        );
        ComplexMatrix { data, rows, cols }
    }

    /// Get element at (row, col).
    ///
    /// # Panics
    /// If the index is out of bounds. Use [`try_get`](Self::try_get) for a
    /// checked variant.
    pub fn get(&self, row: usize, col: usize) -> Complex {
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    ///
    /// # Panics
    /// If the index is out of bounds. Use [`try_set`](Self::try_set) for a
    /// checked variant.
    pub fn set(&mut self, row: usize, col: usize, value: Complex) {
        self.data[row * self.cols + col] = value;
    }

    /// Checked element access: `None` when `(row, col)` is out of bounds.
    pub fn try_get(&self, row: usize, col: usize) -> Option<Complex> {
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Checked element write: `Err` when `(row, col)` is out of bounds.
    pub fn try_set(&mut self, row: usize, col: usize, value: Complex) -> MathResult<()> {
        if row >= self.rows || col >= self.cols {
            Err(MathError::OutOfRange)
        } else {
            self.data[row * self.cols + col] = value;
            Ok(())
        }
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
    ///
    /// # Errors
    /// [`MathError::DimensionMismatch`] if the shapes differ.
    pub fn add(&self, other: &ComplexMatrix) -> MathResult<ComplexMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MathError::DimensionMismatch);
        }
        let mut result = ComplexMatrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }
        Ok(result)
    }

    /// Matrix subtraction.
    ///
    /// # Errors
    /// [`MathError::DimensionMismatch`] if the shapes differ.
    pub fn sub(&self, other: &ComplexMatrix) -> MathResult<ComplexMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MathError::DimensionMismatch);
        }
        let mut result = ComplexMatrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - other.data[i];
        }
        Ok(result)
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
    ///
    /// # Errors
    /// [`MathError::DimensionMismatch`] if `self.cols != other.rows`.
    pub fn mul(&self, other: &ComplexMatrix) -> MathResult<ComplexMatrix> {
        if self.cols != other.rows {
            return Err(MathError::DimensionMismatch);
        }
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
        Ok(result)
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
    ///
    /// # Panics
    /// If the matrix is not square.
    pub fn trace(&self) -> Complex {
        assert_eq!(self.rows, self.cols, "trace requires a square matrix");
        let mut sum = Complex::zero();
        for i in 0..self.rows {
            sum = sum + self.get(i, i);
        }
        sum
    }

    /// Matrix determinant (1×1, 2×2, 3×3 closed forms, larger via LU).
    ///
    /// # Panics
    /// If the matrix is not square.
    pub fn determinant(&self) -> Complex {
        assert_eq!(self.rows, self.cols, "determinant requires a square matrix");
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
                let (_, u, pivot) = self.lu_decomposition().expect("matrix is singular");
                let mut det = Complex::one();
                for i in 0..self.rows {
                    det = det * u.get(i, i);
                }
                // Sign from permutation parity: (-1)^(n - #cycles)
                let mut visited = vec![false; self.rows];
                let mut cycles = 0;
                for i in 0..self.rows {
                    if !visited[i] {
                        cycles += 1;
                        let mut j = i;
                        while !visited[j] {
                            visited[j] = true;
                            j = pivot[j];
                        }
                    }
                }
                if (self.rows - cycles) % 2 == 1 {
                    det = -det;
                }
                det
            }
        }
    }

    /// LU decomposition with partial pivoting.
    ///
    /// Singularity is decided with a **relative** threshold: a pivot is
    /// considered zero only when it is below `1e-15` relative to the largest
    /// entry of the matrix, so matrices whose entries are all tiny (e.g.
    /// `1e-20`-scaled physics data) are not falsely rejected.
    pub fn lu_decomposition(&self) -> Option<(ComplexMatrix, ComplexMatrix, Vec<usize>)> {
        if self.rows != self.cols {
            return None;
        }
        let n = self.rows;
        let mut l = ComplexMatrix::identity(n);
        let mut u = self.clone();
        let mut pivot = (0..n).collect::<Vec<_>>();

        // Global scale of the matrix: max |entry| across all of A. The pivot
        // tolerance is purely *relative* to this scale, so matrices whose
        // entries are all tiny (e.g. 1e-20-scaled physics data) are not
        // falsely declared singular.
        let global_scale = self.data.iter().map(|c| c.norm()).fold(0.0, f64::max);
        let pivot_tol = 1e-15 * global_scale;

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
            if max_val <= pivot_tol {
                return None; // Singular matrix (relative to matrix scale)
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

    /// Check if matrix is unitary (A · Aᴴ = I).
    pub fn is_unitary(&self, tolerance: f64) -> bool {
        if self.rows != self.cols {
            return false;
        }
        let product = self
            .mul(&self.hermitian())
            .expect("unitary check on square matrix");
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
    ///
    /// # Panics
    /// If the matrix is not square.
    pub fn power(&self, n: usize) -> ComplexMatrix {
        assert_eq!(self.rows, self.cols, "power requires a square matrix");
        if n == 0 {
            return ComplexMatrix::identity(self.rows);
        }
        let mut result = ComplexMatrix::identity(self.rows);
        let mut base = self.clone();
        let mut exp = n;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result
                    .mul(&base)
                    .expect("internal: square matrix dimensions");
            }
            base = base.mul(&base).expect("internal: square matrix dimensions");
            exp /= 2;
        }
        result
    }

    /// Matrix exponential e^A using Taylor series with scaling-and-squaring:
    /// compute e^(A/2^k) then square k times. The series terminates early
    /// once a Taylor term falls below `1e-20` in magnitude (so the iteration
    /// count is a safety cap rather than the accuracy control).
    ///
    /// # Panics
    /// If the matrix is not square.
    pub fn exp(&self, iterations: usize) -> ComplexMatrix {
        assert_eq!(self.rows, self.cols, "exp requires a square matrix");
        let norm = self.frobenius_norm();
        let mut scale_pow = 0;
        if norm > 1.0 {
            scale_pow = norm.log2().ceil() as usize;
        }
        let scaled = self.scale(Complex::real(1.0 / 2.0_f64.powi(scale_pow as i32)));

        let mut result = ComplexMatrix::identity(self.rows);
        let mut term = ComplexMatrix::identity(self.rows);
        let mut factorial: f64 = 1.0;

        for n in 1..=iterations {
            factorial *= n as f64;
            term = term
                .mul(&scaled)
                .expect("internal: square matrix dimensions");
            let factor = 1.0 / factorial;
            // Convergence check: once the current Taylor term is far below
            // machine-visible magnitude, the remaining terms (which decay
            // factorially) can never matter again.
            if term.frobenius_norm() * factor < 1e-20 {
                break;
            }
            for i in 0..term.data.len() {
                result.data[i] = result.data[i] + term.data[i] * Complex::real(factor);
            }
        }

        for _ in 0..scale_pow {
            result = result
                .mul(&result)
                .expect("internal: square matrix dimensions");
        }
        result
    }

    /// Matrix logarithm ln(A) using Taylor series around identity.
    /// Returns `None` if the series cannot converge, i.e. if any eigenvalue
    /// of A is outside the disk |λ − 1| < 1 (checked via a Frobenius-norm
    /// bound on A − I, which is conservative but safe).
    ///
    /// # Panics
    /// If the matrix is not square.
    pub fn ln(&self, iterations: usize) -> Option<ComplexMatrix> {
        assert_eq!(self.rows, self.cols, "ln requires a square matrix");
        let identity = ComplexMatrix::identity(self.rows);
        let a_minus_i = self
            .sub(&identity)
            .expect("internal: square matrix dimensions");

        // ||A - I||_F >= spectral radius of (A - I); series needs it < 1
        if a_minus_i.frobenius_norm() >= 1.0 {
            return None;
        }

        let mut result = ComplexMatrix::zeros(self.rows, self.cols);
        let mut term = a_minus_i.clone();
        for n in 1..=iterations {
            let sign = if n % 2 == 0 {
                -Complex::one()
            } else {
                Complex::one()
            };
            let scalar = sign / Complex::real(n as f64);
            for i in 0..result.data.len() {
                result.data[i] = result.data[i] + term.data[i] * scalar;
            }
            term = term
                .mul(&a_minus_i)
                .expect("internal: square matrix dimensions");
        }
        Some(result)
    }

    /// QR decomposition via modified Gram–Schmidt: `A = Q·R` with `Q`
    /// unitary and `R` upper triangular. Rank-deficient columns are completed
    /// with orthonormal basis vectors (QR always exists), so `R` may have
    /// zero diagonal entries for singular input.
    ///
    /// # Errors
    /// [`MathError::DimensionMismatch`] for non-square input.
    pub fn qr_decomposition(&self) -> MathResult<(ComplexMatrix, ComplexMatrix)> {
        if self.rows != self.cols {
            return Err(MathError::DimensionMismatch);
        }
        let n = self.rows;
        if n == 0 {
            return Ok((ComplexMatrix::new(0, 0), ComplexMatrix::new(0, 0)));
        }
        let mut q = ComplexMatrix::new(n, n);
        let mut r = ComplexMatrix::new(n, n);
        for j in 0..n {
            let mut v: Vec<Complex> = (0..n).map(|i| self.get(i, j)).collect();
            for k in 0..j {
                let mut dot = Complex::zero();
                for i in 0..n {
                    dot = dot + q.get(i, k).conjugate() * v[i];
                }
                r.set(k, j, dot);
                for i in 0..n {
                    v[i] = v[i] - dot * q.get(i, k);
                }
            }
            let mut norm_sq = 0.0;
            for i in 0..n {
                norm_sq += v[i].norm_sq();
            }
            let norm = norm_sq.sqrt();
            if norm <= 1e-300 {
                // Column is numerically dependent (e.g. the shift-subtracted
                // matrix inside the eigenvalue QR iteration is singular).
                // Complete the basis with an arbitrary orthonormal vector so
                // that Q stays unitary — QR always exists.
                let mut completed = false;
                for k in 0..n {
                    let mut cand = vec![Complex::zero(); n];
                    cand[k] = Complex::one();
                    for m in 0..j {
                        let mut dot = Complex::zero();
                        for i in 0..n {
                            dot = dot + q.get(i, m).conjugate() * cand[i];
                        }
                        for i in 0..n {
                            cand[i] = cand[i] - dot * q.get(i, m);
                        }
                    }
                    let mut cnorm_sq = 0.0;
                    for i in 0..n {
                        cnorm_sq += cand[i].norm_sq();
                    }
                    if cnorm_sq > 1e-300 {
                        let cnorm = cnorm_sq.sqrt();
                        for i in 0..n {
                            q.set(i, j, cand[i] / Complex::real(cnorm));
                        }
                        completed = true;
                        break;
                    }
                }
                if !completed {
                    return Err(MathError::Singular);
                }
                r.set(j, j, Complex::zero());
                continue;
            }
            r.set(j, j, Complex::real(norm));
            for i in 0..n {
                q.set(i, j, v[i] / Complex::real(norm));
            }
        }
        Ok((q, r))
    }

    /// Eigenvalues of a square matrix via the QR algorithm with Wilkinson
    /// shifts. Returns them in arbitrary order; convergence is judged by the
    /// magnitude of the subdiagonal entries.
    ///
    /// ```
    /// use mathverse_complex::ComplexMatrix;
    /// use mathverse_complex::Complex;
    /// let mut m = ComplexMatrix::new(2, 2);
    /// m.set(0, 0, Complex::real(2.0));
    /// m.set(1, 1, Complex::real(3.0));
    /// let e = m.eigenvalues(1000, 1e-12).unwrap();
    /// assert_eq!(e.len(), 2);
    /// ```
    ///
    /// # Errors
    /// [`MathError::DimensionMismatch`] for non-square input;
    /// [`MathError::NotConverged`] when `max_iterations` is exhausted.
    pub fn eigenvalues(&self, max_iterations: usize, tolerance: f64) -> MathResult<Vec<Complex>> {
        if self.rows != self.cols {
            return Err(MathError::DimensionMismatch);
        }
        let n = self.rows;
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut a = self.clone();
        for _ in 0..max_iterations {
            let mut converged = true;
            for i in 0..n.saturating_sub(1) {
                if a.get(i + 1, i).norm() > tolerance {
                    converged = false;
                    break;
                }
            }
            if converged {
                return Ok((0..n).map(|i| a.get(i, i)).collect());
            }
            // Wilkinson shift: eigenvalue of the trailing 2x2 block closest
            // to the bottom-right entry (avoid the Rayleigh shift's cycling
            // on equal-modulus eigenvalue pairs).
            let shift = if n >= 2 {
                Self::wilkinson_shift(
                    a.get(n - 2, n - 2),
                    a.get(n - 2, n - 1),
                    a.get(n - 1, n - 2),
                    a.get(n - 1, n - 1),
                )
            } else {
                a.get(0, 0)
            };
            let shifted = a.sub(&ComplexMatrix::identity(n).scale(shift))?;
            let (q, r) = shifted.qr_decomposition()?;
            a = r.mul(&q)?.add(&ComplexMatrix::identity(n).scale(shift))?;
        }
        Err(MathError::NotConverged("eigenvalues: QR iteration"))
    }

    /// Wilkinson shift for the trailing 2×2 block `[[a, b], [c, d]]`:
    /// the eigenvalue of that block nearer to `d`.
    fn wilkinson_shift(a: Complex, b: Complex, c: Complex, d: Complex) -> Complex {
        let m = (a + d) / Complex::real(2.0);
        let disc = ((a - d) / Complex::real(2.0)).powf(2.0) + b * c;
        let s = disc.sqrt();
        let l1 = m + s;
        let l2 = m - s;
        if (l1 - d).norm() < (l2 - d).norm() {
            l1
        } else {
            l2
        }
    }
}

impl Index<(usize, usize)> for ComplexMatrix {
    type Output = Complex;
    fn index(&self, (row, col): (usize, usize)) -> &Complex {
        assert!(
            row < self.rows && col < self.cols,
            "index out of bounds: ({row}, {col}) for {}×{} matrix",
            self.rows,
            self.cols
        );
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for ComplexMatrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Complex {
        assert!(
            row < self.rows && col < self.cols,
            "index out of bounds: ({row}, {col}) for {}×{} matrix",
            self.rows,
            self.cols
        );
        &mut self.data[row * self.cols + col]
    }
}

/// `scipy.linalg`-style convenience wrappers around [`ComplexMatrix`].
pub mod linalg {
    use super::*;

    /// `scipy.linalg.expm` equivalent: matrix exponential.
    pub fn expm(a: &ComplexMatrix) -> ComplexMatrix {
        a.exp(50)
    }

    /// `scipy.linalg.logm` equivalent: principal matrix logarithm.
    ///
    /// # Errors
    /// [`MathError::NotConverged`] when `ln` cannot converge (an eigenvalue
    /// of `A` lies outside the disk |λ − 1| < 1).
    pub fn logm(a: &ComplexMatrix) -> MathResult<ComplexMatrix> {
        a.ln(50).ok_or(MathError::NotConverged("logm"))
    }

    /// `scipy.linalg.eig` (values only) equivalent.
    ///
    /// # Errors
    /// See [`ComplexMatrix::eigenvalues`].
    pub fn eig(a: &ComplexMatrix) -> MathResult<Vec<Complex>> {
        a.eigenvalues(1000, 1e-12)
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

        let result = m1.add(&m2).unwrap();
        assert_eq!(result.get(0, 0), Complex::real(4.0));
        assert_eq!(result.get(0, 1), Complex::real(6.0));

        // Shape mismatch is an error, not a panic
        let bad = ComplexMatrix::new(2, 3);
        assert!(matches!(m1.add(&bad), Err(MathError::DimensionMismatch)));
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

        let result = m1.mul(&m2).unwrap();
        assert_eq!(result.get(0, 0), Complex::real(19.0)); // 1*5 + 2*7
        assert_eq!(result.get(0, 1), Complex::real(22.0)); // 1*6 + 2*8
        assert_eq!(result.get(1, 0), Complex::real(43.0)); // 3*5 + 4*7
        assert_eq!(result.get(1, 1), Complex::real(50.0)); // 3*6 + 4*8

        let bad = ComplexMatrix::new(3, 2);
        assert!(matches!(m1.mul(&bad), Err(MathError::DimensionMismatch)));
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
        let product = m.mul(&inv).unwrap();
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

    #[test]
    fn test_determinant_permutation_sign() {
        let mut m = ComplexMatrix::identity(4);
        for j in 0..4 {
            let t = m.get(0, j);
            m.set(0, j, m.get(1, j));
            m.set(1, j, t);
        }
        let det = m.determinant();
        assert!((det.re + 1.0).abs() < 1e-12);
        assert!(det.im.abs() < 1e-12);

        let mut m2 = ComplexMatrix::new(4, 4);
        m2.set(0, 0, Complex::real(1.0));
        m2.set(1, 1, Complex::real(2.0));
        m2.set(2, 2, Complex::real(3.0));
        m2.set(3, 3, Complex::real(4.0));
        for j in 0..4 {
            let t = m2.get(0, j);
            m2.set(0, j, m2.get(1, j));
            m2.set(1, j, t);
        }
        let det2 = m2.determinant();
        assert!((det2.re + 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_exp() {
        // exp(0) = I
        let zero = ComplexMatrix::new(2, 2);
        let e0 = zero.exp(20);
        assert!((e0.get(0, 0) - Complex::one()).norm() < 1e-10);
        assert!((e0.get(1, 1) - Complex::one()).norm() < 1e-10);

        // exp([[0,1],[0,0]]) = [[1,1],[0,1]]
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 1, Complex::one());
        let em = m.exp(20);
        assert!((em.get(0, 0) - Complex::one()).norm() < 1e-10);
        assert!((em.get(0, 1) - Complex::one()).norm() < 1e-10);
        assert!((em.get(1, 0) - Complex::zero()).norm() < 1e-10);
        assert!((em.get(1, 1) - Complex::one()).norm() < 1e-10);

        // exp of large-norm matrix needs scaling-and-squaring: exp(diag(10,10)) = diag(e^10, e^10)
        let mut big = ComplexMatrix::new(2, 2);
        big.set(0, 0, Complex::real(10.0));
        big.set(1, 1, Complex::real(10.0));
        let ebig = big.exp(30);
        let e10 = 10.0_f64.exp();
        assert!((ebig.get(0, 0).re - e10).abs() < 1e-6 * e10);
        assert!((ebig.get(1, 1).re - e10).abs() < 1e-6 * e10);
        assert!((ebig.get(0, 1) - Complex::zero()).norm() < 1e-6 * e10);
    }

    #[test]
    fn test_matrix_exp_large_norm_does_not_overflow() {
        // e^600 ≈ 3.8e260: scaling-and-squaring must keep intermediates in range
        let mut big = ComplexMatrix::new(2, 2);
        big.set(0, 0, Complex::real(600.0));
        big.set(1, 1, Complex::real(600.0));
        let ebig = big.exp(30);
        let e600 = 600.0_f64.exp();
        assert!(ebig.get(0, 0).re.is_finite());
        assert!((ebig.get(0, 0).re - e600).abs() < 1e-6 * e600);
    }

    #[test]
    fn test_matrix_ln() {
        // ln(I) = 0
        let i = ComplexMatrix::identity(2);
        let ln_i = i.ln(20).unwrap();
        assert!((ln_i.get(0, 0) - Complex::zero()).norm() < 1e-10);

        // exp(ln(A)) = A for A near identity
        let mut a = ComplexMatrix::new(2, 2);
        a.set(0, 0, Complex::real(1.2));
        a.set(0, 1, Complex::real(0.1));
        a.set(1, 0, Complex::real(-0.2));
        a.set(1, 1, Complex::real(0.8));
        let ln_a = a.ln(40).unwrap();
        let round = ln_a.exp(40);
        for i in 0..2 {
            for j in 0..2 {
                assert!((round.get(i, j) - a.get(i, j)).norm() < 1e-8);
            }
        }

        // ln(10·I) diverges: must return None (regression: silently returned garbage)
        let mut big = ComplexMatrix::new(2, 2);
        big.set(0, 0, Complex::real(10.0));
        big.set(1, 1, Complex::real(10.0));
        assert!(big.ln(20).is_none());
    }

    #[test]
    fn test_lu_ill_conditioned_scale_invariance() {
        // All entries ~1e-20: the old absolute 1e-15 threshold wrongly
        // declared this singular. A relative threshold must accept it.
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, Complex::real(1e-20));
        m.set(0, 1, Complex::real(2e-20));
        m.set(1, 0, Complex::real(3e-20));
        m.set(1, 1, Complex::real(4e-20));
        let (l, u, pivot) = m
            .lu_decomposition()
            .expect("scaled matrix must not be singular");
        assert_eq!(pivot.len(), 2);
        // L·U = P·A: row k of the product is original row pivot[k]
        let lu = l.mul(&u).unwrap();
        for k in 0..2 {
            for j in 0..2 {
                assert!(
                    (lu.get(k, j) - m.get(pivot[k], j)).norm() < 1e-30,
                    "L·U row {k} col {j} mismatch"
                );
            }
        }

        // Genuinely singular matrices are still rejected
        let mut s = ComplexMatrix::new(2, 2);
        s.set(0, 0, Complex::real(1.0));
        s.set(1, 0, Complex::real(2.0));
        assert!(s.lu_decomposition().is_none());
    }

    #[test]
    fn test_indexing_and_checked_access() {
        let mut m = ComplexMatrix::new(2, 2);
        m[(0, 1)] = Complex::real(7.0);
        assert_eq!(m[(0, 1)], Complex::real(7.0));
        assert_eq!(m.try_get(0, 1), Some(Complex::real(7.0)));
        assert_eq!(m.try_get(5, 5), None);
        assert!(m.try_set(5, 5, Complex::one()).is_err());
        assert!(m.try_set(1, 1, Complex::real(3.0)).is_ok());
        assert_eq!(m.get(1, 1), Complex::real(3.0));
    }

    #[test]
    fn test_qr_decomposition() {
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, Complex::real(1.0));
        m.set(0, 1, Complex::real(2.0));
        m.set(1, 0, Complex::real(3.0));
        m.set(1, 1, Complex::real(4.0));
        let (q, r) = m.qr_decomposition().unwrap();
        // Q unitary
        assert!(q.is_unitary(1e-12));
        // Q·R = A
        let qr = q.mul(&r).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((qr.get(i, j) - m.get(i, j)).norm() < 1e-12);
            }
        }
        // R upper triangular
        assert!((r.get(1, 0) - Complex::zero()).norm() < 1e-12);
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        let mut m = ComplexMatrix::new(3, 3);
        m.set(0, 0, Complex::real(1.0));
        m.set(1, 1, Complex::real(2.0));
        m.set(2, 2, Complex::real(3.0));
        let e = m.eigenvalues(100, 1e-12).unwrap();
        let mut vals: Vec<f64> = e.iter().map(|c| c.re).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        assert!((vals[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_complex_pair() {
        // [[0, -1], [1, 0]] has eigenvalues ±i
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 1, Complex::real(-1.0));
        m.set(1, 0, Complex::real(1.0));
        let e = m.eigenvalues(2000, 1e-10).unwrap();
        let d_i = |c: &Complex| (*c - Complex::i()).norm();
        let d_ni = |c: &Complex| (*c + Complex::i()).norm();
        assert!(e.iter().map(d_i).fold(f64::MAX, f64::min) < 1e-6);
        assert!(e.iter().map(d_ni).fold(f64::MAX, f64::min) < 1e-6);
    }

    #[test]
    fn test_eigenvalues_2x2_real() {
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, Complex::real(2.0));
        m.set(0, 1, Complex::real(1.0));
        m.set(1, 0, Complex::real(1.0));
        m.set(1, 1, Complex::real(2.0));
        let e = m.eigenvalues(2000, 1e-10).unwrap();
        let mut vals: Vec<f64> = e.iter().map(|c| c.re).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_wrappers() {
        use crate::matrix::linalg;
        // expm and eig work on any square matrix
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 1, Complex::one());
        let em = linalg::expm(&m);
        assert!((em.get(0, 1) - Complex::one()).norm() < 1e-10);
        let e = linalg::eig(&m).unwrap();
        assert_eq!(e.len(), 2);
        // logm requires ||A - I|| < 1: use a matrix near the identity
        let mut a = ComplexMatrix::new(2, 2);
        a.set(0, 0, Complex::real(1.1));
        a.set(0, 1, Complex::real(0.05));
        a.set(1, 0, Complex::real(-0.05));
        a.set(1, 1, Complex::real(0.9));
        let ln = linalg::logm(&a).unwrap();
        let round = ln.exp(40);
        for i in 0..2 {
            for j in 0..2 {
                assert!((round.get(i, j) - a.get(i, j)).norm() < 1e-8);
            }
        }
        // logm on a matrix outside the convergence disk errors cleanly
        assert!(linalg::logm(&m).is_err());
    }
}
