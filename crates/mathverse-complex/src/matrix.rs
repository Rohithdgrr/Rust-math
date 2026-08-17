//! Complex matrix operations: arithmetic, decomposition, eigenvalues.

use crate::Complex;
use mathverse_core::error::{MathError, MathResult};
use std::ops::{Index, IndexMut};

/// Complex matrix with row-major storage.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    #[must_use]
    pub fn scale(&self, scalar: Complex) -> ComplexMatrix {
        let mut result = ComplexMatrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] * scalar;
        }
        result
    }

    /// Matrix multiplication.
    ///
    /// Uses cache-blocked (tiling) algorithm for better performance on large
    /// matrices. Block size is 64 elements (fits L1 cache for `Complex<f64>`).
    ///
    /// # Errors
    /// [`MathError::DimensionMismatch`] if `self.cols != other.rows`.
    pub fn mul(&self, other: &ComplexMatrix) -> MathResult<ComplexMatrix> {
        // block size — 64 Complex<f64> = 1024 bytes ≈ L1 line
        const B: usize = 64;
        if self.cols != other.rows {
            return Err(MathError::DimensionMismatch);
        }
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let mut result = ComplexMatrix::new(m, n);

        for ii in (0..m).step_by(B) {
            let ii_end = (ii + B).min(m);
            for jj in (0..n).step_by(B) {
                let jj_end = (jj + B).min(n);
                for kk in (0..k).step_by(B) {
                    let kk_end = (kk + B).min(k);
                    // Multiply the ii×kk block of self by the kk×jj block of
                    // other, accumulating into the ii×jj block of result.
                    for i in ii..ii_end {
                        let row_a = i * k;
                        let row_c = i * n;
                        for kk2 in kk..kk_end {
                            let a_ik = self.data[row_a + kk2];
                            let row_b = kk2 * n;
                            for j in jj..jj_end {
                                result.data[row_c + j] =
                                    result.data[row_c + j] + a_ik * other.data[row_b + j];
                            }
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    /// Matrix multiplication with a register-blocked GEMM kernel.
    ///
    /// Equivalent to [`mul`](Self::mul) but uses the classic i-k-j loop
    /// ordering with 4-wide output register blocks, which improves cache
    /// reuse of the right operand and the accumulator row. Available with
    /// the `blas` feature (which also enables the `ndarray` conversions).
    /// A general complex GEMM from `matrixmultiply` is not used because its
    /// public entry points are `unsafe`, and this crate forbids `unsafe`.
    ///
    /// # Errors
    /// [`MathError::DimensionMismatch`] if `self.cols != other.rows`.
    #[cfg(feature = "blas")]
    pub fn blas_mul(&self, other: &ComplexMatrix) -> MathResult<ComplexMatrix> {
        const B: usize = 64;
        if self.cols != other.rows {
            return Err(MathError::DimensionMismatch);
        }
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let mut result = ComplexMatrix::new(m, n);

        for ii in (0..m).step_by(B) {
            let ii_end = (ii + B).min(m);
            for kk in (0..k).step_by(B) {
                let kk_end = (kk + B).min(k);
                for jj in (0..n).step_by(B) {
                    let jj_end = (jj + B).min(n);
                    for i in ii..ii_end {
                        let row_c = i * n;
                        let row_a = i * k;
                        // 4-wide register block along j: accumulate into
                        // locals, write back once per block.
                        let mut j = jj;
                        while j + 4 <= jj_end {
                            let mut cr = [0.0f64; 4];
                            let mut ci = [0.0f64; 4];
                            for kk2 in kk..kk_end {
                                let a = self.data[row_a + kk2];
                                let row_b = kk2 * n;
                                for t in 0..4 {
                                    let b = other.data[row_b + j + t];
                                    cr[t] += a.re * b.re - a.im * b.im;
                                    ci[t] += a.re * b.im + a.im * b.re;
                                }
                            }
                            for t in 0..4 {
                                let acc = result.data[row_c + j + t];
                                result.data[row_c + j + t] =
                                    Complex::new(cr[t] + acc.re, ci[t] + acc.im);
                            }
                            j += 4;
                        }
                        // Scalar remainder columns
                        while j < jj_end {
                            let mut acc = Complex::zero();
                            for kk2 in kk..kk_end {
                                acc = acc + self.data[row_a + kk2] * other.data[kk2 * n + j];
                            }
                            result.data[row_c + j] = result.data[row_c + j] + acc;
                            j += 1;
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    /// Matrix transpose.
    #[must_use]
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
    #[must_use]
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
        let global_scale = self.data.iter().map(super::Complex::norm).fold(0.0, f64::max);
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
            {
                let col = self.solve(&b)?;
                for i in 0..n {
                    result.set(i, j, col[i]);
                }
            }
        }
        Some(result)
    }

    /// Cholesky decomposition of a Hermitian positive-definite matrix:
    /// returns the lower-triangular factor `L` such that `A = L·Lᴴ`.
    ///
    /// ```
    /// use mathverse_complex::{Complex, ComplexMatrix};
    /// let mut m = ComplexMatrix::new(2, 2);
    /// m.set(0, 0, Complex::real(4.0));
    /// m.set(0, 1, Complex::new(1.0, 1.0));
    /// m.set(1, 0, Complex::new(1.0, -1.0));
    /// m.set(1, 1, Complex::real(3.0));
    /// let l = m.cholesky().unwrap();
    /// let llh = l.mul(&l.hermitian()).unwrap();
    /// assert!((llh.get(0, 0) - Complex::real(4.0)).norm() < 1e-12);
    /// ```
    ///
    /// # Errors
    /// [`MathError::DimensionMismatch`] for non-square input;
    /// [`MathError::NotConverged`] when the matrix is not Hermitian
    /// positive-definite (a non-positive diagonal pivot is encountered).
    pub fn cholesky(&self) -> MathResult<ComplexMatrix> {
        if self.rows != self.cols {
            return Err(MathError::DimensionMismatch);
        }
        let n = self.rows;
        let mut l = ComplexMatrix::new(n, n);
        for i in 0..n {
            for j in 0..=i {
                // sum = A[i][j] − Σ_{k<j} L[i][k]·conj(L[j][k])
                let mut sum = self.get(i, j);
                for k in 0..j {
                    sum = sum - l.get(i, k) * l.get(j, k).conjugate();
                }
                if i == j {
                    // Diagonal pivots are real in exact arithmetic
                    if sum.re <= 0.0 {
                        return Err(MathError::NotConverged(
                            "cholesky: matrix is not positive definite",
                        ));
                    }
                    l.set(i, i, Complex::real(sum.re.sqrt()));
                } else {
                    l.set(i, j, sum / l.get(j, j));
                }
            }
        }
        Ok(l)
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
        let Ok(product) = self.mul(&self.hermitian()) else {
            return false;
        };
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
        self.data.iter().map(super::Complex::norm_sq).sum::<f64>().sqrt()
    }

    /// Frobenius norm (alias for [`frobenius_norm`](Self::frobenius_norm)).
    pub fn norm(&self) -> f64 {
        self.frobenius_norm()
    }

    /// Matrix power A^n.
    ///
    /// # Panics
    /// If the matrix is not square.
    #[must_use]
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
    #[must_use]
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

        for n in 1..=iterations {
            term = term
                .mul(&scaled)
                .expect("internal: square matrix dimensions");
            // Recurrence: term = A^n / n! = (A^(n-1) / (n-1)!) * A / n
            // This avoids computing n! explicitly, which overflows at n > 170.
            let scale = Complex::real(1.0 / n as f64);
            for i in 0..term.data.len() {
                term.data[i] = term.data[i] * scale;
            }
            // Convergence check: once the current Taylor term is far below
            // machine-visible magnitude, the remaining terms (which decay
            // factorially) can never matter again.
            if term.frobenius_norm() < 1e-20 {
                break;
            }
            for i in 0..term.data.len() {
                result.data[i] = result.data[i] + term.data[i];
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
    /// First reduces the matrix to upper Hessenberg form (O(n³)), then
    /// applies the QR iteration with O(n²) per-iteration cost using Givens
    /// rotations.
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
        // Reduce to upper Hessenberg form once — O(n³).
        let mut a = self.hessenberg_reduction();

        for _ in 0..max_iterations {
            // Check convergence: all subdiagonal entries below tolerance.
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
            // to the bottom-right entry.
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
            // Shift, QR decompose (O(n²) on Hessenberg), and recover.
            let shifted = a.sub(&ComplexMatrix::identity(n).scale(shift))?;
            let (q, r) = shifted.qr_hessenberg();
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

    /// Reduce a square matrix to upper Hessenberg form via Householder
    /// reflections. An upper Hessenberg matrix has `h[i][j] = 0` for
    /// `i > j + 1`, meaning all entries below the first subdiagonal are zero.
    ///
    /// This is an O(n³) operation that is a prerequisite for the O(n²)-per-
    /// iteration QR eigenvalue algorithm.
    ///
    /// # Panics
    /// If the matrix is not square.
    #[must_use]
    pub fn hessenberg_reduction(&self) -> ComplexMatrix {
        assert_eq!(self.rows, self.cols, "hessenberg requires a square matrix");
        let n = self.rows;
        if n <= 2 {
            return self.clone();
        }
        let mut a = self.clone();

        for k in 0..n - 2 {
            // Extract the column below the subdiagonal: x = A[k+1..n, k].
            let x: Vec<Complex> = (k + 1..n).map(|i| a.get(i, k)).collect();
            let x_norm: f64 = x.iter().map(super::Complex::norm_sq).sum::<f64>().sqrt();

            if x_norm < 1e-15 {
                continue; // Column is already zero below subdiagonal.
            }

            // Householder vector v = x - alpha * e1, where alpha = -sign(x[0]) * ||x||.
            let alpha = if x[0].re >= 0.0 {
                -Complex::real(x_norm)
            } else {
                Complex::real(x_norm)
            };
            let mut v = x.clone();
            v[0] = v[0] - alpha;
            let v_norm_sq: f64 = v.iter().map(super::Complex::norm_sq).sum();
            if v_norm_sq < 1e-30 {
                continue;
            }

            // Apply similarity: A = (I - 2vv^H/||v||^2) * A * (I - 2vv^H/||v||^2)
            // Left multiply: A rows k+1..n -= (2/||v||^2) * v * (v^H * A rows k+1..n)
            let scale = Complex::real(2.0 / v_norm_sq);
            for j in 0..n {
                // Compute dot = v^H * A[(k+1..n, j)]
                let dot: Complex = (0..v.len())
                    .map(|i| v[i].conjugate() * a.get(k + 1 + i, j))
                    .fold(Complex::zero(), |acc, x| acc + x);
                let scaled_dot = scale * dot;
                for i in 0..v.len() {
                    a.data[(k + 1 + i) * n + j] =
                        a.data[(k + 1 + i) * n + j] - v[i] * scaled_dot;
                }
            }

            // Right multiply: A columns k+1..n -= (2/||v||^2) * (A columns k+1..n * v) * v^H
            for i in 0..n {
                // Compute dot = A[i, k+1..n] * v
                let dot: Complex = (0..v.len())
                    .map(|j| a.get(i, k + 1 + j) * v[j])
                    .fold(Complex::zero(), |acc, x| acc + x);
                let scaled_dot = scale * dot;
                for j in 0..v.len() {
                    a.data[i * n + k + 1 + j] =
                        a.data[i * n + k + 1 + j] - scaled_dot * v[j].conjugate();
                }
            }
        }
        a
    }

    /// QR decomposition of an upper Hessenberg matrix using Givens rotations.
    /// Returns (Q, R) where Q is unitary and R is upper triangular.
    ///
    /// This is O(n²) instead of the general O(n³) QR, because a Hessenberg
    /// matrix only needs n-1 Givens rotations to reduce to upper triangular.
    fn qr_hessenberg(&self) -> (ComplexMatrix, ComplexMatrix) {
        let n = self.rows;
        let mut r = self.clone();
        let mut q = ComplexMatrix::identity(n);

        for j in 0..n - 1 {
            let a = r.get(j, j);
            let b = r.get(j + 1, j);
            let hypot = Complex::real(a.norm().sqrt() + b.norm()); // avoid overflow
            if hypot.norm() < 1e-300 {
                continue;
            }
            // Givens rotation parameters: c = a/h, s = b/h
            let h = Complex::real((a.norm_sq() + b.norm_sq()).sqrt());
            let c = a / h;
            let s = b / h;

            // Apply G to rows j and j+1 of R: [c, s; -s*, c] * R
            for k in j..n {
                let r_j = r.get(j, k);
                let r_j1 = r.get(j + 1, k);
                r.data[j * n + k] = c.conjugate() * r_j + s.conjugate() * r_j1;
                r.data[(j + 1) * n + k] = -s * r_j + c * r_j1;
            }
            // Apply G to columns j and j+1 of Q: Q * [c, s; -s*, c]^H
            for i in 0..n {
                let q_ij = q.get(i, j);
                let q_ij1 = q.get(i, j + 1);
                q.data[i * n + j] = c * q_ij + s * q_ij1;
                q.data[i * n + j + 1] = -s.conjugate() * q_ij + c.conjugate() * q_ij1;
            }
        }
        (q, r)
    }

    /// Eigenvectors of the matrix via inverse iteration.
    /// Returns a matrix whose columns are the (right) eigenvectors.
    pub fn eigenvectors(&self, max_iterations: usize, tolerance: f64) -> MathResult<ComplexMatrix> {
        let n = self.rows;
        if n == 0 {
            return Ok(ComplexMatrix::new(0, 0));
        }
        let eigenvals = self.eigenvalues(max_iterations, tolerance)?;
        let mut result = ComplexMatrix::new(n, n);

        for (k, &lambda) in eigenvals.iter().enumerate() {
            // Shift away from exact eigenvalue so the matrix is non-singular
            let shifted = self.sub(&ComplexMatrix::identity(n).scale(lambda + Complex::real(1e-8)))?;
            let mut x = ComplexMatrix::new(n, 1);
            x.set(0, 0, Complex::one());
            for _ in 0..30 {
                // Create augmented matrix [shifted | x]
                let mut aug = ComplexMatrix::new(n, n + 1);
                for r in 0..n {
                    for c in 0..n {
                        aug.set(r, c, shifted.get(r, c));
                    }
                    aug.set(r, n, x.get(r, 0));
                }
                for col in 0..n {
                    let mut max_val = aug.get(col, col).norm();
                    let mut max_row = col;
                    for row in (col + 1)..n {
                        if aug.get(row, col).norm() > max_val {
                            max_val = aug.get(row, col).norm();
                            max_row = row;
                        }
                    }
                    if max_val < 1e-15 { break; }
                    if max_row != col {
                        for c in col..=n {
                            let tmp = aug.get(col, c);
                            aug.set(col, c, aug.get(max_row, c));
                            aug.set(max_row, c, tmp);
                        }
                    }
                    for row in (col + 1)..n {
                        let factor = aug.get(row, col) / aug.get(col, col);
                        for c in col..=n {
                            aug.set(row, c, aug.get(row, c) - factor * aug.get(col, c));
                        }
                    }
                }
                for i in (0..n).rev() {
                    let mut sum = aug.get(i, n);
                    for j in (i + 1)..n {
                        sum = sum - aug.get(i, j) * x.get(j, 0);
                    }
                    if aug.get(i, i).norm() > 1e-15 {
                        x.set(i, 0, sum / aug.get(i, i));
                    }
                }
                let norm = (0..n).map(|i| x.get(i, 0).norm()).fold(0.0f64, f64::max);
                if norm > 1e-15 {
                    for i in 0..n {
                        x.set(i, 0, x.get(i, 0) / Complex::real(norm));
                    }
                }
            }
            for i in 0..n {
                result.set(i, k, x.get(i, 0));
            }
        }
        Ok(result)
    }

    /// Singular Value Decomposition. Returns (U, S, Vh) where A = U diag(S) Vh.
    pub fn svd(&self, max_iterations: usize, tolerance: f64) -> MathResult<(ComplexMatrix, Vec<f64>, ComplexMatrix)> {
        let m = self.rows;
        let n = self.cols;
        if m == 0 || n == 0 {
            return Ok((ComplexMatrix::new(m, m), Vec::new(), ComplexMatrix::new(n, n)));
        }
        let (u, b, vh) = self.householder_bidiagonalize();
        let mut d: Vec<f64> = (0..n).map(|i| b.get(i, i).re.abs()).collect();
        let mut e: Vec<f64> = if n > 1 { (0..n - 1).map(|i| b.get(i, i + 1).re.abs()).collect() } else { vec![] };

        for _ in 0..max_iterations {
            let mut converged = true;
            for i in 0..n {
                if i < n - 1 && e[i].abs() > tolerance * (d[i].abs() + d[i + 1].abs()) {
                    converged = false;
                }
            }
            if converged { break; }
            let mut shift = 0.0;
            for i in (0..n - 1).rev() {
                if e[i].abs() > tolerance {
                    shift = (d[i + 1] * d[i + 1] - d[i] * d[i] + e[i] * e[i]) / (2.0 * e[i] * d[i + 1]);
                    break;
                }
            }
            for i in 0..n - 1 {
                let f = e[i];
                let h = d[i + 1];
                let y = ((d[i] + shift) * (d[i] + shift) + f * f).sqrt();
                let z = (h * h + f * f).sqrt();
                if y > 1e-15 { d[i] = y; }
                if z > 1e-15 { e[i] = z; } else { e[i] = 0.0; }
            }
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).unwrap_or(std::cmp::Ordering::Equal));
        let sorted_s: Vec<f64> = indices.iter().map(|&i| d[i]).collect();
        let mut sorted_vh = ComplexMatrix::new(n, n);
        for (new_col, &old_col) in indices.iter().enumerate() {
            for row in 0..n {
                sorted_vh.set(row, new_col, vh.get(row, old_col));
            }
        }
        Ok((u, sorted_s, sorted_vh))
    }

    fn householder_bidiagonalize(&self) -> (ComplexMatrix, ComplexMatrix, ComplexMatrix) {
        let m = self.rows;
        let n = self.cols;
        let mut u = ComplexMatrix::identity(m);
        let mut v = ComplexMatrix::identity(n);
        let mut b = self.clone();

        let k = if m < n { m } else { n };
        for i in 0..k {
            // Left Householder
            if i < m {
                let x: Vec<Complex> = (i..m).map(|row| b.get(row, i)).collect();
                let alpha = Complex::real(-(if x[0].re >= 0.0 { 1.0 } else { -1.0 })) * Complex::real(
                    x.iter().map(super::Complex::norm_sq).sum::<f64>().sqrt()
                );
                let v0 = x[0] - alpha;
                if v0.norm() > 1e-15 {
                    let w = x.iter().map(super::Complex::norm_sq).sum::<f64>().sqrt();
                    for j in i..m {
                        b.set(j, i, if j == i { alpha } else { Complex::zero() });
                        for col in 0..n {
                            let old = b.get(j, col);
                            let vj = if j == i { v0 } else { x[j - i] };
                            let factor = Complex::real(2.0) * vj * v0.conjugate() / Complex::real(w * w);
                            b.set(j, col, old - factor * b.get(i, col));
                        }
                    }
                    for col in 0..m {
                        let mut sum = Complex::zero();
                        for row in i..m {
                            let vj = if row == i { v0 } else { x[row - i] };
                            let factor = Complex::real(2.0) * vj * v0.conjugate() / Complex::real(w * w);
                            sum = sum + factor.conjugate() * u.get(col, row);
                        }
                        u.set(col, i, u.get(col, i) - sum);
                    }
                }
            }
            // Right Householder
            if i < n - 1 {
                let x: Vec<Complex> = (i..n).map(|col| b.get(i, col)).collect();
                let alpha = Complex::real(-(if x[0].re >= 0.0 { 1.0 } else { -1.0 })) * Complex::real(
                    x.iter().map(super::Complex::norm_sq).sum::<f64>().sqrt()
                );
                let v0 = x[0] - alpha;
                if v0.norm() > 1e-15 {
                    let w = x.iter().map(super::Complex::norm_sq).sum::<f64>().sqrt();
                    for col in i..n {
                        b.set(i, col, if col == i { alpha } else { Complex::zero() });
                        for row in 0..m {
                            let old = b.get(row, col);
                            let vj = if col == i { v0 } else { x[col - i] };
                            let factor = Complex::real(2.0) * v0.conjugate() * vj / Complex::real(w * w);
                            b.set(row, col, old - factor * b.get(row, i));
                        }
                    }
                    for row in 0..n {
                        let mut sum = Complex::zero();
                        for col in i..n {
                            let vj = if col == i { v0 } else { x[col - i] };
                            let factor = Complex::real(2.0) * v0.conjugate() * vj / Complex::real(w * w);
                            sum = sum + v.get(row, col) * factor;
                        }
                        v.set(row, i, v.get(row, i) - sum);
                    }
                }
            }
        }
        (u, b, v)
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

/// Zero-copy transfer into an `ndarray` 2-D array (row-major, same layout).
#[cfg(feature = "blas")]
impl From<ComplexMatrix> for ndarray::Array2<Complex> {
    fn from(mat: ComplexMatrix) -> Self {
        ndarray::Array2::from_shape_vec((mat.rows, mat.cols), mat.data)
            .expect("rows * cols matches data length")
    }
}

/// Adoption of an `ndarray` 2-D array of [`Complex`] values.
#[cfg(feature = "blas")]
impl From<ndarray::Array2<Complex>> for ComplexMatrix {
    fn from(arr: ndarray::Array2<Complex>) -> Self {
        let rows = arr.nrows();
        let cols = arr.ncols();
        ComplexMatrix::from_data(arr.into_raw_vec(), rows, cols)
    }
}

/// `scipy.linalg`-style convenience wrappers around [`ComplexMatrix`].
pub mod linalg {
    use super::{ComplexMatrix, MathResult, MathError, Complex};

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
        let dist_to = |target: Complex| {
            e.iter()
                .map(|c| (*c - target).norm())
                .fold(f64::MAX, f64::min)
        };
        assert!(dist_to(Complex::i()) < 1e-6);
        assert!(dist_to(-Complex::i()) < 1e-6);
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
    fn test_cholesky() {
        // Hermitian positive-definite: A = [[4, 1+i], [1-i, 3]]
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, Complex::real(4.0));
        m.set(0, 1, Complex::new(1.0, 1.0));
        m.set(1, 0, Complex::new(1.0, -1.0));
        m.set(1, 1, Complex::real(3.0));
        let l = m.cholesky().unwrap();
        // L is lower triangular
        assert!((l.get(0, 1) - Complex::zero()).norm() < 1e-12);
        // L·Lᴴ = A
        let llh = l.mul(&l.hermitian()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (llh.get(i, j) - m.get(i, j)).norm() < 1e-12,
                    "L·Lᴴ mismatch at ({i}, {j})"
                );
            }
        }

        // 3×3: L·Lᴴ = A with a non-diagonal Hermitian matrix
        let mut m3 = ComplexMatrix::new(3, 3);
        m3.set(0, 0, Complex::real(4.0));
        m3.set(0, 1, Complex::new(1.0, -2.0));
        m3.set(0, 2, Complex::new(0.0, 1.0));
        m3.set(1, 0, Complex::new(1.0, 2.0));
        m3.set(1, 1, Complex::real(6.0));
        m3.set(1, 2, Complex::new(-1.0, 0.5));
        m3.set(2, 0, Complex::new(0.0, -1.0));
        m3.set(2, 1, Complex::new(-1.0, -0.5));
        m3.set(2, 2, Complex::real(5.0));
        let l3 = m3.cholesky().unwrap();
        let llh3 = l3.mul(&l3.hermitian()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((llh3.get(i, j) - m3.get(i, j)).norm() < 1e-10);
            }
        }

        // Not positive-definite errors cleanly
        let mut bad = ComplexMatrix::new(2, 2);
        bad.set(0, 0, Complex::real(1.0));
        bad.set(1, 1, Complex::real(-1.0));
        assert!(bad.cholesky().is_err());

        // Non-square errors cleanly
        let rect = ComplexMatrix::new(2, 3);
        assert!(rect.cholesky().is_err());
    }

    #[test]
    fn test_norm_alias() {
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, Complex::real(3.0));
        m.set(1, 1, Complex::real(4.0));
        assert!((m.norm() - m.frobenius_norm()).abs() < 1e-15);
        assert!((m.norm() - 5.0).abs() < 1e-12);
    }

    #[cfg(feature = "blas")]
    #[test]
    fn test_blas_mul_matches_mul() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for (m, k, n) in [(2usize, 3usize, 4usize), (7, 5, 6), (1, 1, 1)] {
            let mut a = ComplexMatrix::new(m, k);
            let mut b = ComplexMatrix::new(k, n);
            for i in 0..m * k {
                a.data[i] = Complex::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
            }
            for i in 0..k * n {
                b.data[i] = Complex::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
            }
            let ref_mul = a.mul(&b).unwrap();
            let blas = a.blas_mul(&b).unwrap();
            for i in 0..m * n {
                assert!(
                    (ref_mul.data[i] - blas.data[i]).norm() < 1e-12,
                    "blas_mul mismatch at {m}x{k}x{n}, element {i}"
                );
            }
        }
        // Dimension mismatch errors identically
        let a = ComplexMatrix::new(2, 3);
        let b = ComplexMatrix::new(2, 2);
        assert!(a.blas_mul(&b).is_err());
    }

    #[cfg(feature = "blas")]
    #[test]
    fn test_ndarray_roundtrip() {
        use ndarray::Array2;
        let mut m = ComplexMatrix::new(2, 3);
        for (i, c) in m.data.iter_mut().enumerate() {
            *c = Complex::new(i as f64, -(i as f64));
        }
        let arr: Array2<Complex> = m.clone().into();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[1, 2]], m.get(1, 2));

        let back: ComplexMatrix = arr.into();
        assert_eq!(back.rows, 2);
        assert_eq!(back.cols, 3);
        for (a, b) in back.data.iter().zip(m.data.iter()) {
            assert!((a - b).norm() < 1e-15);
        }
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
