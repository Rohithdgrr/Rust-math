//! Decompositions: LU, Cholesky, QR (Householder), SVD (one-sided Jacobi),
//! symmetric eigen (Jacobi).

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Maximum number of full sweeps for the one-sided Jacobi SVD.
///
/// One-sided Jacobi converges quadratically and typical matrices reach the
/// `off < 1e-14` stopping criterion within ~6–12 sweeps; the cap only bounds
/// worst-case runtime. If the cap is hit before convergence the best available
/// factorization is returned (accuracy may be degraded for extreme inputs).
pub const MAX_SVD_SWEEPS: usize = 30;

/// Convergence threshold on the relative off-diagonal energy for the
/// one-sided Jacobi SVD.
pub const SVD_OFF_DIAGONAL_TOL: f64 = 1e-14;

/// Maximum number of sweeps for the cyclic Jacobi symmetric eigensolver.
///
/// Same quadratic-convergence behaviour as [`MAX_SVD_SWEEPS`]: the cap is a
/// worst-case guard, not the expected iteration count.
pub const MAX_JACOBI_SWEEPS: usize = 50;

/// Stopping threshold on the largest off-diagonal magnitude for the
/// symmetric Jacobi eigensolver.
pub const JACOBI_OFF_DIAGONAL_TOL: f64 = 1e-14;

/// LU decomposition with partial pivoting: `P A = L U` (unit-diagonal L).
#[derive(Debug, Clone)]
pub struct Lu {
    pub l: Matrix,
    pub u: Matrix,
    /// Row permutation: row `i` of `P A` is row `pivots[i]` of `A`.
    pub pivots: Vec<usize>,
    /// Parity of the permutation (determinant sign).
    pub sign: f64,
}

impl Matrix {
    /// LU with partial pivoting.
    ///
    /// Fails with [`MathError::Singular`] when a pivot drops to (or below)
    /// `ε · n · ‖A‖∞`, i.e. when elimination proves the matrix numerically
    /// singular at machine precision *relative to the input scale*. Merely
    /// ill-conditioned inputs (e.g. condition ~1e13) still factor fine — use
    /// [`crate::condition`] to assess quality separately.
    pub fn lu(&self) -> MathResult<Lu> {
        if !self.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        let n = self.rows;
        let mut a = self.data.clone();
        let mut pivots: Vec<usize> = (0..n).collect();
        let mut sign = 1.0f64;
        // Relative singularity threshold: scales with the largest input
        // magnitude so both huge and tiny well-formed matrices are treated
        // consistently.
        let anorm = self.data.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let tol = f64::EPSILON * (n.max(1) as f64) * anorm;
        for k in 0..n {
            let mut p = k;
            let mut max = a[k * n + k].abs();
            for i in (k + 1)..n {
                let v = a[i * n + k].abs();
                if v > max {
                    max = v;
                    p = i;
                }
            }
            if !(max > tol) {
                // Covers exact zeros and numerically-dead pivots alike.
                return Err(MathError::Singular);
            }
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
                pivots.swap(k, p);
                sign = -sign;
            }
            for i in (k + 1)..n {
                let f = a[i * n + k] / a[k * n + k];
                a[i * n + k] = f;
                for j in (k + 1)..n {
                    a[i * n + j] -= f * a[k * n + j];
                }
            }
        }
        let mut l = Matrix::zeros(n, n);
        let mut u = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let v = a[i * n + j];
                if i > j {
                    l.set(i, j, v);
                } else if i == j {
                    l.set(i, j, 1.0);
                    u.set(i, j, v);
                } else {
                    u.set(i, j, v);
                }
            }
        }
        Ok(Lu { l, u, pivots, sign })
    }
}

impl Matrix {
    /// Cholesky `L` with `A = L Lᵀ`; errors unless `A` is positive definite.
    pub fn cholesky(&self) -> MathResult<Matrix> {
        if !self.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        let n = self.rows;
        let mut l = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut s = self.get(i, j);
                for k in 0..j {
                    s -= l.get(i, k) * l.get(j, k);
                }
                if i == j {
                    if s <= 0.0 {
                        return Err(MathError::InvalidArgument("matrix not positive definite"));
                    }
                    l.set(i, j, s.sqrt());
                } else {
                    l.set(i, j, s / l.get(j, j));
                }
            }
        }
        Ok(l)
    }
}

/// QR decomposition via Householder reflectors: `A = Q R`.
/// `Q` is `m×m` orthogonal; `R` is `m×n` upper triangular.
#[derive(Debug, Clone)]
pub struct Qr {
    pub q: Matrix,
    pub r: Matrix,
}

impl Matrix {
    pub fn qr(&self) -> MathResult<Qr> {
        let (m, n) = (self.rows, self.cols);
        let mut r = self.clone();
        let mut q = Matrix::identity(m);
        for k in 0..n.min(m) {
            let mut x: Vec<f64> = (k..m).map(|i| r.get(i, k)).collect();
            let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_x == 0.0 {
                continue;
            }
            let alpha = if x[0] >= 0.0 { -norm_x } else { norm_x };
            x[0] -= alpha;
            let vn = x.iter().map(|w| w * w).sum::<f64>().sqrt();
            if vn == 0.0 {
                continue;
            }
            for w in &mut x {
                *w /= vn;
            }
            // R <- H R
            for j in k..n {
                let dotv: f64 = x.iter().enumerate().map(|(o, &vv)| vv * r.get(k + o, j)).sum();
                for (o, &vv) in x.iter().enumerate() {
                    r.set(k + o, j, r.get(k + o, j) - 2.0 * vv * dotv);
                }
            }
            // Q <- Q H
            for i in 0..m {
                let dotv: f64 = x.iter().enumerate().map(|(o, &vv)| q.get(i, k + o) * vv).sum();
                for (o, &vv) in x.iter().enumerate() {
                    q.set(i, k + o, q.get(i, k + o) - 2.0 * dotv * vv);
                }
            }
        }
        // Scrub numerical dust below the diagonal.
        for i in 0..m {
            for j in 0..n.min(i) {
                r.set(i, j, 0.0);
            }
        }
        Ok(Qr { q, r })
    }
}

/// SVD via one-sided Jacobi: `A = U S Vᵀ`.
/// `U` is `m×n` with orthonormal columns (zero columns for zero singular
/// values), `S` the descending singular values, `Vᵀ` is `n×n` orthogonal.
#[derive(Debug, Clone)]
pub struct Svd {
    pub u: Matrix,
    pub s: Vec<f64>,
    pub vt: Matrix,
}

impl Matrix {
    /// SVD via one-sided Jacobi rotations.
    ///
    /// Sweeps until the relative off-diagonal column energy drops below
    /// [`SVD_OFF_DIAGONAL_TOL`], capped at [`MAX_SVD_SWEEPS`] sweeps (see the
    /// constant's notes on convergence).
    pub fn svd(&self) -> MathResult<Svd> {
        let (m, n) = (self.rows, self.cols);
        let mut b = self.clone();
        let mut v = Matrix::identity(n);
        for _ in 0..MAX_SVD_SWEEPS {
            let mut off = 0.0;
            for p in 0..n {
                for q in (p + 1)..n {
                    let bp = b.col(p);
                    let bq = b.col(q);
                    let (mut aa, mut bb, mut g) = (0.0, 0.0, 0.0);
                    for i in 0..m {
                        aa += bp[i] * bp[i];
                        bb += bq[i] * bq[i];
                        g += bp[i] * bq[i];
                    }
                    if aa > 0.0 && bb > 0.0 {
                        off += g * g / (aa * bb);
                    }
                }
            }
            if off < SVD_OFF_DIAGONAL_TOL {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    let bp = b.col(p);
                    let bq = b.col(q);
                    let (mut aa, mut bb, mut g) = (0.0, 0.0, 0.0);
                    for i in 0..m {
                        aa += bp[i] * bp[i];
                        bb += bq[i] * bq[i];
                        g += bp[i] * bq[i];
                    }
                    if g == 0.0 {
                        continue;
                    }
                    let zeta = (bb - aa) / (2.0 * g);
                    let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = c * t;
                    for i in 0..m {
                        let x = b.get(i, p);
                        let y = b.get(i, q);
                        b.set(i, p, c * x - s * y);
                        b.set(i, q, s * x + c * y);
                    }
                    for i in 0..n {
                        let x = v.get(i, p);
                        let y = v.get(i, q);
                        v.set(i, p, c * x - s * y);
                        v.set(i, q, s * x + c * y);
                    }
                }
            }
        }
        let s: Vec<f64> = (0..n).map(|j| b.col(j).iter().map(|x| x * x).sum::<f64>().sqrt()).collect();
        let mut u = Matrix::zeros(m, n);
        for (j, &sj) in s.iter().enumerate() {
            if sj > 0.0 {
                for (i, &val) in b.col(j).iter().enumerate() {
                    u.set(i, j, val / sj);
                }
            }
        }
        // Sort by descending singular value; permute U columns and V rows.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
        let mut us = Matrix::zeros(m, n);
        let mut vt = Matrix::zeros(n, n);
        let ss: Vec<f64> = order.iter().map(|&j| s[j]).collect();
        for (k, &j) in order.iter().enumerate() {
            for (i, &val) in u.col(j).iter().enumerate() {
                us.set(i, k, val);
            }
            for (i, &val) in v.col(j).iter().enumerate() {
                vt.set(k, i, val);
            }
        }
        Ok(Svd { u: us, s: ss, vt })
    }
}

impl Matrix {
    /// Eigenvalues and eigenvectors (as columns) of a symmetric matrix,
    /// via cyclic Jacobi rotations. Errors if `A` is not symmetric.
    ///
    /// Sweeps until the largest off-diagonal magnitude drops below
    /// [`JACOBI_OFF_DIAGONAL_TOL`], capped at [`MAX_JACOBI_SWEEPS`] sweeps
    /// (see that constant's notes on convergence). Eigenpairs are returned
    /// sorted by descending eigenvalue.
    pub fn eigen_symmetric(&self) -> MathResult<(Vec<f64>, Matrix)> {
        if !self.is_symmetric(1e-12) {
            return Err(MathError::InvalidArgument("eigen_symmetric requires a symmetric matrix"));
        }
        let n = self.rows;
        let mut a = self.clone();
        let mut v = Matrix::identity(n);
        for _ in 0..MAX_JACOBI_SWEEPS {
            let (mut p, mut q, mut mx) = (0usize, 1usize, 0.0f64);
            for i in 0..n {
                for j in (i + 1)..n {
                    let av = a.get(i, j).abs();
                    if av > mx {
                        mx = av;
                        p = i;
                        q = j;
                    }
                }
            }
            if mx < JACOBI_OFF_DIAGONAL_TOL {
                break;
            }
            let (app, aqq, apq) = (a.get(p, p), a.get(q, q), a.get(p, q));
            let theta = (aqq - app) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = c * t;
            for k in 0..n {
                if k != p && k != q {
                    let akp = a.get(k, p);
                    let akq = a.get(k, q);
                    a.set(k, p, c * akp - s * akq);
                    a.set(p, k, c * akp - s * akq);
                    a.set(k, q, s * akp + c * akq);
                    a.set(q, k, s * akp + c * akq);
                }
            }
            a.set(p, p, c * c * app - 2.0 * s * c * apq + s * s * aqq);
            a.set(q, q, s * s * app + 2.0 * s * c * apq + c * c * aqq);
            a.set(p, q, 0.0);
            a.set(q, p, 0.0);
            for i in 0..n {
                let vip = v.get(i, p);
                let viq = v.get(i, q);
                v.set(i, p, c * vip - s * viq);
                v.set(i, q, s * vip + c * viq);
            }
        }
        let mut vals: Vec<f64> = (0..n).map(|i| a.get(i, i)).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| vals[j].total_cmp(&vals[i]));
        let mut vecs = Matrix::zeros(n, n);
        for (k, &j) in order.iter().enumerate() {
            for (i, &val) in v.col(j).iter().enumerate() {
                vecs.set(i, k, val);
            }
        }
        vals.sort_by(|x, y| y.total_cmp(x));
        Ok((vals, vecs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a() -> Matrix {
        Matrix::from_rows(&[&[4.0, 3.0], &[6.0, 3.0]]).unwrap()
    }
    fn spd() -> Matrix {
        Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap()
    }

    fn rel(a: f64, b: f64) -> f64 {
        let diff = (a - b).abs();
        if b == 0.0 {
            diff
        } else {
            diff / b.abs()
        }
    }

    #[test]
    fn lu_round_trip() {
        let lu = a().lu().unwrap();
        let pa = lu.l.mul(&lu.u).unwrap();
        // row i of P A is row pivots[i] of A
        for i in 0..2 {
            for j in 0..2 {
                assert!(rel(pa.get(i, j), a().get(lu.pivots[i], j)) < 1e-14);
            }
        }
    }

    #[test]
    fn cholesky_round_trip() {
        let l = spd().cholesky().unwrap();
        let back = l.mul(&l.transpose()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(rel(back.get(i, j), spd().get(i, j)) < 1e-14);
            }
        }
        let not_spd = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]).unwrap();
        assert!(not_spd.cholesky().is_err());
    }

    #[test]
    fn qr_round_trip() {
        for m in [&a(), &spd(), &Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]).unwrap()] {
            let qr = m.qr().unwrap();
            let back = qr.q.mul(&qr.r).unwrap();
            for i in 0..m.rows {
                for j in 0..m.cols {
                    assert!(rel(back.get(i, j), m.get(i, j)) < 1e-10, "round trip ({i},{j})");
                }
            }
            // Q orthonormal: Qᵀ Q = I
            let qtq = qr.q.transpose().mul(&qr.q).unwrap();
            for i in 0..m.rows {
                for j in 0..m.rows {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(rel(qtq.get(i, j), want) < 1e-10, "ortho ({i},{j})");
                }
            }
        }
        // wide matrix (2x3)
        let wide = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();
        let qr = wide.qr().unwrap();
        let back = qr.q.mul(&qr.r).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                assert!(rel(back.get(i, j), wide.get(i, j)) < 1e-10);
            }
        }
    }

    #[test]
    fn svd_round_trip() {
        for m in [&a(), &spd(), &Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]).unwrap()] {
            let s = m.svd().unwrap();
            let back = s.u.mul(&Matrix::diagonal(&s.s)).unwrap().mul(&s.vt).unwrap();
            for i in 0..m.rows {
                for j in 0..m.cols {
                    assert!(rel(back.get(i, j), m.get(i, j)) < 1e-10, "svd round trip ({i},{j})");
                }
            }
            let utu = s.u.transpose().mul(&s.u).unwrap();
            for i in 0..s.u.cols {
                for j in 0..s.u.cols {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(rel(utu.get(i, j), want) < 1e-10, "U ortho ({i},{j})");
                }
            }
            let vt = s.vt.transpose().mul(&s.vt).unwrap();
            for i in 0..s.vt.rows {
                for j in 0..s.vt.rows {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(rel(vt.get(i, j), want) < 1e-10, "V ortho ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn svd_ill_conditioned() {
        let m = Matrix::diagonal(&[1e8, 1.0, 1e-8]);
        let s = m.svd().unwrap();
        assert!(rel(s.s[0], 1e8) < 1e-10);
        assert!(rel(s.s[1], 1.0) < 1e-10);
        assert!(rel(s.s[2], 1e-8) < 1e-10);
        let back = s.u.mul(&Matrix::diagonal(&s.s)).unwrap().mul(&s.vt).unwrap();
        assert!(rel(back.get(0, 0), 1e8) < 1e-10);
        assert!(rel(back.get(2, 2), 1e-8) < 1e-10);
    }

    #[test]
    fn eigen_symmetric() {
        let m = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 2.0]]).unwrap(); // eigen 3, 1
        let (vals, vecs) = m.eigen_symmetric().unwrap();
        assert!(rel(vals[0], 3.0) < 1e-12 && rel(vals[1], 1.0) < 1e-12);
        for (k, &lam) in vals.iter().enumerate() {
            let v = mathverse_vector::Vector::new(vecs.col(k));
            let av = m.mul_vec(&v).unwrap();
            let lamv = v.scale(lam);
            assert!(rel(av.get(0), lamv.get(0)) < 1e-10);
            assert!(rel(av.get(1), lamv.get(1)) < 1e-10);
        }
        let nonsym = Matrix::from_rows(&[&[1.0, 2.0], &[0.0, 1.0]]).unwrap();
        assert!(nonsym.eigen_symmetric().is_err());
    }

    #[test]
    fn lu_detects_exact_dependency() {
        // Rank-deficient: second row is a multiple of the first.
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]).unwrap();
        assert!(m.lu().is_err());
        assert!(Matrix::zeros(3, 3).lu().is_err());
    }

    #[test]
    fn lu_tolerates_ill_conditioned_but_invertible() {
        // Condition number ~1e13: numerically singular at single precision,
        // but perfectly factorable in f64 — must NOT be rejected.
        let m = Matrix::diagonal(&[1.0, 1e-13]);
        assert!(m.lu().is_ok());
        // Scale-invariance: the same shape scaled by 1e-150 factors too.
        let tiny = Matrix::diagonal(&[1e-150, 1e-163]);
        assert!(tiny.lu().is_ok());
    }
}
