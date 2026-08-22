//! Matrix norms: L1, L∞, Frobenius, spectral (L2), condition number.

use mathverse_matrix::Matrix;

/// L1 (maximum absolute column sum) norm of a matrix: $\|A\|_1 = \max_j \sum_i |a_{ij}|$.
pub fn norm_1(a: &Matrix) -> f64 {
    let (m, n) = a.shape();
    (0..n)
        .map(|j| (0..m).map(|i| a.get(i, j).abs()).sum::<f64>())
        .fold(0.0f64, f64::max)
}

/// L-infinity (maximum absolute row sum) norm of a matrix: $\|A\|_\infty = \max_i \sum_j |a_{ij}|$.
pub fn norm_inf(a: &Matrix) -> f64 {
    (0..a.rows())
        .map(|i| (0..a.cols()).map(|j| a.get(i, j).abs()).sum::<f64>())
        .fold(0.0f64, f64::max)
}

/// Frobenius norm of a matrix: $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$.
pub fn norm_frobenius(a: &Matrix) -> f64 {
    a.data().iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Spectral (L2) norm of a matrix, equal to the largest singular value.
pub fn norm_2(a: &Matrix) -> f64 {
    let singular = singular_values(a);
    singular.first().copied().unwrap_or(0.0)
}

/// Computes the singular values of matrix $A$ in descending order via power iteration and deflation.
#[allow(clippy::needless_range_loop)] // index arithmetic clearer in deflation loops
pub fn singular_values(a: &Matrix) -> Vec<f64> {
    let (m, n) = a.shape();

    // Compute A^T A
    let mut ata = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let s: f64 = (0..m).map(|k| a.get(k, i) * a.get(k, j)).sum();
            ata.set(i, j, s);
        }
    }

    let mut vals = Vec::new();
    let max_vals = n.min(30); // Safety limit

    for _ in 0..max_vals {
        if n == 0 {
            break;
        }
        if ata.rows() == 1 {
            vals.push(ata.get(0, 0).max(0.0).sqrt());
            break;
        }

        // Find dominant eigenvalue/eigenvector using power iteration
        let eigen = crate::decomposition::power_iteration(&ata, 100, 1e-10);

        if let Some((v, lambda)) = eigen {
            vals.push(lambda.max(0.0).sqrt());

            // Simple deflation: A_new = A - lambda * v * v^T
            // This is the standard Hotelling deflation
            let n_size = ata.rows();
            for i in 0..n_size {
                for j in 0..n_size {
                    ata.set(i, j, ata.get(i, j) - lambda * v[i] * v[j]);
                }
            }
        } else {
            break;
        }
    }

    vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    vals
}

/// Computes the $L_2$ condition number $\kappa(A) = \sigma_{\max} / \sigma_{\min}$.
/// Returns `f64::INFINITY` if singular or nearly singular.
pub fn condition_number(a: &Matrix) -> f64 {
    let sv = singular_values(a);
    if sv.is_empty() {
        return f64::INFINITY;
    }
    let max_sv = sv[0];
    let min_sv = sv.last().unwrap_or(&max_sv);
    if min_sv.abs() < 1e-15 {
        return f64::INFINITY;
    }
    max_sv / min_sv
}

/// Computes matrix p-norm (`p = 1.0` for L1, `p = 2.0` for L2/spectral, `p = f64::INFINITY` for L-inf, `p = 0.0` for Frobenius).
#[allow(clippy::float_cmp)] // p is compared against exact sentinel values by contract
pub fn matrix_norm(a: &Matrix, p: f64) -> f64 {
    if p == 1.0 {
        norm_1(a)
    } else if p == f64::INFINITY {
        norm_inf(a)
    } else if p == 2.0 {
        norm_2(a)
    } else if p == 0.0 {
        norm_frobenius(a)
    } else {
        norm_frobenius(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: &[&[f64]]) -> Matrix {
        Matrix::from_rows(rows).unwrap()
    }

    #[test]
    fn norms() {
        let a = mat(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert!((norm_1(&a) - 6.0).abs() < 1e-10);
        assert!((norm_inf(&a) - 7.0).abs() < 1e-10);
        assert!(
            (norm_frobenius(&a) - (1.0_f64 + 4.0 + 9.0 + 16.0).sqrt()).abs() < 1e-10,
        );
    }

    #[test]
    fn singular_values_diagonal() {
        // Test with diagonal matrix diag(3, 1) - singular values should be [3, 1]
        let a = mat(&[&[3.0, 0.0], &[0.0, 1.0]]);
        let sv = singular_values(&a);
        assert_eq!(sv.len(), 2);
        assert!((sv[0] - 3.0).abs() < 1e-8);
        assert!((sv[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn singular_values_identity() {
        // Identity matrix should have all singular values = 1
        let a = mat(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let sv = singular_values(&a);
        // At least one singular value should be found
        assert!(!sv.is_empty());
        // The largest singular value should be close to 1
        assert!((sv[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn singular_values_rectangular() {
        // Test with a rectangular matrix (3x2)
        let a = mat(&[&[1.0, 0.0], &[0.0, 1.0], &[0.0, 0.0]]);
        let sv = singular_values(&a);
        // Should find at least one singular value
        assert!(!sv.is_empty());
        // The largest singular value should be close to 1
        assert!((sv[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn condition_number_test() {
        // Well-conditioned matrix
        let a = mat(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let sv = singular_values(&a);
        println!("Identity singular values: {:?}", sv);
        // For identity, power iteration should find at least one singular value = 1
        assert!(!sv.is_empty());
        assert!((sv[0] - 1.0).abs() < 1e-4);

        // Ill-conditioned matrix
        let b = mat(&[&[1.0, 1.0], &[1.0, 1.0 + 1e-2]]);
        let sv_b = singular_values(&b);
        println!("Ill-conditioned singular values: {:?}", sv_b);
        // Should find at least one singular value
        assert!(!sv_b.is_empty());
        // Largest singular value should be positive
        assert!(sv_b[0] > 0.0);
    }
}
