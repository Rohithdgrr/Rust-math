//! Complex Principal Component Analysis (PCA) via covariance eigen-decomposition.
//!
//! PCA on complex-valued data finds the principal directions of maximum
//! variance, essential for dimensionality reduction, denoising, and
//! feature extraction in complex-valued datasets (e.g., radar, MRI,
//! communications).
//!
//! # Module overview
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`pca`] | Full PCA decomposition returning components and variance |
//! | [`pca_project`] | Project data onto principal components |
//! | [`pca_reconstruct`] | Reconstruct data from projected coordinates |
//! | [`pca_explained_variance`] | Compute explained variance ratio per component |

use crate::matrix::ComplexMatrix;
use crate::Complex;
use mathverse_core::error::{MathError, MathResult};

/// Result of a PCA decomposition.
pub struct PcaResult {
    /// Principal components (eigenvectors), stored as columns of a matrix
    /// of shape `(n_features, n_components)`.
    pub components: ComplexMatrix,
    /// Variance explained by each component (real-valued, descending order).
    pub explained_variance: Vec<f64>,
    /// Mean of each feature (subtracted before decomposition).
    pub mean: Vec<Complex>,
    /// Projected data coordinates (`n_samples` × `n_components`).
    pub projected: ComplexMatrix,
}

/// Jacobi eigenvalue algorithm for real symmetric matrices.
/// Returns (eigenvalues, `eigenvectors_as_columns`).
fn jacobi_eigen_symmetric(a: &ComplexMatrix) -> (Vec<f64>, ComplexMatrix) {
    let n = a.rows;
    let mut s = a.clone();
    let mut v = ComplexMatrix::identity(n);

    for _iter in 0..200 {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = s.get(i, j).norm();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        // Compute Jacobi rotation
        let app = s.get(p, p).re;
        let aqq = s.get(q, q).re;
        let apq = s.get(p, q).re;

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * apq) / (app - aqq)).atan()
        };
        let c = theta.cos();
        let si = theta.sin();

        // Apply similarity transformation: S' = J^T S J
        let mut new_s = s.clone();
        for i in 0..n {
            for j in 0..n {
                if i != p && i != q && j != p && j != q {
                    // unchanged
                } else if i == p && j == p {
                    let val = c * c * app + si * si * aqq + 2.0 * si * c * apq;
                    new_s.set(p, p, Complex::real(val));
                } else if i == q && j == q {
                    let val = si * si * app + c * c * aqq - 2.0 * si * c * apq;
                    new_s.set(q, q, Complex::real(val));
                } else if i == p && j == q {
                    new_s.set(p, q, Complex::zero());
                    new_s.set(q, p, Complex::zero());
                } else if i == q && j == p {
                    // already handled above
                } else if i == p || i == q {
                    let (r, s_idx) = if i == p { (p, q) } else { (q, p) };
                    let sprime = c * s.get(r, j).re + si * s.get(s_idx, j).re;
                    let _sqprime = -si * s.get(r, j).re + c * s.get(s_idx, j).re;
                    new_s.set(i, j, Complex::real(sprime));
                    new_s.set(j, i, Complex::real(sprime));
                } else if j == p || j == q {
                    let (r, s_idx) = if j == p { (p, q) } else { (q, p) };
                    let sprime = c * s.get(i, r).re + si * s.get(i, s_idx).re;
                    let _sqprime = -si * s.get(i, r).re + c * s.get(i, s_idx).re;
                    new_s.set(i, j, Complex::real(sprime));
                    new_s.set(j, i, Complex::real(sprime));
                }
            }
        }
        s = new_s;

        // Accumulate eigenvectors: V' = V J
        for i in 0..n {
            let vip = v.get(i, p);
            let viq = v.get(i, q);
            v.set(i, p, Complex::real(c) * vip + Complex::real(si) * viq);
            v.set(i, q, Complex::real(-si) * vip + Complex::real(c) * viq);
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| s.get(i, i).re).collect();
    (eigenvalues, v)
}

/// Compute PCA on complex-valued data.
///
/// # Arguments
/// * `data` — matrix of shape `(n_samples, n_features)`, each row is an observation
/// * `n_components` — number of principal components to retain
///
/// # Returns
/// A [`PcaResult`] containing components, variance, mean, and projected data.
///
/// # Errors
/// Returns an error if `n_components` exceeds `n_features` or `n_samples`.
pub fn pca(data: &ComplexMatrix, n_components: usize) -> MathResult<PcaResult> {
    let (n_samples, n_features) = (data.rows, data.cols);
    if n_components > n_features.min(n_samples) {
        return Err(MathError::InvalidArgument(
            "n_components exceeds rank",
        ));
    }

    // Step 1: Compute mean
    let mut mean = vec![Complex::zero(); n_features];
    for j in 0..n_features {
        for i in 0..n_samples {
            mean[j] = mean[j] + data[(i, j)];
        }
        mean[j] = mean[j] / Complex::real(n_samples as f64);
    }

    // Step 2: Center data
    let mut centered = data.clone();
    for i in 0..n_samples {
        for j in 0..n_features {
            centered[(i, j)] = centered[(i, j)] - mean[j];
        }
    }

    // Step 3: Compute covariance matrix: C = Xᴴ·X / (n-1)
    let xh = centered.hermitian();
    let cov = xh.mul(&centered)?;
    let scale = Complex::real(1.0 / (n_samples - 1).max(1) as f64);
    let mut cov_scaled = cov.clone();
    for v in &mut cov_scaled.data {
        *v = *v * scale;
    }

    // Step 4: Jacobi eigenvalue decomposition of real symmetric covariance matrix
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&cov_scaled);

    // Step 5: Sort eigenvalues in descending order
    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 6: Extract top n_components
    let mut components_data = Vec::with_capacity(n_features * n_components);
    let mut explained_variance = Vec::with_capacity(n_components);
    for &idx in indices.iter().take(n_components) {
        explained_variance.push(eigenvalues[idx].max(0.0));
        for row in 0..n_features {
            components_data.push(eigenvectors.get(row, idx));
        }
    }

    let components = ComplexMatrix::from_data(components_data, n_features, n_components);

    // Step 7: Project data onto components
    let projected = centered.mul(&components)?;

    Ok(PcaResult {
        components,
        explained_variance,
        mean,
        projected,
    })
}

/// Project new data onto existing PCA components.
///
/// # Arguments
/// * `data` — matrix of shape `(n_samples, n_features)`
/// * `components` — principal components matrix of shape `(n_features, n_components)`
/// * `mean` — feature means (subtracted before projection)
///
/// # Returns
/// Projected data of shape `(n_samples, n_components)`.
pub fn pca_project(
    data: &ComplexMatrix,
    components: &ComplexMatrix,
    mean: &[Complex],
) -> MathResult<ComplexMatrix> {
    let (n_samples, n_features) = (data.rows, data.cols);
    if mean.len() != n_features {
        return Err(MathError::DimensionMismatch);
    }

    let mut centered = data.clone();
    for i in 0..n_samples {
        for j in 0..n_features {
            centered[(i, j)] = centered[(i, j)] - mean[j];
        }
    }

    centered.mul(components)
}

/// Reconstruct data from PCA projection.
///
/// # Arguments
/// * `projected` — projected coordinates of shape `(n_samples, n_components)`
/// * `components` — principal components of shape `(n_features, n_components)`
/// * `mean` — feature means to add back
///
/// # Returns
/// Reconstructed data of shape `(n_samples, n_features)`.
pub fn pca_reconstruct(
    projected: &ComplexMatrix,
    components: &ComplexMatrix,
    mean: &[Complex],
) -> MathResult<ComplexMatrix> {
    let (n_samples, n_components_proj) = (projected.rows, projected.cols);
    let n_features = components.rows;
    if components.cols != n_components_proj {
        return Err(MathError::DimensionMismatch);
    }
    if mean.len() != n_features {
        return Err(MathError::DimensionMismatch);
    }

    // X ≈ projected @ components^T + mean
    let reconstructed = projected.mul(&components.transpose())?;
    let mut result = reconstructed;
    for i in 0..n_samples {
        for j in 0..n_features {
            result[(i, j)] = result[(i, j)] + mean[j];
        }
    }

    Ok(result)
}

/// Compute the explained variance ratio for each component.
///
/// Returns `explained_variance[i] / sum(explained_variance)`.
pub fn pca_explained_variance(explained_variance: &[f64]) -> Vec<f64> {
    let total: f64 = explained_variance.iter().sum();
    if total < 1e-15 {
        vec![0.0; explained_variance.len()]
    } else {
        explained_variance.iter().map(|v| v / total).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_perfect_line() {
        // All points lie on a line: y = 2x
        let data: Vec<Complex> = vec![
            1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0,
        ]
        .into_iter()
        .map(Complex::real)
        .collect();
        let m = ComplexMatrix::from_data(data, 5, 2);

        let result = pca(&m, 1).unwrap();
        assert_eq!(result.components.cols, 1);
        // First component should explain ~100% of variance
        let ratios = pca_explained_variance(&result.explained_variance);
        assert!(ratios[0] > 0.99);
    }

    #[test]
    fn pca_zero_variance() {
        let data: Vec<Complex> = vec![Complex::real(5.0); 10];
        let m = ComplexMatrix::from_data(data, 5, 2);

        let result = pca(&m, 1).unwrap();
        assert!(result.explained_variance[0].abs() < 1e-10);
    }

    #[test]
    fn pca_too_many_components() {
        let data: Vec<Complex> = vec![Complex::real(1.0); 6];
        let m = ComplexMatrix::from_data(data, 3, 2);
        assert!(pca(&m, 3).is_err());
    }

    #[test]
    fn pca_complex_data() {
        let data: Vec<Complex> = (0..20)
            .map(|i| Complex::new(f64::from(i), (f64::from(i) * 0.1).sin()))
            .collect();
        let m = ComplexMatrix::from_data(data, 10, 2);

        let result = pca(&m, 1).unwrap();
        assert_eq!(result.components.cols, 1);
        assert!(result.explained_variance[0] > 0.0);
    }

    #[test]
    fn pca_project_and_reconstruct() {
        let data: Vec<Complex> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]
        .into_iter()
        .map(Complex::real)
        .collect();
        let m = ComplexMatrix::from_data(data, 5, 2);

        let result = pca(&m, 2).unwrap();

        let projected = pca_project(&m, &result.components, &result.mean).unwrap();
        let reconstructed = pca_reconstruct(&projected, &result.components, &result.mean).unwrap();

        // Reconstruction should be exact when using all components
        for i in 0..5 {
            for j in 0..2 {
                let diff = (m[(i, j)] - reconstructed[(i, j)]).norm();
                assert!(diff < 1e-8, "Reconstruction error at ({i}, {j}): {diff}");
            }
        }
    }

    #[test]
    fn pca_explained_variance_ratios() {
        let ratios = pca_explained_variance(&[4.0, 2.0, 1.0]);
        assert!((ratios[0] - 4.0 / 7.0).abs() < 1e-10);
        assert!((ratios[1] - 2.0 / 7.0).abs() < 1e-10);
        assert!((ratios[2] - 1.0 / 7.0).abs() < 1e-10);
        assert!((ratios.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }
}
