//! Multivariate statistics: covariance matrix, correlation matrix, PCA, Mahalanobis distance.

use mathverse_core::error::{MathError, MathResult};

/// Covariance matrix from data (rows = observations, cols = variables).
/// Returns a `p × p` matrix (row-major).
///
/// # Errors
///
/// Returns [`MathError::InvalidArgument`] if data is empty, has jagged rows, or has fewer than 2 observations.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::covariance_matrix;
///
/// let data: Vec<&[f64]> = vec![&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]];
/// let cov = covariance_matrix(&data).unwrap();
/// assert_eq!(cov.len(), 2);
/// ```
pub fn covariance_matrix(data: &[&[f64]]) -> MathResult<Vec<Vec<f64>>> {
    let n = data.len();
    if n < 2 {
        return Err(MathError::InvalidArgument("need at least 2 observations"));
    }
    if n == 0 {
        return Err(MathError::InvalidArgument("data must not be empty"));
    }
    let p = data[0].len();
    if p == 0 {
        return Err(MathError::InvalidArgument("variables must not be empty"));
    }
    // Check for jagged data
    if !data.iter().all(|row| row.len() == p) {
        return Err(MathError::InvalidArgument("data must be rectangular (jagged input)"));
    }
    let mut means = vec![0.0; p];
    for row in data.iter() {
        for j in 0..p {
            means[j] += row[j];
        }
    }
    for m in means.iter_mut() {
        *m /= n as f64;
    }
    let mut cov = vec![vec![0.0; p]; p];
    for row in data.iter() {
        for i in 0..p {
            for j in 0..p {
                cov[i][j] += (row[i] - means[i]) * (row[j] - means[j]);
            }
        }
    }
    for row in cov.iter_mut().take(p) {
        for val in row.iter_mut().take(p) {
            *val /= (n - 1) as f64;
        }
    }
    Ok(cov)
}

/// Correlation matrix from data.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::correlation_matrix;
///
/// let data: Vec<&[f64]> = vec![&[1.0, 2.0], &[2.0, 4.0], &[3.0, 6.0]];
/// let corr = correlation_matrix(&data);
/// assert!((corr[0][1] - 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn correlation_matrix(data: &[&[f64]]) -> MathResult<Vec<Vec<f64>>> {
    let cov = covariance_matrix(data)?;
    let p = cov.len();
    let mut corr = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            let denom = (cov[i][i] * cov[j][j]).sqrt();
            if denom > 1e-30 {
                corr[i][j] = cov[i][j] / denom;
            } else {
                corr[i][j] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    Ok(corr)
}

/// PCA result.
#[derive(Debug, Clone)]
pub struct PCA {
    /// Principal components (eigenvectors), each is a `p`-length vector.
    pub components: Vec<Vec<f64>>,
    /// Explained variance per component.
    pub explained_variance: Vec<f64>,
    /// Explained variance ratio per component.
    pub explained_variance_ratio: Vec<f64>,
}

/// Perform PCA on data (rows = observations, cols = variables).
/// Returns the top `min(n, p)` components.
///
/// # Errors
///
/// Returns [`MathError::InvalidArgument`] if data is empty or jagged.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::pca;
///
/// let data: Vec<&[f64]> = vec![
///     &[1.0, 2.0], &[2.0, 4.0], &[3.0, 6.0],
///     &[4.0, 8.0], &[5.0, 10.0],
/// ];
/// let result = pca(&data).unwrap();
/// assert!(result.explained_variance[0] > 0.0);
/// ```
pub fn pca(data: &[&[f64]]) -> MathResult<PCA> {
    let n = data.len();
    if n == 0 {
        return Err(MathError::InvalidArgument("data must not be empty"));
    }
    let p = data[0].len();
    if p == 0 {
        return Err(MathError::InvalidArgument("variables must not be empty"));
    }
    // Check for jagged data
    if !data.iter().all(|row| row.len() == p) {
        return Err(MathError::InvalidArgument("data must be rectangular (jagged input)"));
    }
    let mut means = vec![0.0; p];
    for row in data.iter() {
        for j in 0..p {
            means[j] += row[j];
        }
    }
    for m in means.iter_mut() {
        *m /= n as f64;
    }
    let centered: Vec<Vec<f64>> = data
        .iter()
        .map(|row| row.iter().zip(&means).map(|(x, m)| x - m).collect())
        .collect();

    let cov = covariance_matrix(&centered.iter().map(|r| r.as_slice()).collect::<Vec<_>>())?;

    let k = p.min(n);
    let mut components = Vec::new();
    let mut explained_variance = Vec::new();

    let mut matrix = cov;
    for _ in 0..k {
        let (eigenvalue, eigenvector) = power_iteration(&matrix);
        explained_variance.push(eigenvalue);
        for i in 0..p {
            for j in 0..p {
                matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
        components.push(eigenvector);
    }
    let total: f64 = explained_variance.iter().sum();
    let explained_variance_ratio = explained_variance.iter().map(|v| *v / total).collect();

    Ok(PCA {
        components,
        explained_variance,
        explained_variance_ratio,
    })
}

/// Transform data using PCA components.
#[must_use]
pub fn pca_transform(data: &[&[f64]], components: &[Vec<f64>], means: &[f64]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|row| {
            components
                .iter()
                .map(|comp| {
                    row.iter().zip(comp).zip(means)
                        .map(|((x, c), m)| (x - m) * c)
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Mahalanobis distance from point to distribution defined by mean and inverse covariance.
#[must_use]
pub fn mahalanobis(point: &[f64], mean: &[f64], cov_inv: &[Vec<f64>]) -> f64 {
    let n = point.len();
    let mut d = vec![0.0; n];
    for i in 0..n {
        d[i] = point[i] - mean[i];
    }
    let mut dist2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            dist2 += d[i] * cov_inv[i][j] * d[j];
        }
    }
    dist2.sqrt()
}

/// Invert a symmetric positive-definite matrix (Cholesky decomposition).
///
/// # Errors
///
/// Returns [`MathError::Singular`] if the matrix is not positive-definite.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::cholesky_inverse;
///
/// let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
/// let inv = cholesky_inverse(&a).unwrap();
/// assert!((a[0][0] * inv[0][0] + a[0][1] * inv[1][0] - 1.0).abs() < 1e-10);
/// ```
pub fn cholesky_inverse(matrix: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for (k, l_ik) in l[i].iter().take(j).enumerate() {
                sum += *l_ik * l[j][k];
            }
            let diag = matrix[i][i] - sum;
            if diag <= 1e-12 {
                return Err(MathError::Singular);
            }
            l[i][j] = if i == j { diag.sqrt() } else { (matrix[i][j] - sum) / l[j][j] };
        }
    }
    let mut linv = vec![vec![0.0; n]; n];
    for j in 0..n {
        linv[j][j] = 1.0 / l[j][j];
        for i in (j + 1)..n {
            let sum: f64 = (j..i).map(|k| l[i][k] * linv[k][j]).sum();
            linv[i][j] = -sum / l[i][i];
        }
    }
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for linv_k in linv.iter().take(n) {
                sum += linv_k[i] * linv_k[j];
            }
            result[i][j] = sum;
        }
    }
    Ok(result)
}

/// Inverse covariance (precision matrix).
///
/// # Errors
///
/// Returns [`MathError::Singular`] if the covariance matrix is singular.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::precision_matrix;
///
/// let data: Vec<&[f64]> = vec![&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]];
/// let prec = precision_matrix(&data).unwrap();
/// assert_eq!(prec.len(), 2);
/// ```
pub fn precision_matrix(data: &[&[f64]]) -> MathResult<Vec<Vec<f64>>> {
    let cov = covariance_matrix(data)?;
    cholesky_inverse(&cov)
}

// ---------------------------------------------------------------------------
// Power iteration
// ---------------------------------------------------------------------------

fn power_iteration(matrix: &[Vec<f64>]) -> (f64, Vec<f64>) {
    let n = matrix.len();
    let mut b: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    for _ in 0..100 {
        let mut new_b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_b[i] += matrix[i][j] * b[j];
            }
        }
        let norm: f64 = new_b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 {
            break;
        }
        for x in new_b.iter_mut() {
            *x /= norm;
        }
        b = new_b;
    }
    let mut av = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            av[i] += matrix[i][j] * b[j];
        }
    }
    let eigenvalue = b.iter().zip(&av).map(|(bi, avi)| bi * avi).sum();
    (eigenvalue, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_matrix_test() {
        let data: Vec<&[f64]> = vec![&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]];
        let cov = covariance_matrix(&data).unwrap();
        assert_eq!(cov.len(), 2);
        assert_eq!(cov[0].len(), 2);
        assert!((cov[0][0] - cov[1][1]).abs() < 1e-10);
        assert!((cov[0][1] - cov[1][0]).abs() < 1e-10);
    }

    #[test]
    fn correlation_matrix_test() {
        let data: Vec<&[f64]> = vec![&[1.0, 2.0], &[2.0, 4.0], &[3.0, 6.0]];
        let corr = correlation_matrix(&data).unwrap();
        assert!((corr[0][1] - 1.0).abs() < 1e-10);
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pca_test() {
        let data: Vec<&[f64]> = vec![
            &[1.0, 2.0], &[2.0, 4.0], &[3.0, 6.0],
            &[4.0, 8.0], &[5.0, 10.0],
        ];
        let result = pca(&data).unwrap();
        assert!(!result.components.is_empty());
        assert!(result.explained_variance[0] > 0.0);
        assert!(result.explained_variance_ratio[0] > 0.9);
    }

    #[test]
    fn mahalanobis_test() {
        let cov_inv = vec![vec![1.0]];
        let mean = vec![0.0];
        assert!((mahalanobis(&[3.0], &mean, &cov_inv) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn cholesky_inverse_test() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let inv = cholesky_inverse(&a).unwrap();
        let prod = vec![
            vec![a[0][0] * inv[0][0] + a[0][1] * inv[1][0], a[0][0] * inv[0][1] + a[0][1] * inv[1][1]],
            vec![a[1][0] * inv[0][0] + a[1][1] * inv[1][0], a[1][0] * inv[0][1] + a[1][1] * inv[1][1]],
        ];
        assert!((prod[0][0] - 1.0).abs() < 1e-10);
        assert!(prod[0][1].abs() < 1e-10);
        assert!(prod[1][0].abs() < 1e-10);
        assert!((prod[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cholesky_singular_test() {
        let a = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert_eq!(cholesky_inverse(&a), Err(MathError::Singular));
    }

    #[test]
    fn precision_matrix_test() {
        let data: Vec<&[f64]> = vec![&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]];
        let prec = precision_matrix(&data).unwrap();
        assert_eq!(prec.len(), 2);
    }
}
