//! Principal Component Analysis (PCA) for dimensionality reduction.
//!
//! Re-exports the function-based API from `mathverse_statistics::matrix` for
//! convenience.  The struct-based [`PCA`] and [`KernelPCA`] types below provide
//! a fit / transform workflow with `n_components` limiting.

pub use mathverse_statistics::matrix::{covariance_matrix, pca, pca_transform, PCA as StatisticsPCA};

/// PCA model with fitted components.
pub struct PCA {
    pub n_components: usize,
    pub components: Vec<Vec<f64>>, // [n_components, n_features]
    pub mean: Vec<f64>,
    pub explained_variance: Vec<f64>,
    pub explained_variance_ratio: Vec<f64>,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        Self { n_components, components: Vec::new(), mean: Vec::new(),
               explained_variance: Vec::new(), explained_variance_ratio: Vec::new() }
    }

    /// Fit PCA using power iteration for eigendecomposition of covariance matrix.
    pub fn fit(&mut self, x: &[Vec<f64>]) {
        let n = x.len();
        let p = x[0].len();
        let n_comp = self.n_components.min(p);

        // Compute mean
        self.mean = vec![0.0; p];
        for xi in x {
            for (j, &v) in xi.iter().enumerate() {
                self.mean[j] += v;
            }
        }
        for m in &mut self.mean { *m /= n as f64; }

        // Center data
        let centered: Vec<Vec<f64>> = x.iter()
            .map(|xi| xi.iter().zip(&self.mean).map(|(v, m)| v - m).collect())
            .collect();

        // Compute covariance matrix (delegates to mathverse-statistics)
        let centered_refs: Vec<&[f64]> = centered.iter().map(|r| r.as_slice()).collect();
        let cov = mathverse_statistics::covariance_matrix(&centered_refs);

        // Power iteration to find top eigenvectors
        self.components = Vec::new();
        self.explained_variance = Vec::new();
        let mut current_cov = cov;

        for _ in 0..n_comp {
            let mut v: Vec<f64> = (0..p).map(|i| (i as f64 * 0.1 + 0.5).sin()).collect();
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for v_i in &mut v { *v_i /= norm; }

            let mut eigenvalue = 0.0;
            for _ in 0..200 {
                let mut new_v = vec![0.0; p];
                for i in 0..p {
                    for j in 0..p {
                        new_v[i] += current_cov[i][j] * v[j];
                    }
                }
                eigenvalue = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if eigenvalue > 1e-15 {
                    for nv in &mut new_v { *nv /= eigenvalue; }
                }
                v = new_v;
            }

            self.components.push(v.clone());
            self.explained_variance.push(eigenvalue);

            // Deflate: subtract rank-1 component
            for i in 0..p {
                for j in 0..p {
                    current_cov[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        let total_var: f64 = self.explained_variance.iter().sum();
        self.explained_variance_ratio = self.explained_variance.iter()
            .map(|v| if total_var > 0.0 { v / total_var } else { 0.0 })
            .collect();
    }

    /// Transform data to lower dimension.
    pub fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter().map(|xi| {
            let centered: Vec<f64> = xi.iter().zip(&self.mean).map(|(v, m)| v - m).collect();
            self.components.iter().map(|comp| {
                centered.iter().zip(comp).map(|(c, w)| c * w).sum()
            }).collect()
        }).collect()
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.fit(x);
        self.transform(x)
    }
}

/// Kernel PCA for non-linear dimensionality reduction.
pub struct KernelPCA {
    pub n_components: usize,
    pub gamma: f64,
    pub support: Vec<Vec<f64>>,
    pub alphas: Vec<Vec<f64>>,
    pub eigenvalues: Vec<f64>,
}

impl KernelPCA {
    pub fn new(n_components: usize, gamma: f64) -> Self {
        Self { n_components, gamma, support: Vec::new(), alphas: Vec::new(), eigenvalues: Vec::new() }
    }

    fn kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        (-self.gamma * d).exp()
    }

    pub fn fit_transform(&mut self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x.len();
        self.support = x.to_vec();

        // Compute kernel matrix
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let val = self.kernel(&x[i], &x[j]);
                k[i][j] = val;
                k[j][i] = val;
            }
        }

        // Center kernel matrix
        let row_mean: Vec<f64> = k.iter().map(|row| row.iter().sum::<f64>() / n as f64).collect();
        let grand_mean: f64 = row_mean.iter().sum::<f64>() / n as f64;
        for i in 0..n {
            for j in 0..n {
                k[i][j] -= row_mean[i] + row_mean[j] - grand_mean;
            }
        }

        // Power iteration for top eigenvectors
        self.alphas = Vec::new();
        self.eigenvalues = Vec::new();
        let mut current_k = k.clone();

        for _ in 0..self.n_components.min(n) {
            let mut v: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for v_i in &mut v { *v_i /= norm; }

            let mut eigenvalue = 0.0;
            for _ in 0..200 {
                let mut new_v = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        new_v[i] += current_k[i][j] * v[j];
                    }
                }
                eigenvalue = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if eigenvalue > 1e-15 { for nv in &mut new_v { *nv /= eigenvalue; } }
                v = new_v;
            }

            self.alphas.push(v.clone());
            self.eigenvalues.push(eigenvalue);

            for i in 0..n {
                for j in 0..n {
                    current_k[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        // Transform
        (0..n).map(|i| {
            self.alphas.iter().map(|alpha| {
                alpha.iter().enumerate().map(|(j, &a)| a * k[i][j]).sum()
            }).collect()
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_test() {
        let x: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![i as f64 + 0.5, 2.0 * i as f64 + 1.0])
            .collect();
        let mut pca = PCA::new(1);
        let reduced = pca.fit_transform(&x);
        assert_eq!(reduced.len(), 50);
        assert_eq!(reduced[0].len(), 1);
        assert!(pca.explained_variance_ratio[0] > 0.9);
    }

    #[test]
    fn pca_2d_test() {
        let x: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0],
            vec![4.0, 5.0], vec![5.0, 6.0],
        ];
        let mut pca = PCA::new(2);
        let reduced = pca.fit_transform(&x);
        assert_eq!(reduced[0].len(), 2);
    }

    #[test]
    fn kernel_pca_test() {
        let x: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0],
            vec![2.0, 0.0], vec![0.0, 2.0],
        ];
        let mut kpca = KernelPCA::new(1, 0.5);
        let reduced = kpca.fit_transform(&x);
        assert_eq!(reduced.len(), 6);
        assert_eq!(reduced[0].len(), 1);
    }
}
