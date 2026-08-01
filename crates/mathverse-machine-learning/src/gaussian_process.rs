/// Covariance kernel for Gaussian process regression.
#[derive(Debug, Clone)]
pub enum GpKernel {
    /// Radial basis function (squared exponential) kernel.
    RBF {
        /// Length scale parameter.
        length: f64,
    },
    /// Matérn 5/2 kernel.
    Matern52 {
        /// Length scale parameter.
        length: f64,
    },
    /// Rational quadratic kernel.
    RationalQuadratic {
        /// Length scale parameter.
        length: f64,
        /// Mixture weight parameter.
        alpha: f64,
    },
}

impl GpKernel {
    /// Compute kernel similarity between two vectors.
    #[must_use]
    pub fn compute(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let r = dist.sqrt();
        match self {
            GpKernel::RBF { length } => (-dist / (2.0 * length * length)).exp(),
            GpKernel::Matern52 { length } => {
                let s = (5.0_f64).sqrt() * r / length;
                (1.0 + s + s * s / 3.0) * (-s).exp()
            }
            GpKernel::RationalQuadratic { length, alpha } => {
                let ratio = dist / (2.0 * alpha * length * length);
                (1.0 + ratio).powf(-alpha)
            }
        }
    }

    fn compute_matrix(x1: &[Vec<f64>], x2: &[Vec<f64>], kernel: &GpKernel) -> Vec<Vec<f64>> {
        x1.iter()
            .map(|xi| x2.iter().map(|xj| kernel.compute(xi, xj)).collect())
            .collect()
    }
}

/// Gaussian process regression model.
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    /// Kernel function used for covariance.
    pub kernel: GpKernel,
    /// Noise variance added to the diagonal of the kernel matrix.
    pub noise: f64,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k_inv: Vec<Vec<f64>>,
}

impl GaussianProcess {
    /// Fit the GP to training data, returning an error if the kernel matrix is singular.
    #[must_use]
    pub fn fit(
        x: &[Vec<f64>],
        y: &[f64],
        kernel: GpKernel,
        noise: f64,
    ) -> Result<Self, &'static str> {
        let n = x.len();
        let mut k = GpKernel::compute_matrix(x, x, &kernel);

        for i in 0..n {
            k[i][i] += noise;
        }

        let k_inv = invert_matrix(&k).ok_or("singular matrix")?;

        Ok(Self {
            kernel,
            noise,
            x_train: x.to_vec(),
            y_train: y.to_vec(),
            k_inv,
        })
    }

    /// Predict means and variances at the given test points.
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
        let n_train = self.x_train.len();
        let n_test = x.len();

        let k_star = GpKernel::compute_matrix(x, &self.x_train, &self.kernel);
        let k_star_star = GpKernel::compute_matrix(x, x, &self.kernel);

        // alpha = K_inv * y_train
        let mut alpha = vec![0.0; n_train];
        for j in 0..n_train {
            let mut sum = 0.0;
            for k in 0..n_train {
                sum += self.k_inv[j][k] * self.y_train[k];
            }
            alpha[j] = sum;
        }

        let mut means = Vec::with_capacity(n_test);
        let mut variances = Vec::with_capacity(n_test);

        for i in 0..n_test {
            // mean = k_star[i] . alpha
            let mut mu = 0.0;
            for j in 0..n_train {
                mu += k_star[i][j] * alpha[j];
            }
            means.push(mu);

            // variance = k_star_star[i][i] - k_star[i] . K_inv . k_star[i]^T
            let mut var = k_star_star[i][i];
            // k_star[i] . K_inv is a vector of length n_train
            let mut ki_kinv = vec![0.0; n_train];
            for kk in 0..n_train {
                let mut s = 0.0;
                for j in 0..n_train {
                    s += k_star[i][j] * self.k_inv[j][kk];
                }
                ki_kinv[kk] = s;
            }
            // then dot with k_star[i]
            for kk in 0..n_train {
                var -= ki_kinv[kk] * k_star[i][kk];
            }
            variances.push(var.max(0.0));
        }

        (means, variances)
    }
}

fn invert_matrix(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    if n == 0 {
        return None;
    }

    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = m[i].clone();
        row.resize(2 * n, 0.0);
        row[n + i] = 1.0;
        aug.push(row);
    }

    for col in 0..n {
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        result.push(aug[i][n..].to_vec());
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_kernel() {
        let k = GpKernel::RBF { length: 1.0 };
        assert!((k.compute(&[0.0], &[0.0]) - 1.0).abs() < 1e-10);
        assert!((k.compute(&[0.0], &[1.0]) - (-0.5_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn matern52_kernel() {
        let k = GpKernel::Matern52 { length: 1.0 };
        assert!((k.compute(&[0.0], &[0.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rq_kernel() {
        let k = GpKernel::RationalQuadratic {
            length: 1.0,
            alpha: 2.0,
        };
        assert!((k.compute(&[0.0], &[0.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn fit_predict_simple() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let gp = GaussianProcess::fit(&x, &y, GpKernel::RBF { length: 1.0 }, 0.01).unwrap();
        let (means, variances) = gp.predict(&[vec![1.5]]);
        assert!((means[0] - 1.5).abs() < 0.5, "mean={}", means[0]);
        assert!(variances[0] >= 0.0);
        assert!(variances[0] < 1.0, "var={}", variances[0]);
    }

    #[test]
    fn matrix_inversion() {
        let m = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let inv = invert_matrix(&m).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += m[i][k] * inv[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((sum - expected).abs() < 1e-10, "({i},{j}): got {sum}");
            }
        }
    }

    #[test]
    fn singular_matrix() {
        let m = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(invert_matrix(&m).is_none());
    }
}
