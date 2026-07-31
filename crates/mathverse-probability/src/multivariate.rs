//! Joint and multivariate probability: joint distributions, covariance, correlation, copulas.

use crate::rng::Rng;

/// Covariance matrix for multivariate distributions.
#[derive(Debug, Clone)]
pub struct CovarianceMatrix {
    pub data: Vec<Vec<f64>>,
    pub dimension: usize,
}

impl CovarianceMatrix {
    /// Create a new covariance matrix from data.
    pub fn new(data: Vec<Vec<f64>>) -> Result<Self, String> {
        let dim = data.len();
        if dim == 0 {
            return Err("Covariance matrix must have at least one dimension".to_string());
        }
        
        for row in &data {
            if row.len() != dim {
                return Err("Covariance matrix must be square".to_string());
            }
        }
        
        // Check symmetry
        for i in 0..dim {
            for j in 0..dim {
                if (data[i][j] - data[j][i]).abs() > 1e-10 {
                    return Err("Covariance matrix must be symmetric".to_string());
                }
            }
        }
        
        Ok(CovarianceMatrix {
            data,
            dimension: dim,
        })
    }

    /// Compute covariance matrix from samples (rows are observations, columns are variables).
    pub fn from_samples(samples: &[Vec<f64>]) -> Result<Self, String> {
        if samples.is_empty() {
            return Err("No samples provided".to_string());
        }
        
        let n_vars = samples[0].len();
        if n_vars == 0 {
            return Err("Samples must have at least one variable".to_string());
        }
        
        // Check all samples have same dimension
        for sample in samples {
            if sample.len() != n_vars {
                return Err("All samples must have the same dimension".to_string());
            }
        }
        
        let n = samples.len();
        let mut means = vec![0.0; n_vars];
        
        // Compute means
        for sample in samples {
            for (i, &val) in sample.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= n as f64;
        }
        
        // Compute covariance matrix
        let mut cov = vec![vec![0.0; n_vars]; n_vars];
        for sample in samples {
            for i in 0..n_vars {
                for j in 0..n_vars {
                    cov[i][j] += (sample[i] - means[i]) * (sample[j] - means[j]);
                }
            }
        }
        
        let divisor = if n > 1 { (n - 1) as f64 } else { 1.0 };
        for i in 0..n_vars {
            for j in 0..n_vars {
                cov[i][j] /= divisor;
            }
        }
        
        Self::new(cov)
    }

    /// Get the covariance between variables i and j.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    /// Get the variance of variable i.
    pub fn variance(&self, i: usize) -> f64 {
        self.data[i][i]
    }

    /// Check if the matrix is positive definite.
    pub fn is_positive_definite(&self) -> bool {
        // Use Cholesky decomposition attempt to check positive definiteness
        self.cholesky().is_ok()
    }

    /// Cholesky decomposition: returns L such that A = L * L^T.
    pub fn cholesky(&self) -> Result<Vec<Vec<f64>>, String> {
        let n = self.dimension;
        let mut l = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                
                if j == i {
                    for k in 0..j {
                        sum += l[j][k] * l[j][k];
                    }
                    let diag = self.data[i][i] - sum;
                    if diag <= 0.0 {
                        return Err("Matrix is not positive definite".to_string());
                    }
                    l[i][j] = diag.sqrt();
                } else {
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    l[i][j] = (self.data[i][j] - sum) / l[j][j];
                }
            }
        }
        
        Ok(l)
    }
}

/// Correlation matrix.
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    pub data: Vec<Vec<f64>>,
    pub dimension: usize,
}

impl CorrelationMatrix {
    /// Create correlation matrix from covariance matrix.
    pub fn from_covariance(cov: &CovarianceMatrix) -> Self {
        let n = cov.dimension;
        let mut corr = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                let std_i = cov.variance(i).sqrt();
                let std_j = cov.variance(j).sqrt();
                if std_i > 0.0 && std_j > 0.0 {
                    corr[i][j] = cov.get(i, j) / (std_i * std_j);
                } else {
                    corr[i][j] = if i == j { 1.0 } else { 0.0 };
                }
            }
        }
        
        CorrelationMatrix {
            data: corr,
            dimension: n,
        }
    }

    /// Get correlation between variables i and j.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    /// Pearson correlation coefficient between two vectors.
    pub fn pearson(x: &[f64], y: &[f64]) -> Result<f64, String> {
        if x.len() != y.len() || x.is_empty() {
            return Err("Vectors must have the same non-zero length".to_string());
        }
        
        let n = x.len();
        let mean_x = x.iter().sum::<f64>() / n as f64;
        let mean_y = y.iter().sum::<f64>() / n as f64;
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            return Ok(0.0);
        }
        
        Ok(numerator / denominator)
    }

    /// Spearman rank correlation coefficient.
    pub fn spearman(x: &[f64], y: &[f64]) -> Result<f64, String> {
        if x.len() != y.len() || x.is_empty() {
            return Err("Vectors must have the same non-zero length".to_string());
        }
        
        let n = x.len();
        
        // Compute ranks
        let rank_x = Self::ranks(x);
        let rank_y = Self::ranks(y);
        
        // Compute Pearson correlation on ranks
        Self::pearson(&rank_x, &rank_y)
    }

    /// Kendall's tau correlation coefficient.
    pub fn kendall_tau(x: &[f64], y: &[f64]) -> Result<f64, String> {
        if x.len() != y.len() || x.is_empty() {
            return Err("Vectors must have the same non-zero length".to_string());
        }
        
        let n = x.len();
        let mut concordant = 0;
        let mut discordant = 0;
        
        for i in 0..n {
            for j in (i + 1)..n {
                let sign_x = (x[i] - x[j]).signum();
                let sign_y = (y[i] - y[j]).signum();
                
                if sign_x == sign_y && sign_x != 0.0 {
                    concordant += 1;
                } else if sign_x != 0.0 && sign_y != 0.0 {
                    discordant += 1;
                }
            }
        }
        
        let total = concordant + discordant;
        if total == 0 {
            return Ok(0.0);
        }
        
        Ok((concordant - discordant) as f64 / total as f64)
    }

    fn ranks(data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut indexed: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let mut ranks = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && indexed[j].1 == indexed[i].1 {
                j += 1;
            }
            
            // Handle ties: assign average rank
            let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            
            i = j;
        }
        
        ranks
    }
}

/// Multivariate normal distribution.
#[derive(Debug, Clone)]
pub struct MultivariateNormal {
    pub mean: Vec<f64>,
    pub covariance: CovarianceMatrix,
}

impl MultivariateNormal {
    /// Create a new multivariate normal distribution.
    pub fn new(mean: Vec<f64>, covariance: CovarianceMatrix) -> Result<Self, String> {
        if mean.len() != covariance.dimension {
            return Err("Mean dimension must match covariance dimension".to_string());
        }
        
        if !covariance.is_positive_definite() {
            return Err("Covariance matrix must be positive definite".to_string());
        }
        
        Ok(MultivariateNormal {
            mean,
            covariance,
        })
    }

    /// Sample from the multivariate normal distribution.
    pub fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let n = self.mean.len();
        let l = self.covariance.cholesky().unwrap();
        
        // Generate standard normal samples
        let mut z = vec![0.0; n];
        for i in 0..n {
            let u1 = rng.uniform().max(1e-300);
            let u2 = rng.uniform();
            z[i] = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
        }
        
        // Transform: x = μ + L * z
        let mut x = self.mean.clone();
        for i in 0..n {
            for j in 0..=i {
                x[i] += l[i][j] * z[j];
            }
        }
        
        x
    }

    /// PDF at point x.
    pub fn pdf(&self, x: &[f64]) -> f64 {
        let n = self.mean.len();
        let det = self.determinant(&self.covariance.data);
        if det <= 0.0 {
            return 0.0;
        }
        
        let inv = self.inverse(&self.covariance.data);
        let diff: Vec<f64> = x.iter().zip(self.mean.iter()).map(|(a, b)| a - b).collect();
        
        let mut quadratic = 0.0;
        for i in 0..n {
            for j in 0..n {
                quadratic += diff[i] * inv[i][j] * diff[j];
            }
        }
        
        let coeff = 1.0 / ((2.0 * core::f64::consts::PI).powf(n as f64 / 2.0) * det.sqrt());
        coeff * (-0.5 * quadratic).exp()
    }

    fn determinant(&self, matrix: &[Vec<f64>]) -> f64 {
        let n = matrix.len();
        if n == 1 {
            return matrix[0][0];
        }
        if n == 2 {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }
        
        // Use LU decomposition for larger matrices
        let mut lu = matrix.to_vec();
        let mut det = 1.0;
        
        for i in 0..n {
            // Pivot
            let mut max_row = i;
            let mut max_val = lu[i][i].abs();
            for j in (i + 1)..n {
                if lu[j][i].abs() > max_val {
                    max_val = lu[j][i].abs();
                    max_row = j;
                }
            }
            
            if max_val < 1e-10 {
                return 0.0;
            }
            
            if max_row != i {
                lu.swap(i, max_row);
                det *= -1.0;
            }
            
            det *= lu[i][i];
            
            // Eliminate
            for j in (i + 1)..n {
                let factor = lu[j][i] / lu[i][i];
                for k in i..n {
                    lu[j][k] -= factor * lu[i][k];
                }
            }
        }
        
        det
    }

    fn inverse(&self, matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = matrix.len();
        let mut aug = vec![vec![0.0; 2 * n]; n];
        
        // Augment with identity matrix
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = matrix[i][j];
            }
            aug[i][n + i] = 1.0;
        }
        
        // Gaussian elimination
        for i in 0..n {
            // Pivot
            let mut max_row = i;
            let mut max_val = aug[i][i].abs();
            for j in (i + 1)..n {
                if aug[j][i].abs() > max_val {
                    max_val = aug[j][i].abs();
                    max_row = j;
                }
            }
            
            if max_row != i {
                aug.swap(i, max_row);
            }
            
            let pivot = aug[i][i];
            if pivot.abs() < 1e-10 {
                continue;
            }
            
            // Scale row
            for j in 0..2 * n {
                aug[i][j] /= pivot;
            }
            
            // Eliminate column
            for j in 0..n {
                if j != i {
                    let factor = aug[j][i];
                    for k in 0..2 * n {
                        aug[j][k] -= factor * aug[i][k];
                    }
                }
            }
        }
        
        // Extract inverse
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inv[i][j] = aug[i][n + j];
            }
        }
        
        inv
    }
}

/// Gaussian copula for modeling dependence.
pub struct GaussianCopula {
    pub correlation: CorrelationMatrix,
}

impl GaussianCopula {
    /// Create a new Gaussian copula.
    pub fn new(correlation: CorrelationMatrix) -> Result<Self, String> {
        // Check if correlation matrix is valid
        for i in 0..correlation.dimension {
            if (correlation.get(i, i) - 1.0).abs() > 1e-10 {
                return Err("Diagonal elements must be 1.0".to_string());
            }
        }
        
        Ok(GaussianCopula { correlation })
    }

    /// Sample from the copula.
    pub fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let n = self.correlation.dimension;
        let mut mean = vec![0.0; n];
        let mut cov_data = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                cov_data[i][j] = self.correlation.get(i, j);
            }
        }
        
        let cov = CovarianceMatrix::new(cov_data).unwrap();
        let mvn = MultivariateNormal::new(mean, cov).unwrap();
        
        let x = mvn.sample(rng);
        
        // Transform to uniform via normal CDF
        x.iter().map(|&xi| 0.5 * (1.0 + crate::distributions::erf(xi / core::f64::consts::SQRT_2))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance_from_samples() {
        let samples = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
        ];
        
        let cov = CovarianceMatrix::from_samples(&samples).unwrap();
        assert!((cov.variance(0) - 1.6666667).abs() < 1e-6);
        assert!((cov.variance(1) - 1.6666667).abs() < 1e-6);
        assert!((cov.get(0, 1) - 1.6666667).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_from_covariance() {
        let cov_data = vec![
            vec![4.0, 2.0],
            vec![2.0, 9.0],
        ];
        let cov = CovarianceMatrix::new(cov_data).unwrap();
        let corr = CorrelationMatrix::from_covariance(&cov);
        
        assert!((corr.get(0, 1) - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = CorrelationMatrix::pearson(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multivariate_normal() {
        let mean = vec![0.0, 0.0];
        let cov_data = vec![
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ];
        let cov = CovarianceMatrix::new(cov_data).unwrap();
        let mvn = MultivariateNormal::new(mean, cov).unwrap();
        
        let mut rng = Rng::new(42);
        let sample = mvn.sample(&mut rng);
        assert_eq!(sample.len(), 2);
    }
}
