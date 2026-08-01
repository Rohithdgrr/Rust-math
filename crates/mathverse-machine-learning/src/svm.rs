//! Support Vector Machine (SVM) classifier with linear and RBF kernels.

/// SVM classifier using SMO (Sequential Minimal Optimization).
pub struct SVM {
    /// Regularization parameter.
    pub c: f64,
    /// Kernel function used for the dot product.
    pub kernel: Kernel,
    /// Convergence tolerance.
    pub tol: f64,
    /// Maximum number of optimization iterations.
    pub max_iters: usize,
    /// Lagrange multipliers from SMO optimization.
    pub alpha: Vec<f64>,
    /// Decision boundary bias term.
    pub bias: f64,
    /// Training samples that are support vectors.
    pub support_vectors: Vec<Vec<f64>>,
    /// Labels of the support vectors.
    pub support_labels: Vec<f64>,
    /// Alpha values of the support vectors.
    pub support_alpha: Vec<f64>,
}

/// Kernel function for computing similarity between vectors.
#[derive(Clone)]
pub enum Kernel {
    /// Linear kernel: dot product.
    Linear,
    /// Radial basis function kernel.
    RBF {
        /// Kernel bandwidth.
        gamma: f64,
    },
    /// Polynomial kernel.
    Polynomial {
        /// Polynomial degree.
        degree: usize,
        /// Kernel coefficient.
        gamma: f64,
        /// Independent term.
        coef0: f64,
    },
}

impl Kernel {
    /// Compute kernel similarity between two vectors.
    #[must_use]
    pub fn compute(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            Kernel::Linear => a.iter().zip(b).map(|(x, y)| x * y).sum(),
            Kernel::RBF { gamma } => {
                let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
                (-gamma * d).exp()
            }
            Kernel::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                (gamma * dot + coef0).powi(*degree as i32)
            }
        }
    }
}

impl SVM {
    /// Create a new SVM with the given hyperparameters.
    #[must_use]
    #[inline]
    pub fn new(c: f64, kernel: Kernel, tol: f64, max_iters: usize) -> Self {
        Self {
            c,
            kernel,
            tol,
            max_iters,
            alpha: Vec::new(),
            bias: 0.0,
            support_vectors: Vec::new(),
            support_labels: Vec::new(),
            support_alpha: Vec::new(),
        }
    }

    /// Create an SVM with a linear kernel.
    #[must_use]
    #[inline]
    pub fn linear(c: f64) -> Self {
        Self::new(c, Kernel::Linear, 1e-3, 1000)
    }

    /// Create an SVM with an RBF kernel.
    #[must_use]
    #[inline]
    pub fn rbf(c: f64, gamma: f64) -> Self {
        Self::new(c, Kernel::RBF { gamma }, 1e-3, 1000)
    }

    /// Fit SVM using simplified SMO.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = x.len();
        self.alpha = vec![0.0; n];
        self.bias = 0.0;

        for _ in 0..self.max_iters {
            let mut num_changed = 0;
            for i in 0..n {
                let ei = self.predict_single(x, y, i) - y[i];
                if (y[i] * ei < -self.tol && self.alpha[i] < self.c)
                    || (y[i] * ei > self.tol && self.alpha[i] > 0.0)
                {
                    let j = if i + 1 < n {
                        i + 1
                    } else {
                        i.wrapping_sub(1) % n
                    };
                    let ej = self.predict_single(x, y, j) - y[j];
                    let ai_old = self.alpha[i];
                    let aj_old = self.alpha[j];

                    let (l, h) = if y[i] != y[j] {
                        (
                            0.0_f64.max(self.alpha[j] - self.alpha[i]),
                            self.c.min(self.c + self.alpha[j] - self.alpha[i]),
                        )
                    } else {
                        (
                            0.0_f64.max(self.alpha[i] + self.alpha[j] - self.c),
                            self.c.min(self.alpha[i] + self.alpha[j]),
                        )
                    };
                    if (l - h).abs() < 1e-10 {
                        continue;
                    }

                    let eta = 2.0 * self.kernel.compute(&x[i], &x[j])
                        - self.kernel.compute(&x[i], &x[i])
                        - self.kernel.compute(&x[j], &x[j]);
                    if eta >= 0.0 {
                        continue;
                    }

                    self.alpha[j] = aj_old - y[j] * (ei - ej) / eta;
                    self.alpha[j] = self.alpha[j].clamp(l, h);
                    if (self.alpha[j] - aj_old).abs() < 1e-5 {
                        continue;
                    }

                    self.alpha[i] = ai_old + y[i] * y[j] * (aj_old - self.alpha[j]);
                    let b1 = self.bias
                        - ei
                        - y[i] * (self.alpha[i] - ai_old) * self.kernel.compute(&x[i], &x[i])
                        - y[j] * (self.alpha[j] - aj_old) * self.kernel.compute(&x[i], &x[j]);
                    let b2 = self.bias
                        - ej
                        - y[i] * (self.alpha[i] - ai_old) * self.kernel.compute(&x[i], &x[j])
                        - y[j] * (self.alpha[j] - aj_old) * self.kernel.compute(&x[j], &x[j]);
                    self.bias = if self.alpha[i] > 0.0 && self.alpha[i] < self.c {
                        b1
                    } else if self.alpha[j] > 0.0 && self.alpha[j] < self.c {
                        b2
                    } else {
                        (b1 + b2) / 2.0
                    };
                    num_changed += 1;
                }
            }
            if num_changed == 0 {
                break;
            }
        }

        // Extract support vectors
        for i in 0..n {
            if self.alpha[i] > 1e-7 {
                self.support_vectors.push(x[i].clone());
                self.support_labels.push(y[i]);
                self.support_alpha.push(self.alpha[i]);
            }
        }
    }

    fn predict_single(&self, x: &[Vec<f64>], y: &[f64], i: usize) -> f64 {
        let mut sum = 0.0;
        for j in 0..x.len() {
            if self.alpha[j] > 1e-10 {
                sum += self.alpha[j] * y[j] * self.kernel.compute(&x[i], &x[j]);
            }
        }
        sum + self.bias
    }

    /// Predict class labels.
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|xi| {
                let mut sum = 0.0;
                for (sv, (a, l)) in self
                    .support_vectors
                    .iter()
                    .zip(self.support_alpha.iter().zip(self.support_labels.iter()))
                {
                    sum += a * l * self.kernel.compute(xi, sv);
                }
                if sum + self.bias >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svm_linear_test() {
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 3.0],
            vec![6.0, 5.0],
            vec![7.0, 8.0],
            vec![8.0, 7.0],
        ];
        let y = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        let mut svm = SVM::linear(1.0);
        svm.fit(&x, &y);
        let pred = svm.predict(&x);
        let correct = pred
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-6)
            .count();
        assert!(correct >= 4); // at least mostly correct
    }

    #[test]
    fn svm_rbf_test() {
        let x = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![5.0, 5.0],
            vec![6.0, 6.0],
        ];
        let y = vec![1.0, 1.0, -1.0, -1.0];
        let mut svm = SVM::rbf(10.0, 0.5);
        svm.fit(&x, &y);
        assert!(!svm.support_vectors.is_empty());
    }
}
