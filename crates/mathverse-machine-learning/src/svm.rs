//! Support Vector Machine (SVM) classifier with linear, RBF, and polynomial
//! kernels trained via a simplified SMO (Sequential Minimal Optimization).
//!
//! The implementation keeps an exact cache of the decision value
//! `f(x_i) = Σ_j α_j y_j K(x_i, x_j) + b` for every training point and applies
//! incremental updates after each SMO step, so every iteration costs `O(n)`
//! kernel evaluations instead of `O(n²)`.

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
            Self::Linear => a.iter().zip(b).map(|(x, y)| x * y).sum(),
            Self::RBF { gamma } => {
                let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
                (-gamma * d).exp()
            }
            Self::Polynomial {
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

    /// Fit SVM using simplified SMO with cached decision values.
    ///
    /// The cache makes each full examination pass `O(n²)` in kernel
    /// evaluations rather than `O(n³)`, which keeps convergence fast even on
    /// datasets with hundreds of samples.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = x.len();
        self.alpha = vec![0.0; n];
        self.bias = 0.0;
        self.support_vectors.clear();
        self.support_labels.clear();
        self.support_alpha.clear();
        if n < 2 {
            return;
        }

        // Cached decision values: f[i] = Σ_j α_j y_j K(x_i, x_j) + b.
        // With all α zero this is simply `b` (initially 0).
        let mut f_cache: Vec<f64> = vec![self.bias; n];

        for _ in 0..self.max_iters {
            let mut num_changed =
                self.smo_pass(x, y, &mut f_cache, false);
            if num_changed == 0 {
                num_changed = self.smo_pass(x, y, &mut f_cache, true);
                if num_changed == 0 {
                    break; // converged
                }
            }
        }

        // Extract support vectors.
        for (i, &a) in self.alpha.iter().enumerate() {
            if a > 1e-7 {
                self.support_vectors.push(x[i].clone());
                self.support_labels.push(y[i]);
                self.support_alpha.push(a);
            }
        }
    }

    /// One pass of SMO over either all indices (`non_bound_only == false`) or
    /// only non-bound multipliers (`non_bound_only == true`). Returns the
    /// number of multiplier pairs updated.
    fn smo_pass(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        f_cache: &mut [f64],
        non_bound_only: bool,
    ) -> usize {
        let n = x.len();
        let mut num_changed = 0;
        for i in 0..n {
            if non_bound_only && !(self.alpha[i] > 0.0 && self.alpha[i] < self.c) {
                continue;
            }
            let ei = f_cache[i] - y[i];
            let kkt_violated = (y[i] * ei < -self.tol && self.alpha[i] < self.c)
                || (y[i] * ei > self.tol && self.alpha[i] > 0.0);
            if kkt_violated {
                let j = Self::select_second_choice(f_cache, y, i, ei);
                if self.take_step(x, y, i, j, f_cache) {
                    num_changed += 1;
                }
            }
        }
        num_changed
    }

    /// Choose `j != i` maximizing `|E_i - E_j|` using cached errors.
    fn select_second_choice(f_cache: &[f64], y: &[f64], i: usize, ei: f64) -> usize {
        let mut best_j = usize::from(i == 0);
        let mut max_diff = 0.0_f64;
        for (k, (&fk, &yk)) in f_cache.iter().zip(y).enumerate() {
            if k == i {
                continue;
            }
            let diff = (ei - (fk - yk)).abs();
            if diff > max_diff {
                max_diff = diff;
                best_j = k;
            }
        }
        best_j
    }

    /// Attempt the joint SMO update of `(α_i, α_j)`. On success, updates the
    /// bias, both multipliers, and *every* cached decision value exactly:
    ///
    /// `f_k ← f_k + Δα_i·y_i·K(k,i) + Δα_j·y_j·K(k,j) + Δb`
    fn take_step(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        i: usize,
        j: usize,
        f_cache: &mut [f64],
    ) -> bool {
        if i == j {
            return false;
        }
        let ei = f_cache[i] - y[i];
        let ej = f_cache[j] - y[j];
        let ai_old = self.alpha[i];
        let aj_old = self.alpha[j];

        let (l, h) = if y[i] == y[j] {
            (
                0.0_f64.max(ai_old + aj_old - self.c),
                self.c.min(ai_old + aj_old),
            )
        } else {
            (
                0.0_f64.max(aj_old - ai_old),
                self.c.min(self.c + aj_old - ai_old),
            )
        };
        if (l - h).abs() < 1e-10 {
            return false;
        }

        let k_ij = self.kernel.compute(&x[i], &x[j]);
        let eta = 2.0 * k_ij
            - self.kernel.compute(&x[i], &x[i])
            - self.kernel.compute(&x[j], &x[j]);
        if eta >= 0.0 {
            return false;
        }

        let mut aj_new = aj_old - y[j] * (ei - ej) / eta;
        aj_new = aj_new.clamp(l, h);
        if (aj_new - aj_old).abs() < 1e-5 {
            return false;
        }

        let ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new);

        let b1 = self.bias
            - ei
            - y[i] * (ai_new - ai_old) * self.kernel.compute(&x[i], &x[i])
            - y[j] * (aj_new - aj_old) * k_ij;
        let b2 = self.bias
            - ej
            - y[i] * (ai_new - ai_old) * k_ij
            - y[j] * (aj_new - aj_old) * self.kernel.compute(&x[j], &x[j]);
        let b_new = if ai_new > 0.0 && ai_new < self.c {
            b1
        } else if aj_new > 0.0 && aj_new < self.c {
            b2
        } else {
            f64::midpoint(b1, b2)
        };

        // Exact O(n) refresh of every cached decision value.
        let dai_y = y[i] * (ai_new - ai_old);
        let daj_y = y[j] * (aj_new - aj_old);
        let db = b_new - self.bias;
        for (k, fk) in f_cache.iter_mut().enumerate() {
            *fk += dai_y * self.kernel.compute(&x[k], &x[i])
                + daj_y * self.kernel.compute(&x[k], &x[j])
                + db;
        }

        self.alpha[i] = ai_new;
        self.alpha[j] = aj_new;
        self.bias = b_new;
        true
    }

    /// Predict class labels (`+1.0` or `-1.0`).
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|xi| {
                let mut sum = 0.0;
                for ((sv, &a), &l) in self
                    .support_vectors
                    .iter()
                    .zip(&self.support_alpha)
                    .zip(&self.support_labels)
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

    #[test]
    fn svm_scales_to_larger_dataset_quickly() {
        // Regression guard: SMO used to be O(n³) per pass and effectively hung
        // on inputs of this size. Deterministic pseudo-random separable data.
        let n = 200;
        let x: Vec<Vec<f64>> = (0..n)
            .map(|t| {
                let s = (t as f64 * 12.9898).fract() * 43758.545_3;
                let u = s.fract();
                let s2 = ((t as f64 + 0.5) * 78.233).fract() * 12345.678_9;
                let v = s2.fract();
                if t % 2 == 0 {
                    vec![u, v]
                } else {
                    vec![u + 3.0, v + 3.0]
                }
            })
            .collect();
        let y: Vec<f64> = (0..n).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let start = std::time::Instant::now();
        let mut svm = SVM::linear(1.0);
        svm.fit(&x, &y);
        assert!(
            start.elapsed().as_secs() < 30,
            "SMO fit regressed to super-linear runtime"
        );

        let pred = svm.predict(&x);
        let correct = pred.iter().zip(&y).filter(|(p, t)| **p == **t).count();
        assert!(correct > n / 2, "accuracy too low: {correct}/{n}");
        assert_eq!(svm.alpha.len(), n);
    }

    #[test]
    fn svm_handles_degenerate_input() {
        let mut svm = SVM::linear(1.0);
        svm.fit(&[], &[]);
        assert!(svm.predict(&[]).is_empty());
        svm.fit(&[vec![1.0]], &[1.0]);
        assert!(svm.support_vectors.is_empty());
    }

    #[test]
    fn polynomial_kernel_matches_definition() {
        let k = Kernel::Polynomial {
            degree: 3,
            gamma: 2.0,
            coef0: 1.0,
        };
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        assert!((k.compute(&a, &b) - (2.0 * 11.0_f64 + 1.0).powi(3)).abs() < 1e-12);
    }
}
