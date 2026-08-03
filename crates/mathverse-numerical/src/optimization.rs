//! Optimization methods.
//!
//! This module provides a Result-based API wrapper around [`mathverse_optimization`].
//! For the full suite including Adam, RMSProp, linear programming, and genetic algorithms,
//! use [`mathverse_optimization`] directly.

use mathverse_core::error::{MathError, MathResult};
use rand::Rng;

// Re-export the canonical optimization implementations
pub use mathverse_optimization::gradient::{
    gradient_descent as gd_simple, sgd, adam, rmsprop, adagrad, nadam
};

/// Gradient descent optimization.
pub struct GradientDescent {
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64, max_iterations: usize, tolerance: f64) -> Self {
        GradientDescent {
            learning_rate,
            max_iterations,
            tolerance,
        }
    }

    /// Minimize function f(x) using gradient descent.
    pub fn minimize(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        grad: &dyn Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
    ) -> MathResult<(Vec<f64>, f64, usize)> {
        let mut x = x0.to_vec();
        let mut best_value = f(&x);
        let mut best_x = x.clone();
        
        for iteration in 0..self.max_iterations {
            let gradient = grad(&x);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < self.tolerance {
                return Ok((x, best_value, iteration));
            }
            
            // Update x
            for i in 0..x.len() {
                x[i] -= self.learning_rate * gradient[i];
            }
            
            let current_value = f(&x);
            
            if current_value < best_value {
                best_value = current_value;
                best_x = x.clone();
            }
        }
        
        Ok((best_x, best_value, self.max_iterations))
    }

    /// Minimize with momentum.
    pub fn minimize_with_momentum(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        grad: &dyn Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
        momentum: f64,
    ) -> MathResult<(Vec<f64>, f64, usize)> {
        let mut x = x0.to_vec();
        let mut velocity = vec![0.0; x.len()];
        let mut best_value = f(&x);
        let mut best_x = x.clone();
        
        for iteration in 0..self.max_iterations {
            let gradient = grad(&x);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < self.tolerance {
                return Ok((x, best_value, iteration));
            }
            
            // Update velocity and position
            for i in 0..x.len() {
                velocity[i] = momentum * velocity[i] - self.learning_rate * gradient[i];
                x[i] += velocity[i];
            }
            
            let current_value = f(&x);
            
            if current_value < best_value {
                best_value = current_value;
                best_x = x.clone();
            }
        }
        
        Ok((best_x, best_value, self.max_iterations))
    }
}

// Re-export advanced optimization methods from mathverse-optimization
// These have richer APIs and are maintained in the dedicated optimization crate
pub use mathverse_optimization::{
    unconstrained::bfgs_min,
    combinatorial,
    linear_programming,
};


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_descent() {
        let gd = GradientDescent::new(0.1, 1000, 1e-10);
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        
        let (result, value, _) = gd.minimize(&f, &grad, &[1.0, 1.0]).unwrap();
        
        assert!(value < 1e-6);
        assert!(result[0].abs() < 0.1);
        assert!(result[1].abs() < 0.1);
    }

    #[test]
    fn test_adam_re_export() {
        // Test that re-exported Adam works
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        
        let result = adam(&grad, &[1.0, 1.0], 0.1, 0.9, 0.999, 1e-8, 1e-10, 1000);
        
        // Adam should converge reasonably close to origin
        assert!(result[0].abs() < 0.1);
        assert!(result[1].abs() < 0.1);
    }
}
