//! Matrix calculus: gradients, Jacobians, Hessians, and automatic differentiation.

use crate::Matrix;
use mathverse_core::error::MathResult;

/// Matrix derivative result.
#[derive(Debug, Clone)]
pub struct MatrixDerivative {
    pub derivative: Matrix,
    pub variable: String,
}

/// Matrix calculus operations.
pub struct MatrixCalculus;

impl MatrixCalculus {
    /// Gradient of scalar function f: R^n → R.
    pub fn gradient(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
        let mut grad = Vec::with_capacity(x.len());
        
        for i in 0..x.len() {
            let mut x_plus = x.to_vec();
            x_plus[i] += h;
            let f_plus = f(&x_plus);
            
            let mut x_minus = x.to_vec();
            x_minus[i] -= h;
            let f_minus = f(&x_minus);
            
            grad.push((f_plus - f_minus) / (2.0 * h));
        }
        
        grad
    }

    /// Jacobian of vector function f: R^n → R^m.
    pub fn jacobian(f: impl Fn(&[f64]) -> Vec<f64>, x: &[f64], h: f64) -> Matrix {
        let m = f(x).len();
        let n = x.len();
        let mut jacobian = Matrix::zeros(m, n);
        
        for j in 0..n {
            let mut x_plus = x.to_vec();
            x_plus[j] += h;
            let f_plus = f(&x_plus);
            
            let mut x_minus = x.to_vec();
            x_minus[j] -= h;
            let f_minus = f(&x_minus);
            
            for i in 0..m {
                jacobian.set(i, j, (f_plus[i] - f_minus[i]) / (2.0 * h));
            }
        }
        
        jacobian
    }

    /// Hessian of scalar function f: R^n → R.
    pub fn hessian(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Matrix {
        let n = x.len();
        let mut hessian = Matrix::zeros(n, n);
        
        for i in 0..n {
            for j in 0..=i {
                let mut x_pp = x.to_vec();
                x_pp[i] += h;
                x_pp[j] += h;
                let f_pp = f(&x_pp);
                
                let mut x_pm = x.to_vec();
                x_pp[i] += h;
                x_pm[j] -= h;
                let f_pm = f(&x_pm);
                
                let mut x_mp = x.to_vec();
                x_mp[i] -= h;
                x_mp[j] += h;
                let f_mp = f(&x_mp);
                
                let mut x_mm = x.to_vec();
                x_mm[i] -= h;
                x_mm[j] -= h;
                let f_mm = f(&x_mm);
                
                let h_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
                hessian.set(i, j, h_ij);
                hessian.set(j, i, h_ij);
            }
        }
        
        hessian
    }

    /// Partial derivative of matrix function.
    pub fn partial_derivative(
        f: impl Fn(&Matrix) -> Matrix,
        m: &Matrix,
        row: usize,
        col: usize,
        h: f64,
    ) -> Matrix {
        let mut m_plus = m.clone();
        m_plus.set(row, col, m.get(row, col) + h);
        let f_plus = f(&m_plus);
        
        let mut m_minus = m.clone();
        m_minus.set(row, col, m.get(row, col) - h);
        let f_minus = f(&m_minus);
        
        let diff = f_plus.sub(&f_minus).unwrap();
        diff.scale(1.0 / (2.0 * h))
    }

    /// Directional derivative.
    pub fn directional_derivative(
        f: impl Fn(&[f64]) -> f64,
        x: &[f64],
        direction: &[f64],
        h: f64,
    ) -> f64 {
        let n = x.len();
        let mut x_plus = Vec::with_capacity(n);
        
        for i in 0..n {
            x_plus.push(x[i] + h * direction[i]);
        }
        
        let f_plus = f(&x_plus);
        let f_x = f(x);
        
        (f_plus - f_x) / h
    }

    /// Gâteaux derivative (generalized directional derivative).
    pub fn gateaux_derivative(
        f: impl Fn(&Matrix) -> Matrix,
        m: &Matrix,
        direction: &Matrix,
        h: f64,
    ) -> Matrix {
        let m_plus = m.add(&direction.scale(h)).unwrap();
        let f_plus = f(&m_plus);
        let f_m = f(m);
        
        let diff = f_plus.sub(&f_m).unwrap();
        diff.scale(1.0 / h)
    }

    /// Fréchet derivative (linear approximation).
    pub fn frechet_derivative(
        f: impl Fn(&Matrix) -> Matrix,
        m: &Matrix,
        h: f64,
    ) -> MathResult<Matrix> {
        let n = m.rows * m.cols;
        let mut jacobian = Matrix::zeros(m.rows, m.cols);

        for k in 0..n {
            let row = k / m.cols;
            let col = k % m.cols;

            let partial = Self::partial_derivative(&f, m, row, col, h);
            jacobian = jacobian.add(&partial)?;
        }

        Ok(jacobian)
    }
}

/// Automatic differentiation for matrix operations.
pub struct AutoDiff;

impl AutoDiff {
    /// Derivative of matrix product: d(AB)/dA.
    pub fn matrix_product_derivative_a(a: &Matrix, b: &Matrix) -> MathResult<Vec<Matrix>> {
        let (m, n) = (a.rows, a.cols);
        let p = b.cols;
        
        let mut derivatives = Vec::with_capacity(m * n);
        
        for i in 0..m {
            for j in 0..n {
                let mut deriv = Matrix::zeros(m, p);
                for k in 0..p {
                    deriv.set(i, k, b.get(j, k));
                }
                derivatives.push(deriv);
            }
        }
        
        Ok(derivatives)
    }

    /// Derivative of matrix product: d(AB)/dB.
    pub fn matrix_product_derivative_b(a: &Matrix, b: &Matrix) -> MathResult<Vec<Matrix>> {
        let (m, n) = (a.rows, a.cols);
        let p = b.cols;
        
        let mut derivatives = Vec::with_capacity(n * p);
        
        for j in 0..n {
            for k in 0..p {
                let mut deriv = Matrix::zeros(m, n);
                for i in 0..m {
                    deriv.set(i, j, a.get(i, j));
                }
                derivatives.push(deriv);
            }
        }
        
        Ok(derivatives)
    }

    /// Derivative of matrix inverse: d(A^{-1})/dA.
    pub fn matrix_inverse_derivative(a: &Matrix) -> MathResult<Vec<Matrix>> {
        let n = a.rows;
        let inv = a.inverse()?;
        
        let mut derivatives = Vec::with_capacity(n * n);
        
        for i in 0..n {
            for j in 0..n {
                let mut deriv = Matrix::zeros(n, n);
                for k in 0..n {
                    for l in 0..n {
                        deriv.set(k, l, -inv.get(k, i) * inv.get(j, l));
                    }
                }
                derivatives.push(deriv);
            }
        }
        
        Ok(derivatives)
    }

    /// Derivative of determinant: d(det(A))/dA.
    pub fn determinant_derivative(a: &Matrix) -> MathResult<Matrix> {
        let n = a.rows;
        let det = a.det()?;
        let adj = a.inverse()?.scale(det);
        
        Ok(adj.transpose())
    }

    /// Derivative of trace: d(tr(A))/dA = I.
    pub fn trace_derivative(a: &Matrix) -> Matrix {
        Matrix::identity(a.rows)
    }

    /// Derivative of quadratic form: d(x^T A x)/dx = (A + A^T) x.
    pub fn quadratic_form_derivative(a: &Matrix, x: &[f64]) -> MathResult<Vec<f64>> {
        let a_plus_at = a.add(&a.transpose())?;
        let x_vec = mathverse_vector::Vector::new(x.to_vec());
        let result = a_plus_at.mul_vec(&x_vec)?;
        Ok(result.data)
    }

    /// Second derivative of quadratic form: d^2(x^T A x)/dx^2 = A + A^T.
    pub fn quadratic_form_hessian(a: &Matrix) -> MathResult<Matrix> {
        a.add(&a.transpose())
    }
}

/// Gradient-based optimization.
pub struct GradientOptimization;

impl GradientOptimization {
    /// Gradient descent for scalar function.
    pub fn gradient_descent(
        f: impl Fn(&[f64]) -> f64,
        grad: impl Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> (Vec<f64>, usize, f64) {
        let mut x = x0.to_vec();
        
        for iteration in 0..max_iterations {
            let gradient = grad(&x);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < tolerance {
                return (x, iteration, grad_norm);
            }
            
            for i in 0..x.len() {
                x[i] -= learning_rate * gradient[i];
            }
        }
        
        let gradient = grad(&x);
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        
        (x, max_iterations, grad_norm)
    }

    /// Newton's method for optimization.
    pub fn newton_method(
        f: impl Fn(&[f64]) -> f64,
        grad: impl Fn(&[f64]) -> Vec<f64>,
        hess: impl Fn(&[f64]) -> Matrix,
        x0: &[f64],
        max_iterations: usize,
        tolerance: f64,
    ) -> MathResult<(Vec<f64>, usize, f64)> {
        let mut x = x0.to_vec();
        
        for iteration in 0..max_iterations {
            let gradient = mathverse_vector::Vector::new(grad(&x));
            let hessian = hess(&x);
            
            let grad_norm: f64 = gradient.data.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < tolerance {
                return Ok((x, iteration, grad_norm));
            }
            
            let delta = hessian.solve(&gradient)?;
            
            for i in 0..x.len() {
                x[i] -= delta.get(i);
            }
        }
        
        let gradient = mathverse_vector::Vector::new(grad(&x));
        let grad_norm: f64 = gradient.data.iter().map(|g| g * g).sum::<f64>().sqrt();
        
        Ok((x, max_iterations, grad_norm))
    }

    /// BFGS optimization (simplified).
    pub fn bfgs(
        f: impl Fn(&[f64]) -> f64,
        grad: impl Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
        max_iterations: usize,
        tolerance: f64,
    ) -> MathResult<(Vec<f64>, usize, f64)> {
        let n = x0.len();
        let mut x = x0.to_vec();
        let mut h = Matrix::identity(n);  // Approximate inverse Hessian
        
        for iteration in 0..max_iterations {
            let gradient = mathverse_vector::Vector::new(grad(&x));
            let grad_norm: f64 = gradient.data.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < tolerance {
                return Ok((x, iteration, grad_norm));
            }
            
            // Search direction
            let p_vec = h.mul_vec(&gradient)?;
            let p = p_vec.data;
            
            // Line search (simplified)
            let alpha = 1.0;
            let x_new: Vec<f64> = x.iter().zip(p.iter()).map(|(&xi, &pi)| xi - alpha * pi).collect();
            
            let gradient_new = mathverse_vector::Vector::new(grad(&x_new));
            
            // BFGS update
            let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(&xn, &xi)| xn - xi).collect();
            let y: Vec<f64> = gradient_new.data.iter().zip(gradient.data.iter())
                .map(|(&gn, &gi)| gn - gi).collect();
            
            let s_vec = mathverse_vector::Vector::new(s.clone());
            let y_vec = mathverse_vector::Vector::new(y.clone());
            
            let sy = s_vec.dot(&y_vec);
            
            if sy > 1e-15 {
                let n = x.len();
                let mut a = Matrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        a.set(i, j, s[i] * s[j]);
                    }
                }
                
                let mut b = Matrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for k in 0..n {
                            sum += h.get(i, k) * y[k] * s[j];
                        }
                        b.set(i, j, sum);
                    }
                }
                
                let a_scaled = a.scale(1.0 / sy);
                let b_scaled = b.scale(1.0 / sy);
                
                h = h.add(&a_scaled)?.sub(&b_scaled)?;
            }
            
            x = x_new;
        }
        
        let gradient = mathverse_vector::Vector::new(grad(&x));
        let grad_norm: f64 = gradient.data.iter().map(|g| g * g).sum::<f64>().sqrt();
        
        Ok((x, max_iterations, grad_norm))
    }
}

/// Matrix differential calculus.
pub struct MatrixDifferential;

impl MatrixDifferential {
    /// Differential of matrix exponential: d(exp(A)) = exp(A) dA (for commuting matrices).
    pub fn exp_differential(a: &Matrix, da: &Matrix) -> MathResult<Matrix> {
        let exp_a = crate::functions::MatrixExponential::compute(a)?;
        exp_a.mul(da)
    }

    /// Differential of matrix logarithm: d(log(A)) = A^{-1} dA.
    pub fn log_differential(a: &Matrix, da: &Matrix) -> MathResult<Matrix> {
        let inv_a = a.inverse()?;
        inv_a.mul(da)
    }

    /// Differential of matrix inverse: d(A^{-1}) = -A^{-1} dA A^{-1}.
    pub fn inverse_differential(a: &Matrix, da: &Matrix) -> MathResult<Matrix> {
        let inv_a = a.inverse()?;
        let inv_da = inv_a.mul(da)?;
        let neg_inv_da = inv_da.scale(-1.0);
        neg_inv_da.mul(&inv_a)
    }

    /// Chain rule for matrix functions.
    pub fn chain_rule(
        outer: impl Fn(&Matrix) -> Matrix,
        outer_derivative: impl Fn(&Matrix, &Matrix) -> MathResult<Matrix>,
        inner: impl Fn(&[f64]) -> Matrix,
        inner_derivative: impl Fn(&[f64]) -> Vec<f64>,
        x: &[f64],
    ) -> MathResult<Vec<f64>> {
        let inner_result = inner(x);
        let inner_grad = inner_derivative(x);
        
        let outer_diff = outer_derivative(&inner_result, &Matrix::identity(inner_result.rows))?;
        
        let mut result = Vec::new();
        for i in 0..inner_grad.len() {
            let mut sum = 0.0;
            for j in 0..outer_diff.cols {
                sum += outer_diff.get(i, j) * inner_grad[j];
            }
            result.push(sum);
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let x = vec![1.0, 2.0];
        let grad = MatrixCalculus::gradient(&f, &x, 1e-6);
        
        assert!((grad[0] - 2.0).abs() < 1e-5);
        assert!((grad[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_jacobian() {
        let f = |x: &[f64]| vec![x[0] + x[1], x[0] * x[1]];
        let x = vec![2.0, 3.0];
        let jac = MatrixCalculus::jacobian(&f, &x, 1e-6);
        
        assert!((jac.get(0, 0) - 1.0).abs() < 1e-5);
        assert!((jac.get(0, 1) - 1.0).abs() < 1e-5);
        assert!((jac.get(1, 0) - 3.0).abs() < 1e-5);
        assert!((jac.get(1, 1) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_hessian() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let x = vec![1.0, 2.0];
        let hess = MatrixCalculus::hessian(&f, &x, 1e-6);
        
        assert!((hess.get(0, 0) - 2.0).abs() < 1e-5);
        assert!((hess.get(1, 1) - 2.0).abs() < 1e-5);
        assert!((hess.get(0, 1) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_descent() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)];
        let x0 = vec![0.0, 0.0];
        
        let (x, iterations, _) = GradientOptimization::gradient_descent(&f, &grad, &x0, 0.1, 100, 1e-10);
        
        assert!(iterations < 100);
        assert!((x[0] - 1.0).abs() < 0.1);
        assert!((x[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_determinant_derivative() {
        let a = Matrix::identity(2);
        let deriv = AutoDiff::determinant_derivative(&a).unwrap();
        
        assert!((deriv.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((deriv.get(1, 1) - 1.0).abs() < 1e-10);
        assert!((deriv.get(0, 1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_form_derivative() {
        let a = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 2.0]]).unwrap();
        let x = vec![1.0, 1.0];
        
        let grad = AutoDiff::quadratic_form_derivative(&a, &x).unwrap();
        
        assert!((grad[0] - 3.0).abs() < 1e-10);
        assert!((grad[1] - 3.0).abs() < 1e-10);
    }
}
