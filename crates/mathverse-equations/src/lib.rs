//! # mathverse-equations
//!
//! Equation solvers for the MathVerse ecosystem.
//!
//! Provides:
//! - **Polynomial**: quadratic, cubic, and quartic root finders
//! - **Linear systems**: Gaussian elimination solver via [`solve_linear_system`]
//! - **Nonlinear**: Newton-Raphson and bisection root finders
//! - **Differential equations**: Euler and RK4 ODE integrators
//! - **Optimization**: golden-section and Brent's method minimizers
//! - **Matrix equations**: `Ax = b` solvers
//! - **Dynamical systems**: fixed-point iteration and cobweb analysis

pub mod polynomial;
pub mod linear_system;
pub mod nonlinear;
pub mod differential;
pub mod optimization;
pub mod matrix_eq;
pub mod dynamical;

/// Solves a linear system `Ax = b` using Gaussian elimination with partial pivoting.
///
/// Returns `None` if the matrix is singular or nearly singular.
///
/// # Arguments
/// * `a` - coefficient matrix (row-major `n×n`)
/// * `b` - right-hand side vector (length `n`)
///
/// # Examples
/// ```
/// use mathverse_equations::solve_linear_system;
/// let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
/// let b = vec![5.0, 7.0];
/// let x = solve_linear_system(&a, &b).unwrap();
/// assert!((x[0] - 1.6).abs() < 1e-10);
/// ```
pub fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    matrix_eq::solve_gauss(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined() {
        let r = polynomial::solve_quadratic(1.0, -3.0, 2.0);
        assert_eq!(r.len(), 2);
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = solve_linear_system(&a, &b).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10);
    }
}
