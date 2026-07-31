pub mod polynomial;
pub mod linear_system;
pub mod nonlinear;
pub mod differential;
pub mod optimization;
pub mod matrix_eq;
pub mod dynamical;

pub use polynomial::*;
pub use linear_system::*;
pub use differential::*;
pub use matrix_eq::*;

pub fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    linear_system::solve_gauss(a, b)
}

pub use polynomial::solve_quadratic;
pub use polynomial::solve_cubic;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined() {
        let r = solve_quadratic(1.0, -3.0, 2.0);
        assert_eq!(r.len(), 2);
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = solve_linear_system(&a, &b).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10);
    }
}
