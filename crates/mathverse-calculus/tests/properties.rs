//! Property-based tests using proptest.
//!
//! Verifies mathematical properties that should hold for all valid inputs.

use proptest::prelude::*;
use mathverse_calculus::prelude::*;
use core::f64::consts::PI;

proptest! {
    /// Derivative of a constant function is zero.
    #[test]
    fn derivative_of_constant_is_zero(
        x in -100.0_f64..100.0_f64,
    ) {
        let result = derivative(&|_: f64| 42.0, x);
        prop_assert!(result.abs() < 1e-6, "d/dx(c) = {result}");
    }

    /// Derivative of x² is 2x.
    #[test]
    fn derivative_of_square_is_2x(
        x in -50.0_f64..50.0_f64,
    ) {
        let result = derivative(&|t| t * t, x);
        let expected = 2.0 * x;
        prop_assert!((result - expected).abs() < 1e-4, "at {x}: got {result}, expected {expected}");
    }

    /// Second derivative of x³ is 6x.
    #[test]
    fn second_derivative_of_cube(
        x in -20.0_f64..20.0_f64,
    ) {
        let result = second_derivative(&|t| t * t * t, x);
        let expected = 6.0 * x;
        prop_assert!((result - expected).abs() < 1e-2, "at {x}: got {result}, expected {expected}");
    }

    /// Integration is linear: ∫(af + bg) = a∫f + b∫g.
    #[test]
    fn integration_is_linear(
        a in -5.0_f64..5.0_f64,
        b in -5.0_f64..5.0_f64,
    ) {
        let f = |x: f64| x;
        let g = |x: f64| x * x;
        let left = integrate(&|x| a * f(x) + b * g(x), 0.0, 1.0, 1e-10);
        let right = a * integrate(&f, 0.0, 1.0, 1e-10) + b * integrate(&g, 0.0, 1.0, 1e-10);
        prop_assert!((left - right).abs() < 1e-6, "linearity: {left} vs {right}");
    }

    /// Integration over [a, a] is zero.
    #[test]
    fn integration_over_point_is_zero(
        a in -10.0_f64..10.0_f64,
    ) {
        let result = integrate(&|x| x * x * x, a, a, 1e-10);
        prop_assert!(result.abs() < 1e-15, "∫ₐₐ f = {result}");
    }

    /// Integration reverses sign when bounds are swapped.
    #[test]
    fn integration_sign_flip(
        a in 0.0_f64..5.0_f64,
        b in 6.0_f64..10.0_f64,
    ) {
        let forward = integrate(&|x| x * x, a, b, 1e-10);
        let reverse = integrate(&|x| x * x, b, a, 1e-10);
        prop_assert!((forward + reverse).abs() < 1e-6, "sign flip: {forward} + {reverse}");
    }

    /// Gradient of x₁² + x₂² + ... + xₙ² is (2x₁, 2x₂, ..., 2xₙ).
    #[test]
    fn gradient_of_sum_of_squares(
        x in prop::collection::vec(-10.0_f64..10.0_f64, 1..=5),
    ) {
        let n = x.len();
        let f = move |p: &[f64]| p.iter().map(|&xi| xi * xi).sum::<f64>();
        let g = gradient(&f, &x);
        prop_assert_eq!(g.len(), n);
        for i in 0..n {
            let expected = 2.0 * x[i];
            prop_assert!((g[i] - expected).abs() < 1e-4, "dim {i}: got {}, expected {}", g[i], expected);
        }
    }

    /// Laplacian of x² + y² + z² is 6.
    #[test]
    fn laplacian_of_sum_of_squares(
        x in prop::collection::vec(-5.0_f64..5.0_f64, 2..=4),
    ) {
        let f = |p: &[f64]| p.iter().map(|&xi| xi * xi).sum::<f64>();
        let lap = laplacian(&f, &x);
        let expected = 2.0 * x.len() as f64;
        prop_assert!((lap - expected).abs() < 1e-2, "laplacian: {lap} vs {expected}");
    }

    /// Directional derivative in zero direction is zero.
    #[test]
    fn directional_derivative_zero_direction(
        x in prop::collection::vec(-5.0_f64..5.0_f64, 1..=3),
    ) {
        let f = |p: &[f64]| p.iter().map(|&xi| xi * xi).sum::<f64>();
        let zero = vec![0.0; x.len()];
        let result = directional_derivative(&f, &x, &zero).unwrap();
        prop_assert!(result.abs() < 1e-15, "zero dir: {result}");
    }

    /// RK4 for dy/dt = 0 stays constant.
    #[test]
    fn ode_constant_solution(
        y0 in -10.0_f64..10.0_f64,
    ) {
        let result = runge_kutta_4(&|_, _| 0.0, 0.0, y0, 1.0, 10).unwrap();
        for (_, y) in &result {
            prop_assert!((y - y0).abs() < 1e-12, "constant solution drifted: {y} vs {y0}");
        }
    }

    /// Discrete gradient of linear function is constant.
    #[test]
    fn discrete_gradient_linear(
        slope in -5.0_f64..5.0_f64,
        intercept in -5.0_f64..5.0_f64,
    ) {
        if slope.abs() < 0.01 {
            return Ok(());
        }
        let n = 10;
        let dx = 0.5;
        let y: Vec<f64> = (0..n).map(|i| slope * (i as f64 * dx) + intercept).collect();
        let g = discrete_gradient(&y, dx);
        for &val in &g {
            prop_assert!((val - slope).abs() < 1e-6, "gradient: {val} vs {slope}");
        }
    }

    /// Gaussian quadrature is exact for polynomials up to degree 2n-1.
    #[test]
    fn gaussian_exactness(
        n in 1_usize..=10,
    ) {
        let degree = 2 * n - 1;
        let result = gaussian_quadrature(&|x| x.powi(degree as i32), -1.0, 1.0, n).unwrap();
        let expected = if degree % 2 == 0 { 2.0 / (degree as f64 + 1.0) } else { 0.0 };
        prop_assert!((result - expected).abs() < 1e-10, "n={n}, deg={degree}: {result} vs {expected}");
    }
}
