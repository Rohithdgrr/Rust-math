//! Integration tests for mathverse-calculus.
//!
//! Tests the public API across modules, verifying cross-module workflows
//! and real-world usage patterns.

use mathverse_calculus::prelude::*;
use core::f64::consts::PI;

mod derivative_integration {
    use super::*;

    #[test]
    fn derivative_then_integrate_roundtrip() {
        // d/dx(x³) = 3x², then ∫₀² 3x² dx = 8
        let d = |x: f64| derivative(&|t| t * t * t, x);
        let integral = integrate(&d, 0.0, 2.0, 1e-8);
        assert!((integral - 8.0).abs() < 1e-4, "roundtrip: {integral}");
    }

    #[test]
    fn partial_then_gradient_consistency() {
        let f = |x: &[f64]| x[0] * x[0] * x[1];
        let p0 = partial_derivative(&f, &[2.0, 3.0], 0);
        let p1 = partial_derivative(&f, &[2.0, 3.0], 1);
        let g = gradient(&f, &[2.0, 3.0]);
        assert!((g[0] - p0).abs() < 1e-10);
        assert!((g[1] - p1).abs() < 1e-10);
    }

    #[test]
    fn nth_derivative_of_polynomial() {
        // 5th derivative of x^5 should be 120
        let (v, e) = nth_derivative(&|x| x.powi(5), 1.0, 5);
        assert!((v - 120.0).abs() < e * 10.0 + 0.01, "5th deriv: {v} ± {e}");
    }
}

mod integration_integration {
    use super::*;

    #[test]
    fn all_methods_agree_on_polynomial() {
        // ∫₀¹ x³ dx = 0.25
        let f = &|x: f64| x * x * x;
        let expected = 0.25;

        let trap = trapezoid(f, 0.0, 1.0, 1000).unwrap();
        let simp = simpson(f, 0.0, 1.0, 100).unwrap();
        let adap = integrate(f, 0.0, 1.0, 1e-10);
        let gauss = gaussian_quadrature(f, 0.0, 1.0, 5).unwrap();
        let rom = romberg(f, 0.0, 1.0, 10, 1e-12).unwrap();

        assert!((trap - expected).abs() < 1e-5, "trapezoid: {trap}");
        assert!((simp - expected).abs() < 1e-8, "simpson: {simp}");
        assert!((adap - expected).abs() < 1e-8, "adaptive: {adap}");
        assert!((gauss - expected).abs() < 1e-10, "gauss: {gauss}");
        assert!((rom - expected).abs() < 1e-10, "romberg: {rom}");
    }

    #[test]
    fn gaussian_exact_for_polynomials() {
        // 5-point Gauss is exact for degree ≤ 9
        for deg in 0..=9 {
            let result = gaussian_quadrature(&|x| x.powi(deg), -1.0, 1.0, 5).unwrap();
            let expected = if deg % 2 == 0 { 2.0 / (deg as f64 + 1.0) } else { 0.0 };
            assert!((result - expected).abs() < 1e-12, "degree {deg}: {result} vs {expected}");
        }
    }

    #[test]
    fn integration_error_cases() {
        assert!(trapezoid(&|x| x, 0.0, 1.0, 0).is_err());
        assert!(simpson(&|x| x, 0.0, 1.0, 0).is_err());
        assert!(romberg(&|x| x, 0.0, 1.0, 0, 1e-10).is_err());
        assert!(gaussian_quadrature(&|x| x, 0.0, 1.0, 0).is_err());
    }
}

mod ode_integration {
    use super::*;

    #[test]
    fn exponential_growth_all_methods() {
        // dy/dt = y, y(0) = 1 → y(1) = e
        let expected = 1.0_f64.exp();

        let euler = euler(&|_, y| y, 0.0, 1.0, 1.0, 1000).unwrap();
        let mid = midpoint(&|_, y| y, 0.0, 1.0, 1.0, 100).unwrap();
        let rk4 = runge_kutta_4(&|_, y| y, 0.0, 1.0, 1.0, 10).unwrap();

        assert!((euler.last().unwrap().1 - expected).abs() < 0.02, "euler");
        assert!((mid.last().unwrap().1 - expected).abs() < 0.001, "midpoint");
        assert!((rk4.last().unwrap().1 - expected).abs() < 1e-5, "rk4");
    }

    #[test]
    fn harmonic_oscillator_period() {
        // d²x/dt² = -x, period = 2π
        let f = |_: f64, y: &[f64]| vec![y[1], -y[0]];
        let result = runge_kutta_4_system(&f, 0.0, &[1.0, 0.0], 2.0 * PI, 1000).unwrap();
        let y_final = &result.last().unwrap().1;
        assert!((y_final[0] - 1.0).abs() < 1e-4, "position after 1 period");
        assert!((y_final[1] - 0.0).abs() < 1e-4, "velocity after 1 period");
    }

    #[test]
    fn ode_builder_matches_direct() {
        let direct = runge_kutta_4(&|_, y| -y, 0.0, 1.0, 1.0, 50).unwrap();
        let builder = OdeProblem::new(&|_, y| -y, (0.0, 1.0), 1.0)
            .method(OdeMethod::Rk4)
            .steps(50)
            .solve()
            .unwrap();
        assert_eq!(direct.len(), builder.len());
        for (a, b) in direct.iter().zip(builder.iter()) {
            assert!((a.1 - b.1).abs() < 1e-12);
        }
    }

    #[test]
    fn ode_error_cases() {
        assert!(euler(&|_, y| y, 0.0, 1.0, 1.0, 0).is_err());
        assert!(runge_kutta_4(&|_, y| y, 0.0, 1.0, 1.0, 0).is_err());
        assert!(midpoint(&|_, y| y, 0.0, 1.0, 1.0, 0).is_err());
        assert!(runge_kutta_4_system(&|_, y: &[f64]| y.to_vec(), 0.0, &[1.0], 1.0, 0).is_err());
    }
}

mod vector_calculus_integration {
    use super::*;

    #[test]
    fn gradient_is_inverse_of_directional() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let x = &[1.0, 2.0];
        let grad = gradient(&f, x);

        // Directional derivative in direction of gradient should equal |grad|
        let dir_div = directional_derivative(&f, x, &grad).unwrap();
        let grad_norm = (grad[0] * grad[0] + grad[1] * grad[1]).sqrt();
        assert!((dir_div - grad_norm).abs() < 1e-6);
    }

    #[test]
    fn divergence_of_gradient_is_laplacian() {
        // For f(x,y) = x² + y²: ∇·∇f = ∇²f = 4
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let x = &[1.0, 2.0, 3.0];

        let grad_f = |p: &[f64]| gradient(&f, p);
        let div_grad = divergence(&grad_f, x).unwrap();
        let lap = laplacian(&f, x);

        assert!((div_grad - lap).abs() < 1e-2, "div(grad)={div_grad} vs lap={lap}");
    }

    #[test]
    fn jacobian_of_linear_map() {
        // F(x,y) = (2x + 3y, 4x + 5y) → J = [[2,3],[4,5]]
        let f = |x: &[f64]| vec![2.0 * x[0] + 3.0 * x[1], 4.0 * x[0] + 5.0 * x[1]];
        let j = jacobian(&f, &[0.0, 0.0]);
        assert!((j[0] - 2.0).abs() < 1e-10);
        assert!((j[1] - 3.0).abs() < 1e-10);
        assert!((j[2] - 4.0).abs() < 1e-10);
        assert!((j[3] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn hessian_symmetry() {
        let f = |x: &[f64]| x[0] * x[0] * x[1] + x[1] * x[1] * x[0];
        let h = hessian(&f, &[1.0, 2.0]);
        // H[0][1] should equal H[1][0]
        assert!((h[1] - h[2]).abs() < 1e-10, "H[0][1]={} vs H[1][0]={}", h[1], h[2]);
    }
}

mod root_finding_integration {
    use super::*;

    #[test]
    fn newton_auto_finds_roots() {
        let root = newton_raphson_auto(&|x| x * x - 9.0, 5.0, 1e-12, 100).unwrap();
        assert!((root - 3.0).abs() < 1e-8, "root: {root}");
    }

    #[test]
    fn critical_point_of_cubic() {
        // f(x) = x³ - 3x has critical points at x = ±1
        let f = |x: f64| x * x * x - 3.0 * x;
        let cp1 = find_critical_point(&f, 0.5, 1e-10, 100).unwrap();
        let cp2 = find_critical_point(&f, -0.5, 1e-10, 100).unwrap();
        assert!((cp1 - 1.0).abs() < 1e-8, "positive cp: {cp1}");
        assert!((cp2 + 1.0).abs() < 1e-8, "negative cp: {cp2}");
    }
}

mod cross_module_workflows {
    use super::*;

    #[test]
    fn optimize_using_gradient_and_hessian() {
        // Minimize f(x,y) = (x-3)² + (y-4)² using Newton's method
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2);
        let mut x = vec![0.0, 0.0];

        for _ in 0..20 {
            let g = gradient(&f, &x);
            let h = hessian(&f, &x);
            // Newton step: x -= H^{-1} ∇f (for diagonal H, this is simple)
            x[0] -= g[0] / h[0];
            x[1] -= g[1] / h[3];
        }

        assert!((x[0] - 3.0).abs() < 1e-6, "x: {}", x[0]);
        assert!((x[1] - 4.0).abs() < 1e-6, "y: {}", x[1]);
    }

    #[test]
    fn derivative_matches_ode_for_autonomous() {
        // dy/dx = 2x at x=3 should be 6
        // Also: solve dy/dx = 2x with y(0)=0, then y(3) = 9
        let deriv = derivative(&|x| x * x, 3.0);
        assert!((deriv - 6.0).abs() < 1e-6);

        let _sol = runge_kutta_4(&|_, y| 2.0 * y, 0.0, 0.0, 3.0, 100).unwrap();
        // Actually this solves dy/dx = 2y, not dy/dx = 2x
        // Let me fix: dy/dx = 2x → y = x²
        let sol2 = runge_kutta_4(&|x, _| 2.0 * x, 0.0, 0.0, 3.0, 100).unwrap();
        let y_final = sol2.last().unwrap().1;
        assert!((y_final - 9.0).abs() < 0.01, "ODE solution: {y_final}");
    }
}
