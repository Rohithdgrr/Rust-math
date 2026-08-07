//! Advanced usage examples for mathverse-calculus.
//!
//! Demonstrates cross-module workflows: optimization, physics simulation,
//! and numerical analysis patterns.
//!
//! Run with: cargo run -p mathverse-calculus --example advanced

use mathverse_calculus::prelude::*;
use core::f64::consts::PI;

fn main() {
    println!("=== mathverse-calculus: Advanced Examples ===\n");

    gradient_descent_optimization();
    pendulum_simulation();
    newton_fractals();
    verify_fundamental_theorem();
}

/// Minimize f(x,y) = (x-2)² + (y-3)² + sin(x)cos(y) using gradient descent.
fn gradient_descent_optimization() {
    println!("1. Gradient Descent Optimization");
    println!("   Minimize: f(x,y) = (x-2)² + (y-3)² + sin(x)cos(y)");

    let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) + x[0].sin() * x[1].cos();
    let mut x = vec![0.0, 0.0];
    let lr = 0.1;

    for step in 0..100 {
        let g = gradient(&f, &x);
        let grad_norm = (g[0] * g[0] + g[1] * g[1]).sqrt();
        if grad_norm < 1e-8 {
            println!("   Converged at step {step}: x = ({:.8}, {:.8})", x[0], x[1]);
            break;
        }
        x[0] -= lr * g[0];
        x[1] -= lr * g[1];

        if step % 20 == 0 {
            let fx = f(&x);
            println!("   Step {step:3}: x=({:.6}, {:.6}), f={:.8}", x[0], x[1], fx);
        }
    }
    println!();
}

/// Simulate a damped pendulum: d²θ/dt² = -sin(θ) - b·dθ/dt.
fn pendulum_simulation() {
    println!("2. Damped Pendulum Simulation");
    println!("   d²θ/dt² = -sin(θ) - 0.1·dθ/dt, θ(0)=π/4, ω(0)=0");

    let damping = 0.1;
    let f = move |_: f64, y: &[f64]| vec![y[1], -y[0].sin() - damping * y[1]];
    let result = runge_kutta_4_system(&f, 0.0, &[PI / 4.0, 0.0], 10.0, 1000).unwrap();

    // Print every 100th step
    for (i, (t, state)) in result.iter().step_by(100).enumerate() {
        let energy = 0.5 * state[1] * state[1] + (1.0 - state[0].cos());
        println!("   t={:5.2}: θ={:8.5}, ω={:8.5}, E={:.6}", t, state[0], state[1], energy);
    }
    println!();
}

/// Use Newton's method to find roots of z³ - 1 in the real line,
/// demonstrating convergence basins.
fn newton_fractals() {
    println!("3. Newton's Method: Roots of x³ - 1 = 0");
    println!("   (Real root: x=1, complex roots not found by real Newton)");

    let f = |x: f64| x * x * x - 1.0;
    let starts = vec![-2.0, -1.0, -0.5, 0.5, 1.5, 2.0, 5.0];

    for x0 in starts {
        match newton_raphson_auto(&f, x0, 1e-12, 100) {
            Ok(root) => println!("   x₀={:5.2} → root={:.12}", x0, root),
            Err(e) => println!("   x₀={:5.2} → failed: {}", x0, e),
        }
    }
    println!();
}

/// Verify the Fundamental Theorem of Calculus:
/// d/dx ∫ₐˣ f(t) dt = f(x).
fn verify_fundamental_theorem() {
    println!("4. Fundamental Theorem of Calculus");
    println!("   Verify: d/dx ∫₀ˣ sin(t)·exp(-t²) dt = sin(x)·exp(-x²)");

    let f = |t: f64| t.sin() * (-t * t).exp();
    let test_points = vec![0.0, 0.5, 1.0, 1.5, 2.0];

    for x in test_points {
        // Compute d/dx ∫₀ˣ f(t) dt numerically
        let integral = |x: f64| integrate(&f, 0.0, x, 1e-10);
        let lhs = derivative(&integral, x);
        let rhs = f(x);
        let error = (lhs - rhs).abs();
        println!("   x={:.1}: d/dx∫f={:.8}, f(x)={:.8}, error={:.2e}", x, lhs, rhs, error);
    }
    println!();
}
