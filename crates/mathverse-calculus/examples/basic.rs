//! Basic usage examples for mathverse-calculus.
//!
//! Run with: cargo run -p mathverse-calculus --example basic

use mathverse_calculus::prelude::*;
use core::f64::consts::PI;

fn main() {
    println!("=== mathverse-calculus: Basic Examples ===\n");

    // --- Derivatives ---
    println!("Derivatives:");
    let d_sin = derivative(&f64::sin, 0.0);
    println!("  d/dx sin(x) at x=0: {:.10} (expected: 1.0)", d_sin);

    let d_poly = derivative(&|x| x * x * x, 2.0);
    println!("  d/dx x³ at x=2: {:.10} (expected: 12.0)", d_poly);

    let (d5, e5) = nth_derivative(&|x| x.powi(5), 1.0, 5);
    println!("  5th derivative of x⁵ at x=1: {:.10} ± {:.2e} (expected: 120.0)", d5, e5);

    // --- Integration ---
    println!("\nIntegration:");
    let i_sin = integrate(&f64::sin, 0.0, PI, 1e-12);
    println!("  ∫₀^π sin(x) dx: {:.10} (expected: 2.0)", i_sin);

    let i_gauss = gaussian_quadrature(&|x| x * x, 0.0, 1.0, 3).unwrap();
    println!("  ∫₀¹ x² dx (Gauss 3-pt): {:.10} (expected: 0.333...)", i_gauss);

    let i_2d = integrate_2d(&|x, y| x * y, 0.0, 1.0, 0.0, 1.0, 5).unwrap();
    println!("  ∫₀¹∫₀¹ x·y dx dy: {:.10} (expected: 0.25)", i_2d);

    // --- ODEs ---
    println!("\nODE Solvers:");
    let sol = OdeProblem::new(&|_, y| y, (0.0, 1.0), 1.0)
        .method(OdeMethod::Rk4)
        .steps(100)
        .solve()
        .unwrap();
    println!("  dy/dt=y, y(0)=1 at t=1: {:.10} (expected: e = {:.10})",
        sol.last().unwrap().1, 1.0_f64.exp());

    // Harmonic oscillator
    let osc = runge_kutta_4_system(
        &|_, y| vec![y[1], -y[0]],
        0.0, &[1.0, 0.0], 2.0 * PI, 1000,
    ).unwrap();
    let final_state = &osc.last().unwrap().1;
    println!("  Harmonic oscillator after 2π: x={:.6}, v={:.6} (expected: 1.0, 0.0)",
        final_state[0], final_state[1]);

    // --- Vector Calculus ---
    println!("\nVector Calculus:");
    let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let grad = gradient(&f, &[3.0, 4.0]);
    println!("  ∇(x²+y²) at (3,4): ({:.6}, {:.6}) (expected: 6, 8)", grad[0], grad[1]);

    let lap = laplacian(&f, &[1.0, 2.0]);
    println!("  ∇²(x²+y²) at (1,2): {:.6} (expected: 4)", lap);

    let dir = directional_derivative(&f, &[1.0, 2.0], &[1.0, 0.0]).unwrap();
    println!("  D_(1,0)(x²+y²) at (1,2): {:.6} (expected: 2)", dir);

    // --- Root Finding ---
    println!("\nRoot Finding:");
    let root = newton_raphson_auto(&|x| x * x - 2.0, 1.5, 1e-12, 100).unwrap();
    println!("  √2 ≈ {:.15} (expected: {:.15})", root, 2.0_f64.sqrt());

    let cp = find_critical_point(&|x| x * x * x - 3.0 * x, 0.5, 1e-10, 100).unwrap();
    println!("  Critical point of x³-3x near 0.5: {:.10} (expected: 1.0)", cp);

    println!("\n=== Done ===");
}
