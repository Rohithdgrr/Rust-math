//! Gradient-based optimization examples.
//!
//! Demonstrates gradient descent, SGD with momentum, Adam, and RMSProp on the
//! convex quadratic `f(x) = x² + y²`.

use mathverse_optimization::{adam, gradient_descent, rmsprop, sgd};

fn main() {
    // Minimize f(x) = x² + y², gradient = (2x, 2y)
    let grad = |x: &[f64]| x.iter().map(|v| 2.0 * v).collect::<Vec<f64>>();

    println!("Minimizing f(x) = x² + y²");
    println!("Starting point: [10.0, -10.0]\n");

    // Gradient Descent
    let gd = gradient_descent(&grad, &[10.0, -10.0], 0.1, 1e-10, 10_000);
    println!("Gradient Descent result: {gd:?}");
    println!("Final value: {:.6}\n", gd.iter().map(|v| v * v).sum::<f64>());

    // SGD with momentum
    let sgd = sgd(&grad, &[10.0, -10.0], 0.1, 0.9, 1e-10, 10_000);
    println!("SGD (momentum 0.9) result: {sgd:?}");
    println!("Final value: {:.6}\n", sgd.iter().map(|v| v * v).sum::<f64>());

    // Adam
    let adam = adam(&grad, &[10.0, -10.0], 0.1, 0.9, 0.999, 1e-8, 1e-10, 10_000);
    println!("Adam result: {adam:?}");
    println!("Final value: {:.6}\n", adam.iter().map(|v| v * v).sum::<f64>());

    // RMSProp
    let rms = rmsprop(&grad, &[10.0, -10.0], 0.01, 0.99, 1e-8, 1e-10, 10_000);
    println!("RMSProp result: {rms:?}");
    println!("Final value: {:.6}", rms.iter().map(|v| v * v).sum::<f64>());
}
