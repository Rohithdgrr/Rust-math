//! Stochastic optimization examples.
//!
//! Demonstrates simulated annealing and a genetic algorithm on the convex
//! quadratic `f(x) = x² + y²`.

use mathverse_optimization::{genetic_algorithm, simulated_annealing};

fn main() {
    // Minimize f(x) = x² + y²
    let f = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();

    println!("Minimizing f(x) = x² + y² using stochastic methods\n");

    // Simulated Annealing (seed 42)
    let bounds = [(-10.0, 10.0), (-10.0, 10.0)];
    let sa_best = simulated_annealing(&f, &bounds, 100.0, 1e-4, 1.0, 1000, 42);
    println!("Simulated Annealing result: {sa_best:?}");
    println!("Final value: {:.6}\n", f(&sa_best));

    // Genetic Algorithm (seed 7)
    let ga_best = genetic_algorithm(&f, &bounds, 200, 500, 0.05, 7);
    println!("Genetic Algorithm result: {ga_best:?}");
    println!("Final value: {:.6}", f(&ga_best));
}
