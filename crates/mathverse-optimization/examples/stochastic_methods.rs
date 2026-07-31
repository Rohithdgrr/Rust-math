//! Stochastic optimization examples.
//!
//! This example demonstrates simulated annealing and genetic algorithms.

use mathverse_optimization::{simulated_annealing, genetic, AnnealingConfig, GeneticConfig};
use mathverse_probability::Rng;

fn main() {
    // Minimize f(x) = x² + y²
    let f = |x: &[f64]| x.iter().map(|v| v * v).sum();
    
    println!("Minimizing f(x) = x² + y² using stochastic methods");
    
    // Simulated Annealing
    let mut rng = Rng::new(42);
    let bounds = [(-10.0, 10.0), (-10.0, 10.0)];
    let sa_cfg = AnnealingConfig::default();
    let sa_best = simulated_annealing(&f, &bounds, &sa_cfg, &mut rng);
    println!("Simulated Annealing result: {:?}", sa_best);
    println!("Final value: {:.6}", f(&sa_best));
    
    // Genetic Algorithm
    let mut rng = Rng::new(7);
    let ga_cfg = GeneticConfig::new(vec![(-5.0, 5.0), (-5.0, 5.0)]);
    let ga_best = genetic(&f, &ga_cfg, &mut rng);
    println!("Genetic Algorithm result: {:?}", ga_best);
    println!("Final value: {:.6}", f(&ga_best));
}
