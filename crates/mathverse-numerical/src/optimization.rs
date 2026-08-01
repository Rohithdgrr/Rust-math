//! Optimization methods: gradient descent, BFGS, simulated annealing, genetic algorithms.

use mathverse_core::error::{MathError, MathResult};
use rand::Rng;

/// Gradient descent optimization.
pub struct GradientDescent {
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64, max_iterations: usize, tolerance: f64) -> Self {
        GradientDescent {
            learning_rate,
            max_iterations,
            tolerance,
        }
    }

    /// Minimize function f(x) using gradient descent.
    pub fn minimize(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        grad: &dyn Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
    ) -> MathResult<(Vec<f64>, f64, usize)> {
        let mut x = x0.to_vec();
        let mut best_value = f(&x);
        let mut best_x = x.clone();
        
        for iteration in 0..self.max_iterations {
            let gradient = grad(&x);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < self.tolerance {
                return Ok((x, best_value, iteration));
            }
            
            // Update x
            for i in 0..x.len() {
                x[i] -= self.learning_rate * gradient[i];
            }
            
            let current_value = f(&x);
            
            if current_value < best_value {
                best_value = current_value;
                best_x = x.clone();
            }
        }
        
        Ok((best_x, best_value, self.max_iterations))
    }

    /// Minimize with momentum.
    pub fn minimize_with_momentum(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        grad: &dyn Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
        momentum: f64,
    ) -> MathResult<(Vec<f64>, f64, usize)> {
        let mut x = x0.to_vec();
        let mut velocity = vec![0.0; x.len()];
        let mut best_value = f(&x);
        let mut best_x = x.clone();
        
        for iteration in 0..self.max_iterations {
            let gradient = grad(&x);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < self.tolerance {
                return Ok((x, best_value, iteration));
            }
            
            // Update velocity and position
            for i in 0..x.len() {
                velocity[i] = momentum * velocity[i] - self.learning_rate * gradient[i];
                x[i] += velocity[i];
            }
            
            let current_value = f(&x);
            
            if current_value < best_value {
                best_value = current_value;
                best_x = x.clone();
            }
        }
        
        Ok((best_x, best_value, self.max_iterations))
    }
}

/// BFGS (Broyden-Fletcher-Goldfarb-Shanno) quasi-Newton method.
pub struct BFGS {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl BFGS {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        BFGS {
            max_iterations,
            tolerance,
        }
    }

    /// Minimize using BFGS.
    pub fn minimize(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        grad: &dyn Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
    ) -> MathResult<(Vec<f64>, f64, usize)> {
        let n = x0.len();
        let mut x = x0.to_vec();
        let mut hessian_inv = vec![vec![0.0; n]; n];
        
        // Initialize Hessian inverse as identity
        for i in 0..n {
            hessian_inv[i][i] = 1.0;
        }
        
        let mut grad_old = grad(&x);
        let mut best_value = f(&x);
        let mut best_x = x.clone();
        
        for iteration in 0..self.max_iterations {
            let grad_norm: f64 = grad_old.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < self.tolerance {
                return Ok((x, best_value, iteration));
            }
            
            // Compute search direction: p = -H * g
            let mut p = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    p[i] -= hessian_inv[i][j] * grad_old[j];
                }
            }
            
            // Line search (simplified backtracking)
            let alpha = self.line_search(f, &x, &p, &grad_old)?;
            
            // Update x
            let x_old = x.clone();
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            
            let grad_new = grad(&x);
            let current_value = f(&x);
            
            if current_value < best_value {
                best_value = current_value;
                best_x = x.clone();
            }
            
            // Update Hessian inverse using BFGS formula
            let s: Vec<f64> = x.iter().zip(&x_old).map(|(&xi, &xoi)| xi - xoi).collect();
            let y: Vec<f64> = grad_new.iter().zip(&grad_old).map(|(&gi, &goi)| gi - goi).collect();
            
            let ys: f64 = y.iter().zip(&s).map(|(&yi, &si)| yi * si).sum();
            
            if ys > 1e-15 {
                // BFGS update
                let mut st_y = vec![vec![0.0; n]; n];
                let mut y_st = vec![vec![0.0; n]; n];
                
                for i in 0..n {
                    for j in 0..n {
                        st_y[i][j] = s[i] * y[j] / ys;
                        y_st[i][j] = y[i] * s[j] / ys;
                    }
                }
                
                // H_new = (I - st_y) * H * (I - y_st) + st_y
                let mut h_new = vec![vec![0.0; n]; n];
                
                for i in 0..n {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for k in 0..n {
                            let mut temp1 = 0.0;
                            let mut temp2 = 0.0;
                            for l in 0..n {
                                temp1 += (if k == l { 1.0 } else { 0.0 } - st_y[k][l]) * hessian_inv[l][j];
                                temp2 += hessian_inv[i][l] * (if l == k { 1.0 } else { 0.0 } - y_st[l][k]);
                            }
                            sum += temp1 * (if k == j { 1.0 } else { 0.0 } - y_st[k][j]);
                        }
                        h_new[i][j] = sum + st_y[i][j];
                    }
                }
                
                hessian_inv = h_new;
            }
            
            grad_old = grad_new;
        }
        
        Ok((best_x, best_value, self.max_iterations))
    }

    fn line_search(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        x: &[f64],
        p: &[f64],
        grad: &[f64],
    ) -> MathResult<f64> {
        let mut alpha = 1.0;
        let rho = 0.5;
        let c1 = 0.1;
        
        for _ in 0..20 {
            let x_new: Vec<f64> = x.iter().zip(p).map(|(&xi, &pi)| xi + alpha * pi).collect();
            let f_new = f(&x_new);
            let f_x = f(x);
            
            let directional_derivative: f64 = grad.iter().zip(p).map(|(&gi, &pi)| gi * pi).sum();
            
            if f_new <= f_x + c1 * alpha * directional_derivative {
                return Ok(alpha);
            }
            
            alpha *= rho;
        }
        
        Ok(alpha)
    }
}

/// Simulated annealing optimization.
pub struct SimulatedAnnealing {
    pub initial_temp: f64,
    pub cooling_rate: f64,
    pub min_temp: f64,
    pub max_iterations: usize,
}

impl SimulatedAnnealing {
    pub fn new(initial_temp: f64, cooling_rate: f64, min_temp: f64, max_iterations: usize) -> Self {
        SimulatedAnnealing {
            initial_temp,
            cooling_rate,
            min_temp,
            max_iterations,
        }
    }

    /// Minimize using simulated annealing.
    pub fn minimize(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        x0: &[f64],
        bounds: &[(f64, f64)],
        rng: &mut impl rand::Rng,
    ) -> (Vec<f64>, f64) {
        let mut x = x0.to_vec();
        let mut current_value = f(&x);
        let mut best_x = x.clone();
        let mut best_value = current_value;
        let mut temp = self.initial_temp;
        
        for _ in 0..self.max_iterations {
            if temp < self.min_temp {
                break;
            }
            
            // Generate neighbor
            let mut neighbor = x.clone();
            for i in 0..neighbor.len() {
                let range = bounds[i].1 - bounds[i].0;
                neighbor[i] += (rng.gen::<f64>() - 0.5) * range * 0.1 * temp / self.initial_temp;
                neighbor[i] = neighbor[i].max(bounds[i].0).min(bounds[i].1);
            }
            
            let neighbor_value = f(&neighbor);
            let delta = neighbor_value - current_value;
            
            // Accept or reject
            if delta < 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                x = neighbor;
                current_value = neighbor_value;
                
                if current_value < best_value {
                    best_value = current_value;
                    best_x = x.clone();
                }
            }
            
            temp *= self.cooling_rate;
        }
        
        (best_x, best_value)
    }
}

/// Genetic algorithm optimization.
pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub max_generations: usize,
}

impl GeneticAlgorithm {
    pub fn new(population_size: usize, mutation_rate: f64, crossover_rate: f64, max_generations: usize) -> Self {
        GeneticAlgorithm {
            population_size,
            mutation_rate,
            crossover_rate,
            max_generations,
        }
    }

    /// Minimize using genetic algorithm.
    pub fn minimize(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        bounds: &[(f64, f64)],
        rng: &mut impl rand::Rng,
    ) -> (Vec<f64>, f64) {
        let dim = bounds.len();
        
        // Initialize population
        let mut population: Vec<Vec<f64>> = (0..self.population_size)
            .map(|_| {
                bounds.iter().map(|&(min, max)| rng.gen::<f64>() * (max - min) + min).collect()
            })
            .collect();
        
        let mut best_individual = population[0].clone();
        let mut best_fitness = f(&best_individual);
        
        for _generation in 0..self.max_generations {
            // Evaluate fitness
            let fitness: Vec<f64> = population.iter().map(|ind| f(ind)).collect();
            
            // Find best
            for (i, &fit) in fitness.iter().enumerate() {
                if fit < best_fitness {
                    best_fitness = fit;
                    best_individual = population[i].clone();
                }
            }
            
            // Selection (tournament)
            let mut new_population = Vec::new();
            
            while new_population.len() < self.population_size {
                let parent1 = Self::tournament_select(&population, &fitness, rng);
                let parent2 = Self::tournament_select(&population, &fitness, rng);
                
                // Crossover
                let mut child = if rng.gen::<f64>() < self.crossover_rate {
                    Self::crossover(&parent1, &parent2, rng)
                } else {
                    parent1.clone()
                };
                
                // Mutation
                for i in 0..dim {
                    if rng.gen::<f64>() < self.mutation_rate {
                        let range = bounds[i].1 - bounds[i].0;
                        child[i] += (rng.gen::<f64>() - 0.5) * range * 0.1;
                        child[i] = child[i].max(bounds[i].0).min(bounds[i].1);
                    }
                }
                
                new_population.push(child);
            }
            
            population = new_population;
        }
        
        (best_individual, best_fitness)
    }

    fn tournament_select(population: &[Vec<f64>], fitness: &[f64], rng: &mut impl rand::Rng) -> Vec<f64> {
        let tournament_size = 3;
        let mut best = population[rng.gen_range(0..population.len())].clone();
        let mut best_fit = f64::INFINITY;
        
        for _ in 0..tournament_size {
            let idx = rng.gen_range(0..population.len());
            if fitness[idx] < best_fit {
                best_fit = fitness[idx];
                best = population[idx].clone();
            }
        }
        
        best
    }

    fn crossover(parent1: &[f64], parent2: &[f64], rng: &mut impl rand::Rng) -> Vec<f64> {
        let crossover_point = rng.gen_range(1..parent1.len());
        let mut child = parent1[..crossover_point].to_vec();
        child.extend_from_slice(&parent2[crossover_point..]);
        child
    }
}

/// Nelder-Mead simplex method (derivative-free optimization).
pub struct NelderMead {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl NelderMead {
    pub fn new(tolerance: f64, max_iterations: usize) -> Self {
        NelderMead {
            tolerance,
            max_iterations,
            alpha: 1.0,
            beta: 2.0,
            gamma: 0.5,
        }
    }

    /// Minimize using Nelder-Mead.
    pub fn minimize(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        x0: &[f64],
    ) -> MathResult<(Vec<f64>, f64)> {
        let n = x0.len();
        
        // Initialize simplex
        let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
        
        for i in 0..n {
            let mut vertex = x0.to_vec();
            vertex[i] += 0.1;
            simplex.push(vertex);
        }
        
        let mut best_value = f(x0);
        let mut best_point = x0.to_vec();
        
        for _iteration in 0..self.max_iterations {
            // Evaluate all vertices
            let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
            
            // Sort by function value
            let mut indices: Vec<usize> = (0..values.len()).collect();
            indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
            
            let worst = indices[n];
            let second_worst = indices[n - 1];
            let best = indices[0];
            
            // Check convergence
            let simplex_size: f64 = simplex.iter()
                .map(|v| {
                    let diff: f64 = v.iter().zip(&simplex[best]).map(|(&vi, &bi)| (vi - bi).powi(2)).sum();
                    diff.sqrt()
                })
                .sum::<f64>() / (n + 1) as f64;
            
            if simplex_size < self.tolerance {
                return Ok((simplex[best].clone(), values[best]));
            }
            
            // Centroid (excluding worst)
            let centroid: Vec<f64> = (0..=n)
                .filter(|&i| i != worst)
                .map(|i| {
                    simplex[i].iter().map(|&v| v / n as f64).sum::<f64>()
                })
                .collect();
            
            // Reflection
            let reflected: Vec<f64> = centroid.iter()
                .zip(&simplex[worst])
                .map(|(&c, &w)| c + self.alpha * (c - w))
                .collect();
            
            let reflected_value = f(&reflected);
            
            if reflected_value < values[best] {
                // Expansion
                let expanded: Vec<f64> = centroid.iter()
                    .zip(&reflected)
                    .map(|(&c, &r)| c + self.beta * (r - c))
                    .collect();
                
                let expanded_value = f(&expanded);
                
                if expanded_value < reflected_value {
                    simplex[worst] = expanded;
                } else {
                    simplex[worst] = reflected;
                }
            } else if reflected_value < values[second_worst] {
                simplex[worst] = reflected;
            } else {
                // Contraction
                let contracted: Vec<f64> = centroid.iter()
                    .zip(&simplex[worst])
                    .map(|(&c, &w)| c + self.gamma * (w - c))
                    .collect();
                
                let contracted_value = f(&contracted);
                
                if contracted_value < values[worst] {
                    simplex[worst] = contracted;
                } else {
                    // Shrink
                    for i in 1..=n {
                        simplex[i] = simplex[best].iter()
                            .zip(&simplex[i])
                            .map(|(&b, &v)| b + self.gamma * (v - b))
                            .collect();
                    }
                }
            }
            
            // Update best
            if values[best] < best_value {
                best_value = values[best];
                best_point = simplex[best].clone();
            }
        }
        
        Ok((best_point, best_value))
    }
}

/// Particle swarm optimization.
pub struct ParticleSwarm {
    pub num_particles: usize,
    pub max_iterations: usize,
    pub inertia: f64,
    pub cognitive_weight: f64,
    pub social_weight: f64,
}

impl ParticleSwarm {
    pub fn new(num_particles: usize, max_iterations: usize) -> Self {
        ParticleSwarm {
            num_particles,
            max_iterations,
            inertia: 0.7,
            cognitive_weight: 1.5,
            social_weight: 1.5,
        }
    }

    /// Minimize using particle swarm optimization.
    pub fn minimize(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        bounds: &[(f64, f64)],
        rng: &mut impl rand::Rng,
    ) -> (Vec<f64>, f64) {
        let dim = bounds.len();
        
        // Initialize particles
        let mut positions: Vec<Vec<f64>> = (0..self.num_particles)
            .map(|_| {
                bounds.iter().map(|&(min, max)| rng.gen::<f64>() * (max - min) + min).collect()
            })
            .collect();
        
        let mut velocities: Vec<Vec<f64>> = (0..self.num_particles)
            .map(|_| vec![0.0; dim])
            .collect();
        
        let mut personal_best = positions.clone();
        let mut personal_best_values: Vec<f64> = positions.iter().map(|p| f(p)).collect();
        
        let mut global_best = personal_best[0].clone();
        let mut global_best_value = personal_best_values[0];
        
        for (i, &value) in personal_best_values.iter().enumerate() {
            if value < global_best_value {
                global_best_value = value;
                global_best = personal_best[i].clone();
            }
        }
        
        for _iteration in 0..self.max_iterations {
            for i in 0..self.num_particles {
                // Update velocity
                for j in 0..dim {
                    let r1 = rng.gen::<f64>();
                    let r2 = rng.gen::<f64>();
                    
                    velocities[i][j] = self.inertia * velocities[i][j]
                        + self.cognitive_weight * r1 * (personal_best[i][j] - positions[i][j])
                        + self.social_weight * r2 * (global_best[j] - positions[i][j]);
                    
                    // Clamp velocity
                    let max_vel = (bounds[j].1 - bounds[j].0) * 0.5;
                    velocities[i][j] = velocities[i][j].max(-max_vel).min(max_vel);
                }
                
                // Update position
                for j in 0..dim {
                    positions[i][j] += velocities[i][j];
                    positions[i][j] = positions[i][j].max(bounds[j].0).min(bounds[j].1);
                }
                
                // Update personal best
                let current_value = f(&positions[i]);
                if current_value < personal_best_values[i] {
                    personal_best_values[i] = current_value;
                    personal_best[i] = positions[i].clone();
                    
                    if current_value < global_best_value {
                        global_best_value = current_value;
                        global_best = positions[i].clone();
                    }
                }
            }
        }
        
        (global_best, global_best_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_descent() {
        let gd = GradientDescent::new(0.1, 1000, 1e-10);
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        
        let (result, value, _) = gd.minimize(&f, &grad, &[1.0, 1.0]).unwrap();
        
        assert!(value < 1e-6);
        assert!(result[0].abs() < 0.1);
        assert!(result[1].abs() < 0.1);
    }

    #[test]
    fn test_bfgs() {
        let bfgs = BFGS::new(100, 1e-10);
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        
        let (result, value, _) = bfgs.minimize(&f, &grad, &[1.0, 1.0]).unwrap();
        
        assert!(value < 1e-10);
        assert!(result[0].abs() < 1e-8);
        assert!(result[1].abs() < 1e-8);
    }

    #[test]
    fn test_simulated_annealing() {
        let sa = SimulatedAnnealing::new(100.0, 0.95, 0.01, 1000);
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let bounds = [(0.0, 3.0), (0.0, 3.0)];
        
        let mut rng = rand::thread_rng();
        let (result, value) = sa.minimize(&f, &[0.5, 0.5], &bounds, &mut rng);
        
        assert!(value < 0.1);
        assert!((result[0] - 1.0).abs() < 0.5);
        assert!((result[1] - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_genetic_algorithm() {
        let ga = GeneticAlgorithm::new(50, 0.1, 0.8, 100);
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let bounds = [(0.0, 3.0), (0.0, 3.0)];
        
        let mut rng = rand::thread_rng();
        let (result, value) = ga.minimize(&f, &bounds, &mut rng);
        
        assert!(value < 1.0);
    }

    #[test]
    fn test_nelder_mead() {
        let nm = NelderMead::new(1e-10, 1000);
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        
        let (result, value) = nm.minimize(&f, &[1.0, 1.0]).unwrap();
        
        assert!(value < 1e-8);
        assert!(result[0].abs() < 0.01);
        assert!(result[1].abs() < 0.01);
    }

    #[test]
    fn test_particle_swarm() {
        let pso = ParticleSwarm::new(30, 100);
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let bounds = [(0.0, 3.0), (0.0, 3.0)];
        
        let mut rng = rand::thread_rng();
        let (result, value) = pso.minimize(&f, &bounds, &mut rng);
        
        assert!(value < 0.1);
        assert!((result[0] - 1.0).abs() < 0.5);
        assert!((result[1] - 2.0).abs() < 0.5);
    }
}
