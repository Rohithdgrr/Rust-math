//! Combinatorial optimizers: simulated annealing, genetic algorithm, particle swarm.

/// Simulated annealing with exponential cooling schedule.
pub fn simulated_annealing(f: &dyn Fn(&[f64]) -> f64, bounds: &[(f64, f64)], t0: f64, t_min: f64, step: f64, iters_per_t: usize, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    let mut next_rng = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let mut x: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * (next_rng() as f64 / u64::MAX as f64)).collect();
    let mut best = x.clone();
    let mut fx = f(&x);
    let mut t = t0;
    while t > t_min {
        for _ in 0..iters_per_t {
            let mut cand = x.clone();
            for (i, &(lo, hi)) in bounds.iter().enumerate() {
                let r = (next_rng() as f64 / u64::MAX as f64 - 0.5) * 2.0;
                cand[i] = (cand[i] + step * r).clamp(lo, hi);
            }
            let fc = f(&cand);
            let d = fc - fx;
            if d < 0.0 || (next_rng() as f64 / u64::MAX as f64) < (-d / t).exp() {
                x = cand; fx = fc;
                if fx < f(&best) { best = x.clone(); }
            }
        }
        t *= 0.95;
    }
    best
}

pub fn genetic_algorithm(f: &dyn Fn(&[f64]) -> f64, bounds: &[(f64, f64)], pop_size: usize, generations: usize, mutation_rate: f64, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    let mut next_rng = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let _d = bounds.len();
    let mut pop: Vec<(Vec<f64>, f64)> = (0..pop_size).map(|_| {
        let x: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * (next_rng() as f64 / u64::MAX as f64)).collect();
        let val = f(&x);
        (x, val)
    }).collect();
    for _ in 0..generations {
        pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let elite = (pop_size / 4).max(1);
        let mut next_pop: Vec<(Vec<f64>, f64)> = pop[..elite].to_vec();
        while next_pop.len() < pop_size {
            let a = &pop[(next_rng() as usize) % elite].0;
            let b = &pop[(next_rng() as usize) % elite].0;
            let mut child: Vec<f64> = a.iter().zip(b).map(|(x, y)| if next_rng() % 2 == 0 { *x } else { *y }).collect();
            if (next_rng() as f64 / u64::MAX as f64) < mutation_rate {
                for (i, &(lo, hi)) in bounds.iter().enumerate() {
                    child[i] = (child[i] + 0.1 * (next_rng() as f64 / u64::MAX as f64 - 0.5) * 2.0).clamp(lo, hi);
                }
            }
            let val = f(&child);
            next_pop.push((child, val));
        }
        pop = next_pop;
    }
    pop.into_iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
}

pub fn particle_swarm(f: &dyn Fn(&[f64]) -> f64, bounds: &[(f64, f64)], n_particles: usize, max_iters: usize, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    let mut next_rng = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let d = bounds.len();
    let mut pos: Vec<Vec<f64>> = (0..n_particles).map(|_| bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * (next_rng() as f64 / u64::MAX as f64)).collect()).collect();
    let mut vel: Vec<Vec<f64>> = vec![vec![0.0; d]; n_particles];
    let mut best_pos = pos.clone();
    let mut best_val: Vec<f64> = pos.iter().map(|p| f(p)).collect();
    let mut global_best = best_pos[0].clone();
    let mut global_val = best_val[0];
    for i in 0..n_particles { if best_val[i] < global_val { global_best = best_pos[i].clone(); global_val = best_val[i]; } }
    for _ in 0..max_iters {
        for i in 0..n_particles {
            for j in 0..d {
                let r1 = next_rng() as f64 / u64::MAX as f64;
                let r2 = next_rng() as f64 / u64::MAX as f64;
                vel[i][j] = 0.7 * vel[i][j] + 1.5 * r1 * (best_pos[i][j] - pos[i][j]) + 1.5 * r2 * (global_best[j] - pos[i][j]);
                pos[i][j] = (pos[i][j] + vel[i][j]).clamp(bounds[j].0, bounds[j].1);
            }
            let val = f(&pos[i]);
            if val < best_val[i] { best_pos[i] = pos[i].clone(); best_val[i] = val; }
            if val < global_val { global_best = pos[i].clone(); global_val = val; }
        }
    }
    global_best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sa_test() {
        let best = simulated_annealing(&|x: &[f64]| (x[0]-2.0).powi(2)+(x[1]-3.0).powi(2), &[(-10.0,10.0),(-10.0,10.0)], 10.0, 0.001, 1.0, 100, 42);
        assert!((best[0]-2.0).abs() < 1.0 && (best[1]-3.0).abs() < 1.0);
    }
}
