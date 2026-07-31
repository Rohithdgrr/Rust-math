//! Simulation and RNG: Monte Carlo simulation, event-driven simulation, random number generation utilities.

use crate::rng::Rng;

/// Monte Carlo simulation.
pub struct MonteCarloSimulation;

impl MonteCarloSimulation {
    /// Simple Monte Carlo integration.
    pub fn integrate(
        f: impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        n_samples: usize,
        rng: &mut Rng,
    ) -> (f64, f64) {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..n_samples {
            let x = a + (b - a) * rng.uniform();
            let y = f(x);
            sum += y;
            sum_sq += y * y;
        }
        
        let mean = sum / n_samples as f64;
        let variance = (sum_sq / n_samples as f64 - mean * mean) / n_samples as f64;
        let integral = mean * (b - a);
        
        (integral, variance.sqrt() * (b - a))
    }

    /// Monte Carlo estimation of probability.
    pub fn estimate_probability(
        event: impl Fn(&mut Rng) -> bool,
        n_samples: usize,
        rng: &mut Rng,
    ) -> (f64, f64) {
        let mut count = 0;
        
        for _ in 0..n_samples {
            if event(rng) {
                count += 1;
            }
        }
        
        let p = count as f64 / n_samples as f64;
        let variance = p * (1.0 - p) / n_samples as f64;
        
        (p, variance.sqrt())
    }

    /// Monte Carlo for high-dimensional integration.
    pub fn multidimensional_integrate(
        f: impl Fn(&[f64]) -> f64,
        bounds: &[(f64, f64)],
        n_samples: usize,
        rng: &mut Rng,
    ) -> (f64, f64) {
        let dim = bounds.len();
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..n_samples {
            let mut x = Vec::with_capacity(dim);
            for &(a, b) in bounds {
                x.push(a + (b - a) * rng.uniform());
            }
            
            let y = f(&x);
            sum += y;
            sum_sq += y * y;
        }
        
        let mean = sum / n_samples as f64;
        let variance = (sum_sq / n_samples as f64 - mean * mean) / n_samples as f64;
        
        let volume: f64 = bounds.iter().map(|&(a, b)| b - a).product();
        let integral = mean * volume;
        
        (integral, variance.sqrt() * volume)
    }

    /// Monte Carlo option pricing (European call).
    pub fn european_call_option(
        s0: f64,
        k: f64,
        r: f64,
        sigma: f64,
        t: f64,
        n_samples: usize,
        rng: &mut Rng,
    ) -> (f64, f64) {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..n_samples {
            let z = crate::distributions::Normal { mu: 0.0, sigma: 1.0 }.sample(rng);
            let st = s0 * ((r - 0.5 * sigma * sigma) * t + sigma * t.sqrt() * z).exp();
            let payoff = (st - k).max(0.0);
            let discounted = payoff * (-r * t).exp();
            
            sum += discounted;
            sum_sq += discounted * discounted;
        }
        
        let mean = sum / n_samples as f64;
        let variance = (sum_sq / n_samples as f64 - mean * mean) / n_samples as f64;
        
        (mean, variance.sqrt())
    }
}

/// Event-driven simulation (discrete event simulation).
pub struct EventDrivenSimulation {
    pub current_time: f64,
    pub event_queue: Vec<SimulationEvent>,
}

#[derive(Clone)]
pub struct SimulationEvent {
    pub time: f64,
    pub event_type: String,
    pub data: Vec<f64>,
}

impl EventDrivenSimulation {
    pub fn new() -> Self {
        EventDrivenSimulation {
            current_time: 0.0,
            event_queue: Vec::new(),
        }
    }

    /// Schedule an event.
    pub fn schedule(&mut self, event: SimulationEvent) {
        self.event_queue.push(event);
        self.event_queue.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Get next event.
    pub fn next_event(&mut self) -> Option<SimulationEvent> {
        if self.event_queue.is_empty() {
            return None;
        }
        
        let event = self.event_queue.remove(0);
        self.current_time = event.time;
        Some(event)
    }

    /// Run simulation until time limit.
    pub fn run_until(&mut self, time_limit: f64, event_handler: impl Fn(&SimulationEvent)) {
        while self.current_time < time_limit {
            if let Some(event) = self.next_event() {
                event_handler(&event);
            } else {
                break;
            }
        }
    }

    /// Run simulation for n events.
    pub fn run_n_events(&mut self, n_events: usize, event_handler: impl Fn(&SimulationEvent)) {
        for _ in 0..n_events {
            if let Some(event) = self.next_event() {
                event_handler(&event);
            } else {
                break;
            }
        }
    }
}

/// Random number generation utilities.
pub struct RNGUtils;

impl RNGUtils {
    /// Box-Muller transform for normal random variables.
    pub fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    /// Marsaglia polar method for normal random variables.
    pub fn marsaglia_polar(rng: &mut Rng) -> Option<(f64, f64)> {
        loop {
            let u1 = 2.0 * rng.uniform() - 1.0;
            let u2 = 2.0 * rng.uniform() - 1.0;
            let s = u1 * u1 + u2 * u2;
            
            if s < 1.0 && s > 0.0 {
                let mult = (-2.0 * s.ln() / s).sqrt();
                return Some((u1 * mult, u2 * mult));
            }
        }
    }

    /// Inverse transform sampling.
    pub fn inverse_transform(
        cdf: impl Fn(f64) -> f64,
        quantile: impl Fn(f64) -> f64,
        u: f64,
    ) -> f64 {
        quantile(u)
    }

    /// Acceptance-rejection sampling.
    pub fn acceptance_rejection(
        target_pdf: impl Fn(f64) -> f64,
        proposal_pdf: impl Fn(f64) -> f64,
        proposal_sample: impl Fn(&mut Rng) -> f64,
        m: f64,
        rng: &mut Rng,
    ) -> f64 {
        loop {
            let x = proposal_sample(rng);
            let u = rng.uniform();
            
            if u < target_pdf(x) / (m * proposal_pdf(x)) {
                return x;
            }
        }
    }

    /// Alias method for discrete distributions (simplified).
    pub fn alias_method(
        probabilities: &[f64],
        rng: &mut Rng,
    ) -> usize {
        let n = probabilities.len();
        let u = rng.uniform();
        let mut acc = 0.0;
        
        for (i, &p) in probabilities.iter().enumerate() {
            acc += p;
            if u <= acc {
                return i;
            }
        }
        
        n - 1
    }

    /// Ziggurat algorithm for normal distribution (simplified).
    pub fn ziggurat(rng: &mut Rng) -> f64 {
        // Simplified: use Box-Muller as fallback
        let u1 = rng.uniform();
        let u2 = rng.uniform();
        let (z1, _) = Self::box_muller(u1, u2);
        z1
    }
}

/// Random variate generation for specific distributions.
pub struct RandomVariateGeneration;

impl RandomVariateGeneration {
    /// Generate exponential random variate.
    pub fn exponential(lambda: f64, rng: &mut Rng) -> f64 {
        let u = rng.uniform().max(1e-300);
        -u.ln() / lambda
    }

    /// Generate Poisson random variate.
    pub fn poisson(lambda: f64, rng: &mut Rng) -> i64 {
        let l = (-lambda).exp();
        let mut k = 0i64;
        let mut p = 1.0;
        
        loop {
            k += 1;
            p *= rng.uniform();
            
            if p <= l {
                return k - 1;
            }
        }
    }

    /// Generate binomial random variate.
    pub fn binomial(n: u64, p: f64, rng: &mut Rng) -> i64 {
        let mut count = 0i64;
        
        for _ in 0..n {
            if rng.uniform() < p {
                count += 1;
            }
        }
        
        count
    }

    /// Generate geometric random variate.
    pub fn geometric(p: f64, rng: &mut Rng) -> i64 {
        let u = rng.uniform().max(1e-300);
        (u.ln() / (1.0 - p).ln()).ceil() as i64
    }

    /// Generate gamma random variate (Marsaglia-Tsang).
    pub fn gamma(shape: f64, rate: f64, rng: &mut Rng) -> f64 {
        if shape < 1.0 {
            return Self::gamma(shape + 1.0, rate, rng) * rng.uniform().powf(1.0 / shape);
        }
        
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        
        loop {
            let mut x;
            let mut v;
            
            loop {
                x = crate::distributions::Normal { mu: 0.0, sigma: 1.0 }.sample(rng);
                v = (1.0 + c * x).powi(3);
                
                if v > 0.0 {
                    break;
                }
            }
            
            let u = rng.uniform();
            
            if u < 1.0 - 0.0331 * (x * x).powi(2) {
                return d * v / rate;
            }
            
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v / rate;
            }
        }
    }

    /// Generate beta random variate.
    pub fn beta(alpha: f64, beta_param: f64, rng: &mut Rng) -> f64 {
        let x = Self::gamma(alpha, 1.0, rng);
        let y = Self::gamma(beta_param, 1.0, rng);
        x / (x + y)
    }

    /// Generate chi-squared random variate.
    pub fn chi_squared(df: f64, rng: &mut Rng) -> f64 {
        Self::gamma(df / 2.0, 0.5, rng)
    }

    /// Generate Student's t random variate.
    pub fn students_t(df: f64, rng: &mut Rng) -> f64 {
        let z = crate::distributions::Normal { mu: 0.0, sigma: 1.0 }.sample(rng);
        let chi2 = Self::chi_squared(df, rng);
        z / (chi2 / df).sqrt()
    }

    /// Generate F-distribution random variate.
    pub fn f_distribution(df1: f64, df2: f64, rng: &mut Rng) -> f64 {
        let chi2_1 = Self::chi_squared(df1, rng);
        let chi2_2 = Self::chi_squared(df2, rng);
        (chi2_1 / df1) / (chi2_2 / df2)
    }
}

/// Variance reduction for simulation.
pub struct SimulationVarianceReduction;

impl SimulationVarianceReduction {
    /// Antithetic variates for simulation.
    pub fn antithetic_variates<F>(
        estimator: F,
        n_samples: usize,
        rng: &mut Rng,
    ) -> (f64, f64)
    where
        F: Fn(&mut Rng) -> f64,
    {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..n_samples {
            let y1 = estimator(rng);
            let y2 = estimator(rng);  // Would use complementary RNG in practice
            let avg = (y1 + y2) / 2.0;
            
            sum += avg;
            sum_sq += avg * avg;
        }
        
        let mean = sum / n_samples as f64;
        let variance = (sum_sq / n_samples as f64 - mean * mean) / n_samples as f64;
        
        (mean, variance.sqrt())
    }

    /// Control variates for simulation.
    pub fn control_variates<F, G>(
        estimator: F,
        control: G,
        control_mean: f64,
        n_samples: usize,
        rng: &mut Rng,
    ) -> (f64, f64)
    where
        F: Fn(&mut Rng) -> f64,
        G: Fn(&mut Rng) -> f64,
    {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_y_sq = 0.0;
        
        for _ in 0..n_samples {
            let x = estimator(rng);
            let y = control(rng);
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_y_sq += y * y;
        }
        
        let mean_x = sum_x / n_samples as f64;
        let mean_y = sum_y / n_samples as f64;
        
        let cov_xy = (sum_xy / n_samples as f64 - mean_x * mean_y);
        let var_y = (sum_y_sq / n_samples as f64 - mean_y * mean_y);
        
        let c = if var_y > 0.0 { cov_xy / var_y } else { 0.0 };
        let controlled_mean = mean_x - c * (mean_y - control_mean);
        let variance = var_y * (1.0 - c * c) / n_samples as f64;
        
        (controlled_mean, variance.sqrt())
    }
}

/// Quasi-random sequences for simulation.
pub struct QuasiRandomSimulation;

impl QuasiRandomSimulation {
    /// Halton sequence for quasi-Monte Carlo.
    pub fn halton_sequence(dim: usize, n: usize) -> Vec<Vec<f64>> {
        let mut sequence = Vec::new();
        let mut bases = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        
        for i in 0..n {
            let mut point = Vec::with_capacity(dim);
            for d in 0..dim {
                let base = bases[d % bases.len()];
                point.push(Self::halton_number(i, base));
            }
            sequence.push(point);
        }
        
        sequence
    }

    fn halton_number(index: usize, base: usize) -> f64 {
        let mut result = 0.0;
        let mut f = 1.0 / base as f64;
        let mut i = index;
        
        while i > 0 {
            result += f * (i % base) as f64;
            i /= base;
            f /= base as f64;
        }
        
        result
    }

    /// Sobol sequence (simplified 1D).
    pub fn sobol_sequence_1d(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i + 1) as f64 / n as f64).collect()
    }
}

/// Parallel simulation utilities.
pub struct ParallelSimulation;

impl ParallelSimulation {
    /// Split simulation into independent chunks.
    pub fn split_simulation(
        n_samples: usize,
        n_chunks: usize,
    ) -> Vec<(usize, usize)> {
        let chunk_size = n_samples / n_chunks;
        let mut chunks = Vec::new();
        
        for i in 0..n_chunks {
            let start = i * chunk_size;
            let end = if i < n_chunks - 1 {
                start + chunk_size
            } else {
                n_samples
            };
            chunks.push((start, end));
        }
        
        chunks
    }

    /// Combine results from parallel simulations.
    pub fn combine_results(results: &[(f64, f64)]) -> (f64, f64) {
        let n = results.len();
        let mut sum = 0.0;
        let mut sum_weights = 0.0;
        
        for &(value, variance) in results {
            let weight = if variance > 0.0 { 1.0 / variance } else { 1.0 };
            sum += value * weight;
            sum_weights += weight;
        }
        
        let combined_mean = sum / sum_weights;
        let combined_variance = 1.0 / sum_weights;
        
        (combined_mean, combined_variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monte_carlo_integration() {
        let mut rng = Rng::new(42);
        let (integral, error) = MonteCarloSimulation::integrate(
            |x| x * x,
            0.0,
            1.0,
            10000,
            &mut rng,
        );
        assert!((integral - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_probability_estimation() {
        let mut rng = Rng::new(42);
        let (p, error) = MonteCarloSimulation::estimate_probability(
            |r| r.uniform() < 0.5,
            10000,
            &mut rng,
        );
        assert!((p - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_box_muller() {
        let (z1, z2) = RNGUtils::box_muller(0.5, 0.5);
        assert!(z1.is_finite());
        assert!(z2.is_finite());
    }

    #[test]
    fn test_exponential_generation() {
        let mut rng = Rng::new(42);
        let x = RandomVariateGeneration::exponential(1.0, &mut rng);
        assert!(x > 0.0);
    }

    #[test]
    fn test_poisson_generation() {
        let mut rng = Rng::new(42);
        let k = RandomVariateGeneration::poisson(5.0, &mut rng);
        assert!(k >= 0);
    }

    #[test]
    fn test_event_driven_simulation() {
        let mut sim = EventDrivenSimulation::new();
        sim.schedule(SimulationEvent {
            time: 1.0,
            event_type: "arrival".to_string(),
            data: vec![1.0],
        });
        
        let event = sim.next_event();
        assert!(event.is_some());
        assert!((sim.current_time - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_halton_sequence() {
        let sequence = QuasiRandomSimulation::halton_sequence(2, 10);
        assert_eq!(sequence.len(), 10);
        assert_eq!(sequence[0].len(), 2);
    }
}
