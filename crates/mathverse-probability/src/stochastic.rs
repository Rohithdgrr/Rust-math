//! Stochastic processes: random walks, Brownian motion, Poisson processes, Gaussian processes.

use crate::{distributions::DiscreteDist, rng::Rng};

/// Simple random walk.
#[must_use]
pub struct RandomWalk {
    pub step_size: f64,
    pub current: f64,
}

impl RandomWalk {
    #[must_use]
    pub fn new(step_size: f64, initial: f64) -> Self {
        RandomWalk {
            step_size,
            current: initial,
        }
    }

    /// Take one step (±step_size with equal probability).
    pub fn step(&mut self, rng: &mut Rng) -> f64 {
        let direction = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        self.current += direction * self.step_size;
        self.current
    }

    /// Generate n steps.
    pub fn generate(&mut self, n: usize, rng: &mut Rng) -> Vec<f64> {
        let mut path = vec![self.current];
        for _ in 0..n {
            path.push(self.step(rng));
        }
        path
    }
}

/// Brownian motion (Wiener process).
#[must_use]
pub struct BrownianMotion {
    pub dt: f64,
    pub current: f64,
    pub time: f64,
}

impl BrownianMotion {
    #[must_use]
    pub fn new(dt: f64, initial: f64) -> Self {
        BrownianMotion {
            dt,
            current: initial,
            time: 0.0,
        }
    }

    /// Simulate one step of Brownian motion.
    pub fn step(&mut self, rng: &mut Rng) -> f64 {
        let increment = (self.dt).sqrt()
            * crate::distributions::Normal {
                mu: 0.0,
                sigma: 1.0,
            }
            .sample(rng);
        self.current += increment;
        self.time += self.dt;
        self.current
    }

    /// Generate n steps.
    pub fn generate(&mut self, n: usize, rng: &mut Rng) -> Vec<f64> {
        let mut path = vec![self.current];
        for _ in 0..n {
            path.push(self.step(rng));
        }
        path
    }

    /// Geometric Brownian motion (used in Black-Scholes).
    pub fn geometric_step(&mut self, drift: f64, volatility: f64, rng: &mut Rng) -> f64 {
        let z = crate::distributions::Normal {
            mu: 0.0,
            sigma: 1.0,
        }
        .sample(rng);
        let increment =
            (drift - 0.5 * volatility * volatility) * self.dt + volatility * (self.dt).sqrt() * z;
        self.current *= increment.exp();
        self.time += self.dt;
        self.current
    }
}

/// Poisson process.
#[must_use]
pub struct PoissonProcess {
    pub rate: f64,
    pub current_time: f64,
    pub event_count: u64,
}

impl PoissonProcess {
    #[must_use]
    pub fn new(rate: f64) -> Self {
        PoissonProcess {
            rate,
            current_time: 0.0,
            event_count: 0,
        }
    }

    /// Time until next event (exponentially distributed).
    #[must_use]
    pub fn time_to_next_event(&self, rng: &mut Rng) -> f64 {
        let u = rng.uniform().max(1e-300);
        -u.ln() / self.rate
    }

    /// Simulate events up to time T.
    pub fn simulate(&mut self, t_max: f64, rng: &mut Rng) -> Vec<f64> {
        let mut event_times = Vec::new();

        while self.current_time < t_max {
            let dt = self.time_to_next_event(rng);
            self.current_time += dt;

            if self.current_time <= t_max {
                event_times.push(self.current_time);
                self.event_count += 1;
            }
        }

        event_times
    }

    /// Number of events in time interval [0, t].
    #[must_use]
    pub fn count_in_interval(&self, t: f64, rng: &mut Rng) -> u64 {
        let poisson = crate::distributions::Poisson {
            lambda: self.rate * t,
        };
        poisson.sample(rng) as u64
    }
}

/// Gaussian process.
#[must_use]
pub struct GaussianProcess {
    pub mean_fn: Box<dyn Fn(f64) -> f64>,
    pub kernel_fn: Box<dyn Fn(f64, f64) -> f64>,
}

impl GaussianProcess {
    #[must_use]
    pub fn new<F1, F2>(mean_fn: F1, kernel_fn: F2) -> Self
    where
        F1: Fn(f64) -> f64 + 'static,
        F2: Fn(f64, f64) -> f64 + 'static,
    {
        GaussianProcess {
            mean_fn: Box::new(mean_fn),
            kernel_fn: Box::new(kernel_fn),
        }
    }

    /// Squared exponential (RBF) kernel.
    #[must_use]
    pub fn rbf_kernel(length_scale: f64, variance: f64) -> Box<dyn Fn(f64, f64) -> f64> {
        Box::new(move |x1, x2| {
            let diff = x1 - x2;
            variance * (-0.5 * (diff / length_scale).powi(2)).exp()
        })
    }

    /// Matérn kernel.
    #[must_use]
    pub fn matern_kernel(
        length_scale: f64,
        variance: f64,
        nu: f64,
    ) -> Box<dyn Fn(f64, f64) -> f64> {
        Box::new(move |x1, x2| {
            let diff = (x1 - x2).abs();
            let sqrt_2_nu_diff = 2.0_f64.sqrt() * nu.sqrt() * diff / length_scale;
            let _bessel = if sqrt_2_nu_diff < 1e-10 {
                1.0
            } else {
                // Simplified Bessel function approximation
                (sqrt_2_nu_diff).exp() / sqrt_2_nu_diff.sqrt()
            };
            variance * (1.0 + sqrt_2_nu_diff) * (-sqrt_2_nu_diff).exp()
        })
    }

    /// Sample from the GP at given points.
    #[must_use]
    pub fn sample(&self, x_points: &[f64], rng: &mut Rng) -> Vec<f64> {
        let n = x_points.len();
        if n == 0 {
            return Vec::new();
        }

        // Build covariance matrix
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                cov[i][j] = (self.kernel_fn)(x_points[i], x_points[j]);
            }
        }

        // Build mean vector
        let mean: Vec<f64> = x_points.iter().map(|&x| (self.mean_fn)(x)).collect();

        // Sample from multivariate normal
        let cov_matrix = crate::multivariate::CovarianceMatrix::new(cov).unwrap();
        let mvn = crate::multivariate::MultivariateNormal::new(mean, cov_matrix).unwrap();
        mvn.sample(rng)
    }
}

/// Ornstein-Uhlenbeck process (mean-reverting).
#[must_use]
pub struct OrnsteinUhlenbeck {
    pub theta: f64, // Mean reversion level
    pub mu: f64,    // Mean
    pub sigma: f64, // Volatility
    pub current: f64,
    pub dt: f64,
}

impl OrnsteinUhlenbeck {
    #[must_use]
    pub fn new(theta: f64, mu: f64, sigma: f64, initial: f64, dt: f64) -> Self {
        OrnsteinUhlenbeck {
            theta,
            mu,
            sigma,
            current: initial,
            dt,
        }
    }

    /// Simulate one step using Euler-Maruyama.
    pub fn step(&mut self, rng: &mut Rng) -> f64 {
        let z = crate::distributions::Normal {
            mu: 0.0,
            sigma: 1.0,
        }
        .sample(rng);
        let drift = self.theta * (self.mu - self.current) * self.dt;
        let diffusion = self.sigma * (self.dt).sqrt() * z;
        self.current += drift + diffusion;
        self.current
    }

    /// Generate n steps.
    pub fn generate(&mut self, n: usize, rng: &mut Rng) -> Vec<f64> {
        let mut path = vec![self.current];
        for _ in 0..n {
            path.push(self.step(rng));
        }
        path
    }
}

/// Jump diffusion process (Merton's model).
#[must_use]
pub struct JumpDiffusion {
    pub drift: f64,
    pub volatility: f64,
    pub jump_intensity: f64, // Poisson rate
    pub jump_mean: f64,
    pub jump_vol: f64,
    pub current: f64,
    pub dt: f64,
}

impl JumpDiffusion {
    #[must_use]
    pub fn new(
        drift: f64,
        volatility: f64,
        jump_intensity: f64,
        jump_mean: f64,
        jump_vol: f64,
        initial: f64,
        dt: f64,
    ) -> Self {
        JumpDiffusion {
            drift,
            volatility,
            jump_intensity,
            jump_mean,
            jump_vol,
            current: initial,
            dt,
        }
    }

    /// Simulate one step.
    pub fn step(&mut self, rng: &mut Rng) -> f64 {
        // Continuous part
        let z = crate::distributions::Normal {
            mu: 0.0,
            sigma: 1.0,
        }
        .sample(rng);
        let continuous = (self.drift - 0.5 * self.volatility * self.volatility) * self.dt
            + self.volatility * (self.dt).sqrt() * z;

        // Jump part
        let n_jumps = crate::distributions::Poisson {
            lambda: self.jump_intensity * self.dt,
        }
        .sample(rng);
        let mut jump_sum = 0.0;
        for _ in 0..n_jumps.max(0) {
            let jump_z = crate::distributions::Normal {
                mu: self.jump_mean,
                sigma: self.jump_vol,
            }
            .sample(rng);
            jump_sum += jump_z;
        }

        self.current *= (continuous + jump_sum).exp();
        self.current
    }
}

/// Levy process (simplified).
#[must_use]
pub struct LevyProcess {
    pub alpha: f64, // Stability parameter
    pub beta: f64,  // Skewness
    pub scale: f64,
    pub current: f64,
    pub dt: f64,
}

impl LevyProcess {
    #[must_use]
    pub fn new(alpha: f64, beta: f64, scale: f64, initial: f64, dt: f64) -> Self {
        LevyProcess {
            alpha,
            beta,
            scale,
            current: initial,
            dt,
        }
    }

    /// Simulate one step using simplified approximation.
    pub fn step(&mut self, rng: &mut Rng) -> f64 {
        // Simplified: use normal for alpha=2, Cauchy for alpha=1
        let increment = if (self.alpha - 2.0).abs() < 0.01 {
            crate::distributions::Normal {
                mu: 0.0,
                sigma: self.scale * (self.dt).sqrt(),
            }
            .sample(rng)
        } else if (self.alpha - 1.0).abs() < 0.01 {
            crate::distributions::Cauchy {
                x0: 0.0,
                gamma: self.scale * self.dt,
            }
            .sample(rng)
        } else {
            // General case: use normal approximation
            crate::distributions::Normal {
                mu: 0.0,
                sigma: self.scale * (self.dt).sqrt(),
            }
            .sample(rng)
        };

        self.current += increment;
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_walk() {
        let mut rng = Rng::new(42);
        let mut rw = RandomWalk::new(1.0, 0.0);
        let path = rw.generate(10, &mut rng);
        assert_eq!(path.len(), 11);
        assert_eq!(path[0], 0.0);
    }

    #[test]
    fn test_brownian_motion() {
        let mut rng = Rng::new(42);
        let mut bm = BrownianMotion::new(0.1, 0.0);
        let path = bm.generate(10, &mut rng);
        assert_eq!(path.len(), 11);
    }

    #[test]
    fn test_poisson_process() {
        let mut rng = Rng::new(42);
        let mut pp = PoissonProcess::new(2.0);
        let events = pp.simulate(5.0, &mut rng);
        assert!(!events.is_empty());
        for &t in &events {
            assert!(t <= 5.0);
        }
    }

    #[test]
    fn test_ornstein_uhlenbeck() {
        let mut rng = Rng::new(42);
        let mut ou = OrnsteinUhlenbeck::new(0.5, 0.0, 0.1, 1.0, 0.01);
        let path = ou.generate(10, &mut rng);
        assert_eq!(path.len(), 11);
    }

    #[test]
    fn test_gaussian_process() {
        let mut rng = Rng::new(42);
        let gp = GaussianProcess::new(|_| 0.0, GaussianProcess::rbf_kernel(1.0, 1.0));
        let x_points = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let samples = gp.sample(&x_points, &mut rng);
        assert_eq!(samples.len(), 5);
    }
}
