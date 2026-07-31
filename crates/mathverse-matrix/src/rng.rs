//! Random number generator for matrix operations.

/// Simple deterministic RNG for testing and reproducibility.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create new RNG with seed.
    pub fn new(seed: u64) -> Self {
        Rng { state: seed.wrapping_add(1) }
    }

    /// Generate random u64 using xorshift64*.
    pub fn gen(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let result = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        self.state
    }

    /// Generate uniform f64 in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.gen() as f64) / (u64::MAX as f64)
    }

    /// Generate uniform f64 in [min, max).
    pub fn uniform_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.uniform() * (max - min)
    }

    /// Generate standard normal f64 (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}
