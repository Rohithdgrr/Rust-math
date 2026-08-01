//! Deterministic pseudo-random numbers (xorshift64*), dependency-free.
//!
//! Seed for reproducible simulations; not for cryptography.

/// xorshift64* generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `0..n` (rejection sampling, no modulo bias).
    pub fn below(&mut self, n: u64) -> u64 {
        let limit = u64::MAX - (u64::MAX % n);
        loop {
            let r = self.next_u64();
            if r < limit {
                return r % n;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Rng::new(123);
        let mut b = Rng::new(123);
        assert_eq!(a.next_u64(), b.next_u64());
        for _ in 0..1000 {
            let u = a.uniform();
            assert!((0.0..1.0).contains(&u));
        }
        assert!(a.below(7) < 7);
    }
}
