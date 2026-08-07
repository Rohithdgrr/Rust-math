//! Deterministic pseudo-random numbers (xoshiro256\*\*), dependency-free.
//!
//! Seed for reproducible simulations; not for cryptography. xoshiro256** is a
//! fast, high-quality PRNG that passes BigCrush.

/// SplitMix64 mixer used to seed the xoshiro256** state from a single seed.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256\*\* generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Seeds the 128-bit state via splitmix64 from `seed`.
    pub fn new(seed: u64) -> Rng {
        let mut sm = seed;
        // A zero seed is allowed; splitmix64 expands it into 4 non-zero-ish words.
        Rng {
            state: [0u64; 4].map(|_| splitmix64(&mut sm)),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);

        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `0..n` (rejection sampling, no modulo bias).
    pub fn below(&mut self, n: u64) -> u64 {
        if n <= 1 {
            return 0;
        }
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

    #[test]
    fn uniform_mean_is_balanced() {
        let mut rng = Rng::new(42);
        let n = 500_000;
        let sum: f64 = (0..n).map(|_| rng.uniform()).sum();
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }
}