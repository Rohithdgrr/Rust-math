//! Parallel Monte Carlo simulation utilities.

use rayon::prelude::*;

/// Run N independent Monte Carlo samples in parallel.
///
/// Each sample is an independent call to the closure. Results are collected
/// and reduced using the provided accumulator.
///
/// # Arguments
/// * `n_samples` - Number of Monte Carlo samples
/// * `sample_fn` - Function that generates one sample result
/// * `reduce_fn` - Function that combines two partial results
/// * `identity` - Identity element for the reduction
///
/// # Examples
///
/// ```
/// use mathverse_parallel::par_monte_carlo;
///
/// // Estimate pi: count points inside unit circle
/// let n = 1_000_000u64;
/// let inside: u64 = par_monte_carlo(
///     n,
///     |_| {
///         let x = rand_f64();
///         let y = rand_f64();
///         if x * x + y * y <= 1.0 { 1u64 } else { 0 }
///     },
///     |a, b| a + b,
///     0u64,
/// );
/// let pi_est = 4.0 * inside as f64 / n as f64;
/// assert!((pi_est - std::f64::consts::PI).abs() < 0.01);
/// ```
pub fn par_monte_carlo<T, F, R, I>(n: u64, sample_fn: F, reduce_fn: R, identity: I) -> T
where
    T: Send + Sync,
    F: Fn(u64) -> T + Send + Sync,
    R: fn(T, T) -> T + Send + Sync,
    I: Fn() -> T + Send + Sync,
{
    (0..n)
        .into_par_iter()
        .map(|i| sample_fn(i))
        .reduce(|| identity(), reduce_fn)
}

/// Simple pseudo-random f64 in [0, 1) using xorshift64*.
#[inline]
pub fn rand_f64() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0x1234_5678_9ABC_DEF0) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        s.set(x);
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    })
}

/// Parallel estimation of pi using Monte Carlo.
pub fn par_estimate_pi(n_samples: u64) -> f64 {
    let inside: u64 = par_monte_carlo(
        n_samples,
        |_| {
            let x = rand_f64();
            let y = rand_f64();
            if x * x + y * y <= 1.0 { 1u64 } else { 0 }
        },
        |a, b| a + b,
        || 0u64,
    );
    4.0 * inside as f64 / n_samples as f64
}

/// Parallel Monte Carlo integration of f(x) over [a, b].
pub fn par_mc_integrate(
    f: impl Fn(f64) -> f64 + Send + Sync,
    a: f64,
    b: f64,
    n_samples: u64,
) -> f64 {
    let sum: f64 = par_monte_carlo(
        n_samples,
        |_| {
            let x = a + rand_f64() * (b - a);
            f(x)
        },
        |acc, val| acc + val,
        || 0.0,
    );
    (b - a) * sum / n_samples as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_monte_carlo_sum() {
        let result = par_monte_carlo(1000, |i| i as f64, |a, b| a + b, || 0.0);
        let expected: f64 = (0..1000).map(|i| i as f64).sum();
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_par_estimate_pi() {
        let pi = par_estimate_pi(1_000_000);
        assert!((pi - std::f64::consts::PI).abs() < 0.01);
    }

    #[test]
    fn test_par_mc_integrate() {
        // integral of x^2 from 0 to 1 = 1/3
        let result = par_mc_integrate(|x| x * x, 0.0, 1.0, 1_000_000);
        assert!((result - 1.0 / 3.0).abs() < 0.01);
    }
}
