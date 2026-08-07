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
/// use mathverse_parallel::{par_monte_carlo, rand_f64};
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
///     || 0u64,
/// );
/// let pi_est = 4.0 * inside as f64 / n as f64;
/// assert!((pi_est - std::f64::consts::PI).abs() < 0.01);
/// ```
pub fn par_monte_carlo<T, F, R, I>(n: u64, sample_fn: F, reduce_fn: R, identity: I) -> T
where
    T: Send + Sync,
    F: Fn(u64) -> T + Send + Sync,
    R: Fn(T, T) -> T + Send + Sync,
    I: Fn() -> T + Send + Sync,
{
    (0..n)
        .into_par_iter()
        .map(sample_fn)
        .reduce(identity, reduce_fn)
}

/// `SplitMix64` mixing step (public-domain, from the reference
/// implementation). Passes `BigCrush`; **not** cryptographically secure — use
/// a dedicated CSPRNG for security contexts.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    *state = z;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Draw a fresh, well-mixed seed from a global atomic counter.
///
/// Guarantees every worker thread gets a **distinct** stream: with the old
/// constant thread-local seed, all threads produced the identical sequence,
/// so parallel Monte Carlo silently re-sampled the same points on every
/// thread (no accuracy gain from parallelism). Not cryptographically secure.
fn fresh_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut n = COUNTER.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    splitmix64(&mut n)
}

/// Simple pseudo-random f64 in [0, 1) using a per-thread SplitMix64 stream.
///
/// Each thread derives a distinct seed on first use (see [`fresh_seed`]), so
/// parallel callers never share a sequence. Reproducibility across runs is
/// **not** guaranteed when threads are involved; **not** cryptographically
/// secure.
#[inline]
pub fn rand_f64() -> f64 {
    use std::cell::Cell;
    thread_local! {
        // `None` marks an unseeded stream; the first call draws a seed.
        static STATE: Cell<Option<u64>> = const { Cell::new(None) };
    }
    STATE.with(|s| {
        let mut x = match s.get() {
            Some(seed) => seed,
            None => {
                let seed = fresh_seed();
                s.set(Some(seed));
                seed
            }
        };
        let z = splitmix64(&mut x);
        s.set(Some(x));
        // 53 bits of entropy, matching the f64 mantissa.
        (z >> 11) as f64 / (1u64 << 53) as f64
    })
}

/// Parallel estimation of pi using Monte Carlo.
///
/// Results vary between runs: each worker thread draws from its own seeded
/// stream, and the thread-to-seed assignment depends on scheduling.
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
///
/// Results vary between runs: each worker thread draws from its own seeded
/// stream, and the thread-to-seed assignment depends on scheduling.
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
    fn threads_get_distinct_streams() {
        // Each thread must produce a different first draw; otherwise parallel
        // sampling silently duplicates points (identical streams per thread).
        use std::collections::HashSet;
        use std::thread;
        let firsts: Vec<f64> = (0..16)
            .map(|_| thread::spawn(|| {
                // Drain any pre-existing stream state from this thread so the
                // first call here is genuinely this thread's seed draw.
                rand_f64();
                rand_f64()
            }))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        let unique: HashSet<u64> = firsts.iter().map(|f| f.to_bits() >> 12).collect();
        assert_eq!(unique.len(), firsts.len(), "threads shared a stream: {firsts:?}");
    }

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
