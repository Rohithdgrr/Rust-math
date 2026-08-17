//! Polar representations for complex signals: phase unwrapping,
//! instantaneous frequency, and polar-form storage.
//!
//! These utilities are essential for radar (Doppler processing),
//! communications (carrier phase tracking), audio analysis (instantaneous
//! frequency), and interferometry.
//!
//! # Module overview
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`unwrap_phase`] | Unwrap phase discontinuities at ±π boundaries |
//! | [`instantaneous_frequency`] | Compute instantaneous frequency from unwrapped phase |
//! | [`polar_to_complex`] | Convert (magnitude, phase) pairs to complex values |
//! | [`complex_to_polar`] | Extract magnitude and phase arrays from complex signal |
//! | [`demodulate`] | Complex baseband demodulation: remove carrier frequency |
//! | [`modulate`] | Complex baseband modulation: apply carrier frequency |

use crate::Complex;

/// Unwrap phase discontinuities by adding integer multiples of 2π
/// wherever consecutive phase values jump by more than π.
///
/// Uses a simple sequential unwrapping algorithm.
///
/// # Arguments
/// * `phases` — array of phase values in `[-π, π]`
///
/// # Returns
/// Unwrapped phase array with continuous values (no ±π jumps).
///
/// # Example
/// ```
/// use mathverse_complex::polar::unwrap_phase;
/// let wrapped = vec![0.0, 1.0, -3.0, -2.0, 0.5];
/// let unwrapped = unwrap_phase(&wrapped);
/// // The jump from 1.0 to -3.0 is a 4.0 rad discontinuity > π,
/// // so 2π is added to subsequent values.
/// assert!(unwrapped[2] > unwrapped[1]); // No jump back
/// ```
pub fn unwrap_phase(phases: &[f64]) -> Vec<f64> {
    if phases.is_empty() {
        return Vec::new();
    }

    let two_pi = 2.0 * core::f64::consts::PI;
    let mut unwrapped = Vec::with_capacity(phases.len());
    unwrapped.push(phases[0]);

    let mut cumulative_offset = 0.0;
    for window in phases.windows(2) {
        let diff = window[1] - window[0];

        if diff > core::f64::consts::PI {
            // Wrapped downward: subtract 2π to continue the trend
            cumulative_offset -= two_pi;
        } else if diff < -core::f64::consts::PI {
            // Wrapped upward: add 2π to continue the trend
            cumulative_offset += two_pi;
        }

        unwrapped.push(window[1] + cumulative_offset);
    }

    unwrapped
}

/// Compute instantaneous frequency from a phase signal.
///
/// `f_inst[k] = (φ[k+1] − φ[k]) / (2π · dt)`
///
/// Uses centered differences for interior points, forward/backward
/// differences at boundaries.
///
/// # Arguments
/// * `phases` — unwrapped phase array (use [`unwrap_phase`] first)
/// * `dt` — sampling period in seconds
///
/// # Returns
/// Instantaneous frequency array of the same length as `phases`, in Hz.
pub fn instantaneous_frequency(phases: &[f64], dt: f64) -> Vec<f64> {
    let n = phases.len();
    if n == 0 || dt <= 0.0 {
        return Vec::new();
    }

    let two_pi = 2.0 * core::f64::consts::PI;
    let mut freq = Vec::with_capacity(n);

    if n == 1 {
        freq.push(0.0);
        return freq;
    }

    // Forward difference at start
    freq.push((phases[1] - phases[0]) / (two_pi * dt));

    // Centered differences for interior
    for i in 1..n - 1 {
        freq.push((phases[i + 1] - phases[i - 1]) / (2.0 * two_pi * dt));
    }

    // Backward difference at end
    freq.push((phases[n - 1] - phases[n - 2]) / (two_pi * dt));

    freq
}

/// Convert arrays of magnitudes and phases to complex values.
///
/// `z[k] = mag[k] · e^(i · phase[k])`
///
/// # Panics
/// If `magnitudes.len() != phases.len()`.
pub fn polar_to_complex(magnitudes: &[f64], phases: &[f64]) -> Vec<Complex> {
    assert_eq!(
        magnitudes.len(),
        phases.len(),
        "magnitudes and phases must have equal length"
    );
    magnitudes
        .iter()
        .zip(phases.iter())
        .map(|(m, p)| Complex::polar(*m, *p))
        .collect()
}

/// Extract magnitude and phase arrays from complex signal.
///
/// Returns `(magnitudes, phases)` where `phases` are in `[-π, π]`.
pub fn complex_to_polar(signal: &[Complex]) -> (Vec<f64>, Vec<f64>) {
    let magnitudes = signal.iter().map(super::Complex::norm).collect();
    let phases = signal.iter().map(super::Complex::arg).collect();
    (magnitudes, phases)
}

/// Complex baseband demodulation: remove a carrier frequency.
///
/// `y[k] = x[k] · e^(−i · 2π · f_c · k · dt)`
///
/// This shifts the spectrum down by `f_c`, converting a passband signal
/// to baseband.
///
/// # Arguments
/// * `signal` — input passband signal
/// * `carrier_freq` — carrier frequency to remove (Hz)
/// * `dt` — sampling period (seconds)
pub fn demodulate(signal: &[Complex], carrier_freq: f64, dt: f64) -> Vec<Complex> {
    let two_pi = 2.0 * core::f64::consts::PI;
    signal
        .iter()
        .enumerate()
        .map(|(k, &z)| {
            let phase = -two_pi * carrier_freq * k as f64 * dt;
            z * Complex::polar(1.0, phase)
        })
        .collect()
}

/// Complex baseband modulation: apply a carrier frequency.
///
/// `y[k] = x[k] · e^(i · 2π · f_c · k · dt)`
///
/// This shifts the baseband spectrum up by `f_c`.
///
/// # Arguments
/// * `signal` — baseband signal
/// * `carrier_freq` — carrier frequency to apply (Hz)
/// * `dt` — sampling period (seconds)
pub fn modulate(signal: &[Complex], carrier_freq: f64, dt: f64) -> Vec<Complex> {
    let two_pi = 2.0 * core::f64::consts::PI;
    signal
        .iter()
        .enumerate()
        .map(|(k, &z)| {
            let phase = two_pi * carrier_freq * k as f64 * dt;
            z * Complex::polar(1.0, phase)
        })
        .collect()
}

/// Compute the circular mean phase of a set of phase values.
///
/// `mean_phase = arg(Σ e^(iφ_k))`
pub fn circular_mean_phase(phases: &[f64]) -> f64 {
    if phases.is_empty() {
        return 0.0;
    }
    let sum: Complex = phases
        .iter()
        .map(|p| Complex::polar(1.0, *p))
        .fold(Complex::zero(), |a, b| a + b);
    sum.arg()
}

/// Compute the circular standard deviation of phase values.
///
/// Returns a value in `[0, 1]` where 0 means all phases are identical
/// and 1 means phases are uniformly distributed.
pub fn circular_stddev(phases: &[f64]) -> f64 {
    let n = phases.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let sum: Complex = phases
        .iter()
        .map(|p| Complex::polar(1.0, *p))
        .fold(Complex::zero(), |a, b| a + b);
    let r = sum.norm() / n;
    // CircStats: resultant length R̄ → (1 - R̄) as dispersion measure
    (1.0 - r).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = core::f64::consts::PI;
    const EPS: f64 = 1e-10;

    #[test]
    fn unwrap_no_discontinuity() {
        let phases = vec![-0.5, 0.0, 0.5, 1.0];
        let unwrapped = unwrap_phase(&phases);
        for (a, b) in phases.iter().zip(unwrapped.iter()) {
            assert!((a - b).abs() < EPS);
        }
    }

    #[test]
    fn unwrap_single_jump() {
        // Simulate a jump from 0.9π to -0.9π (real jump = -1.8π < -π)
        // Unwrap adds 2π: -0.9π + 2π = 1.1π
        let phases = vec![0.9 * PI, -0.9 * PI];
        let unwrapped = unwrap_phase(&phases);
        assert!((unwrapped[0] - 0.9 * PI).abs() < EPS);
        assert!((unwrapped[1] - (0.9 * PI + (2.0 * PI - 1.8 * PI))).abs() < EPS);
    }

    #[test]
    fn unwrap_multiple_jumps() {
        // Three periods of a sawtooth
        let n = 100;
        let phases: Vec<f64> = (0..n)
            .map(|i| ((f64::from(i) * 1.5) % (2.0 * PI)) - PI)
            .collect();
        let unwrapped = unwrap_phase(&phases);

        // Unwrapped should be monotonically increasing
        for i in 1..unwrapped.len() {
            assert!(
                unwrapped[i] >= unwrapped[i - 1] - EPS,
                "Phase decreased at index {}: {} < {}",
                i,
                unwrapped[i],
                unwrapped[i - 1]
            );
        }
    }

    #[test]
    fn unwrap_empty() {
        assert!(unwrap_phase(&[]).is_empty());
    }

    #[test]
    fn instantaneous_frequency_constant_phase() {
        let phases = vec![0.0; 10];
        let freq = instantaneous_frequency(&phases, 0.01);
        for f in &freq {
            assert!(f.abs() < EPS);
        }
    }

    #[test]
    fn instantaneous_frequency_linear_ramp() {
        // Linear phase ramp: φ(t) = 2π·f₀·t → constant frequency f₀
        let f0 = 100.0;
        let dt = 0.001;
        let n = 50;
        let phases: Vec<f64> = (0..n)
            .map(|k| 2.0 * PI * f0 * k as f64 * dt)
            .collect();
        let freq = instantaneous_frequency(&phases, dt);
        // Interior points should be close to f0
        for f in &freq[1..n - 1] {
            assert!((f - f0).abs() < 1.0, "Expected ~{f0}, got {f}");
        }
    }

    #[test]
    fn instantaneous_frequency_empty() {
        let freq = instantaneous_frequency(&[], 0.001);
        assert!(freq.is_empty());
    }

    #[test]
    fn polar_to_complex_roundtrip() {
        let mags = vec![1.0, 2.0, 3.0];
        let phases = vec![0.0, PI / 4.0, PI / 2.0];
        let complex = polar_to_complex(&mags, &phases);
        let (mags2, phases2) = complex_to_polar(&complex);
        for i in 0..3 {
            assert!((mags[i] - mags2[i]).abs() < EPS);
            assert!((phases[i] - phases2[i]).abs() < EPS);
        }
    }

    #[test]
    fn demodulate_modulate_roundtrip() {
        let signal: Vec<Complex> = (0..32)
            .map(|k| Complex::new((f64::from(k) * 0.3).sin(), 0.0))
            .collect();
        let fc = 1000.0;
        let dt = 0.0001;
        let modulated = modulate(&signal, fc, dt);
        let demodulated = demodulate(&modulated, fc, dt);

        for (a, b) in signal.iter().zip(demodulated.iter()) {
            assert!((a - b).norm() < 1e-8, "Roundtrip error: {}", (a - b).norm());
        }
    }

    #[test]
    fn circular_mean_single_direction() {
        let phases = vec![0.1, 0.2, 0.15, 0.25];
        let mean = circular_mean_phase(&phases);
        assert!((mean - 0.175).abs() < 0.1);
    }

    #[test]
    fn circular_mean_ambiguous() {
        // Phases at opposite sides: mean should be undefined (high stddev)
        let phases = vec![0.0, PI];
        let stddev = circular_stddev(&phases);
        assert!(stddev > 0.5, "Expected high circular stddev, got {stddev}");
    }

    #[test]
    fn circular_stddev_identical() {
        let phases = vec![1.0; 10];
        let stddev = circular_stddev(&phases);
        assert!(stddev < EPS);
    }
}
