//! Batch trigonometric operations over slices.
//!
//! These helpers avoid per-element function call overhead in tight loops and
//! are aimed at DSP, audio synthesis, and graphics code. All functions are
//! generic over [`Real`] and work with `f32`/`f64` slices.
//!
//! Where an output slice is provided, functions return `true` on success and
//! `false` when the slice lengths do not match, so callers can avoid
//! allocation in `no_std` contexts.

use mathverse_core::traits::{Real, Trig};

/// Map sine over `xs`, writing into `out`.
///
/// Returns `false` (leaving `out` untouched) when the lengths differ.
///
/// ```
/// let xs = [0.0f64, 1.0];
/// let mut out = [0.0; 2];
/// assert!(mathverse_trigonometry::batched::map_sin(&xs, &mut out));
/// assert_eq!(out[0], 0.0);
/// ```
#[must_use]
pub fn map_sin<T: Real + Trig>(xs: &[T], out: &mut [T]) -> bool {
    if xs.len() != out.len() {
        return false;
    }
    for (o, &x) in out.iter_mut().zip(xs.iter()) {
        *o = x.sin();
    }
    true
}

/// Map cosine over `xs`, writing into `out`.
///
/// Returns `false` (leaving `out` untouched) when the lengths differ.
#[must_use]
pub fn map_cos<T: Real + Trig>(xs: &[T], out: &mut [T]) -> bool {
    if xs.len() != out.len() {
        return false;
    }
    for (o, &x) in out.iter_mut().zip(xs.iter()) {
        *o = x.cos();
    }
    true
}

/// Map `(sin, cos)` pairs over `xs` into two output slices.
///
/// Returns `false` (leaving outputs untouched) when any length differs.
#[must_use]
pub fn map_sin_cos<T: Real + Trig>(xs: &[T], sin_out: &mut [T], cos_out: &mut [T]) -> bool {
    if xs.len() != sin_out.len() || xs.len() != cos_out.len() {
        return false;
    }
    for ((&x, so), co) in xs.iter().zip(sin_out.iter_mut()).zip(cos_out.iter_mut()) {
        let (s, c) = x.sin_cos();
        *so = s;
        *co = c;
    }
    true
}

/// Replace each element with its sine, in place.
pub fn sin_inplace<T: Real + Trig>(xs: &mut [T]) {
    for x in xs.iter_mut() {
        *x = x.sin();
    }
}

/// `Σ sin(xᵢ)`.
#[must_use]
pub fn sum_sin<T: Real + Trig>(xs: &[T]) -> T {
    let mut acc = T::zero();
    for &x in xs {
        acc = acc + x.sin();
    }
    acc
}

/// `Σ cos(xᵢ)`.
#[must_use]
pub fn sum_cos<T: Real + Trig>(xs: &[T]) -> T {
    let mut acc = T::zero();
    for &x in xs {
        acc = acc + x.cos();
    }
    acc
}

/// `(Σ sin(xᵢ), Σ cos(xᵢ))` computed with one pass over the slice.
#[must_use]
pub fn sum_sin_cos<T: Real + Trig>(xs: &[T]) -> (T, T) {
    let mut s = T::zero();
    let mut c = T::zero();
    for &x in xs {
        let (si, ci) = x.sin_cos();
        s = s + si;
        c = c + ci;
    }
    (s, c)
}

/// Additive synthesis: `out[i] = Σₖ amps[k] · sin(freq·i + phases[k])`.
///
/// Returns `false` (leaving `out` untouched) when `phases` and `amps` have
/// different lengths.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
#[must_use]
pub fn accumulate_sine<T: Real + Trig>(freq: T, phases: &[T], amps: &[T], out: &mut [T]) -> bool {
    if phases.len() != amps.len() {
        return false;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let t = freq * T::from_i64(i as i64);
        let mut acc = T::zero();
        for (k, &phase) in phases.iter().enumerate() {
            acc = acc + amps[k] * (t + phase).sin();
        }
        *slot = acc;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    const EPS: f64 = 1e-12;

    #[test]
    fn map_and_inplace() {
        let xs = [0.0f64, PI / 2.0, PI];
        let mut out = [0.0; 3];
        assert!(map_sin(&xs, &mut out));
        assert!((out[0]).abs() < EPS);
        assert!((out[1] - 1.0).abs() < EPS);
        assert!(out[2].abs() < 1e-12);

        assert!((map_sin(&xs[..2], &mut out) == false));
        assert_eq!(out[0], 0.0, "output must be untouched on mismatch");

        let mut vals = [0.0f64, PI / 2.0, PI];
        sin_inplace(&mut vals);
        assert!((vals[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn map_sin_cos_pairs() {
        let xs = [0.0f64, PI / 2.0];
        let mut s = [0.0; 2];
        let mut c = [0.0; 2];
        assert!(map_sin_cos(&xs, &mut s, &mut c));
        assert!((s[0]).abs() < EPS && (c[0] - 1.0).abs() < EPS);
        assert!((s[1] - 1.0).abs() < EPS && c[1].abs() < EPS);

        let mut bad = [0.0; 1];
        assert!(!map_sin_cos(&xs, &mut s, &mut bad));
    }

    #[test]
    fn sums() {
        let xs = [0.0f64, PI / 2.0];
        assert!((sum_sin(&xs) - 1.0).abs() < EPS);
        assert!((sum_cos(&xs) - 1.0).abs() < EPS);
        let (s, c) = sum_sin_cos(&xs);
        assert!((s - 1.0).abs() < EPS && (c - 1.0).abs() < EPS);
    }

    #[test]
    fn accumulate() {
        // Single harmonic: out[i] = sin(i).
        let mut out = [0.0; 3];
        assert!(accumulate_sine(1.0f64, &[0.0], &[1.0], &mut out));
        assert!((out[0]).abs() < EPS);
        assert!((out[1] - 1.0f64.sin()).abs() < EPS);
        assert!((out[2] - 2.0f64.sin()).abs() < EPS);

        assert!(!accumulate_sine(1.0f64, &[0.0, 1.0], &[1.0], &mut out));
    }
}
