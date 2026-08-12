//! Tick locators and formatters — the matplotlib analogue of
//! `MaxNLocator` / `MultipleLocator` / `FixedLocator` and
//! `ScalarFormatter` / `FuncFormatter`.
//!
//! Locators decide *where* ticks go; formatters decide how their labels are
//! rendered. Both are trait objects so users can plug custom behaviour in.

use std::sync::Arc;

/// Chooses tick positions in data space for a range `[lo, hi]`.
///
/// Implementations should return a sorted, ascending list of values in
/// `[lo, hi]`. The `max_ticks` hint is a soft target.
pub trait TickLocator: Send + Sync + std::fmt::Debug {
    /// Compute tick positions.
    fn locate(&self, lo: f64, hi: f64, max_ticks: usize) -> Vec<f64>;
}

/// Formats a tick value into its label string.
pub trait TickFormatter: Send + Sync + std::fmt::Debug {
    /// Format one tick value.
    fn format(&self, value: f64) -> String;
}

/// "Nice" locator: picks steps from `1, 2, 5 × 10^k` so labels stay readable.
/// This is the default behaviour of matplotlib's `MaxNLocator` (nbins = auto).
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxNLocator;

impl MaxNLocator {
    /// Create a new max-n locator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl TickLocator for MaxNLocator {
    fn locate(&self, lo: f64, hi: f64, max_ticks: usize) -> Vec<f64> {
        nice_ticks(lo, hi, max_ticks)
    }
}

/// Fixed step locator: ticks at multiples of `step` inside `[lo, hi]`.
#[derive(Debug, Clone, Copy)]
pub struct MultipleLocator {
    /// Step between ticks.
    pub step: f64,
}

impl MultipleLocator {
    /// Create a locator with the given step.
    ///
    /// Returns `None` for a non-positive or non-finite step.
    #[must_use]
    pub fn new(step: f64) -> Option<Self> {
        (step.is_finite() && step > 0.0).then_some(Self { step })
    }
}

impl TickLocator for MultipleLocator {
    fn locate(&self, lo: f64, hi: f64, max_ticks: usize) -> Vec<f64> {
        if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
            return Vec::new();
        }
        let mut out = Vec::new();
        let start = (lo / self.step).ceil() * self.step;
        let mut v = start;
        let limit = hi * (1.0 + 1e-12) + self.step;
        while v <= limit {
            if v >= lo && v <= hi {
                out.push(v);
            }
            v += self.step;
            if out.len() >= max_ticks.max(1) * 2 {
                break;
            }
        }
        out
    }
}

/// Explicit tick positions supplied by the caller.
#[derive(Debug, Clone)]
pub struct FixedLocator {
    /// Ticks to show, in any order (sorted on use).
    pub ticks: Vec<f64>,
}

impl FixedLocator {
    /// Create a fixed locator.
    #[must_use]
    pub fn new(ticks: Vec<f64>) -> Self {
        Self { ticks }
    }
}

impl TickLocator for FixedLocator {
    fn locate(&self, lo: f64, hi: f64, _max_ticks: usize) -> Vec<f64> {
        let mut out: Vec<f64> = self
            .ticks
            .iter()
            .copied()
            .filter(|&v| v >= lo && v <= hi && v.is_finite())
            .collect();
        out.sort_by(|a, b| a.total_cmp(b));
        out
    }
}

/// Default scalar formatter: compact decimals, scientific notation for very
/// large / small magnitudes (mirrors the crate's existing tick rendering).
#[derive(Debug, Clone, Copy, Default)]
pub struct ScalarFormatter;

impl ScalarFormatter {
    /// Create a new scalar formatter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl TickFormatter for ScalarFormatter {
    fn format(&self, value: f64) -> String {
        format_tick(value)
    }
}

/// Format ticks through an arbitrary function (matplotlib `FuncFormatter`).
#[derive(Clone)]
pub struct FuncFormatter {
    f: Arc<dyn Fn(f64) -> String + Send + Sync>,
}

impl std::fmt::Debug for FuncFormatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuncFormatter").finish_non_exhaustive()
    }
}

impl FuncFormatter {
    /// Wrap a formatting function.
    pub fn new(f: impl Fn(f64) -> String + Send + Sync + 'static) -> Self {
        Self { f: Arc::new(f) }
    }
}

impl TickFormatter for FuncFormatter {
    fn format(&self, value: f64) -> String {
        (self.f)(value)
    }
}

/// Ticks at a fixed step, formatted by a custom function (convenience combo).
pub fn ticks_at(step: f64, formatter: impl Fn(f64) -> String + Send + Sync + 'static) -> (Box<dyn TickLocator>, Box<dyn TickFormatter>) {
    match MultipleLocator::new(step) {
        Some(m) => (Box::new(m), Box::new(FuncFormatter::new(formatter))),
        None => (Box::new(MaxNLocator::new()), Box::new(FuncFormatter::new(formatter))),
    }
}

/// The "nice step" algorithm: round `raw` up to `1, 2, or 5 × 10^k`.
#[must_use]
pub fn nice_step(raw: f64) -> f64 {
    if !raw.is_finite() || raw <= 0.0 {
        return 1.0;
    }
    let mag = 10.0f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = if norm <= 1.0 {
        1.0
    } else if norm <= 2.0 {
        2.0
    } else if norm <= 5.0 {
        5.0
    } else {
        10.0
    };
    step * mag
}

/// Compute "nice" ticks in `[lo, hi]` targeting `max_ticks` intervals.
#[must_use]
pub fn nice_ticks(lo: f64, hi: f64, max_ticks: usize) -> Vec<f64> {
    if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
        return Vec::new();
    }
    let count = max_ticks.max(1) as f64;
    let step = nice_step((hi - lo) / count);
    let start = (lo / step).ceil() * step;
    let mut out = Vec::new();
    let mut v = start;
    let limit = hi * (1.0 + 1e-12) + step;
    while v <= limit {
        if v >= lo && v <= hi {
            out.push(v);
        }
        v += step;
    }
    out
}

/// Compact numeric label: plain decimal below 1e6, scientific above.
#[must_use]
pub fn format_tick(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if v.abs() >= 1e6 || (v.abs() < 1e-4 && v != 0.0) {
        format!("{v:.1e}")
    } else {
        format!("{v:.3}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_ticks_cover_range() {
        let ticks = nice_ticks(0.0, 10.0, 5);
        assert!(!ticks.is_empty());
        assert!(ticks.iter().all(|&v| v >= 0.0 && v <= 10.0));
        assert!(ticks.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn multiple_locator_even_spacing() {
        let loc = MultipleLocator::new(2.0).unwrap();
        let ticks = loc.locate(0.5, 9.5, 8);
        assert_eq!(ticks, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn fixed_locator_filters_and_sorts() {
        let loc = FixedLocator::new(vec![9.0, -1.0, 3.0, 5.0]);
        let ticks = loc.locate(0.0, 6.0, 10);
        assert_eq!(ticks, vec![3.0, 5.0]);
    }

    #[test]
    fn func_formatter_applies() {
        let f = FuncFormatter::new(|v| format!("{v:.0}%"));
        assert_eq!(f.format(42.0), "42%");
    }

    #[test]
    fn multiple_locator_rejects_bad_step() {
        assert!(MultipleLocator::new(0.0).is_none());
        assert!(MultipleLocator::new(-1.0).is_none());
        assert!(MultipleLocator::new(f64::NAN).is_none());
    }

    #[test]
    fn max_n_locator_impl() {
        let loc = MaxNLocator::new();
        let ticks = loc.locate(0.0, 1.0, 4);
        assert!(ticks.len() >= 2);
    }

    #[test]
    fn format_tick_variants() {
        assert_eq!(format_tick(0.0), "0");
        assert_eq!(format_tick(2.0), "2");
        assert_eq!(format_tick(2.5), "2.5");
        assert_eq!(format_tick(1e7), "1.0e7");
    }
}
