//! Error bar statistics via `mathverse-statistics`.

use crate::error::{PlotError, PlotResult};

/// A sample mean with a symmetric confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorBar {
    /// Sample center (mean).
    pub center: f64,
    /// Lower bound.
    pub lo: f64,
    /// Upper bound.
    pub hi: f64,
}

impl ErrorBar {
    /// Mean plus a `z`-scaled confidence interval, both from
    /// `mathverse_statistics`.
    ///
    /// # Errors
    ///
    /// Returns `PlotError::InvalidData` for empty or non-finite input.
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_plot::errorbar::ErrorBar;
    /// let bar = ErrorBar::ci(&[1.0, 2.0, 3.0, 4.0, 5.0], 1.96).unwrap();
    /// assert_eq!(bar.center, 3.0);
    /// ```
    pub fn ci(xs: &[f64], z: f64) -> PlotResult<Self> {
        if xs.is_empty() {
            return Err(PlotError::InvalidData("empty data".into()));
        }
        if xs.iter().any(|x| !x.is_finite()) {
            return Err(PlotError::InvalidData("non-finite data value".into()));
        }
        let (lo, hi) = mathverse_statistics::mean_ci(xs, z);
        Ok(Self {
            center: mathverse_statistics::mean(xs),
            lo,
            hi,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_centered_on_mean() {
        let bar = ErrorBar::ci(&[1.0, 2.0, 3.0, 4.0, 5.0], 1.96).unwrap();
        assert_eq!(bar.center, 3.0);
        assert!(bar.lo < bar.center && bar.center < bar.hi);
    }

    #[test]
    fn widening_z_widens_interval() {
        let z1 = ErrorBar::ci(&[1.0, 2.0, 3.0], 1.0).unwrap();
        let z2 = ErrorBar::ci(&[1.0, 2.0, 3.0], 2.0).unwrap();
        assert!((z2.hi - z2.lo) > (z1.hi - z1.lo));
    }

    #[test]
    fn empty_and_nonfinite_error() {
        assert!(ErrorBar::ci(&[], 1.96).is_err());
        assert!(ErrorBar::ci(&[1.0, f64::NAN], 1.96).is_err());
    }
}
