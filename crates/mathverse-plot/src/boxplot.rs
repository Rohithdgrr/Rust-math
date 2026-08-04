//! Box plot (Tukey) statistics, computed via `mathverse-statistics`.

use crate::error::{PlotError, PlotResult};
use mathverse_statistics::iqr;

/// Tukey five-number summary with whiskers and labeled outliers.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxStats {
    /// Lower whisker (most extreme inside value below the fence).
    pub min: f64,
    /// First quartile.
    pub q1: f64,
    /// Median.
    pub median: f64,
    /// Third quartile.
    pub q3: f64,
    /// Upper whisker (most extreme inside value above the fence).
    pub max: f64,
    /// Values beyond the `1.5 * IQR` fences.
    pub outliers: Vec<f64>,
}

impl BoxStats {
    /// Compute Tukey box stats. Whiskers extend to the most extreme values
    /// inside the `1.5 * IQR` fences; values beyond are outliers.
    ///
    /// # Errors
    ///
    /// Returns `PlotError::InvalidData` for empty or non-finite input.
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_plot::boxplot::BoxStats;
    /// let stats = BoxStats::compute(&[1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(stats.median, 2.0);
    /// ```
    pub fn compute(xs: &[f64]) -> PlotResult<Self> {
        if xs.is_empty() {
            return Err(PlotError::InvalidData("empty data".into()));
        }
        if xs.iter().any(|x| !x.is_finite()) {
            return Err(PlotError::InvalidData("non-finite data value".into()));
        }
        let (q1, median, q3) = mathverse_statistics::quartiles(xs);
        let spread = iqr(xs);
        let (lo, hi) = (q1 - 1.5 * spread, q3 + 1.5 * spread);
        let (mut lo_w, mut hi_w) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut outliers = Vec::new();
        for &x in xs {
            if x < lo || x > hi {
                outliers.push(x);
            } else {
                lo_w = lo_w.min(x);
                hi_w = hi_w.max(x);
            }
        }
        // Degenerate fences (IQR == 0): neither whisker may stay infinite.
        if lo_w.is_infinite() {
            lo_w = q1;
            hi_w = q1;
        }
        Ok(Self {
            min: lo_w,
            q1,
            median,
            q3,
            max: hi_w,
            outliers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_data_no_outliers() {
        let stats = BoxStats::compute(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        assert_eq!(stats.q1, 3.0);
        assert_eq!(stats.median, 5.0);
        assert_eq!(stats.q3, 7.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 9.0);
        assert!(stats.outliers.is_empty());
    }

    #[test]
    fn labels_single_outlier() {
        let stats =
            BoxStats::compute(&[1.0; 8].into_iter().chain([100.0]).collect::<Vec<_>>()).unwrap();
        assert_eq!(stats.outliers, vec![100.0]);
        assert_eq!(stats.max, 1.0);
    }

    #[test]
    fn empty_and_nonfinite_error() {
        assert!(BoxStats::compute(&[]).is_err());
        assert!(BoxStats::compute(&[1.0, f64::NAN]).is_err());
    }
}
