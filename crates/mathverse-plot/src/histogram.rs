//! Histogram chart: binning delegated to `mathverse-statistics`.

use mathverse_statistics::{fd_rule, scott_rule, sqrt_rule, sturges_rule};

use crate::error::{PlotError, PlotResult};

/// Bin-count selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinningMethod {
    /// Pick a rule automatically (Sturges).
    Auto,
    /// Sturges' rule: `ceil(1 + log2(n))`.
    Sturges,
    /// Scott's normal reference rule.
    Scott,
    /// Freedman-Diaconis rule (IQR based).
    FreedmanDiaconis,
    /// Square-root rule: `ceil(sqrt(n))`.
    Sqrt,
    /// A fixed number of bins.
    Bins(usize),
}

/// Binned histogram data: edges (len = bins + 1) and per-bin counts.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    edges: Vec<f64>,
    counts: Vec<usize>,
}

impl Histogram {
    /// Bin `data` with the given method. Returns an error for empty data or
    /// when no binning rule applies.
    pub fn bin(data: &[f64], method: BinningMethod) -> PlotResult<Self> {
        if data.is_empty() {
            return Err(PlotError::InvalidData("empty data".into()));
        }
        let bins = match method {
            BinningMethod::Auto | BinningMethod::Sturges => sturges_rule(data),
            BinningMethod::Scott => scott_rule(data),
            BinningMethod::FreedmanDiaconis => fd_rule(data),
            BinningMethod::Sqrt => sqrt_rule(data),
            BinningMethod::Bins(n) => Some(n.max(1)),
        }
        .ok_or_else(|| PlotError::InvalidData("binning rule returned no bin count".into()))?;

        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !(min.is_finite() && max.is_finite()) {
            return Err(PlotError::InvalidData("non-finite data value".into()));
        }

        let span = max - min;
        if span <= 0.0 {
            return Ok(Self {
                edges: vec![min, min + 1.0],
                counts: vec![data.len()],
            });
        }
        let width = span / bins as f64;
        let mut edges = Vec::with_capacity(bins + 1);
        for i in 0..=bins {
            edges.push(min + i as f64 * width);
        }
        edges[bins] = max; // absorb float drift on the final edge

        let mut counts = vec![0usize; bins];
        for &v in data {
            let mut b = ((v - min) / width) as usize;
            if b >= bins {
                b = bins - 1;
            }
            counts[b] += 1;
        }
        Ok(Self { edges, counts })
    }

    /// Half-open bin edges, length `bins + 1`.
    #[must_use]
    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    /// Per-bin counts, length `bins`.
    #[must_use]
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_counts() {
        let h = Histogram::bin(&[1.0, 2.0, 2.0, 3.0, 3.0, 3.0], BinningMethod::Bins(3)).unwrap();
        assert_eq!(h.counts, vec![1, 2, 3]);
    }

    #[test]
    fn auto_bins_uses_sturges() {
        let data: Vec<f64> = (0..100).map(f64::from).collect();
        let h = Histogram::bin(&data, BinningMethod::Auto).unwrap();
        assert_eq!(h.edges().len(), h.counts().len() + 1);
        assert_eq!(h.counts().iter().sum::<usize>(), data.len());
        assert_eq!(h.counts().len(), 8); // sturges(100) = 1 + ceil(log2 100) = 8
    }

    #[test]
    fn constant_data_single_bin() {
        let h = Histogram::bin(&[5.0, 5.0, 5.0], BinningMethod::Bins(5)).unwrap();
        assert_eq!(h.counts, vec![3]);
    }

    #[test]
    fn empty_data_errors() {
        assert!(matches!(
            Histogram::bin(&[], BinningMethod::Auto),
            Err(PlotError::InvalidData(_))
        ));
    }
}
