//! Convenience PDF overlay generators via `mathverse-probability`.

use mathverse_probability::ContinuousDist;

/// Sample a generic PDF function on a uniform grid over `[lo, hi]` with `n`
/// points. Returns `(x, pdf(x))` pairs, suitable for `SvgPlot::add_series`.
#[must_use]
pub fn sample_pdf(pdf: &dyn Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> Vec<(f64, f64)> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(lo, pdf(lo))];
    }
    (0..n)
        .map(|i| {
            let x = lo + (hi - lo) * i as f64 / (n - 1) as f64;
            (x, pdf(x))
        })
        .collect()
}

/// Sample a normal PDF with given `mu` and `sigma`.
#[must_use]
pub fn sample_normal(mu: f64, sigma: f64, lo: f64, hi: f64, n: usize) -> Vec<(f64, f64)> {
    let dist = mathverse_probability::distributions::Normal { mu, sigma };
    sample_pdf(&|x| dist.pdf(x), lo, hi, n)
}

/// Sample a standard normal PDF.
#[must_use]
pub fn sample_standard_normal(lo: f64, hi: f64, n: usize) -> Vec<(f64, f64)> {
    let dist = mathverse_probability::distributions::Normal {
        mu: 0.0,
        sigma: 1.0,
    };
    sample_pdf(&|x| dist.pdf(x), lo, hi, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_peak_at_mean() {
        let curve = sample_normal(0.0, 1.0, -3.0, 3.0, 101);
        let peak = curve
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert!((peak.0 - 0.0).abs() < 0.01);
    }

    #[test]
    fn pdf_curve_spans_grid() {
        let curve = sample_standard_normal(-2.0, 2.0, 5);
        assert_eq!(curve.len(), 5);
        assert!((curve[0].0 - (-2.0)).abs() < 1e-9);
        assert!((curve[4].0 - 2.0).abs() < 1e-9);
    }
}
