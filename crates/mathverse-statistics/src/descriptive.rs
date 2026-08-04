//! Descriptive statistics: central tendency, dispersion, shape, percentiles,
//! quantiles, weighted and robust means, correlation, and linear regression.

use crate::error::{MathError, MathResult};

/// Arithmetic mean: sum(x) / n.
/// Returns NaN for empty input.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::mean;
///
/// assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
/// ```
#[must_use]
#[inline]
pub fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Sample variance (n - 1 denominator).
/// Returns NaN when n <= 1 (mathematically undefined).
///
/// # Examples
///
/// ```
/// use mathverse_statistics::variance_sample;
///
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!((variance_sample(&v) - 2.5).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn variance_sample(xs: &[f64]) -> f64 {
    if xs.len() <= 1 {
        return f64::NAN;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
}

/// Population variance (n denominator).
///
/// # Examples
///
/// ```
/// use mathverse_statistics::variance_pop;
///
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!((variance_pop(&v) - 2.0).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn variance_pop(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
}

/// Sample standard deviation: sqrt(variance_sample).
///
/// # Examples
///
/// ```
/// use mathverse_statistics::std_dev_sample;
///
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!((std_dev_sample(&v) - 1.5811388300841898).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn std_dev_sample(xs: &[f64]) -> f64 {
    variance_sample(xs).sqrt()
}

/// Population standard deviation: sqrt(variance_pop).
///
/// # Examples
///
/// ```
/// use mathverse_statistics::std_dev_pop;
///
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!((std_dev_pop(&v) - 1.4142135623730951).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn std_dev_pop(xs: &[f64]) -> f64 {
    variance_pop(xs).sqrt()
}

/// Median (average of two middle values for even-length input).
/// Returns NaN for empty input.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::median;
///
/// assert_eq!(median(&[1.0, 3.0, 2.0]), 2.0);
/// assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
/// ```
#[must_use]
#[inline]
pub fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// General percentile (0-100) by linear interpolation.
/// Returns NaN for empty input.
///
/// # Panics
///
/// Panics if p is not in [0.0, 100.0].
///
/// # Examples
///
/// ```
/// use mathverse_statistics::percentile;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(percentile(&data, 0.0), 1.0);
/// assert_eq!(percentile(&data, 50.0), 3.0);
/// assert_eq!(percentile(&data, 100.0), 5.0);
/// ```
#[must_use]
pub fn percentile(xs: &[f64], p: f64) -> f64 {
    assert!((0.0..=100.0).contains(&p), "percentile must be in [0, 100]");
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let rank = p / 100.0 * (v.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let frac = rank - lo as f64;
        v[lo] * (1.0 - frac) + v[hi] * frac
    }
}

/// General quantile (0.0-1.0) by linear interpolation.
/// Returns NaN for empty input.
///
/// # Panics
///
/// Panics if q is not in [0.0, 1.0].
///
/// # Examples
///
/// ```
/// use mathverse_statistics::quantile;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(quantile(&data, 0.5), 3.0);
/// assert_eq!(quantile(&data, 0.25), 2.0);
/// ```
#[must_use]
pub fn quantile(xs: &[f64], q: f64) -> f64 {
    percentile(xs, q * 100.0)
}

/// Weighted mean: sum(w_i * x_i) / sum(w_i).
///
/// # Errors
///
/// Returns MathError::DimensionMismatch if xs and weights differ in length.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::weighted_mean;
///
/// let xs = [1.0, 2.0, 3.0];
/// let w = [3.0, 2.0, 1.0];
/// assert!((weighted_mean(&xs, &w).unwrap() - 1.6666666666666667).abs() < 1e-12);
/// ```
pub fn weighted_mean(xs: &[f64], weights: &[f64]) -> MathResult<f64> {
    if xs.len() != weights.len() {
        return Err(MathError::DimensionMismatch);
    }
    let num: f64 = xs.iter().zip(weights).map(|(x, w)| x * w).sum();
    let den: f64 = weights.iter().sum();
    Ok(num / den)
}

/// Geometric mean: (product(x_i))^(1/n).
///
/// # Errors
///
/// Returns MathError::InvalidArgument if xs is empty.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::geometric_mean;
///
/// assert!((geometric_mean(&[2.0, 8.0]).unwrap() - 4.0).abs() < 1e-12);
/// ```
pub fn geometric_mean(xs: &[f64]) -> MathResult<f64> {
    if xs.is_empty() {
        return Err(MathError::InvalidArgument(
            "geometric_mean requires non-empty input",
        ));
    }
    let log_sum: f64 = xs.iter().map(|x| x.ln()).sum();
    Ok((log_sum / xs.len() as f64).exp())
}

/// Harmonic mean: n / sum(1 / x_i).
///
/// # Errors
///
/// Returns MathError::InvalidArgument if xs is empty.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::harmonic_mean;
///
/// assert!((harmonic_mean(&[2.0, 8.0]).unwrap() - 3.2).abs() < 1e-12);
/// ```
pub fn harmonic_mean(xs: &[f64]) -> MathResult<f64> {
    if xs.is_empty() {
        return Err(MathError::InvalidArgument(
            "harmonic_mean requires non-empty input",
        ));
    }
    let sum: f64 = xs.iter().map(|x| 1.0 / x).sum();
    Ok(xs.len() as f64 / sum)
}

/// Trimmed mean: discard trim fraction from both tails, then average.
///
/// # Errors
///
/// Returns MathError::InvalidArgument if trim is not in [0.0, 0.5).
///
/// # Examples
///
/// ```
/// use mathverse_statistics::trimmed_mean;
///
/// let data = [1.0, 1.0, 2.0, 3.0, 100.0];
/// assert!((trimmed_mean(&data, 0.2).unwrap() - 2.0).abs() < 1e-12);
/// ```
pub fn trimmed_mean(xs: &[f64], trim: f64) -> MathResult<f64> {
    if !(0.0..0.5).contains(&trim) {
        return Err(MathError::InvalidArgument("trim must be in [0, 0.5)"));
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let k = (trim * n as f64).floor() as usize;
    let trimmed = &v[k..n - k];
    Ok(trimmed.iter().sum::<f64>() / trimmed.len() as f64)
}

/// Winsorized mean: replace trim fraction from both tails with nearest value.
///
/// # Errors
///
/// Returns MathError::InvalidArgument if trim is not in [0.0, 0.5).
///
/// # Examples
///
/// ```
/// use mathverse_statistics::winsorized_mean;
///
/// let data = [1.0, 1.0, 2.0, 3.0, 100.0];
/// assert!((winsorized_mean(&data, 0.2).unwrap() - 2.0).abs() < 1e-12);
/// ```
pub fn winsorized_mean(xs: &[f64], trim: f64) -> MathResult<f64> {
    if !(0.0..0.5).contains(&trim) {
        return Err(MathError::InvalidArgument("trim must be in [0, 0.5)"));
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let k = (trim * n as f64).floor() as usize;
    let low = v[k];
    let high = v[n - 1 - k];
    let winsorized: Vec<f64> = v.iter().map(|&x| x.clamp(low, high)).collect();
    Ok(winsorized.iter().sum::<f64>() / n as f64)
}

/// Standard error of the mean: s / sqrt(n).
///
/// # Examples
///
/// ```
/// use mathverse_statistics::standard_error;
///
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let se = standard_error(&v);
/// assert!(se > 0.0);
/// ```
#[must_use]
#[inline]
pub fn standard_error(xs: &[f64]) -> f64 {
    std_dev_sample(xs) / (xs.len() as f64).sqrt()
}

/// Sample skewness (adjusted Fisher-Pearson, G1).
/// Returns 0.0 when n < 3 or variance is zero.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::skewness;
///
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// assert!(skewness(&v).abs() < 1e-10);
/// ```
#[must_use]
pub fn skewness(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 3 {
        return 0.0;
    }
    let m = mean(xs);
    let s = std_dev_sample(xs);
    if s == 0.0 {
        return 0.0;
    }
    let sum3: f64 = xs.iter().map(|x| ((x - m) / s).powi(3)).sum();
    let nf = n as f64;
    sum3 * nf / ((nf - 1.0) * (nf - 2.0))
}

/// Sample excess kurtosis (Fisher, G2).
/// Returns 0.0 when n < 4 or variance is zero.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::kurtosis;
///
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// assert!((kurtosis(&v) + 1.2).abs() < 0.1);
/// ```
#[must_use]
pub fn kurtosis(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 4 {
        return 0.0;
    }
    let m = mean(xs);
    let s = std_dev_sample(xs);
    if s == 0.0 {
        return 0.0;
    }
    let sum4: f64 = xs.iter().map(|x| ((x - m) / s).powi(4)).sum();
    let nf = n as f64;
    (sum4 * nf * (nf + 1.0)) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0))
        - 3.0 * (nf - 1.0).powi(2) / ((nf - 2.0) * (nf - 3.0))
}

/// Median absolute deviation (MAD).
///
/// # Examples
///
/// ```
/// use mathverse_statistics::{mad, median};
///
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!((mad(&v) - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn mad(xs: &[f64]) -> f64 {
    let m = median(xs);
    let devs: Vec<f64> = xs.iter().map(|x| (x - m).abs()).collect();
    median(&devs)
}

/// Most frequent value; returns None when every value is unique.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::mode;
///
/// assert_eq!(mode(&[1.0, 1.0, 2.0, 3.0]), Some(1.0));
/// assert_eq!(mode(&[1.0, 2.0, 3.0]), None);
/// ```
#[must_use]
pub fn mode(xs: &[f64]) -> Option<f64> {
    use alloc::collections::BTreeMap;
    let mut counts: BTreeMap<u64, (f64, usize)> = BTreeMap::new();
    for &x in xs {
        // Normalize -0.0 to 0.0 so they are counted together
        let normalized = if x == 0.0 { 0.0 } else { x };
        let e = counts
            .entry(normalized.to_bits())
            .or_insert((normalized, 0));
        e.1 += 1;
    }
    counts
        .values()
        .max_by_key(|(_, c)| *c)
        .filter(|(_, c)| *c > 1)
        .map(|(v, _)| *v)
}

/// Quartiles by linear interpolation: (q1, median, q3).
/// Consistent with the `percentile` function.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::quartiles;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let (q1, q2, q3) = quartiles(&data);
/// assert!((q1 - 3.0).abs() < 0.1);
/// assert_eq!(q2, 5.0);
/// assert!((q3 - 7.0).abs() < 0.1);
/// ```
#[must_use]
pub fn quartiles(xs: &[f64]) -> (f64, f64, f64) {
    (percentile(xs, 25.0), median(xs), percentile(xs, 75.0))
}

/// Population covariance: sum((x_i - x_bar)(y_i - y_bar)) / n.
/// Note: This uses the population formula (n denominator), not the sample formula (n-1).
///
/// # Errors
///
/// Returns MathError::DimensionMismatch if xs and ys differ in length.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::covariance;
///
/// let xs = [1.0, 2.0, 3.0, 4.0];
/// let ys = [2.0, 4.0, 6.0, 8.0];
/// assert!((covariance(&xs, &ys).unwrap() - 2.5).abs() < 1e-12);
/// ```
pub fn covariance(xs: &[f64], ys: &[f64]) -> MathResult<f64> {
    if xs.len() != ys.len() {
        return Err(MathError::DimensionMismatch);
    }
    let (mx, my) = (mean(xs), mean(ys));
    let cov: f64 = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (x - mx) * (y - my))
        .sum::<f64>()
        / xs.len() as f64;
    Ok(cov)
}

/// Pearson correlation coefficient in [-1, 1].
/// Returns NaN if either sequence has zero variance.
///
/// # Errors
///
/// Returns MathError::DimensionMismatch if xs and ys differ in length.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::pearson;
///
/// let xs = [1.0, 2.0, 3.0, 4.0];
/// let ys = [3.0, 5.0, 7.0, 9.0];
/// assert!((pearson(&xs, &ys).unwrap() - 1.0).abs() < 1e-12);
/// ```
pub fn pearson(xs: &[f64], ys: &[f64]) -> MathResult<f64> {
    if xs.len() != ys.len() {
        return Err(MathError::DimensionMismatch);
    }
    let (sx, sy) = (std_dev_sample(xs), std_dev_sample(ys));
    if sx == 0.0 || sy == 0.0 {
        return Ok(f64::NAN);
    }
    let n = xs.len() as f64;
    Ok(covariance(xs, ys)? / (sx * sy) * n / (n - 1.0).max(1.0))
}

/// Least-squares fit y = slope * x + intercept.
/// Returns (slope, intercept, r_squared).
///
/// # Errors
///
/// Returns MathError::DimensionMismatch if xs and ys differ in length.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::linear_regression;
///
/// let xs = [1.0, 2.0, 3.0, 4.0];
/// let ys = [3.0, 5.0, 7.0, 9.0];
/// let (slope, intercept, r2) = linear_regression(&xs, &ys).unwrap();
/// assert!((slope - 2.0).abs() < 1e-12);
/// assert!((intercept - 1.0).abs() < 1e-12);
/// assert!((r2 - 1.0).abs() < 1e-12);
/// ```
pub fn linear_regression(xs: &[f64], ys: &[f64]) -> MathResult<(f64, f64, f64)> {
    if xs.len() != ys.len() {
        return Err(MathError::DimensionMismatch);
    }
    let (mx, my) = (mean(xs), mean(ys));
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        sxx = (x - mx).powi(2).mul_add(1.0, sxx);
        sxy = (x - mx).mul_add(y - my, sxy);
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let mut sst = 0.0;
    let mut ssr = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        sst = (y - my).powi(2).mul_add(1.0, sst);
        ssr = (y - (slope * x + intercept)).powi(2).mul_add(1.0, ssr);
    }
    let r2 = 1.0 - ssr / sst;
    Ok((slope, intercept, r2))
}

/// Normal-approximation confidence interval for the mean.
/// Returns (lower, upper) bounds.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::mean_ci;
///
/// let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let (lo, hi) = mean_ci(&xs, 1.96);
/// assert!(lo < 3.0 && hi > 3.0);
/// ```
#[must_use]
#[inline]
pub fn mean_ci(xs: &[f64], z: f64) -> (f64, f64) {
    let se = standard_error(xs);
    let m = mean(xs);
    (m - z * se, m + z * se)
}

/// Two-sample z statistic: (m1 - m2) / sqrt(v1/n1 + v2/n2).
///
/// # Examples
///
/// ```
/// use mathverse_statistics::z_test;
///
/// let z = z_test(5.0, 4.0, 10, 5.0, 4.0, 10);
/// assert_eq!(z, 0.0);
/// ```
#[must_use]
#[inline]
pub fn z_test(mean_a: f64, var_a: f64, n_a: usize, mean_b: f64, var_b: f64, n_b: usize) -> f64 {
    (mean_a - mean_b) / (var_a / n_a as f64 + var_b / n_b as f64).sqrt()
}

/// Interquartile range: Q3 - Q1.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::iqr;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// assert!((iqr(&data) - 4.0).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn iqr(xs: &[f64]) -> f64 {
    percentile(xs, 75.0) - percentile(xs, 25.0)
}

/// Coefficient of variation: std_dev / |mean|.
/// Returns NaN if the mean is zero.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::coefficient_of_variation;
///
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let cv = coefficient_of_variation(&v);
/// assert!(cv > 0.0);
/// ```
#[must_use]
#[inline]
pub fn coefficient_of_variation(xs: &[f64]) -> f64 {
    let m = mean(xs);
    if m == 0.0 {
        return f64::NAN;
    }
    std_dev_sample(xs) / m.abs()
}

/// Range: max - min.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::range;
///
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// assert!((range(&v) - 8.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn range(xs: &[f64]) -> f64 {
    let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max - min
}

/// Summary statistics struct produced by [`describe`].
#[derive(Debug, Clone)]
#[must_use]
pub struct Summary {
    /// Number of observations.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// First quartile (25th percentile).
    pub q1: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// Third quartile (75th percentile).
    pub q3: f64,
    /// Maximum value.
    pub max: f64,
    /// Sample skewness (adjusted Fisher-Pearson G1).
    pub skewness: f64,
    /// Sample excess kurtosis (Fisher G2).
    pub kurtosis: f64,
}

/// Sturges' rule for histogram bin count: `ceil(1 + log2(n))`.
///
/// Returns `None` for empty input.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::sturges_rule;
///
/// assert_eq!(sturges_rule(&[1.0, 2.0, 3.0]), Some(3));
/// ```
#[must_use]
pub fn sturges_rule(xs: &[f64]) -> Option<usize> {
    if xs.is_empty() {
        return None;
    }
    Some(1 + (xs.len() as f64).log2().ceil() as usize)
}

/// Scott's normal reference rule for histogram bin count:
/// `ceil((max - min) / (3.49 * sigma / n^(1/3)))`.
///
/// Returns `None` for empty data or data with zero range.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::scott_rule;
///
/// let data = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
/// assert!(scott_rule(&data).unwrap() >= 1);
/// ```
#[must_use]
pub fn scott_rule(xs: &[f64]) -> Option<usize> {
    if xs.is_empty() {
        return None;
    }
    let r = range(xs);
    if r <= 0.0 {
        return Some(1);
    }
    let n = xs.len() as f64;
    let width = 3.49 * std_dev_sample(xs) / n.cbrt();
    if width <= 0.0 || width.is_nan() {
        return None;
    }
    Some((r / width).ceil().max(1.0) as usize)
}

/// Freedman-Diaconis rule for histogram bin count using the IQR:
/// `ceil((max - min) / (2 * IQR / n^(1/3)))`.
///
/// Returns `None` for empty data or a zero IQR.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::fd_rule;
///
/// let data = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// assert!(fd_rule(&data).unwrap() >= 1);
/// ```
#[must_use]
pub fn fd_rule(xs: &[f64]) -> Option<usize> {
    if xs.is_empty() {
        return None;
    }
    let r = range(xs);
    if r <= 0.0 {
        return Some(1);
    }
    let n = xs.len() as f64;
    let width = 2.0 * iqr(xs) / n.cbrt();
    if width <= 0.0 || width.is_nan() {
        return None;
    }
    Some((r / width).ceil().max(1.0) as usize)
}

/// Square-root rule for histogram bin count: `ceil(sqrt(n))`.
///
/// Returns `None` for empty data.
#[must_use]
pub fn sqrt_rule(xs: &[f64]) -> Option<usize> {
    if xs.is_empty() {
        return None;
    }
    Some((xs.len() as f64).sqrt().ceil() as usize)
}

/// Compute a full summary of the data.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::describe;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let s = describe(&data);
/// assert_eq!(s.n, 5);
/// assert_eq!(s.mean, 3.0);
/// assert_eq!(s.min, 1.0);
/// assert_eq!(s.max, 5.0);
/// ```
#[must_use]
pub fn describe(xs: &[f64]) -> Summary {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    Summary {
        n: xs.len(),
        mean: mean(xs),
        std_dev: std_dev_sample(xs),
        min: v.first().copied().unwrap_or(f64::NAN),
        q1: percentile(xs, 25.0),
        median: median(xs),
        q3: percentile(xs, 75.0),
        max: v.last().copied().unwrap_or(f64::NAN),
        skewness: skewness(xs),
        kurtosis: kurtosis(xs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xs() -> Vec<f64> {
        (1..=9).map(f64::from).collect()
    }

    #[test]
    fn percentile_test() {
        let v = xs();
        assert!((percentile(&v, 0.0) - 1.0).abs() < 1e-12);
        assert!((percentile(&v, 50.0) - 5.0).abs() < 1e-12);
        assert!((percentile(&v, 100.0) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn percentile_empty() {
        assert!(percentile(&[], 50.0).is_nan());
    }

    #[test]
    fn quantile_test() {
        let v = xs();
        assert!((quantile(&v, 0.5) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn weighted_mean_test() {
        let xs = [1.0, 2.0, 3.0];
        let w = [3.0, 2.0, 1.0];
        assert!(
            (weighted_mean(&xs, &w).unwrap() - (1.0 * 3.0 + 2.0 * 2.0 + 3.0 * 1.0) / 6.0).abs()
                < 1e-12
        );
    }

    #[test]
    fn weighted_mean_dim_mismatch() {
        assert_eq!(
            weighted_mean(&[1.0, 2.0], &[1.0]),
            Err(MathError::DimensionMismatch)
        );
    }

    #[test]
    fn geometric_harmonic_test() {
        let xs = [2.0, 8.0];
        assert!((geometric_mean(&xs).unwrap() - 4.0).abs() < 1e-12);
        assert!((harmonic_mean(&xs).unwrap() - 3.2).abs() < 1e-12);
    }

    #[test]
    fn geometric_empty() {
        assert_eq!(
            geometric_mean(&[]),
            Err(MathError::InvalidArgument(
                "geometric_mean requires non-empty input"
            ))
        );
    }

    #[test]
    fn harmonic_empty() {
        assert_eq!(
            harmonic_mean(&[]),
            Err(MathError::InvalidArgument(
                "harmonic_mean requires non-empty input"
            ))
        );
    }

    #[test]
    fn trimmed_mean_test() {
        let xs = vec![1.0, 1.0, 2.0, 3.0, 100.0];
        assert!((trimmed_mean(&xs, 0.2).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn trimmed_mean_invalid_trim() {
        assert_eq!(
            trimmed_mean(&[1.0, 2.0], 0.6),
            Err(MathError::InvalidArgument("trim must be in [0, 0.5)"))
        );
    }

    #[test]
    fn winsorized_mean_test() {
        let xs = vec![1.0, 1.0, 2.0, 3.0, 100.0];
        assert!((winsorized_mean(&xs, 0.2).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn skewness_kurtosis_test() {
        let v = xs();
        assert!(skewness(&v).abs() < 1e-10);
        assert!((kurtosis(&v) + 1.2).abs() < 0.1);
    }

    #[test]
    fn iqr_cv_range_test() {
        let v = xs();
        assert!((iqr(&v) - 4.0).abs() < 1e-12);
        assert!((range(&v) - 8.0).abs() < 1e-12);
        assert!((coefficient_of_variation(&v) - std_dev_sample(&v) / 5.0).abs() < 1e-12);
    }

    #[test]
    fn describe_test() {
        let s = describe(&xs());
        assert_eq!(s.n, 9);
        assert!((s.mean - 5.0).abs() < 1e-12);
        assert!((s.min - 1.0).abs() < 1e-12);
        assert!((s.max - 9.0).abs() < 1e-12);
    }

    #[test]
    fn linear_regression_test() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [3.0, 5.0, 7.0, 9.0];
        let (s, i, r2) = linear_regression(&xs, &ys).unwrap();
        assert!((s - 2.0).abs() < 1e-12);
        assert!((i - 1.0).abs() < 1e-12);
        assert!((r2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn linear_regression_dim_mismatch() {
        assert_eq!(
            linear_regression(&[1.0, 2.0], &[3.0]),
            Err(MathError::DimensionMismatch)
        );
    }

    #[test]
    fn covariance_test() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [2.0, 4.0, 6.0, 8.0];
        // population covariance: sum((x-mx)(y-my)) / n
        assert!((covariance(&xs, &ys).unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn covariance_dim_mismatch() {
        assert_eq!(
            covariance(&[1.0, 2.0], &[3.0]),
            Err(MathError::DimensionMismatch)
        );
    }

    #[test]
    fn pearson_test() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [3.0, 5.0, 7.0, 9.0];
        assert!((pearson(&xs, &ys).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_dim_mismatch() {
        assert_eq!(
            pearson(&[1.0, 2.0], &[3.0]),
            Err(MathError::DimensionMismatch)
        );
    }

    #[test]
    fn mode_test() {
        assert_eq!(mode(&[1.0, 1.0, 2.0, 3.0]), Some(1.0));
        assert_eq!(mode(&[1.0, 2.0, 3.0]), None);
    }

    #[test]
    fn quartiles_test() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(quartiles(&data), (3.0, 5.0, 7.0));
    }

    #[test]
    fn binning_rules_test() {
        assert_eq!(sturges_rule(&[]), None);
        assert_eq!(sturges_rule(&[1.0, 2.0, 3.0]), Some(3));
        assert_eq!(sqrt_rule(&[1.0; 100]), Some(10));
        assert_eq!(scott_rule(&[1.0; 10]), Some(1));
        assert_eq!(fd_rule(&[1.0; 10]), Some(1));
        let data = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0];
        assert!(scott_rule(&data).is_some() && scott_rule(&data).unwrap() >= 1);
        assert!(fd_rule(&data).is_some() && fd_rule(&data).unwrap() >= 1);
    }
}
