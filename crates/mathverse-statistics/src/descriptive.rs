//! Descriptive statistics: percentile, quantile, weighted mean, geometric/harmonic mean,
//! trimmed mean, skewness, kurtosis, standard error, summary.

/// General percentile (0-100) by linear interpolation.
pub fn percentile(xs: &[f64], p: f64) -> f64 {
    assert!(p >= 0.0 && p <= 100.0, "percentile must be 0..100");
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

/// General quantile (0.0–1.0) by linear interpolation.
pub fn quantile(xs: &[f64], q: f64) -> f64 {
    percentile(xs, q * 100.0)
}

/// Weighted mean: Σ(wᵢ·xᵢ) / Σ(wᵢ).
pub fn weighted_mean(xs: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(xs.len(), weights.len());
    let num: f64 = xs.iter().zip(weights).map(|(x, w)| x * w).sum();
    let den: f64 = weights.iter().sum();
    num / den
}

/// Geometric mean: (Πxᵢ)^(1/n). Requires positive values.
pub fn geometric_mean(xs: &[f64]) -> f64 {
    assert!(!xs.is_empty(), "geometric_mean requires non-empty input");
    let log_sum: f64 = xs.iter().map(|x| x.ln()).sum();
    (log_sum / xs.len() as f64).exp()
}

/// Harmonic mean: n / Σ(1/xᵢ). Requires positive values.
pub fn harmonic_mean(xs: &[f64]) -> f64 {
    assert!(!xs.is_empty(), "harmonic_mean requires non-empty input");
    let sum: f64 = xs.iter().map(|x| 1.0 / x).sum();
    xs.len() as f64 / sum
}

/// Trimmed mean: discard `trim` fraction from both tails, then average.
pub fn trimmed_mean(xs: &[f64], trim: f64) -> f64 {
    assert!(trim >= 0.0 && trim < 0.5, "trim must be in [0, 0.5)");
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let k = (trim * n as f64).floor() as usize;
    let trimmed = &v[k..n - k];
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

/// Winsorized mean: replace `trim` fraction from both tails with nearest value.
pub fn winsorized_mean(xs: &[f64], trim: f64) -> f64 {
    assert!(trim >= 0.0 && trim < 0.5);
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let k = (trim * n as f64).floor() as usize;
    let low = v[k];
    let high = v[n - 1 - k];
    let winsorized: Vec<f64> = v.iter().map(|&x| x.clamp(low, high)).collect();
    winsorized.iter().sum::<f64>() / n as f64
}

/// Population standard deviation (n denominator).
pub fn std_dev_pop(xs: &[f64]) -> f64 {
    variance_pop(xs).sqrt()
}

/// Sample variance (n-1 denominator).
pub fn variance_sample(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1).max(1) as f64
}

/// Population variance (n denominator).
pub fn variance_pop(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
}

/// Arithmetic mean.
pub fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Standard error of the mean: s / √n.
pub fn standard_error(xs: &[f64]) -> f64 {
    std_dev_sample(xs) / (xs.len() as f64).sqrt()
}

/// Sample standard deviation.
pub fn std_dev_sample(xs: &[f64]) -> f64 {
    variance_sample(xs).sqrt()
}

/// Sample skewness (adjusted Fisher-Pearson, G1).
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

/// Median absolute deviation.
pub fn mad(xs: &[f64]) -> f64 {
    let m = median(xs);
    let devs: Vec<f64> = xs.iter().map(|x| (x - m).abs()).collect();
    median(&devs)
}

/// Most frequent value; `None` when every value is unique.
pub fn mode(xs: &[f64]) -> Option<f64> {
    use std::collections::HashMap;
    let mut counts: HashMap<u64, (f64, usize)> = HashMap::new();
    for &x in xs {
        let e = counts.entry(x.to_bits()).or_insert((x, 0));
        e.1 += 1;
    }
    counts
        .values()
        .max_by_key(|(_, c)| *c)
        .filter(|(_, c)| *c > 1)
        .map(|(v, _)| *v)
}

/// Quartiles by nearest-rank: `q1 = sorted[n/4]`, `q2 = median`, `q3 = sorted[3n/4]`.
pub fn quartiles(xs: &[f64]) -> (f64, f64, f64) {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    (v[n / 4], median(&v), v[3 * n / 4])
}

/// Sample covariance.
pub fn covariance(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "covariance needs equal-length inputs");
    let (mx, my) = (mean(xs), mean(ys));
    xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / xs.len() as f64
}

/// Pearson correlation in `[-1, 1]`.
pub fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let (sx, sy) = (std_dev_sample(xs), std_dev_sample(ys));
    if sx == 0.0 || sy == 0.0 {
        return f64::NAN;
    }
    covariance(xs, ys) / (sx * sy) * xs.len() as f64 / (xs.len() - 1).max(1) as f64
}

/// Least-squares fit `y = slope·x + intercept`, returns `(slope, intercept, r²)`.
pub fn linear_regression(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    assert_eq!(xs.len(), ys.len());
    let (mx, my) = (mean(xs), mean(ys));
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        sxx += (x - mx).powi(2);
        sxy += (x - mx) * (y - my);
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let mut sst = 0.0;
    let mut ssr = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        sst += (y - my).powi(2);
        ssr += (y - (slope * x + intercept)).powi(2);
    }
    let r2 = 1.0 - ssr / sst;
    (slope, intercept, r2)
}

/// Normal-approximation confidence interval for the mean: `(lo, hi)`.
pub fn mean_ci(xs: &[f64], z: f64) -> (f64, f64) {
    let se = std_dev_sample(xs) / (xs.len() as f64).sqrt();
    (mean(xs) - z * se, mean(xs) + z * se)
}

/// Two-sample z statistic: `(m1 - m2) / sqrt(v1/n1 + v2/n2)`.
pub fn z_test(mean_a: f64, var_a: f64, n_a: usize, mean_b: f64, var_b: f64, n_b: usize) -> f64 {
    (mean_a - mean_b) / (var_a / n_a as f64 + var_b / n_b as f64).sqrt()
}

/// Median.
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

/// Interquartile range: Q3 - Q1.
pub fn iqr(xs: &[f64]) -> f64 {
    percentile(xs, 75.0) - percentile(xs, 25.0)
}

/// Coefficient of variation: std_dev / |mean|.
pub fn coefficient_of_variation(xs: &[f64]) -> f64 {
    let m = mean(xs);
    if m == 0.0 {
        return f64::NAN;
    }
    std_dev_sample(xs) / m.abs()
}

/// Range: max - min.
pub fn range(xs: &[f64]) -> f64 {
    let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max - min
}

/// Summary statistics struct.
#[derive(Debug, Clone)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Compute a full summary of the data.
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

    fn xs() -> Vec<f64> { vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] }

    #[test]
    fn percentile_test() {
        let v = xs();
        assert!((percentile(&v, 0.0) - 1.0).abs() < 1e-12);
        assert!((percentile(&v, 50.0) - 5.0).abs() < 1e-12);
        assert!((percentile(&v, 100.0) - 9.0).abs() < 1e-12);
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
        assert!((weighted_mean(&xs, &w) - (1.0*3.0 + 2.0*2.0 + 3.0*1.0) / 6.0).abs() < 1e-12);
    }

    #[test]
    fn geometric_harmonic_test() {
        let xs = [2.0, 8.0];
        assert!((geometric_mean(&xs) - 4.0).abs() < 1e-12);
        assert!((harmonic_mean(&xs) - 3.2).abs() < 1e-12);
    }

    #[test]
    fn trimmed_mean_test() {
        let xs = vec![1.0, 1.0, 2.0, 3.0, 100.0];
        assert!((trimmed_mean(&xs, 0.2) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn skewness_kurtosis_test() {
        let v = xs();
        assert!((skewness(&v)).abs() < 1e-10);
        assert!((kurtosis(&v) + 1.2).abs() < 0.1); // uniform-ish data has excess kurtosis ≈ -1.2
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
}
