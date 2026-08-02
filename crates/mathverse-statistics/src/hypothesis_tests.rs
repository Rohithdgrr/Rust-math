//! Hypothesis tests: t-test, Welch's t-test, paired t-test, chi-squared, F-test,
//! binomial test, Mann-Whitney U, Wilcoxon signed-rank.

use crate::descriptive::{mean, std_dev_sample, variance_sample};
use crate::distributions::{chi_squared_cdf, f_cdf, normal_cdf, student_t_cdf};
use crate::error::{MathError, MathResult};

/// Two-sample t-test (equal variance).
/// Returns (t-statistic, two-tailed p-value).
#[must_use]
pub fn t_test_two_sample(a: &[f64], b: &[f64]) -> (f64, f64) {
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (variance_sample(a), variance_sample(b));
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let df = na + nb - 2.0;
    let pooled = (((na - 1.0) * va + (nb - 1.0) * vb) / df).sqrt();
    let se = pooled * (1.0 / na + 1.0 / nb).sqrt();
    let t = (ma - mb) / se;
    let p = 2.0 * (1.0 - student_t_cdf(t.abs(), df));
    (t, p)
}

/// Welch's t-test (unequal variance).
/// Returns (t-statistic, approximate df, two-tailed p-value).
#[must_use]
pub fn welch_t_test(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (variance_sample(a), variance_sample(b));
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let se = (va / na + vb / nb).sqrt();
    let t = (ma - mb) / se;
    // Welch–Satterthwaite df
    let num = (va / na + vb / nb).powi(2);
    let den = (va / na).powi(2) / (na - 1.0) + (vb / nb).powi(2) / (nb - 1.0);
    let df = num / den;
    let p = 2.0 * (1.0 - student_t_cdf(t.abs(), df));
    (t, df, p)
}

/// Paired t-test.
/// Returns (t-statistic, two-tailed p-value).
///
/// # Errors
///
/// Returns [`MathError::DimensionMismatch`] if `a` and `b` differ in length.
pub fn paired_t_test(a: &[f64], b: &[f64]) -> MathResult<(f64, f64)> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch);
    }
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let m = mean(&diffs);
    let s = std_dev_sample(&diffs);
    let n = diffs.len() as f64;
    let t = m / (s / n.sqrt());
    let df = n - 1.0;
    let p = 2.0 * (1.0 - student_t_cdf(t.abs(), df));
    Ok((t, p))
}

/// One-sample t-test: test if mean equals `mu0`.
/// Returns (t-statistic, two-tailed p-value).
#[must_use]
pub fn one_sample_t_test(xs: &[f64], mu0: f64) -> (f64, f64) {
    let m = mean(xs);
    let s = std_dev_sample(xs);
    let n = xs.len() as f64;
    let t = (m - mu0) / (s / n.sqrt());
    let df = n - 1.0;
    let p = 2.0 * (1.0 - student_t_cdf(t.abs(), df));
    (t, p)
}

/// Two-sample F-test for equal variances.
/// Returns (F-statistic, two-tailed p-value).
#[must_use]
pub fn f_test_variance(a: &[f64], b: &[f64]) -> (f64, f64) {
    let va = variance_sample(a);
    let vb = variance_sample(b);
    let f = if va > vb { va / vb } else { vb / va };
    let df1 = (if va > vb { a.len() } else { b.len() }) as f64 - 1.0;
    let df2 = (if va > vb { b.len() } else { a.len() }) as f64 - 1.0;
    let p = 2.0 * (1.0 - f_cdf(f, df1, df2));
    (f, p)
}

/// One-way ANOVA F statistic: `(SSB/(k-1)) / (SSW/(N-k))`.
/// Returns (F-statistic, two-tailed p-value).
#[must_use]
pub fn one_way_anova(groups: &[&[f64]]) -> (f64, f64) {
    let k = groups.len();
    let n: usize = groups.iter().map(|g| g.len()).sum();
    let all: Vec<f64> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let grand = mean(&all);
    let mut ssb = 0.0;
    let mut ssw = 0.0;
    for g in groups {
        let mg = mean(g);
        ssb += g.len() as f64 * (mg - grand).powi(2);
        ssw += g.iter().map(|x| (x - mg).powi(2)).sum::<f64>();
    }
    let f = (ssb / (k - 1) as f64) / (ssw / (n - k) as f64);
    let p = 1.0 - f_cdf(f, (k - 1) as f64, (n - k) as f64);
    (f, p)
}

/// Chi-squared goodness-of-fit test.
/// Returns (χ² statistic, p-value). `observed` and `expected` must have same length.
///
/// # Errors
///
/// Returns [`MathError::DimensionMismatch`] if `observed` and `expected` differ in length.
/// Returns [`MathError::InvalidArgument`] if any expected value is zero.
pub fn chi_squared_gof(observed: &[f64], expected: &[f64]) -> MathResult<(f64, f64)> {
    if observed.len() != expected.len() {
        return Err(MathError::DimensionMismatch);
    }
    if expected.iter().any(|&e| e <= 0.0) {
        return Err(MathError::InvalidArgument(
            "expected values must be positive",
        ));
    }
    let chi2: f64 = observed
        .iter()
        .zip(expected)
        .map(|(o, e)| (o - e).powi(2) / e)
        .sum();
    let df = observed.len() as f64 - 1.0;
    let p = 1.0 - chi_squared_cdf(chi2, df);
    Ok((chi2, p))
}

/// Chi-squared test of independence (contingency table).
/// Returns (χ² statistic, p-value, degrees of freedom).
///
/// # Errors
///
/// Returns [`MathError::InvalidArgument`] if the table is jagged or empty.
pub fn chi_squared_independence(table: &[&[f64]]) -> MathResult<(f64, f64, f64)> {
    let rows = table.len();
    if rows == 0 {
        return Err(MathError::InvalidArgument("table must not be empty"));
    }
    let cols = table[0].len();
    if cols == 0 {
        return Err(MathError::InvalidArgument("table must not be empty"));
    }
    // Check for jagged table
    if !table.iter().all(|row| row.len() == cols) {
        return Err(MathError::InvalidArgument(
            "table must be rectangular (jagged input)",
        ));
    }
    let n: f64 = table.iter().flat_map(|r| r.iter()).sum();
    if n == 0.0 {
        return Err(MathError::InvalidArgument("table sum must be positive"));
    }
    let row_totals: Vec<f64> = table.iter().map(|r| r.iter().sum()).collect();
    let col_totals: Vec<f64> = (0..cols)
        .map(|j| table.iter().map(|r| r[j]).sum())
        .collect();
    let mut chi2 = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let expected = row_totals[i] * col_totals[j] / n;
            if expected <= 0.0 {
                return Err(MathError::InvalidArgument(
                    "expected cell values must be positive",
                ));
            }
            chi2 += (table[i][j] - expected).powi(2) / expected;
        }
    }
    let df = ((rows - 1) * (cols - 1)) as f64;
    let p = 1.0 - chi_squared_cdf(chi2, df);
    Ok((chi2, p, df))
}

/// Binomial test: probability of `k` or more extreme successes in `n` trials with probability `p0`.
/// Returns (two-tailed p-value).
#[must_use]
pub fn binomial_test(k: u64, n: u64, p0: f64) -> f64 {
    use crate::distributions::binomial_pmf;
    let observed_pmf = binomial_pmf(k, n, p0);
    let mut p = 0.0;
    for i in 0..=n {
        if binomial_pmf(i, n, p0) <= observed_pmf + 1e-15 {
            p += binomial_pmf(i, n, p0);
        }
    }
    p.min(1.0)
}

/// Mann-Whitney U test (two-sample, non-parametric).
/// Returns (U statistic, approximate two-tailed p-value).
#[must_use]
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n1 = a.len() as f64;
    let n2 = b.len() as f64;
    let mut all: Vec<(f64, usize)> = Vec::new();
    for &x in a {
        all.push((x, 0));
    }
    for &x in b {
        all.push((x, 1));
    }
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Assign ranks (handle ties with average rank)
    let mut ranks = vec![0.0; all.len()];
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j < all.len() && all[j].0 == all[i].0 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 0.5; // 1-indexed average
        for rank_idx in ranks.iter_mut().take(j).skip(i) {
            *rank_idx = avg_rank;
        }
        i = j;
    }
    let r1: f64 = ranks
        .iter()
        .zip(&all)
        .filter(|(_, (_, g))| *g == 0)
        .map(|(r, _)| r)
        .sum();
    let u1 = r1 - n1 * (n1 + 1.0) / 2.0;
    let u2 = n1 * n2 - u1;
    let u = u1.min(u2);
    // Normal approximation for large samples
    let mu = n1 * n2 / 2.0;
    let sigma = (n1 * n2 * (n1 + n2 + 1.0) / 12.0).sqrt();
    let z = (u - mu) / sigma;
    let p = 2.0 * (1.0 - normal_cdf(z.abs())); // two-tailed
    (u, p.max(0.0))
}

/// Wilcoxon signed-rank test (paired, non-parametric).
/// Returns (W statistic, approximate two-tailed p-value).
///
/// # Errors
///
/// Returns [`MathError::DimensionMismatch`] if `a` and `b` differ in length.
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> MathResult<(f64, f64)> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch);
    }
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let non_zero: Vec<(f64, usize)> = diffs
        .iter()
        .enumerate()
        .filter(|(_, d)| d.abs() > 1e-15)
        .map(|(i, &d)| (d, i))
        .collect();
    if non_zero.is_empty() {
        return Ok((0.0, 1.0));
    }
    let mut ranked: Vec<(f64, f64)> = non_zero.iter().map(|(d, _)| (*d, d.abs())).collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    // Assign ranks
    let mut ranks = vec![0.0; ranked.len()];
    let mut i = 0;
    while i < ranked.len() {
        let mut j = i;
        while j < ranked.len() && (ranked[j].1 - ranked[i].1).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 0.5;
        for rank_idx in ranks.iter_mut().take(j).skip(i) {
            *rank_idx = avg_rank;
        }
        i = j;
    }
    let mut w_pos = 0.0;
    let mut w_neg = 0.0;
    for (i, (d, _)) in ranked.iter().enumerate() {
        if *d > 0.0 {
            w_pos += ranks[i];
        } else {
            w_neg += ranks[i];
        }
    }
    let w = w_pos.min(w_neg);
    let n = non_zero.len() as f64;
    let mu = n * (n + 1.0) / 4.0;
    let sigma = (n * (n + 1.0) * (2.0 * n + 1.0) / 24.0).sqrt();
    let z = (w - mu) / sigma;
    let p = 2.0 * (1.0 - normal_cdf(z.abs()));
    Ok((w, p.max(0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_test_two_sample_test() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [6.0, 7.0, 8.0, 9.0, 10.0];
        let (t, p) = t_test_two_sample(&a, &b);
        assert!(t <= -5.0); // strongly different
        assert!(p < 0.01);
    }

    #[test]
    fn welch_test() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        let (t, df, p) = welch_t_test(&a, &b);
        assert!(t < 0.0);
        assert!(p < 0.05);
        assert!(df > 0.0);
    }

    #[test]
    fn paired_t_test_test() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.1, 2.2, 3.3, 4.4, 5.6];
        let (t, p) = paired_t_test(&a, &b).unwrap();
        assert!(t < 0.0); // b > a on average, so t = mean(a-b)/se < 0
        assert!(p < 0.05);
    }

    #[test]
    fn chi_squared_gof_test() {
        let observed = [20.0, 30.0, 50.0];
        let expected = [25.0, 25.0, 50.0];
        let (chi2, p) = chi_squared_gof(&observed, &expected).unwrap();
        assert!(chi2 > 0.0);
        assert!(p > 0.0);
    }

    #[test]
    fn binomial_test_test() {
        let p = binomial_test(8, 10, 0.5);
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn mann_whitney_test() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let (u, p) = mann_whitney_u(&a, &b);
        assert!(u >= 0.0);
        assert!(p < 0.1);
    }
}
