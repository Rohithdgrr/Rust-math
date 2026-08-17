//! Survival analysis: Kaplan-Meier and Nelson-Aalen estimators, Cox
//! proportional-hazards regression, Weibull accelerated failure time models,
//! and the two-sample log-rank test. Censoring is right-censoring: `event =
//! false` marks a censored observation.

use crate::distributions::{ChiSquared, ContinuousDist};

/// Sort a copy of `(time, event)` pairs by time (stable, censored first on
/// ties so that no event is counted before the subjects leaving at the same
/// time).
fn sort_samples(times: &[f64], events: &[bool]) -> Vec<(f64, bool)> {
    let mut pairs: Vec<(f64, bool)> = times.iter().zip(events).map(|(&t, &e)| (t, e)).collect();
    pairs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then_with(|| b.1.cmp(&a.1))
    });
    pairs
}

/// Kaplan-Meier product-limit estimate. Returns `(event_time, S(t))` pairs at
/// every distinct event time.
///
/// # Panics
/// Panics if the slices differ in length.
#[must_use]
pub fn kaplan_meier(times: &[f64], events: &[bool]) -> Vec<(f64, f64)> {
    assert_eq!(times.len(), events.len(), "times and events must be equal length");
    let pairs = sort_samples(times, events);
    let n = pairs.len();
    let mut out = Vec::new();
    let mut s = 1.0;
    let mut at_risk = n as f64;
    let mut i = 0;
    while i < n {
        let t = pairs[i].0;
        let mut d = 0.0;
        let mut j = i;
        while j < n && (pairs[j].0 - t).abs() < f64::EPSILON {
            if pairs[j].1 {
                d += 1.0;
            }
            j += 1;
        }
        if d > 0.0 && at_risk > 0.0 {
            s *= 1.0 - d / at_risk;
            out.push((t, s));
        }
        at_risk -= (j - i) as f64;
        i = j;
    }
    out
}

/// Survival probability `S(t)` from the Kaplan-Meier estimate (step function;
/// `S(t) = 1` before the first event, `0` after the last).
#[must_use]
pub fn kaplan_meier_survival(times: &[f64], events: &[bool], t: f64) -> f64 {
    let steps = kaplan_meier(times, events);
    let mut s = 1.0;
    for &(ti, si) in &steps {
        if ti <= t {
            s = si;
        } else {
            break;
        }
    }
    s
}

/// Nelson-Aalen cumulative hazard estimate. Returns `(event_time, H(t))`
/// pairs at every distinct event time.
#[must_use]
pub fn nelson_aalen(times: &[f64], events: &[bool]) -> Vec<(f64, f64)> {
    assert_eq!(times.len(), events.len(), "times and events must be equal length");
    let pairs = sort_samples(times, events);
    let n = pairs.len();
    let mut out = Vec::new();
    let mut h = 0.0;
    let mut at_risk = n as f64;
    let mut i = 0;
    while i < n {
        let t = pairs[i].0;
        let mut d = 0.0;
        let mut j = i;
        while j < n && (pairs[j].0 - t).abs() < f64::EPSILON {
            if pairs[j].1 {
                d += 1.0;
            }
            j += 1;
        }
        if d > 0.0 && at_risk > 0.0 {
            h += d / at_risk;
            out.push((t, h));
        }
        at_risk -= (j - i) as f64;
        i = j;
    }
    out
}

/// Two-sample log-rank test. Returns `(chi-square statistic, p-value)`.
/// Large statistics reject the null hypothesis of equal survival curves.
#[must_use]
pub fn log_rank_test(
    times_a: &[f64],
    events_a: &[bool],
    times_b: &[f64],
    events_b: &[bool],
) -> (f64, f64) {
    assert_eq!(times_a.len(), events_a.len(), "times_a and events_a must be equal length");
    assert_eq!(times_b.len(), events_b.len(), "times_b and events_b must be equal length");
    let mut a: Vec<(f64, bool)> = times_a.iter().zip(events_a).map(|(&t, &e)| (t, e)).collect();
    let mut b: Vec<(f64, bool)> = times_b.iter().zip(events_b).map(|(&t, &e)| (t, e)).collect();
    a.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    b.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

    let (mut ia, mut ib) = (0usize, 0usize);
    let (mut n1, mut n2) = (a.len() as f64, b.len() as f64);
    let (mut z, mut v) = (0.0, 0.0);
    while ia < a.len() || ib < b.len() {
        let ta = a.get(ia).map(|p| p.0).unwrap_or(f64::INFINITY);
        let tb = b.get(ib).map(|p| p.0).unwrap_or(f64::INFINITY);
        let t = ta.min(tb);
        let (mut d1, mut d2) = (0.0, 0.0);
        let (mut c1, mut c2) = (0.0, 0.0);
        while ia < a.len() && a[ia].0 == t {
            if a[ia].1 {
                d1 += 1.0;
            }
            c1 += 1.0;
            ia += 1;
        }
        while ib < b.len() && b[ib].0 == t {
            if b[ib].1 {
                d2 += 1.0;
            }
            c2 += 1.0;
            ib += 1;
        }
        let d = d1 + d2;
        if d > 0.0 {
            let e1 = d * n1 / (n1 + n2);
            z += d1 - e1;
            if n1 + n2 > 1.0 {
                v += n1 * n2 * d * (n1 + n2 - d) / ((n1 + n2).powi(2) * (n1 + n2 - 1.0));
            }
        }
        n1 -= c1;
        n2 -= c2;
    }
    if v <= 0.0 {
        return (0.0, 1.0);
    }
    let chi2 = z * z / v;
    let p = 1.0 - ChiSquared { k: 1.0 }.cdf(chi2);
    (chi2, p)
}

/// Result of a Cox proportional-hazards fit.
#[must_use]
#[derive(Clone, Debug)]
pub struct CoxPHResult {
    /// Regression coefficients `Î²`.
    pub coefficients: Vec<f64>,
    /// Base-line cumulative hazard at the observed event times.
    pub baseline_hazard: Vec<(f64, f64)>,
    /// Value of the partial log-likelihood at the optimum.
    pub log_likelihood: f64,
    /// Number of Newton-Raphson iterations used.
    pub iterations: usize,
}

/// Fit a Cox proportional-hazards model by maximizing the partial likelihood
/// (Breslow tie handling, Newton-Raphson with backtracking).
///
/// `covariates` is `[sample][feature]`; `events[i]` marks an observed failure.
///
/// # Errors
/// Returns an error if the input shapes are inconsistent, no events are
/// observed, or the Hessian is singular.
pub fn cox_ph(
    times: &[f64],
    events: &[bool],
    covariates: &[Vec<f64>],
) -> Result<CoxPHResult, String> {
    let n = times.len();
    if n == 0 {
        return Err("cox_ph needs at least one sample".into());
    }
    if events.len() != n || covariates.len() != n {
        return Err("cox_ph input shapes do not match".into());
    }
    if !events.iter().any(|&e| e) {
        return Err("cox_ph needs at least one observed event".into());
    }
    let p = covariates[0].len();
    if p == 0 {
        return Err("cox_ph needs at least one covariate".into());
    }
    if covariates.iter().any(|row| row.len() != p) {
        return Err("cox_ph covariate rows have unequal length".into());
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| times[i].partial_cmp(&times[j]).unwrap());

    let mut beta = vec![0.0; p];
    let mut ll = partial_log_likelihood(&beta, &idx, times, events, covariates);
    let mut iterations = 0;
    for _ in 0..100 {
        iterations += 1;
        let (g, h) = partial_gradient_hessian(&beta, &idx, times, events, covariates);
        if g.iter().all(|&gi| gi.abs() < 1e-9) {
            break;
        }
        // Solve HÂ·d = -g by Gaussian elimination (H is negative semi-definite).
        let d = match solve_negative(&h, &g, p) {
            Some(d) => d,
            None => break,
        };
        let mut step = 1.0;
        let mut improved = false;
        for _ in 0..20 {
            let trial: Vec<f64> = beta.iter().zip(&d).map(|(&b, &di)| b + step * di).collect();
            let trial_ll = partial_log_likelihood(&trial, &idx, times, events, covariates);
            if trial_ll > ll + 1e-12 {
                beta = trial;
                ll = trial_ll;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            break;
        }
    }

    // Baseline cumulative hazard at event times: Breslow.
    let mut baseline = Vec::new();
    let mut h0 = 0.0;
    let mut i = 0;
    while i < n {
        let si = idx[i];
        let t = times[si];
        let mut d = 0.0;
        let mut j = i;
        // all samples with this event time
        while j < n && times[idx[j]] == t {
            if events[idx[j]] {
                d += 1.0;
            }
            j += 1;
        }
        if d > 0.0 {
            let mut risk = 0.0;
            for &k in &idx {
                if times[k] >= t {
                    let xk: f64 = beta.iter().zip(&covariates[k]).map(|(&b, &x)| b * x).sum();
                    risk += xk.exp();
                }
            }
            if risk > 0.0 {
                h0 += d / risk;
                baseline.push((t, h0));
            }
        }
        i = j;
    }

    Ok(CoxPHResult {
        coefficients: beta,
        baseline_hazard: baseline,
        log_likelihood: ll,
        iterations,
    })
}

fn partial_log_likelihood(
    beta: &[f64],
    idx: &[usize],
    _times: &[f64],
    events: &[bool],
    covariates: &[Vec<f64>],
) -> f64 {
    let n = idx.len();
    let mut ll = 0.0;
    let mut risk_sum = 0.0;
    let mut i = n;
    while i > 0 {
        i -= 1;
        let si = idx[i];
        let xi: f64 = beta.iter().zip(&covariates[si]).map(|(&b, &x)| b * x).sum();
        risk_sum += xi.exp();
        if events[si] {
            ll += xi - risk_sum.ln();
        }
    }
    ll
}

fn partial_gradient_hessian(
    beta: &[f64],
    idx: &[usize],
    _times: &[f64],
    events: &[bool],
    covariates: &[Vec<f64>],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = idx.len();
    let p = beta.len();
    let mut grad = vec![0.0; p];
    let mut hess = vec![vec![0.0; p]; p];
    // Scan risk sets in decreasing time order, maintaining the cumulative
    // weighted sums over the current risk set.
    let mut risk_sum = 0.0;
    let mut wsum = vec![0.0; p];
    let mut wouter = vec![vec![0.0; p]; p];
    let mut i = n;
    while i > 0 {
        i -= 1;
        let si = idx[i];
        let xi: f64 = beta.iter().zip(&covariates[si]).map(|(&b, &x)| b * x).sum();
        let ex = xi.exp();
        risk_sum += ex;
        for a in 0..p {
            wsum[a] += ex * covariates[si][a];
            for b in 0..p {
                wouter[a][b] += ex * covariates[si][a] * covariates[si][b];
            }
        }
        if events[si] {
            let inv = 1.0 / risk_sum;
            let means: Vec<f64> = wsum.iter().map(|&s| s * inv).collect();
            for a in 0..p {
                grad[a] += covariates[si][a] - means[a];
                for b in 0..p {
                    hess[a][b] -= wouter[a][b] * inv - means[a] * means[b];
                }
            }
        }
    }
    (grad, hess)
}

/// Solve `(-H)Â·d = g` for symmetric negative semi-definite H (H is the
/// Hessian). Returns `None` if the system is singular.
fn solve_negative(h: &[Vec<f64>], g: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut a: Vec<Vec<f64>> = (0..p)
        .map(|i| {
            (0..p)
                .map(|j| -h[i][j])
                .chain(std::iter::once(g[i]))
                .collect()
        })
        .collect();
    for col in 0..p {
        let mut pivot = col;
        for r in col + 1..p {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        for r in col + 1..p {
            let factor = a[r][col] / a[col][col];
            for c in col..=p {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    let mut x = vec![0.0; p];
    for r in (0..p).rev() {
        let mut sum = a[r][p];
        for c in r + 1..p {
            sum -= a[r][c] * x[c];
        }
        x[r] = sum / a[r][r];
    }
    Some(x)
}

/// Result of a Weibull accelerated-failure-time fit.
#[must_use]
#[derive(Clone, Debug)]
pub struct WeibullAFTResult {
    /// Shape parameter `k` (log-scale: `1/k` is the rate of `ln T`).
    pub shape: f64,
    /// Coefficients `Î²` with `Î»_i = exp(x_iÂ·Î²)`.
    pub coefficients: Vec<f64>,
    /// Value of the log-likelihood at the optimum.
    pub log_likelihood: f64,
}

/// Fit a Weibull AFT model `S(t|x) = exp(-(t/Î»(x))^k)` with
/// `Î»(x) = exp(xÂ·Î²)` by maximum likelihood (gradient ascent with
/// backtracking line search).
///
/// `covariates` is `[sample][feature]`. If every sample is censored the fit
/// is degenerate and an error is returned.
pub fn weibull_aft(
    times: &[f64],
    events: &[bool],
    covariates: &[Vec<f64>],
) -> Result<WeibullAFTResult, String> {
    let n = times.len();
    if n == 0 || events.len() != n || covariates.len() != n {
        return Err("weibull_aft input shapes do not match".into());
    }
    if !events.iter().any(|&e| e) {
        return Err("weibull_aft needs at least one observed event".into());
    }
    let p = covariates[0].len();
    if covariates.iter().any(|row| row.len() != p) {
        return Err("weibull_aft covariate rows have unequal length".into());
    }

    // Parameters: (ln k, Î²_0..Î²_{p-1}).
    let mut params = vec![0.0; p + 1];
    params[0] = (1.0f64).ln(); // k = 1
    let mut ll = weibull_ll(&params, times, events, covariates);
    for _ in 0..10_000 {
        let grad = weibull_grad(&params, times, events, covariates);
        if grad.iter().all(|&g| g.abs() < 1e-9) {
            break;
        }
        let mut step = 1.0;
        let mut improved = false;
        for _ in 0..30 {
            let trial: Vec<f64> = params
                .iter()
                .zip(&grad)
                .map(|(&p, &g)| p + step * g)
                .collect();
            let trial_ll = weibull_ll(&trial, times, events, covariates);
            if trial_ll > ll + 1e-12 {
                params = trial;
                ll = trial_ll;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            break;
        }
    }
    Ok(WeibullAFTResult {
        shape: params[0].exp(),
        coefficients: params[1..].to_vec(),
        log_likelihood: ll,
    })
}

fn weibull_ll(params: &[f64], times: &[f64], events: &[bool], covariates: &[Vec<f64>]) -> f64 {
    let k = params[0].exp();
    let mut ll = 0.0;
    for (i, &t) in times.iter().enumerate() {
        if t <= 0.0 {
            continue;
        }
        let eta: f64 = params[1..]
            .iter()
            .zip(&covariates[i])
            .map(|(&b, &x)| b * x)
            .sum();
        let lambda = eta.exp();
        let z = t / lambda;
        if events[i] {
            ll += k.ln() - k * eta + (k - 1.0) * t.ln() - z.powf(k);
        } else {
            ll -= z.powf(k);
        }
    }
    ll
}

fn weibull_grad(params: &[f64], times: &[f64], events: &[bool], covariates: &[Vec<f64>]) -> Vec<f64> {
    let k = params[0].exp();
    let p = params.len();
    let mut grad = vec![0.0; p];
    for (i, &t) in times.iter().enumerate() {
        if t <= 0.0 {
            continue;
        }
        let eta: f64 = params[1..]
            .iter()
            .zip(&covariates[i])
            .map(|(&b, &x)| b * x)
            .sum();
        let lambda = eta.exp();
        let z = t / lambda;
        let zk = z.powf(k);
        let delta = events[i];
        // d l / d(ln k) = k Â· d l / dk
        grad[0] += k
            * (delta as i32 as f64 * (1.0 / k - eta + t.ln()) - zk * z.ln());
        for j in 0..p - 1 {
            grad[j + 1] += k * (zk - delta as i32 as f64) * covariates[i][j];
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kaplan_meier_exact_exponential_large_n() {
        // Exp(rate 1): S(t) = e^{-t}. 4000 samples, no censoring.
        let mut rng = crate::rng::Rng::new(7);
        let n = 4000;
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        for _ in 0..n {
            times.push(-rng.uniform().ln());
            events.push(true);
        }
        let s1 = kaplan_meier_survival(&times, &events, 1.0);
        assert!((s1 - (-1.0f64).exp()).abs() < 0.03, "S(1) {s1}");
        let s2 = kaplan_meier_survival(&times, &events, 0.0);
        assert!((s2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kaplan_meier_hand_computation() {
        // t: 1(e), 2(e), 2(c), 3(e): S = 3/4 Â· 1/2 Â· 1/2 = 0.1875
        let times = vec![1.0, 2.0, 2.0, 3.0];
        let events = vec![true, true, false, true];
        let steps = kaplan_meier(&times, &events);
        assert_eq!(steps.len(), 3);
        assert!((steps[0].1 - 0.75).abs() < 1e-12);
        assert!((steps[1].1 - 0.5).abs() < 1e-12);
        assert!((steps[2].1 - 0.1875).abs() < 1e-12);
        let h = nelson_aalen(&times, &events);
        assert!((h[0].1 - 0.25).abs() < 1e-12);
        assert!((h[1].1 - 0.25 - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn log_rank_detects_difference() {
        let mut rng = crate::rng::Rng::new(3);
        // Group A: slow hazard; group B: fast hazard.
        let a_times: Vec<f64> = (0..300).map(|_| -rng.uniform().ln() / 0.5).collect();
        let b_times: Vec<f64> = (0..300).map(|_| -rng.uniform().ln() / 2.0).collect();
        let a_ev = vec![true; 300];
        let b_ev = vec![true; 300];
        let (chi2, p) = log_rank_test(&a_times, &a_ev, &b_times, &b_ev);
        assert!(chi2 > 50.0, "chi2 {chi2}");
        assert!(p < 1e-6, "p {p}");
        // Identical groups -> high p.
        let (_, p3) = log_rank_test(&a_times, &a_ev, &a_times, &a_ev);
        assert!(p3 > 0.05, "identical groups p {p3}");
    }

    #[test]
    fn cox_ph_recovers_coefficients() {
        let mut rng = crate::rng::Rng::new(11);
        let n = 1500;
        let beta_true = [0.8, -0.5];
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut covs = Vec::with_capacity(n);
        for _ in 0..n {
            let x1 = if rng.uniform() < 0.5 { 1.0 } else { 0.0 };
            let x2 = rng.uniform();
            let eta = beta_true[0] * x1 + beta_true[1] * x2;
            let t = -rng.uniform().ln() / eta.exp();
            let censor_t = 5.0;
            if t < censor_t {
                times.push(t);
                events.push(true);
            } else {
                times.push(censor_t);
                events.push(false);
            }
            covs.push(vec![x1, x2]);
        }
        let fit = cox_ph(&times, &events, &covs).unwrap();
        assert!(
            (fit.coefficients[0] - beta_true[0]).abs() < 0.3,
            "beta1 {}",
            fit.coefficients[0]
        );
        assert!(
            (fit.coefficients[1] - beta_true[1]).abs() < 0.3,
            "beta2 {}",
            fit.coefficients[1]
        );
        assert!(!fit.baseline_hazard.is_empty());
    }

    #[test]
    fn weibull_aft_exponential_is_mean_time() {
        let times = vec![2.0, 4.0, 6.0];
        let events = vec![true, true, true];
        let covs = vec![Vec::new(), Vec::new(), Vec::new()];
        let fit = weibull_aft(&times, &events, &covs).unwrap();
        assert!((fit.shape - 1.0).abs() < 0.2, "shape {}", fit.shape);
        assert!((fit.coefficients[0] - 4.0f64.ln()).abs() < 0.2, "beta0 {}", fit.coefficients[0]);
    }

    #[test]
    fn weibull_aft_recovers_shape_and_beta() {
        let mut rng = crate::rng::Rng::new(5);
        let k_true = 2.0;
        let beta_true = [0.5, -0.3];
        let n = 2000;
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut covs = Vec::with_capacity(n);
        for _ in 0..n {
            let x1 = rng.uniform();
            let x2 = rng.uniform();
            let eta = beta_true[0] * x1 + beta_true[1] * x2;
            let lambda = eta.exp();
            let t = lambda * (-rng.uniform().ln()).powf(1.0 / k_true);
            times.push(t);
            events.push(true);
            covs.push(vec![x1, x2]);
        }
        let fit = weibull_aft(&times, &events, &covs).unwrap();
        assert!((fit.shape - k_true).abs() < 0.25, "shape {}", fit.shape);
        assert!((fit.coefficients[0] - beta_true[0]).abs() < 0.2, "b1 {}", fit.coefficients[0]);
        assert!((fit.coefficients[1] - beta_true[1]).abs() < 0.2, "b2 {}", fit.coefficients[1]);
    }

    #[test]
    fn log_rank_p_value_uses_chisq() {
        // All censored: statistic must be 0 -> p = 1.
        let (chi2, p) = log_rank_test(&[1.0, 2.0], &[false, false], &[1.5, 3.0], &[false, false]);
        assert_eq!(chi2, 0.0);
        assert_eq!(p, 1.0);
    }
}
