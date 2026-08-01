//! Reliability theory: survival functions, hazard functions, failure rates, MTTF, MTBF, censored data.

use crate::F64Ext;

/// Survival function (reliability function).
pub struct SurvivalFunction;

impl SurvivalFunction {
    /// Survival function S(t) = P(T > t) = 1 - F(t).
    pub fn from_cdf(cdf: impl Fn(f64) -> f64, t: f64) -> f64 {
        1.0 - cdf(t)
    }

    /// Survival function from hazard function.
    pub fn from_hazard(hazard: impl Fn(f64) -> f64, t: f64) -> f64 {
        // S(t) = exp(-∫₀ᵗ λ(s) ds)
        let n = 1000;
        let dt = t / n as f64;
        let mut integral = 0.0;

        for i in 0..n {
            let s = (i as f64 + 0.5) * dt;
            integral += hazard(s) * dt;
        }

        (-integral).exp()
    }

    /// Empirical survival function from data.
    pub fn empirical(failure_times: &[f64], t: f64) -> f64 {
        if failure_times.is_empty() {
            return 1.0;
        }

        let n = failure_times.len();
        let count = failure_times.iter().filter(|&&x| x > t).count();
        count as f64 / n as f64
    }

    /// Kaplan-Meier estimator for censored data.
    pub fn kaplan_meier(times: &[f64], events: &[bool]) -> Vec<(f64, f64)> {
        let n = times.len();
        if n == 0 {
            return Vec::new();
        }

        // Sort by time
        let mut indexed: Vec<(usize, f64, bool)> = times
            .iter()
            .zip(events.iter())
            .enumerate()
            .map(|(i, (&t, &e))| (i, t, e))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut survival = Vec::new();
        let mut s = 1.0;
        let mut at_risk = n as f64;

        for (_, t, event) in indexed {
            if event {
                s *= (at_risk - 1.0) / at_risk;
            }
            survival.push((t, s));
            at_risk -= 1.0;
        }

        survival
    }
}

/// Hazard function (failure rate).
pub struct HazardFunction;

impl HazardFunction {
    /// Hazard function h(t) = f(t) / S(t).
    pub fn from_pdf_survival(
        pdf: impl Fn(f64) -> f64,
        survival: impl Fn(f64) -> f64,
        t: f64,
    ) -> f64 {
        let s = survival(t);
        if s > 0.0 {
            pdf(t) / s
        } else {
            0.0
        }
    }

    /// Cumulative hazard function H(t) = ∫₀ᵗ h(s) ds.
    pub fn cumulative(hazard: impl Fn(f64) -> f64, t: f64) -> f64 {
        let n = 1000;
        let dt = t / n as f64;
        let mut integral = 0.0;

        for i in 0..n {
            let s = (i as f64 + 0.5) * dt;
            integral += hazard(s) * dt;
        }

        integral
    }

    /// Nelson-Aalen estimator for cumulative hazard.
    pub fn nelson_aalen(times: &[f64], events: &[bool]) -> Vec<(f64, f64)> {
        let n = times.len();
        if n == 0 {
            return Vec::new();
        }

        let mut indexed: Vec<(usize, f64, bool)> = times
            .iter()
            .zip(events.iter())
            .enumerate()
            .map(|(i, (&t, &e))| (i, t, e))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut cumulative_hazard = Vec::new();
        let mut h = 0.0;
        let mut at_risk = n;

        for (_, t, event) in indexed {
            if event {
                h += 1.0 / at_risk as f64;
            }
            cumulative_hazard.push((t, h));
            at_risk -= 1;
        }

        cumulative_hazard
    }

    /// Constant hazard (exponential distribution).
    pub fn constant(lambda: f64) -> impl Fn(f64) -> f64 {
        move |_t| lambda
    }

    /// Weibull hazard function.
    pub fn weibull(shape: f64, scale: f64) -> impl Fn(f64) -> f64 {
        move |t| {
            if t <= 0.0 {
                0.0
            } else {
                (shape / scale) * (t / scale).powf(shape - 1.0)
            }
        }
    }

    /// Bathtub hazard function (early failures, constant, wear-out).
    pub fn bathtub(beta1: f64, beta2: f64, lambda: f64) -> impl Fn(f64) -> f64 {
        move |t| {
            if t <= 0.0 {
                0.0
            } else {
                beta1 / t + lambda + beta2 * t
            }
        }
    }
}

/// Mean Time To Failure (MTTF).
pub struct MTTF;

impl MTTF {
    /// MTTF from survival function: E\[T\] = integral of S(t) dt.
    pub fn from_survival(survival: impl Fn(f64) -> f64, max_time: f64) -> f64 {
        let n = 10000;
        let dt = max_time / n as f64;
        let mut integral = 0.0;

        for i in 0..n {
            let t = (i as f64 + 0.5) * dt;
            integral += survival(t) * dt;
        }

        integral
    }

    /// MTTF for exponential distribution.
    pub fn exponential(lambda: f64) -> f64 {
        1.0 / lambda
    }

    /// MTTF for Weibull distribution.
    pub fn weibull(shape: f64, scale: f64) -> f64 {
        scale * (1.0 + 1.0 / shape).gamma()
    }

    /// Empirical MTTF from failure data.
    pub fn empirical(failure_times: &[f64]) -> f64 {
        if failure_times.is_empty() {
            return 0.0;
        }
        failure_times.iter().sum::<f64>() / failure_times.len() as f64
    }
}

/// Mean Time Between Failures (MTBF).
pub struct MTBF;

impl MTBF {
    /// MTBF = Total operating time / Number of failures.
    pub fn calculate(total_time: f64, n_failures: usize) -> f64 {
        if n_failures == 0 {
            return total_time;
        }
        total_time / n_failures as f64
    }

    /// MTBF with repair time included.
    pub fn with_repair(operating_time: f64, repair_time: f64, n_failures: usize) -> f64 {
        let total_time = operating_time + repair_time;
        Self::calculate(total_time, n_failures)
    }

    /// Availability from MTBF and MTTR.
    pub fn availability(mtbf: f64, mttr: f64) -> f64 {
        mtbf / (mtbf + mttr)
    }
}

/// Reliability metrics.
pub struct ReliabilityMetrics;

impl ReliabilityMetrics {
    /// System reliability for series system.
    pub fn series(component_reliabilities: &[f64]) -> f64 {
        component_reliabilities.iter().product()
    }

    /// System reliability for parallel system.
    pub fn parallel(component_reliabilities: &[f64]) -> f64 {
        let failure_prob: f64 = component_reliabilities.iter().map(|&r| 1.0 - r).product();
        1.0 - failure_prob
    }

    /// System reliability for k-out-of-n system.
    pub fn k_out_of_n(k: usize, component_reliability: f64, n: usize) -> f64 {
        let mut reliability = 0.0;

        for i in k..=n {
            let combinations = mathverse_core::algorithms::binomial(n as u64, i as u64) as f64;
            reliability += combinations
                * component_reliability.powi(i as i32)
                * (1.0 - component_reliability).powi((n - i) as i32);
        }

        reliability
    }

    /// System reliability for standby system.
    pub fn standby(component_reliability: f64, n_components: usize) -> f64 {
        let lambda = -component_reliability.ln();
        let mut reliability = 0.0;

        for i in 0..n_components {
            let term =
                (lambda * i as f64).exp() * (lambda * i as f64).powi(i as i32) / (i as f64).gamma();
            reliability += term * component_reliability;
        }

        reliability.min(1.0)
    }
}

/// Censored data analysis.
pub struct CensoredData;

impl CensoredData {
    /// Right-censored data analysis.
    pub fn right_censored(observed_times: &[f64], is_censored: &[bool]) -> (f64, f64) {
        let n = observed_times.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        let mut sum_observed = 0.0;
        let mut n_observed = 0;

        for (&t, &censored) in observed_times.iter().zip(is_censored.iter()) {
            if !censored {
                sum_observed += t;
                n_observed += 1;
            }
        }

        let mttf = if n_observed > 0 {
            sum_observed / n_observed as f64
        } else {
            0.0
        };

        let censoring_rate = is_censored.iter().filter(|&&c| c).count() as f64 / n as f64;

        (mttf, censoring_rate)
    }

    /// Left-censored data analysis.
    pub fn left_censored(detection_limit: f64, observed_values: &[f64]) -> (f64, usize) {
        let n = observed_values.len();
        let n_censored = observed_values
            .iter()
            .filter(|&&x| x <= detection_limit)
            .count();

        let mean = if n > n_censored {
            let sum: f64 = observed_values
                .iter()
                .filter(|&&x| x > detection_limit)
                .sum();
            sum / (n - n_censored) as f64
        } else {
            detection_limit
        };

        (mean, n_censored)
    }

    /// Interval-censored data analysis.
    pub fn interval_censored(intervals: &[(f64, f64)]) -> f64 {
        if intervals.is_empty() {
            return 0.0;
        }

        let sum: f64 = intervals.iter().map(|(a, b)| (a + b) / 2.0).sum();

        sum / intervals.len() as f64
    }
}

/// Failure rate analysis.
pub struct FailureRateAnalysis;

impl FailureRateAnalysis {
    /// Failure rate from failure data.
    pub fn calculate(failure_times: &[f64], time_window: f64) -> f64 {
        if failure_times.is_empty() {
            return 0.0;
        }

        let count = failure_times.iter().filter(|&&t| t <= time_window).count();
        count as f64 / time_window
    }

    /// Time between failures analysis.
    pub fn time_between_failures(failure_times: &[f64]) -> Vec<f64> {
        if failure_times.len() < 2 {
            return Vec::new();
        }

        let mut tbf = Vec::new();
        for i in 1..failure_times.len() {
            tbf.push(failure_times[i] - failure_times[i - 1]);
        }

        tbf
    }

    /// Failure rate trend analysis (increasing, decreasing, constant).
    pub fn trend(tbf: &[f64]) -> &'static str {
        if tbf.len() < 3 {
            return "insufficient data";
        }

        let n = tbf.len();
        let first_half = &tbf[..n / 2];
        let second_half = &tbf[n / 2..];

        let mean_first = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let mean_second = second_half.iter().sum::<f64>() / second_half.len() as f64;

        if mean_second > mean_first * 1.1 {
            "decreasing failure rate (improving)"
        } else if mean_second < mean_first * 0.9 {
            "increasing failure rate (worsening)"
        } else {
            "constant failure rate"
        }
    }
}

/// Warranty analysis.
pub struct WarrantyAnalysis;

impl WarrantyAnalysis {
    /// Expected warranty cost.
    pub fn expected_cost(failure_rate: f64, repair_cost: f64, warranty_period: f64) -> f64 {
        let expected_failures = failure_rate * warranty_period;
        expected_failures * repair_cost
    }

    /// Warranty reserve calculation.
    pub fn reserve(
        n_products: usize,
        failure_rate: f64,
        repair_cost: f64,
        warranty_period: f64,
    ) -> f64 {
        n_products as f64 * Self::expected_cost(failure_rate, repair_cost, warranty_period)
    }

    /// Warranty claim probability.
    pub fn claim_probability(survival_function: impl Fn(f64) -> f64, warranty_period: f64) -> f64 {
        1.0 - survival_function(warranty_period)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survival_function() {
        let cdf = |t: f64| -> f64 { t.clamp(0.0, 1.0) };

        let s = SurvivalFunction::from_cdf(cdf, 0.5);
        assert!((s - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_kaplan_meier() {
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let events = vec![true, true, false, true];
        let km = SurvivalFunction::kaplan_meier(&times, &events);
        assert_eq!(km.len(), 4);
    }

    #[test]
    fn test_mttf_exponential() {
        let mttf = MTTF::exponential(0.1);
        assert!((mttf - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_reliability() {
        let components = vec![0.9, 0.95, 0.99];
        let reliability = ReliabilityMetrics::series(&components);
        assert!((reliability - 0.9 * 0.95 * 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_reliability() {
        let components = vec![0.9, 0.9];
        let reliability = ReliabilityMetrics::parallel(&components);
        assert!((reliability - (1.0 - 0.1 * 0.1)).abs() < 1e-10);
    }

    #[test]
    fn test_mtbf() {
        let mtbf = MTBF::calculate(1000.0, 10);
        assert!((mtbf - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_availability() {
        let availability = MTBF::availability(100.0, 10.0);
        assert!((availability - 100.0 / 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_hazard() {
        let hazard = HazardFunction::weibull(2.0, 1.0);
        let h = hazard(1.0);
        assert!((h - 2.0).abs() < 1e-10);
    }
}
