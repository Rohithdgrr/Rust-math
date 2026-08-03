//! Time value of money (TVM) and financial arithmetic.

/// Future value of a single sum: `FV = PV * (1 + r)^n`.
pub fn future_value(present_value: f64, rate: f64, periods: f64) -> f64 {
    present_value * (1.0 + rate).powf(periods)
}

/// Present value of a single sum: `PV = FV / (1 + r)^n`.
pub fn present_value(future_value: f64, rate: f64, periods: f64) -> f64 {
    future_value / (1.0 + rate).powf(periods)
}

/// Future value of an ordinary annuity: `FV = PMT * [((1+r)^n - 1) / r]`.
pub fn annuity_future_value(payment: f64, rate: f64, periods: u32) -> f64 {
    if rate.abs() < f64::EPSILON {
        payment * periods as f64
    } else {
        payment * ((1.0 + rate).powf(periods as f64) - 1.0) / rate
    }
}

/// Present value of an ordinary annuity: `PV = PMT * [(1 - (1+r)^-n) / r]`.
pub fn annuity_present_value(payment: f64, rate: f64, periods: u32) -> f64 {
    if rate.abs() < f64::EPSILON {
        payment * periods as f64
    } else {
        payment * (1.0 - (1.0 + rate).powf(-(periods as f64))) / rate
    }
}

/// Present value of a perpetuity: `PV = PMT / r`.
pub fn perpetuity_present_value(payment: f64, rate: f64) -> f64 {
    payment / rate
}

/// Growing perpetuity: `PV = PMT / (r - g)`, requires `r > g`.
pub fn growing_perpetuity(payment: f64, rate: f64, growth: f64) -> f64 {
    payment / (rate - growth)
}

/// Number of periods to reach a target future value.
pub fn periods_to_reach(present_value: f64, future_value: f64, rate: f64) -> f64 {
    (future_value / present_value).ln() / (1.0 + rate).ln()
}

/// Rate per period needed to reach a target future value.
pub fn rate_for_target(present_value: f64, future_value: f64, periods: f64) -> f64 {
    (future_value / present_value).powf(1.0 / periods) - 1.0
}

/// Continuous compounding future value: `FV = PV * e^(r*t)`.
pub fn continuous_compound(present_value: f64, rate: f64, time: f64) -> f64 {
    present_value * (rate * time).exp()
}

/// Rule of 72: approximate doubling time.
pub fn rule_of_72(rate_percent: f64) -> f64 {
    72.0 / rate_percent
}

/// Rule of 69.3: more accurate doubling time for continuous compounding.
pub fn rule_of_693(rate_percent: f64) -> f64 {
    69.3 / rate_percent
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn fv_pv() {
        assert!((future_value(1000.0, 0.05, 10.0) - 1628.8946267774414).abs() < EPS);
        let fv = future_value(1000.0, 0.05, 10.0);
        assert!((present_value(fv, 0.05, 10.0) - 1000.0).abs() < EPS);
    }

    #[test]
    fn annuity() {
        let fv = annuity_future_value(100.0, 0.05, 10);
        assert!((fv - 1257.7892535548828).abs() < EPS);
        let pv = annuity_present_value(100.0, 0.05, 10);
        assert!((pv - 772.1734929184781).abs() < EPS);
    }

    #[test]
    fn perpetuity() {
        assert!((perpetuity_present_value(100.0, 0.05) - 2000.0).abs() < EPS);
        assert!((growing_perpetuity(100.0, 0.10, 0.03) - 1428.5714).abs() < 1.0);
    }

    #[test]
    fn doubling_time() {
        let t = periods_to_reach(1000.0, 2000.0, 0.07);
        assert!((t - 10.2447683510587).abs() < EPS);
    }

    #[test]
    fn continuous() {
        assert!((continuous_compound(1000.0, 0.05, 10.0) - 1648.7212707001282).abs() < EPS);
    }

    #[test]
    fn rule_of_72_test() {
        assert!((rule_of_72(6.0) - 12.0).abs() < EPS);
    }
}
