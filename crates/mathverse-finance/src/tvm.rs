//! Time value of money calculations

/// Calculate future value with compound interest
/// 
/// # Arguments
/// * `present_value` - Present value
/// * `rate` - Interest rate per period (as decimal, e.g., 0.05 for 5%)
/// * `periods` - Number of periods
/// 
/// # Returns
/// Future value
pub fn future_value(present_value: f64, rate: f64, periods: f64) -> f64 {
    present_value * (1.0 + rate).powf(periods)
}

/// Calculate present value from future value
/// 
/// # Arguments
/// * `future_value` - Future value
/// * `rate` - Discount rate per period (as decimal)
/// * `periods` - Number of periods
/// 
/// # Returns
/// Present value
pub fn present_value(future_value: f64, rate: f64, periods: f64) -> f64 {
    future_value / (1.0 + rate).powf(periods)
}

/// Calculate future value of an annuity (ordinary)
/// 
/// # Arguments
/// * `payment` - Payment per period
/// * `rate` - Interest rate per period (as decimal)
/// * `periods` - Number of periods
/// 
/// # Returns
/// Future value of annuity
pub fn future_value_annuity(payment: f64, rate: f64, periods: f64) -> f64 {
    payment * ((1.0 + rate).powf(periods) - 1.0) / rate
}

/// Calculate present value of an annuity (ordinary)
/// 
/// # Arguments
/// * `payment` - Payment per period
/// * `rate` - Discount rate per period (as decimal)
/// * `periods` - Number of periods
/// 
/// # Returns
/// Present value of annuity
pub fn present_value_annuity(payment: f64, rate: f64, periods: f64) -> f64 {
    payment * (1.0 - (1.0 + rate).powf(-periods)) / rate
}

/// Calculate future value of an annuity due
/// 
/// # Arguments
/// * `payment` - Payment per period
/// * `rate` - Interest rate per period (as decimal)
/// * `periods` - Number of periods
/// 
/// # Returns
/// Future value of annuity due
pub fn future_value_annuity_due(payment: f64, rate: f64, periods: f64) -> f64 {
    future_value_annuity(payment, rate, periods) * (1.0 + rate)
}

/// Calculate present value of an annuity due
/// 
/// # Arguments
/// * `payment` - Payment per period
/// * `rate` - Discount rate per period (as decimal)
/// * `periods` - Number of periods
/// 
/// # Returns
/// Present value of annuity due
pub fn present_value_annuity_due(payment: f64, rate: f64, periods: f64) -> f64 {
    present_value_annuity(payment, rate, periods) * (1.0 + rate)
}

/// Calculate payment for an annuity
/// 
/// # Arguments
/// * `present_value` - Present value
/// * `rate` - Interest rate per period (as decimal)
/// * `periods` - Number of periods
/// 
/// # Returns
/// Payment per period
///
/// With a zero rate the payment is simply `PV / periods` (no interest).
pub fn annuity_payment(present_value: f64, rate: f64, periods: f64) -> f64 {
    if rate == 0.0 {
        if periods == 0.0 {
            return f64::NAN;
        }
        return present_value / periods;
    }
    present_value * rate / (1.0 - (1.0 + rate).powf(-periods))
}

/// Calculate number of periods for an annuity
/// 
/// # Arguments
/// * `present_value` - Present value
/// * `payment` - Payment per period
/// * `rate` - Interest rate per period (as decimal)
/// 
/// # Returns
/// Number of periods
///
/// Uses `n = −ln(1 − PV·r / PMT) / ln(1 + r)`. Returns `f64::INFINITY` when
/// the payment never exceeds the per-period interest accrual (no finite
/// payoff exists), and `PV / PMT` when the rate is zero.
pub fn annuity_periods(present_value: f64, payment: f64, rate: f64) -> f64 {
    if rate == 0.0 {
        if payment == 0.0 {
            return f64::NAN;
        }
        return present_value / payment;
    }
    let base = 1.0 - present_value * rate / payment;
    if base <= 0.0 {
        // PMT ≤ PV·r: interest outgrows the payments, the loan is never repaid.
        return f64::INFINITY;
    }
    -base.ln() / (1.0 + rate).ln()
}

/// Calculate effective annual rate from nominal rate
/// 
/// # Arguments
/// * `nominal_rate` - Nominal annual rate (as decimal)
/// * `compounding_periods` - Number of compounding periods per year
/// 
/// # Returns
/// Effective annual rate (as decimal)
pub fn effective_annual_rate(nominal_rate: f64, compounding_periods: f64) -> f64 {
    (1.0 + nominal_rate / compounding_periods).powf(compounding_periods) - 1.0
}

/// Calculate continuous compounding future value
/// 
/// # Arguments
/// * `present_value` - Present value
/// * `rate` - Annual interest rate (as decimal)
/// * `years` - Time in years
/// 
/// # Returns
/// Future value with continuous compounding
pub fn continuous_compounding_fv(present_value: f64, rate: f64, years: f64) -> f64 {
    present_value * (rate * years).exp()
}

/// Calculate continuous compounding present value
/// 
/// # Arguments
/// * `future_value` - Future value
/// * `rate` - Annual discount rate (as decimal)
/// * `years` - Time in years
/// 
/// # Returns
/// Present value with continuous compounding
pub fn continuous_compounding_pv(future_value: f64, rate: f64, years: f64) -> f64 {
    future_value * (-rate * years).exp()
}

/// Calculate net present value
/// 
/// # Arguments
/// * `cash_flows` - Slice of cash flows (first element is initial investment, negative)
/// * `rate` - Discount rate per period (as decimal)
/// 
/// # Returns
/// Net present value
pub fn net_present_value(cash_flows: &[f64], rate: f64) -> f64 {
    cash_flows
        .iter()
        .enumerate()
        .map(|(i, &cf)| cf / (1.0 + rate).powi(i as i32))
        .sum()
}

/// Calculate internal rate of return using Newton-Raphson method
/// 
/// # Arguments
/// * `cash_flows` - Slice of cash flows
/// * `guess` - Initial guess for IRR (default: 0.1)
/// * `tolerance` - Convergence tolerance (default: 1e-6)
/// * `max_iterations` - Maximum iterations (default: 100)
/// 
/// # Returns
/// Internal rate of return (as decimal) or error if convergence fails
pub fn internal_rate_of_return(
    cash_flows: &[f64],
    guess: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> Result<f64, String> {
    let mut rate = guess.unwrap_or(0.1);
    let tol = tolerance.unwrap_or(1e-6);
    let max_iter = max_iterations.unwrap_or(100);

    for _ in 0..max_iter {
        let npv = net_present_value(cash_flows, rate);
        let npv_derivative: f64 = cash_flows
            .iter()
            .enumerate()
            .map(|(i, &cf)| -(i as f64) * cf / (1.0 + rate).powi(i as i32 + 1))
            .sum();

        if npv_derivative.abs() < 1e-10 {
            return Err("Derivative too small".to_string());
        }

        let new_rate = rate - npv / npv_derivative;

        if (new_rate - rate).abs() < tol {
            return Ok(new_rate);
        }

        rate = new_rate;
    }

    Err("IRR did not converge".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_future_value() {
        assert_relative_eq!(future_value(1000.0, 0.05, 10.0), 1628.894626777442, epsilon = 1e-6);
    }

    #[test]
    fn test_present_value() {
        assert_relative_eq!(present_value(1628.894626777442, 0.05, 10.0), 1000.0, epsilon = 1e-6);
    }

    #[test]
    fn test_effective_annual_rate() {
        assert_relative_eq!(effective_annual_rate(0.12, 12.0), 0.1268250301319697, epsilon = 1e-6);
    }

    #[test]
    fn test_net_present_value() {
        let cash_flows = vec![-1000.0, 300.0, 300.0, 300.0, 300.0];
        assert_relative_eq!(net_present_value(&cash_flows, 0.1), -49.0403660952122, epsilon = 1e-4);
    }

    #[test]
    fn test_annuity_periods_roundtrip() {
        // n = −ln(1 − 1000·0.1/150) / ln(1.1) ≈ 11.5267
        let n = annuity_periods(1000.0, 150.0, 0.1);
        assert_relative_eq!(n, 11.526_72, epsilon = 1e-3);
        // Round-trips: the same payment must reproduce the PV.
        assert_relative_eq!(present_value_annuity(150.0, 0.1, n), 1000.0, epsilon = 1e-6);
        // Zero rate: no interest, so n = PV / PMT.
        assert_relative_eq!(annuity_periods(1000.0, 200.0, 0.0), 5.0, epsilon = 1e-12);
        // Payment ≤ interest accrual: never paid off.
        assert!(annuity_periods(1000.0, 50.0, 0.1).is_infinite());
    }

    #[test]
    fn test_annuity_payment_zero_rate() {
        assert_relative_eq!(annuity_payment(1000.0, 0.0, 5.0), 200.0, epsilon = 1e-12);
    }
}
