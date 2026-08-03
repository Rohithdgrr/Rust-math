//! Options pricing models

/// Calculate Black-Scholes call option price
/// 
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// 
/// # Returns
/// Call option price
pub fn black_scholes_call(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
) -> f64 {
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility * volatility / 2.0) * time_to_expiry)
        / (volatility * time_to_expiry.sqrt());
    let d2 = d1 - volatility * time_to_expiry.sqrt();

    spot_price * normal_cdf(d1) - strike_price * (-risk_free_rate * time_to_expiry).exp() * normal_cdf(d2)
}

/// Calculate Black-Scholes put option price
/// 
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// 
/// # Returns
/// Put option price
pub fn black_scholes_put(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
) -> f64 {
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility * volatility / 2.0) * time_to_expiry)
        / (volatility * time_to_expiry.sqrt());
    let d2 = d1 - volatility * time_to_expiry.sqrt();

    strike_price * (-risk_free_rate * time_to_expiry).exp() * normal_cdf(-d2)
        - spot_price * normal_cdf(-d1)
}

/// Standard normal cumulative distribution function
fn normal_cdf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.2316419;

    let t = 1.0 / (1.0 + p * x.abs());
    let y = 1.0
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x / 2.0).exp();

    0.5 * (1.0 + sign * y)
}

/// Calculate call option delta
/// 
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// 
/// # Returns
/// Call option delta
pub fn call_delta(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
) -> f64 {
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility * volatility / 2.0) * time_to_expiry)
        / (volatility * time_to_expiry.sqrt());
    normal_cdf(d1)
}

/// Calculate put option delta
/// 
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// 
/// # Returns
/// Put option delta
pub fn put_delta(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
) -> f64 {
    call_delta(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility) - 1.0
}

/// Calculate option gamma
/// 
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// 
/// # Returns
/// Option gamma
pub fn option_gamma(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
) -> f64 {
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility * volatility / 2.0) * time_to_expiry)
        / (volatility * time_to_expiry.sqrt());
    normal_pdf(d1) / (spot_price * volatility * time_to_expiry.sqrt())
}

/// Standard normal probability density function
fn normal_pdf(x: f64) -> f64 {
    (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Calculate call option theta
/// 
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// 
/// # Returns
/// Call option theta (per year)
pub fn call_theta(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
) -> f64 {
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility * volatility / 2.0) * time_to_expiry)
        / (volatility * time_to_expiry.sqrt());
    let d2 = d1 - volatility * time_to_expiry.sqrt();

    -(spot_price * normal_pdf(d1) * volatility) / (2.0 * time_to_expiry.sqrt())
        - risk_free_rate * strike_price * (-risk_free_rate * time_to_expiry).exp() * normal_cdf(d2)
}

/// Calculate put option theta
/// 
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// 
/// # Returns
/// Put option theta (per year)
pub fn put_theta(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
) -> f64 {
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility * volatility / 2.0) * time_to_expiry)
        / (volatility * time_to_expiry.sqrt());
    let d2 = d1 - volatility * time_to_expiry.sqrt();

    -(spot_price * normal_pdf(d1) * volatility) / (2.0 * time_to_expiry.sqrt())
        + risk_free_rate * strike_price * (-risk_free_rate * time_to_expiry).exp() * normal_cdf(-d2)
}

/// Calculate option vega
/// 
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// 
/// # Returns
/// Option vega
pub fn option_vega(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
) -> f64 {
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility * volatility / 2.0) * time_to_expiry)
        / (volatility * time_to_expiry.sqrt());
    spot_price * normal_pdf(d1) * time_to_expiry.sqrt()
}

/// Calculate option rho
/// 
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// * `is_call` - true for call option, false for put
/// 
/// # Returns
/// Option rho
pub fn option_rho(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
    is_call: bool,
) -> f64 {
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility * volatility / 2.0) * time_to_expiry)
        / (volatility * time_to_expiry.sqrt());
    let d2 = d1 - volatility * time_to_expiry.sqrt();

    if is_call {
        strike_price * time_to_expiry * (-risk_free_rate * time_to_expiry).exp() * normal_cdf(d2)
    } else {
        -strike_price * time_to_expiry * (-risk_free_rate * time_to_expiry).exp() * normal_cdf(-d2)
    }
}

/// Calculate binomial option price (Cox-Ross-Rubinstein model)
/// 
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// * `steps` - Number of time steps
/// * `is_call` - true for call option, false for put
/// 
/// # Returns
/// Option price
pub fn binomial_option_price(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
    steps: usize,
    is_call: bool,
) -> f64 {
    let dt = time_to_expiry / steps as f64;
    let u = (volatility * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = ((risk_free_rate * dt).exp() - d) / (u - d);

    let mut prices = vec![0.0; steps + 1];

    for i in 0..=steps {
        let spot = spot_price * u.powi(i as i32) * d.powi((steps - i) as i32);
        prices[i] = if is_call {
            (spot - strike_price).max(0.0)
        } else {
            (strike_price - spot).max(0.0)
        };
    }

    for j in (0..steps).rev() {
        let discount = (-risk_free_rate * dt).exp();
        for i in 0..=j {
            prices[i] = discount * (p * prices[i + 1] + (1.0 - p) * prices[i]);
        }
    }

    prices[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_black_scholes_call() {
        let price = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2);
        assert_relative_eq!(price, 10.450583572185565, epsilon = 1e-4);
    }

    #[test]
    fn test_black_scholes_put() {
        let price = black_scholes_put(100.0, 100.0, 1.0, 0.05, 0.2);
        assert_relative_eq!(price, 5.573526022256971, epsilon = 1e-4);
    }

    #[test]
    fn test_call_delta() {
        let delta = call_delta(100.0, 100.0, 1.0, 0.05, 0.2);
        assert!(delta > 0.0 && delta < 1.0);
    }
}
