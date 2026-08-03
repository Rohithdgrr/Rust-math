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

/// Monte Carlo estimate of a European option price under geometric
/// Brownian motion with a constant seed for reproducibility.
///
/// Simulates `paths` independent GBM paths (seed `42`), discounts the mean
/// terminal payoff by `exp(-r·T)`. The statistical error shrinks like
/// `1/√paths`; pass a large `paths` (e.g. 100_000) for ~0.5% accuracy. An
/// empty path count returns `NaN`.
///
/// # Arguments
/// * `spot_price` - Current spot price of underlying
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of underlying (as decimal)
/// * `paths` - Number of Monte Carlo paths
/// * `is_call` - true for call option, false for put
///
/// ```
/// use mathverse_finance::monte_carlo_option_price;
/// let mc = monte_carlo_option_price(100.0, 100.0, 1.0, 0.05, 0.2, 200_000, true);
/// let bs = mathverse_finance::black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2);
/// assert!((mc - bs).abs() < 0.5);
/// ```
pub fn monte_carlo_option_price(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64,
    paths: usize,
    is_call: bool,
) -> f64 {
    if paths == 0 {
        return f64::NAN;
    }
    let drift = (risk_free_rate - 0.5 * volatility * volatility) * time_to_expiry;
    let diffusion = volatility * time_to_expiry.sqrt();
    // LCG with a fixed seed: deterministic across runs.
    let mut state: u64 = 42;
    let mut sum = 0.0;
    for _ in 0..paths {
        let u1 = next_uniform(&mut state);
        let u2 = next_uniform(&mut state);
        // Box-Muller pair from two uniforms.
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
        let terminal = spot_price * (drift + diffusion * z).exp();
        sum += if is_call {
            (terminal - strike_price).max(0.0)
        } else {
            (strike_price - terminal).max(0.0)
        };
    }
    (-risk_free_rate * time_to_expiry).exp() * sum / paths as f64
}

/// Next uniform in (0, 1) from an LCG (Numerical Recipes constants).
fn next_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    // Top 26 bits → (0, 1) with no zero (u1.ln() is taken on it).
    const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
    ((*state >> 11) as f64 + 0.5) * SCALE
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
