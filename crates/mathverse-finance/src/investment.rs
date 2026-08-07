//! Investment analysis

/// Calculate return on investment (ROI)
/// 
/// # Arguments
/// * `initial_investment` - Initial investment amount
/// * `final_value` - Final value of investment
/// 
/// # Returns
/// ROI as a percentage
pub fn return_on_investment(initial_investment: f64, final_value: f64) -> f64 {
    (final_value - initial_investment) / initial_investment * 100.0
}

/// Calculate annualized return (CAGR)
/// 
/// # Arguments
/// * `initial_value` - Initial investment value
/// * `final_value` - Final investment value
/// * `years` - Number of years
/// 
/// # Returns
/// Compound annual growth rate as a percentage
pub fn cagr(initial_value: f64, final_value: f64, years: f64) -> f64 {
    ((final_value / initial_value).powf(1.0 / years) - 1.0) * 100.0
}

/// Calculate holding period return
/// 
/// # Arguments
/// * `initial_price` - Initial price
/// * `final_price` - Final price
/// * `income` - Income received (dividends, interest, etc.)
/// 
/// # Returns
/// Holding period return as a percentage
pub fn holding_period_return(initial_price: f64, final_price: f64, income: f64) -> f64 {
    ((final_price + income - initial_price) / initial_price) * 100.0
}

/// Calculate arithmetic mean return
/// 
/// # Arguments
/// * `returns` - Slice of returns (as percentages)
/// 
/// # Returns
/// Arithmetic mean return as a percentage
pub fn arithmetic_mean_return(returns: &[f64]) -> f64 {
    let sum: f64 = returns.iter().sum();
    sum / returns.len() as f64
}

/// Calculate geometric mean return
/// 
/// # Arguments
/// * `returns` - Slice of returns (as percentages)
/// 
/// # Returns
/// Geometric mean return as a percentage
pub fn geometric_mean_return(returns: &[f64]) -> f64 {
    let product: f64 = returns.iter().map(|&r| 1.0 + r / 100.0).product();
    (product.powf(1.0 / returns.len() as f64) - 1.0) * 100.0
}

/// Calculate Sharpe ratio
/// 
/// # Arguments
/// * `portfolio_return` - Portfolio return (as decimal)
/// * `risk_free_rate` - Risk-free rate (as decimal)
/// * `portfolio_std_dev` - Portfolio standard deviation (as decimal)
/// 
/// # Returns
/// Sharpe ratio
pub fn sharpe_ratio(portfolio_return: f64, risk_free_rate: f64, portfolio_std_dev: f64) -> f64 {
    (portfolio_return - risk_free_rate) / portfolio_std_dev
}

/// Calculate Sortino ratio
/// 
/// # Arguments
/// * `portfolio_return` - Portfolio return (as decimal)
/// * `risk_free_rate` - Risk-free rate (as decimal)
/// * `downside_deviation` - Downside deviation (as decimal)
/// 
/// # Returns
/// Sortino ratio
pub fn sortino_ratio(portfolio_return: f64, risk_free_rate: f64, downside_deviation: f64) -> f64 {
    (portfolio_return - risk_free_rate) / downside_deviation
}

/// Calculate Treynor ratio
/// 
/// # Arguments
/// * `portfolio_return` - Portfolio return (as decimal)
/// * `risk_free_rate` - Risk-free rate (as decimal)
/// * `beta` - Portfolio beta
/// 
/// # Returns
/// Treynor ratio
pub fn treynor_ratio(portfolio_return: f64, risk_free_rate: f64, beta: f64) -> f64 {
    (portfolio_return - risk_free_rate) / beta
}

/// Calculate information ratio from pre-computed scalars
/// 
/// # Arguments
/// * `portfolio_return` - Portfolio return (as decimal)
/// * `benchmark_return` - Benchmark return (as decimal)
/// * `tracking_error` - Tracking error (as decimal)
/// 
/// # Returns
/// Information ratio
///
/// Prefer [`super::portfolio::information_ratio`] when the underlying return
/// series are available; this scalar form exists for callers that already
/// have the tracking error computed.
pub fn information_ratio_from_scalars(portfolio_return: f64, benchmark_return: f64, tracking_error: f64) -> f64 {
    (portfolio_return - benchmark_return) / tracking_error
}

/// Calculate maximum drawdown
/// 
/// # Arguments
/// * `values` - Slice of portfolio values over time
/// 
/// # Returns
/// Maximum drawdown as a percentage
pub fn maximum_drawdown(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut peak = values[0];
    let mut max_dd = 0.0;

    for &value in values.iter().skip(1) {
        if value > peak {
            peak = value;
        } else {
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }

    max_dd * 100.0
}

/// Calculate Calmar ratio
/// 
/// # Arguments
/// * `annual_return` - Annualized return (as decimal)
/// * `max_drawdown` - Maximum drawdown (as decimal)
/// 
/// # Returns
/// Calmar ratio
pub fn calmar_ratio(annual_return: f64, max_drawdown: f64) -> f64 {
    annual_return / max_drawdown
}

/// Calculate alpha (Jensen's alpha)
/// 
/// # Arguments
/// * `portfolio_return` - Portfolio return (as decimal)
/// * `risk_free_rate` - Risk-free rate (as decimal)
/// * `beta` - Portfolio beta
/// * `market_return` - Market return (as decimal)
/// 
/// # Returns
/// Alpha (as decimal)
pub fn jensens_alpha(
    portfolio_return: f64,
    risk_free_rate: f64,
    beta: f64,
    market_return: f64,
) -> f64 {
    portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))
}

/// Calculate beta
/// 
/// # Arguments
/// * `covariance` - Covariance between asset and market
/// * `market_variance` - Variance of market returns
/// 
/// # Returns
/// Beta
pub fn beta(covariance: f64, market_variance: f64) -> f64 {
    covariance / market_variance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_return_on_investment() {
        assert_relative_eq!(return_on_investment(1000.0, 1500.0), 50.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cagr() {
        assert_relative_eq!(cagr(1000.0, 1500.0, 5.0), 8.447177, epsilon = 1e-4);
    }

    #[test]
    fn test_sharpe_ratio() {
        assert_relative_eq!(sharpe_ratio(0.15, 0.02, 0.10), 1.3, epsilon = 1e-6);
    }

    #[test]
    fn test_maximum_drawdown() {
        let values = vec![100.0, 110.0, 105.0, 120.0, 115.0, 130.0, 125.0];
        assert_relative_eq!(maximum_drawdown(&values), 4.545454, epsilon = 1e-4);
    }
}
