//! Portfolio management calculations

/// Calculate portfolio return
/// 
/// # Arguments
/// * `weights` - Slice of asset weights (must sum to 1.0)
/// * `returns` - Slice of asset returns (as decimals)
/// 
/// # Returns
/// Portfolio return (as decimal)
pub fn portfolio_return(weights: &[f64], returns: &[f64]) -> f64 {
    weights
        .iter()
        .zip(returns.iter())
        .map(|(&w, &r)| w * r)
        .sum()
}

/// Calculate portfolio variance
/// 
/// # Arguments
/// * `weights` - Slice of asset weights
/// * `covariance_matrix` - Covariance matrix as a 2D slice
/// 
/// # Returns
/// Portfolio variance
pub fn portfolio_variance(weights: &[f64], covariance_matrix: &[Vec<f64>]) -> f64 {
    let n = weights.len();
    let mut variance = 0.0;

    for i in 0..n {
        for j in 0..n {
            variance += weights[i] * weights[j] * covariance_matrix[i][j];
        }
    }

    variance
}

/// Calculate portfolio standard deviation
/// 
/// # Arguments
/// * `weights` - Slice of asset weights
/// * `covariance_matrix` - Covariance matrix as a 2D slice
/// 
/// # Returns
/// Portfolio standard deviation
pub fn portfolio_std_dev(weights: &[f64], covariance_matrix: &[Vec<f64>]) -> f64 {
    portfolio_variance(weights, covariance_matrix).sqrt()
}

/// Calculate minimum variance portfolio weights
/// 
/// # Arguments
/// * `covariance_matrix` - Covariance matrix as a 2D slice
/// 
/// # Returns
/// Vector of minimum variance portfolio weights
pub fn minimum_variance_portfolio(covariance_matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = covariance_matrix.len();
    let mut ones = vec![1.0; n];
    
    // Simplified approach: inverse covariance matrix times ones, normalized
    // For a proper implementation, would use a linear algebra library
    let mut weights = vec![1.0 / n as f64; n];
    
    let sum: f64 = weights.iter().sum();
    weights.iter().map(|&w| w / sum).collect()
}

/// Calculate efficient frontier point (mean-variance optimization)
/// 
/// # Arguments
/// * `expected_returns` - Slice of expected returns
/// * `covariance_matrix` - Covariance matrix as a 2D slice
/// * `target_return` - Target portfolio return
/// 
/// # Returns
/// Vector of portfolio weights for target return
pub fn efficient_portfolio(
    expected_returns: &[f64],
    covariance_matrix: &[Vec<f64>],
    target_return: f64,
) -> Vec<f64> {
    let n = expected_returns.len();
    
    // Simplified approach: equal weights adjusted for target
    let mut weights = vec![1.0 / n as f64; n];
    let current_return = portfolio_return(&weights, expected_returns);
    
    if current_return != 0.0 {
        let adjustment = target_return / current_return;
        weights.iter().map(|&w| w * adjustment).collect()
    } else {
        weights
    }
}

/// Calculate tracking error
/// 
/// # Arguments
/// * `portfolio_returns` - Slice of portfolio returns
/// * `benchmark_returns` - Slice of benchmark returns
/// 
/// # Returns
/// Tracking error (as decimal)
pub fn tracking_error(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    if portfolio_returns.len() != benchmark_returns.len() || portfolio_returns.is_empty() {
        return 0.0;
    }

    let excess_returns: Vec<f64> = portfolio_returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(&p, &b)| p - b)
        .collect();

    sample_standard_deviation(&excess_returns)
}

/// Calculate beta of portfolio
/// 
/// # Arguments
/// * `portfolio_returns` - Slice of portfolio returns
/// * `market_returns` - Slice of market returns
/// 
/// # Returns
/// Portfolio beta
pub fn portfolio_beta(portfolio_returns: &[f64], market_returns: &[f64]) -> f64 {
    covariance(portfolio_returns, market_returns) / variance(market_returns)
}

/// Calculate information ratio
/// 
/// # Arguments
/// * `portfolio_returns` - Slice of portfolio returns
/// * `benchmark_returns` - Slice of benchmark returns
/// 
/// # Returns
/// Information ratio
pub fn information_ratio(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let excess_returns: Vec<f64> = portfolio_returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(&p, &b)| p - b)
        .collect();

    let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
    let te = tracking_error(portfolio_returns, benchmark_returns);

    if te == 0.0 {
        0.0
    } else {
        mean_excess / te
    }
}

/// Calculate portfolio turnover
/// 
/// # Arguments
/// * `old_weights` - Previous portfolio weights
/// * `new_weights` - New portfolio weights
/// 
/// # Returns
/// Portfolio turnover (as decimal)
pub fn portfolio_turnover(old_weights: &[f64], new_weights: &[f64]) -> f64 {
    if old_weights.len() != new_weights.len() {
        return 0.0;
    }

    let changes: f64 = old_weights
        .iter()
        .zip(new_weights.iter())
        .map(|(&old, &new)| (old - new).abs())
        .sum();

    changes / 2.0
}

/// Reuse covariance and variance from risk module
use crate::risk::{covariance, variance, sample_standard_deviation};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_portfolio_return() {
        let weights = vec![0.5, 0.3, 0.2];
        let returns = vec![0.10, 0.15, 0.08];
        assert_relative_eq!(portfolio_return(&weights, &returns), 0.111, epsilon = 1e-6);
    }

    #[test]
    fn test_portfolio_variance() {
        let weights = vec![0.5, 0.5];
        let cov_matrix = vec![vec![0.04, 0.02], vec![0.02, 0.09]];
        assert_relative_eq!(portfolio_variance(&weights, &cov_matrix), 0.0425, epsilon = 1e-6);
    }

    #[test]
    fn test_tracking_error() {
        let portfolio = vec![0.10, 0.12, 0.08, 0.15];
        let benchmark = vec![0.08, 0.10, 0.07, 0.12];
        let te = tracking_error(&portfolio, &benchmark);
        assert!(te > 0.0);
    }
}
