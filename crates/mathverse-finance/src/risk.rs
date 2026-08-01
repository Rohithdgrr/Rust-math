//! Risk management calculations

/// Calculate variance
/// 
/// # Arguments
/// * `values` - Slice of values
/// 
/// # Returns
/// Variance
pub fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance
}

/// Calculate standard deviation
/// 
/// # Arguments
/// * `values` - Slice of values
/// 
/// # Returns
/// Standard deviation
pub fn standard_deviation(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Calculate sample variance
/// 
/// # Arguments
/// * `values` - Slice of values
/// 
/// # Returns
/// Sample variance
pub fn sample_variance(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance
}

/// Calculate sample standard deviation
/// 
/// # Arguments
/// * `values` - Slice of values
/// 
/// # Returns
/// Sample standard deviation
pub fn sample_standard_deviation(values: &[f64]) -> f64 {
    sample_variance(values).sqrt()
}

/// Calculate covariance
/// 
/// # Arguments
/// * `x` - First slice of values
/// * `y` - Second slice of values (must be same length as x)
/// 
/// # Returns
/// Covariance
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;

    let cov: f64 = x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>() / x.len() as f64;

    cov
}

/// Calculate correlation coefficient
/// 
/// # Arguments
/// * `x` - First slice of values
/// * `y` - Second slice of values (must be same length as x)
/// 
/// # Returns
/// Correlation coefficient (-1 to 1)
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let cov = covariance(x, y);
    let std_x = standard_deviation(x);
    let std_y = standard_deviation(y);

    if std_x == 0.0 || std_y == 0.0 {
        return 0.0;
    }

    cov / (std_x * std_y)
}

/// Calculate Value at Risk (VaR) - parametric method
/// 
/// # Arguments
/// * `portfolio_value` - Portfolio value
/// * `mean_return` - Mean return (as decimal)
/// * `std_dev` - Standard deviation of returns (as decimal)
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
/// 
/// # Returns
/// Value at Risk (absolute amount)
pub fn value_at_risk(
    portfolio_value: f64,
    mean_return: f64,
    std_dev: f64,
    confidence_level: f64,
) -> f64 {
    let z_score = inverse_normal_cdf(confidence_level);
    let var_percent = mean_return - z_score * std_dev;
    portfolio_value * var_percent.abs()
}

/// Approximate inverse normal CDF using Beasley-Springer-Moro algorithm
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }

    let a = [-3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00];

    let b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01];

    let c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00, 2.938163982698783e+00];

    let d = [7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661907416e+00];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    let q: f64;
    let r: f64;

    if p < p_low {
        q = (0.0 - p.ln()).sqrt();
        let num = ((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5];
        let den = (((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0;
        return num / den;
    } else if p <= p_high {
        q = p - 0.5;
        let r = q * q;
        let num = ((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5] * q;
        let den = ((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0;
        return num / den;
    } else {
        q = (0.0 - (1.0 - p).ln()).sqrt();
        let num = ((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5];
        let den = (((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0;
        return -num / den;
    }
}

/// Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
/// 
/// # Arguments
/// * `portfolio_value` - Portfolio value
/// * `mean_return` - Mean return (as decimal)
/// * `std_dev` - Standard deviation of returns (as decimal)
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
/// 
/// # Returns
/// Conditional Value at Risk (absolute amount)
pub fn conditional_var(
    portfolio_value: f64,
    mean_return: f64,
    std_dev: f64,
    confidence_level: f64,
) -> f64 {
    let z_score = inverse_normal_cdf(confidence_level);
    let pdf = (0.0 - z_score.powi(2) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cvar_percent = mean_return - std_dev * pdf / (1.0 - confidence_level);
    portfolio_value * cvar_percent.abs()
}

/// Calculate downside deviation
/// 
/// # Arguments
/// * `returns` - Slice of returns (as decimals)
/// * `minimum_acceptable_return` - Minimum acceptable return (as decimal)
/// 
/// # Returns
/// Downside deviation
pub fn downside_deviation(returns: &[f64], minimum_acceptable_return: f64) -> f64 {
    let downside_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < minimum_acceptable_return)
        .map(|&r| (minimum_acceptable_return - r).powi(2))
        .collect();

    if downside_returns.is_empty() {
        return 0.0;
    }

    (downside_returns.iter().sum::<f64>() / returns.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(variance(&values), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_standard_deviation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(standard_deviation(&values), 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_covariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_relative_eq!(covariance(&x, &y), 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_relative_eq!(correlation(&x, &y), 1.0, epsilon = 1e-6);
    }
}
