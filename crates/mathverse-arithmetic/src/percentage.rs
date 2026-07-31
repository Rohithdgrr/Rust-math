//! Advanced percentage operations: markup, margin, discount, compound interest.

use mathverse_core::error::{MathError, MathResult};

/// Advanced percentage calculations.
pub struct Percentage;

impl Percentage {
    /// Calculate markup: selling price = cost + (cost * markup_rate).
    pub fn markup(cost: f64, markup_rate: f64) -> f64 {
        cost * (1.0 + markup_rate / 100.0)
    }

    /// Calculate markup rate from cost and selling price.
    pub fn markup_rate(cost: f64, selling_price: f64) -> MathResult<f64> {
        if cost == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok((selling_price - cost) / cost * 100.0)
    }

    /// Calculate margin: margin_rate = (selling_price - cost) / selling_price.
    pub fn margin_rate(cost: f64, selling_price: f64) -> MathResult<f64> {
        if selling_price == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok((selling_price - cost) / selling_price * 100.0)
    }

    /// Calculate selling price from cost and margin rate.
    pub fn selling_price_from_margin(cost: f64, margin_rate: f64) -> MathResult<f64> {
        if margin_rate >= 100.0 {
            return Err(MathError::InvalidArgument("margin rate must be less than 100%"));
        }
        Ok(cost / (1.0 - margin_rate / 100.0))
    }

    /// Calculate discount: discounted price = original * (1 - discount_rate).
    pub fn discount(original_price: f64, discount_rate: f64) -> f64 {
        original_price * (1.0 - discount_rate / 100.0)
    }

    /// Calculate discount rate from original and discounted price.
    pub fn discount_rate(original_price: f64, discounted_price: f64) -> MathResult<f64> {
        if original_price == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok((original_price - discounted_price) / original_price * 100.0)
    }

    /// Calculate original price from discounted price and discount rate.
    pub fn original_price(discounted_price: f64, discount_rate: f64) -> MathResult<f64> {
        if discount_rate >= 100.0 {
            return Err(MathError::InvalidArgument("discount rate must be less than 100%"));
        }
        Ok(discounted_price / (1.0 - discount_rate / 100.0))
    }

    /// Calculate compound interest: A = P(1 + r/n)^(nt).
    pub fn compound_interest(
        principal: f64,
        rate: f64,
        times_per_year: u32,
        years: f64,
    ) -> f64 {
        let n = times_per_year as f64;
        principal * (1.0 + rate / 100.0 / n).powf(n * years)
    }

    /// Calculate simple interest: A = P(1 + rt).
    pub fn simple_interest(principal: f64, rate: f64, years: f64) -> f64 {
        principal * (1.0 + rate / 100.0 * years)
    }

    /// Calculate effective annual rate from nominal rate.
    pub fn effective_annual_rate(nominal_rate: f64, compounding_periods: u32) -> f64 {
        let n = compounding_periods as f64;
        ((1.0 + nominal_rate / 100.0 / n).powf(n) - 1.0) * 100.0
    }

    /// Calculate nominal rate from effective annual rate.
    pub fn nominal_rate(effective_rate: f64, compounding_periods: u32) -> f64 {
        let n = compounding_periods as f64;
        ((1.0 + effective_rate / 100.0).powf(1.0 / n) - 1.0) * n * 100.0
    }

    /// Calculate percentage increase.
    pub fn increase(original: f64, percent: f64) -> f64 {
        original * (1.0 + percent / 100.0)
    }

    /// Calculate percentage decrease.
    pub fn decrease(original: f64, percent: f64) -> f64 {
        original * (1.0 - percent / 100.0)
    }

    /// Calculate what percentage A is of B.
    pub fn is_what_percent_of(value: f64, total: f64) -> MathResult<f64> {
        if total == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(value / total * 100.0)
    }

    /// Calculate successive percentage changes.
    pub fn successive_changes(initial: f64, changes: &[f64]) -> f64 {
        let mut result = initial;
        for &change in changes {
            result = result * (1.0 + change / 100.0);
        }
        result
    }

    /// Calculate reverse percentage (find original before percentage change).
    pub fn reverse_percentage(final_value: f64, percent_change: f64) -> MathResult<f64> {
        if percent_change == -100.0 {
            return Err(MathError::InvalidArgument("cannot reverse -100% change"));
        }
        Ok(final_value / (1.0 + percent_change / 100.0))
    }
}

/// Profit and loss calculations.
pub struct ProfitLoss;

impl ProfitLoss {
    /// Calculate gross profit: revenue - cost.
    pub fn gross_profit(revenue: f64, cost: f64) -> f64 {
        revenue - cost
    }

    /// Calculate gross profit margin: (revenue - cost) / revenue.
    pub fn gross_profit_margin(revenue: f64, cost: f64) -> MathResult<f64> {
        if revenue == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok((revenue - cost) / revenue * 100.0)
    }

    /// Calculate net profit: revenue - cost - expenses.
    pub fn net_profit(revenue: f64, cost: f64, expenses: f64) -> f64 {
        revenue - cost - expenses
    }

    /// Calculate net profit margin: net_profit / revenue.
    pub fn net_profit_margin(revenue: f64, net_profit: f64) -> MathResult<f64> {
        if revenue == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(net_profit / revenue * 100.0)
    }

    /// Calculate break-even point in units.
    pub fn break_even_units(fixed_costs: f64, price_per_unit: f64, variable_cost_per_unit: f64) -> MathResult<f64> {
        let contribution_margin = price_per_unit - variable_cost_per_unit;
        if contribution_margin <= 0.0 {
            return Err(MathError::InvalidArgument("contribution margin must be positive"));
        }
        Ok(fixed_costs / contribution_margin)
    }

    /// Calculate break-even point in revenue.
    pub fn break_even_revenue(fixed_costs: f64, contribution_margin_ratio: f64) -> MathResult<f64> {
        if contribution_margin_ratio <= 0.0 {
            return Err(MathError::InvalidArgument("contribution margin ratio must be positive"));
        }
        Ok(fixed_costs / contribution_margin_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markup() {
        assert!((Percentage::markup(100.0, 25.0) - 125.0).abs() < 1e-10);
        assert!((Percentage::markup_rate(100.0, 125.0).unwrap() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_margin() {
        assert!((Percentage::margin_rate(100.0, 125.0).unwrap() - 20.0).abs() < 1e-10);
        assert!((Percentage::selling_price_from_margin(100.0, 20.0).unwrap() - 125.0).abs() < 1e-10);
    }

    #[test]
    fn test_discount() {
        assert!((Percentage::discount(100.0, 20.0) - 80.0).abs() < 1e-10);
        assert!((Percentage::discount_rate(100.0, 80.0).unwrap() - 20.0).abs() < 1e-10);
        assert!((Percentage::original_price(80.0, 20.0).unwrap() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_compound_interest() {
        let result = Percentage::compound_interest(1000.0, 5.0, 12, 1.0);
        assert!((result - 1051.16).abs() < 1e-2);
    }

    #[test]
    fn test_effective_rate() {
        let eff = Percentage::effective_annual_rate(5.0, 12);
        assert!((eff - 5.116).abs() < 1e-3);
    }

    #[test]
    fn test_profit_loss() {
        assert_eq!(ProfitLoss::gross_profit(1000.0, 600.0), 400.0);
        assert!((ProfitLoss::gross_profit_margin(1000.0, 600.0).unwrap() - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_break_even() {
        let units = ProfitLoss::break_even_units(1000.0, 10.0, 6.0).unwrap();
        assert!((units - 250.0).abs() < 1e-10);
    }
}
