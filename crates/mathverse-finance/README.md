# mathverse-finance

[![Crates.io](https://img.shields.io/crates/v/mathverse-finance.svg)](https://crates.io/crates/mathverse-finance)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

Finance domain applications for MathVerse: time value of money, investment analysis, risk metrics, options pricing, and portfolio theory.

## Features

- **Time Value of Money** — FV, PV, annuities, NPV, IRR (Newton-Raphson)
- **Investment Analysis** — ROI, CAGR, Sharpe/Sortino/Treynor ratios, max drawdown
- **Risk Metrics** — VaR, CVaR, covariance, correlation, downside deviation
- **Options Pricing** — Black-Scholes, Greeks (delta, gamma, theta, vega, rho), binomial model
- **Portfolio Theory** — returns, variance, efficient frontier, tracking error, turnover

## Module Overview

| Module | Functions | Description |
|---|---|---|
| `tvm` | 13 | Time value of money: FV, PV, annuities, NPV, IRR |
| `investment` | 13 | ROI, CAGR, risk-adjusted returns, drawdown |
| `risk` | 9 | Variance, VaR, CVaR, covariance, correlation |
| `options` | 11 | Black-Scholes, Greeks, binomial model |
| `portfolio` | 9 | Portfolio returns, variance, efficient frontier |

## Installation

```toml
[dependencies]
mathverse-finance = { path = "../mathverse-finance" }
```

## Quick Start

```rust
use mathverse_finance::*;

fn main() {
    // Time value of money
    let fv = tvm::future_value(1000.0, 0.05, 10);
    println!("FV of $1,000 at 5% for 10 years: ${fv:.2}");

    // NPV
    let cashflows = vec![-1000.0, 300.0, 400.0, 500.0, 600.0];
    let npv = tvm::net_present_value(&cashflows, 0.1);
    println!("NPV at 10%: ${npv:.2}");

    // Sharpe ratio
    let sharpe = investment::sharpe_ratio(0.12, 0.03, 0.15);
    println!("Sharpe ratio: {sharpe:.2}");

    // Black-Scholes call
    let call = options::black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2);
    println!("BS call price: ${call:.2}");
}
```

Expected output:

```
FV of $1,000 at 5% for 10 years: $1628.89
NPV at 10%: $413.22
Sharpe ratio: 0.60
BS call price: $10.45
```

## Per-Module Reference

### `tvm` — Time Value of Money

| Function | Description |
|---|---|
| `future_value(pv, rate, periods)` | FV = PV(1+r)ⁿ |
| `present_value(fv, rate, periods)` | PV = FV/(1+r)ⁿ |
| `future_value_annuity(pmt, rate, periods)` | FV of ordinary annuity |
| `present_value_annuity(pmt, rate, periods)` | PV of ordinary annuity |
| `future_value_annuity_due(pmt, rate, periods)` | FV of annuity due |
| `present_value_annuity_due(pmt, rate, periods)` | PV of annuity due |
| `annuity_payment(pv, rate, periods)` | PMT for loan/annuity |
| `annuity_periods(pv, pmt, rate)` | Number of periods |
| `effective_annual_rate(nominal, periods)` | EAR from nominal rate |
| `continuous_compounding_fv(pv, rate, years)` | FV = PV·e^(rt) |
| `continuous_compounding_pv(fv, rate, years)` | PV = FV·e^(−rt) |
| `net_present_value(cashflows, rate)` | NPV |
| `internal_rate_of_return(cashflows, guess, tol, max_iter)` | IRR via Newton-Raphson |

### `investment` — Investment Analysis

| Function | Description |
|---|---|
| `return_on_investment(initial, final)` | ROI = (final − initial) / initial |
| `cagr(initial, final, years)` | Compound annual growth rate |
| `holding_period_return(initial, final, income)` | HPR |
| `arithmetic_mean_return(returns)` | Arithmetic mean |
| `geometric_mean_return(returns)` | Geometric mean |
| `sharpe_ratio(port_return, rf, std_dev)` | Sharpe ratio |
| `sortino_ratio(port_return, rf, downside_dev)` | Sortino ratio |
| `treynor_ratio(port_return, rf, beta)` | Treynor ratio |
| `information_ratio(port_return, bench_return, tracking_error)` | Information ratio |
| `maximum_drawdown(values)` | Max drawdown % |
| `calmar_ratio(annual_return, max_drawdown)` | Calmar ratio |
| `jensens_alpha(port_return, rf, beta, market_return)` | Jensen's alpha |
| `beta(covariance, market_variance)` | β = Cov/Var |

### `risk` — Risk Metrics

| Function | Description |
|---|---|
| `variance(values)` | Population variance |
| `standard_deviation(values)` | Population std dev |
| `sample_variance(values)` | Sample variance |
| `sample_standard_deviation(values)` | Sample std dev |
| `covariance(x, y)` | Covariance |
| `correlation(x, y)` | Pearson correlation |
| `value_at_risk(port_value, mean, std, confidence)` | Parametric VaR |
| `conditional_var(port_value, mean, std, confidence)` | CVaR / Expected Shortfall |
| `downside_deviation(returns, min_return)` | Downside deviation |

### `options` — Options Pricing

| Function | Description |
|---|---|
| `black_scholes_call(S, K, T, r, σ)` | BS call price |
| `black_scholes_put(S, K, T, r, σ)` | BS put price |
| `call_delta(S, K, T, r, σ)` | Call delta (Δ) |
| `put_delta(S, K, T, r, σ)` | Put delta (Δ) |
| `option_gamma(S, K, T, r, σ)` | Gamma (Γ) |
| `call_theta(S, K, T, r, σ)` | Call theta (Θ) |
| `put_theta(S, K, T, r, σ)` | Put theta (Θ) |
| `option_vega(S, K, T, r, σ)` | Vega (ν) |
| `option_rho(S, K, T, r, σ, is_call)` | Rho (ρ) |
| `binomial_option_price(S, K, T, r, σ, steps, is_call)` | CRR binomial tree |

### `portfolio` — Portfolio Management

| Function | Description |
|---|---|
| `portfolio_return(weights, returns)` | Weighted return |
| `portfolio_variance(weights, cov_matrix)` | Portfolio variance |
| `portfolio_std_dev(weights, cov_matrix)` | Portfolio std dev |
| `minimum_variance_portfolio(cov_matrix)` | Min-variance weights |
| `efficient_portfolio(expected_returns, cov_matrix, target)` | Efficient frontier |
| `tracking_error(port_returns, bench_returns)` | Tracking error |
| `portfolio_beta(port_returns, market_returns)` | Portfolio β |
| `information_ratio(port_returns, bench_returns)` | Information ratio |
| `portfolio_turnover(old_weights, new_weights)` | Turnover |

## Dependencies

- `mathverse-core`
- `mathverse-statistics`
- `mathverse-probability`
- `mathverse-algebra`

## Future Scope

- Monte Carlo simulation for portfolio returns
- Bond pricing (YTM, duration, convexity)
- Credit risk models (Merton, KMV)
- Yield curve fitting (Nelson-Siegel)
- Factor models (Fama-French)

## License

MIT OR Apache-2.0
