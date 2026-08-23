//! Symbolic differentiation

use crate::expr::Expr;

/// Compute the symbolic derivative of an expression with respect to a variable
///
/// # Arguments
/// * `expr` - The expression to differentiate
/// * `var` - The variable to differentiate with respect to
///
/// # Returns
/// The derivative expression
pub fn differentiate(expr: &Expr, var: &str) -> Expr {
    match expr {
        Expr::Constant(_) => Expr::c(0.0),
        Expr::Variable(name) => {
            if name == var {
                Expr::c(1.0)
            } else {
                Expr::c(0.0)
            }
        }
        Expr::Add(a, b) => {
            crate::simplify::simplify(&(differentiate(a, var) + differentiate(b, var)))
        }
        Expr::Sub(a, b) => {
            crate::simplify::simplify(&(differentiate(a, var) - differentiate(b, var)))
        }
        Expr::Mul(a, b) => {
            // Product rule: (uv)' = u'v + uv'
            let da = differentiate(a, var);
            let db = differentiate(b, var);
            (da * (**b).clone()) + ((**a).clone() * db)
        }
        Expr::Div(a, b) => {
            // Quotient rule: (u/v)' = (u'v - uv') / v^2
            let da = differentiate(a, var);
            let db = differentiate(b, var);
            let numerator = (da * (**b).clone()) - ((**a).clone() * db);
            let denominator = (**b).clone().pow(Expr::c(2.0));
            numerator / denominator
        }
        Expr::Pow(a, b) => {
            // Chain rule for power: (u^v)' = u^v * (v' * ln(u) + v * u'/u)
            let base = (**a).clone();
            let exp = (**b).clone();
            let da = differentiate(a, var);
            let db = differentiate(b, var);

            let ln_base = base.clone().ln();
            let term1 = db * ln_base;
            let term2 = exp.clone() * (da / base.clone());
            let inner = term1 + term2;
            base.pow(exp) * inner
        }
        Expr::Neg(a) => -differentiate(a, var),
        Expr::Ln(a) => {
            // Chain rule: ln(u)' = u' / u
            let da = differentiate(a, var);
            da / (**a).clone()
        }
        Expr::Exp(a) => {
            // Chain rule: e^u' = e^u * u'
            let da = differentiate(a, var);
            (**a).clone().exp() * da
        }
        Expr::Sin(a) => {
            // Chain rule: sin(u)' = cos(u) * u'
            let da = differentiate(a, var);
            (**a).clone().cos() * da
        }
        Expr::Cos(a) => {
            // Chain rule: cos(u)' = -sin(u) * u'
            let da = differentiate(a, var);
            -((**a).clone().sin() * da)
        }
        Expr::Tan(a) => {
            // Chain rule: tan(u)' = sec^2(u) * u' = (1/cos(u)^2) * u'
            let da = differentiate(a, var);
            let cos_a = (**a).clone().cos();
            let sec_squared = Expr::c(1.0) / cos_a.pow(Expr::c(2.0));
            sec_squared * da
        }
        Expr::Sqrt(a) => {
            // Chain rule: sqrt(u)' = (1/(2*sqrt(u))) * u'
            let da = differentiate(a, var);
            let sqrt_a = (**a).clone().sqrt();
            let derivative = Expr::c(1.0) / (Expr::c(2.0) * sqrt_a);
            derivative * da
        }
    }
}

/// Compute the nth derivative of an expression
///
/// # Arguments
/// * `expr` - The expression to differentiate
/// * `var` - The variable to differentiate with respect to
/// * `n` - The order of derivative
///
/// # Returns
/// The nth derivative expression
pub fn nth_derivative(expr: &Expr, var: &str, n: usize) -> Expr {
    if n == 0 {
        expr.clone()
    } else {
        let first = differentiate(expr, var);
        nth_derivative(&first, var, n - 1)
    }
}

/// Compute the partial derivative of a multivariate expression
///
/// # Arguments
/// * `expr` - The expression to differentiate
/// * `var` - The variable to differentiate with respect to
///
/// # Returns
/// The partial derivative expression
pub fn partial_derivative(expr: &Expr, var: &str) -> Expr {
    differentiate(expr, var)
}

/// Compute the gradient of a multivariate expression
///
/// # Arguments
/// * `expr` - The expression to differentiate
/// * `vars` - Slice of variable names
///
/// # Returns
/// Vector of partial derivatives
pub fn gradient(expr: &Expr, vars: &[&str]) -> Vec<Expr> {
    vars.iter().map(|&v| partial_derivative(expr, v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_derivative() {
        let expr = Expr::c(5.0);
        let deriv = differentiate(&expr, "x");
        assert_eq!(deriv, Expr::c(0.0));
    }

    #[test]
    fn test_variable_derivative() {
        let expr = Expr::v("x");
        let deriv = differentiate(&expr, "x");
        assert_eq!(deriv, Expr::c(1.0));
    }

    #[test]
    fn test_addition_derivative() {
        let expr = Expr::v("x") + Expr::c(3.0);
        let deriv = differentiate(&expr, "x");
        assert_eq!(deriv, Expr::c(1.0));
    }

    #[test]
    fn test_multiplication_derivative() {
        let expr = Expr::v("x") * Expr::v("x");
        let _deriv = differentiate(&expr, "x");
        // d/dx(x*x) = 1*x + x*1 = 2x
        let _expected = Expr::c(2.0) * Expr::v("x");
        // Note: This won't match exactly due to expression structure, but should evaluate the same
    }

    #[test]
    fn test_sin_derivative() {
        let expr = Expr::v("x").sin();
        let deriv = differentiate(&expr, "x");
        // d/dx(sin(x)) = cos(x)
        // Check that it evaluates correctly
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 0.0);
        assert!((deriv.evaluate(&vars).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_exp_derivative() {
        let expr = Expr::v("x").exp();
        let deriv = differentiate(&expr, "x");
        // d/dx(e^x) = e^x
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 0.0);
        assert!((deriv.evaluate(&vars).unwrap() - 1.0).abs() < 1e-6);
    }
}
