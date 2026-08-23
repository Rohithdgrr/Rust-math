//! LaTeX rendering for symbolic expressions

use crate::expr::Expr;

/// Convert an expression to LaTeX format
///
/// # Arguments
/// * `expr` - The expression to convert
///
/// # Returns
/// LaTeX string representation
pub fn to_latex(expr: &Expr) -> String {
    match expr {
        Expr::Constant(c) => {
            if c.fract() == 0.0 {
                format!("{}", *c as i64)
            } else {
                format!("{c:.6}")
                    .trim_end_matches('0')
                    .trim_end_matches('.')
                    .to_string()
            }
        }
        Expr::Variable(name) => name.clone(),
        Expr::Add(a, b) => {
            let left = to_latex(a);
            let right = to_latex(b);
            format!("{left} + {right}")
        }
        Expr::Sub(a, b) => {
            let left = to_latex(a);
            let right = to_latex(b);
            format!("{left} - {right}")
        }
        Expr::Mul(a, b) => {
            let left = to_latex(a);
            let right = to_latex(b);
            // Add parentheses if needed
            let left_paren = needs_parentheses(a, true);
            let right_paren = needs_parentheses(b, false);

            let left_str = if left_paren {
                format!("({left})")
            } else {
                left
            };
            let right_str = if right_paren {
                format!("({right})")
            } else {
                right
            };

            format!("{left_str} \\cdot {right_str}")
        }
        Expr::Div(a, b) => {
            let left = to_latex(a);
            let right = to_latex(b);
            format!("\\frac{{{left}}}{{{right}}}")
        }
        Expr::Pow(a, b) => {
            let base = to_latex(a);
            let exp = to_latex(b);
            let base_paren = needs_parentheses(a, true);
            let base_str = if base_paren {
                format!("({base})")
            } else {
                base
            };
            format!("{base_str}^{{{exp}}}")
        }
        Expr::Neg(a) => {
            let inner = to_latex(a);
            let inner_paren = needs_parentheses(a, false);
            if inner_paren {
                format!("({inner})")
            } else {
                inner
            }
        }
        Expr::Ln(a) => unary_fn("ln", a),
        Expr::Exp(a) => {
            let inner = to_latex(a);
            format!("e^{{{inner}}}")
        }
        Expr::Sin(a) => unary_fn("sin", a),
        Expr::Cos(a) => unary_fn("cos", a),
        Expr::Tan(a) => unary_fn("tan", a),
        Expr::Sqrt(a) => {
            let inner = to_latex(a);
            format!("\\sqrt{{{inner}}}")
        }
    }
}

/// Render a named unary function, parenthesizing the argument if needed.
fn unary_fn(name: &str, a: &Expr) -> String {
    let inner = to_latex(a);
    if needs_parentheses(a, false) {
        format!("\\{name}({inner})")
    } else {
        format!("\\{name} {inner}")
    }
}

/// Check if an expression needs parentheses in LaTeX output
fn needs_parentheses(expr: &Expr, is_power_base: bool) -> bool {
    match expr {
        Expr::Constant(_)
        | Expr::Variable(_)
        | Expr::Ln(_)
        | Expr::Exp(_)
        | Expr::Sin(_)
        | Expr::Cos(_)
        | Expr::Tan(_)
        | Expr::Sqrt(_) => false,
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Pow(_, _) | Expr::Neg(_) => true,
        Expr::Mul(_, _) | Expr::Div(_, _) => is_power_base,
    }
}

/// Convert an expression to LaTeX with display mode (centered, larger)
///
/// # Arguments
/// * `expr` - The expression to convert
///
/// # Returns
/// LaTeX string with display mode
pub fn to_latex_display(expr: &Expr) -> String {
    format!("\\[ {} \\]", to_latex(expr))
}

/// Convert an expression to LaTeX with inline mode
///
/// # Arguments
/// * `expr` - The expression to convert
///
/// # Returns
/// LaTeX string with inline mode
pub fn to_latex_inline(expr: &Expr) -> String {
    format!("$ {} $", to_latex(expr))
}

/// Create a LaTeX equation environment
///
/// # Arguments
/// * `label` - Optional label for the equation
/// * `expr` - The expression to convert
///
/// # Returns
/// LaTeX equation environment string
pub fn to_latex_equation(label: Option<&str>, expr: &Expr) -> String {
    let latex = to_latex(expr);
    match label {
        Some(l) => format!("\\begin{{equation}}\\label{{{l}}} {latex} \\end{{equation}}"),
        None => format!("\\begin{{equation}} {latex} \\end{{equation}}"),
    }
}

/// Create a LaTeX align environment for multiple expressions
///
/// # Arguments
/// * `exprs` - Slice of expressions to align
///
/// # Returns
/// LaTeX align environment string
pub fn to_latex_align(exprs: &[Expr]) -> String {
    let lines: Vec<String> = exprs.iter().map(to_latex).collect();
    let joined = lines.join(" \\\\\n");
    format!("\\begin{{align}}\n{joined}\\n\\end{{align}}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_latex() {
        let expr = Expr::c(5.0);
        assert_eq!(to_latex(&expr), "5");
    }

    #[test]
    fn test_variable_latex() {
        let expr = Expr::v("x");
        assert_eq!(to_latex(&expr), "x");
    }

    #[test]
    fn test_addition_latex() {
        let expr = Expr::v("x") + Expr::v("y");
        assert_eq!(to_latex(&expr), "x + y");
    }

    #[test]
    fn test_multiplication_latex() {
        let expr = Expr::v("x") * Expr::v("y");
        assert_eq!(to_latex(&expr), "x \\cdot y");
    }

    #[test]
    fn test_division_latex() {
        let expr = Expr::v("x") / Expr::v("y");
        assert_eq!(to_latex(&expr), "\\frac{x}{y}");
    }

    #[test]
    fn test_power_latex() {
        let expr = Expr::v("x").pow(Expr::c(2.0));
        assert_eq!(to_latex(&expr), "x^{2}");
    }

    #[test]
    fn test_sin_latex() {
        let expr = Expr::v("x").sin();
        assert_eq!(to_latex(&expr), "\\sin x");
    }

    #[test]
    fn test_exp_latex() {
        let expr = Expr::v("x").exp();
        assert_eq!(to_latex(&expr), "e^{x}");
    }

    #[test]
    fn test_ln_latex() {
        let expr = Expr::v("x").ln();
        assert_eq!(to_latex(&expr), "\\ln x");
    }

    #[test]
    fn test_sqrt_latex() {
        let expr = Expr::v("x").sqrt();
        assert_eq!(to_latex(&expr), "\\sqrt{x}");
    }
}
