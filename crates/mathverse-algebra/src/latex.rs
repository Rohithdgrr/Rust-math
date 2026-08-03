//! LaTeX rendering for polynomials and their solutions.
//!
//! Bridges numeric results to a typeset representation suitable for
//! Jupyter, Typora, LaTeX documents, or MathJax/KaTeX web pages. This is the
//! light-weight self-contained half of the symbolic bridge; deeper symbolic
//! manipulation lives in `mathverse-symbolic`.
//!
//! # Example
//!
//! ```rust
//! use mathverse_algebra::latex::{polynomial_latex, equation_solution_latex};
//!
//! assert_eq!(polynomial_latex(&[6.0, -5.0, 1.0]), "x^{2} - 5x + 6");
//! assert_eq!(
//!     equation_solution_latex(&[6.0, -5.0, 1.0], &[2.0, 3.0]),
//!     "x^{2} - 5x + 6 = 0 \\quad\\Longrightarrow\\quad x \\in \\{2, 3\\}"
//! );
//! ```

use core::fmt::Write;

/// Format a float without trailing `.0` when it is near-integral.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn fmt_num(x: f64) -> String {
    if x.fract().abs() < 1e-12 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// Render a single monomial `c·x^power` (without the sign).
#[allow(clippy::float_cmp)]
fn term_latex(coeff: f64, power: usize) -> String {
    let abs = coeff.abs();
    match power {
        0 => fmt_num(abs),
        1 => {
            if abs == 1.0 {
                "x".to_string()
            } else {
                format!("{}x", fmt_num(abs))
            }
        }
        _ => {
            if abs == 1.0 {
                format!("x^{{{power}}}")
            } else {
                format!("{}x^{{{power}}}", fmt_num(abs))
            }
        }
    }
}

/// Render a polynomial (coefficients lowest-degree first) as LaTeX.
///
/// Zero coefficients are skipped; a zero polynomial renders as `0`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::latex::polynomial_latex;
///
/// assert_eq!(polynomial_latex(&[6.0, -5.0, 1.0]), "x^{2} - 5x + 6");
/// assert_eq!(polynomial_latex(&[-2.0, 0.0, 3.0]), "3x^{2} - 2");
/// ```
#[must_use]
pub fn polynomial_latex(coeffs: &[f64]) -> String {
    let mut parts: Vec<(bool, String)> = Vec::new(); // (is_negative, term)
    for (i, &c) in coeffs.iter().enumerate().rev() {
        if c.abs() < 1e-12 {
            continue;
        }
        parts.push((c < 0.0, term_latex(c, i)));
    }
    if parts.is_empty() {
        return "0".to_string();
    }
    let mut out = String::new();
    for (idx, (neg, term)) in parts.iter().enumerate() {
        if idx == 0 {
            if *neg {
                out.push('-');
            }
        } else if *neg {
            out.push_str(" - ");
        } else {
            out.push_str(" + ");
        }
        out.push_str(term);
    }
    out
}

/// Render a set of roots in LaTeX set notation.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::latex::roots_latex;
///
/// assert_eq!(roots_latex(&[2.0, 3.0]), "\\{2, 3\\}");
/// assert_eq!(roots_latex(&[]), "\\varnothing");
/// ```
#[must_use]
pub fn roots_latex(roots: &[f64]) -> String {
    if roots.is_empty() {
        return "\\varnothing".to_string();
    }
    let mut out = String::from("\\{");
    for (i, &r) in roots.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&fmt_num(r));
    }
    out.push_str("\\}");
    out
}

/// Render the full equation `p(x) = 0 ⟹ x ∈ {roots}` in LaTeX.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::latex::equation_solution_latex;
///
/// assert_eq!(
///     equation_solution_latex(&[6.0, -5.0, 1.0], &[2.0, 3.0]),
///     "x^{2} - 5x + 6 = 0 \\quad\\Longrightarrow\\quad x \\in \\{2, 3\\}"
/// );
/// ```
#[must_use]
pub fn equation_solution_latex(coeffs: &[f64], roots: &[f64]) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "{} = 0 \\quad\\Longrightarrow\\quad x \\in {}",
        polynomial_latex(coeffs),
        roots_latex(roots)
    );
    out
}

/// Render a list of factored linear factors, e.g. `(x - 2)(x - 3)`.
///
/// Each root contributes one `(x - r)` factor.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::latex::factors_latex;
///
/// assert_eq!(factors_latex(&[2.0, 3.0]), "(x - 2)(x - 3)");
/// ```
#[must_use]
pub fn factors_latex(roots: &[f64]) -> String {
    let mut out = String::new();
    for &r in roots {
        out.push('(');
        out.push_str("x");
        if r >= 0.0 {
            out.push_str(" - ");
        } else {
            out.push_str(" + ");
        }
        out.push_str(&fmt_num(r.abs()));
        out.push(')');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_render() {
        assert_eq!(polynomial_latex(&[6.0, -5.0, 1.0]), "x^{2} - 5x + 6");
        assert_eq!(polynomial_latex(&[-2.0, 0.0, 3.0]), "3x^{2} - 2");
        assert_eq!(polynomial_latex(&[0.0]), "0");
        assert_eq!(polynomial_latex(&[7.0]), "7");
        assert_eq!(polynomial_latex(&[0.0, 1.0]), "x");
        assert_eq!(polynomial_latex(&[-1.0, 0.0, 0.0, 1.0]), "x^{3} - 1");
    }

    #[test]
    fn roots_render() {
        assert_eq!(roots_latex(&[2.0, 3.0]), "\\{2, 3\\}");
        assert_eq!(roots_latex(&[1.5, -0.25]), "\\{1.5, -0.25\\}");
        assert_eq!(roots_latex(&[]), "\\varnothing");
    }

    #[test]
    fn equation_render() {
        assert_eq!(
            equation_solution_latex(&[6.0, -5.0, 1.0], &[2.0, 3.0]),
"x^{2} - 5x + 6 = 0 \\quad\\Longrightarrow\\quad x \\in \\{2, 3\\}"
        );
    }

    #[test]
    fn factors_render() {
        assert_eq!(factors_latex(&[2.0, 3.0]), "(x - 2)(x - 3)");
        assert_eq!(factors_latex(&[-2.0]), "(x + 2)");
        assert_eq!(factors_latex(&[]), "");
    }
}
