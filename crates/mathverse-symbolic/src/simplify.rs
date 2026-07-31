//! Expression simplification

use crate::expr::Expr;

/// Simplify an expression by applying algebraic rules
/// 
/// # Arguments
/// * `expr` - The expression to simplify
/// 
/// # Returns
/// Simplified expression
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Constant(c) => Expr::c(*c),
        Expr::Variable(name) => Expr::v(name),
        Expr::Add(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            simplify_add(&a_simp, &b_simp)
        }
        Expr::Sub(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            simplify_sub(&a_simp, &b_simp)
        }
        Expr::Mul(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            simplify_mul(&a_simp, &b_simp)
        }
        Expr::Div(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            simplify_div(&a_simp, &b_simp)
        }
        Expr::Pow(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            simplify_pow(&a_simp, &b_simp)
        }
        Expr::Neg(a) => {
            let a_simp = simplify(a);
            simplify_neg(&a_simp)
        }
        Expr::Ln(a) => simplify(a).ln(),
        Expr::Exp(a) => simplify(a).exp(),
        Expr::Sin(a) => simplify(a).sin(),
        Expr::Cos(a) => simplify(a).cos(),
        Expr::Tan(a) => simplify(a).tan(),
        Expr::Sqrt(a) => simplify(a).sqrt(),
    }
}

/// Simplify addition
fn simplify_add(a: &Expr, b: &Expr) -> Expr {
    match (a, b) {
        // 0 + x = x
        (Expr::Constant(0.0), _) => b.clone(),
        (_, Expr::Constant(0.0)) => a.clone(),
        // x + (-y) = x - y
        (x, Expr::Neg(y)) => x.clone().sub((**y).clone()),
        (Expr::Neg(x), y) => y.clone().sub((**x).clone()),
        // Constant folding
        (Expr::Constant(c1), Expr::Constant(c2)) => Expr::c(c1 + c2),
        // x + x = 2x
        _ if a == b => Expr::c(2.0).mul(a.clone()),
        _ => a.clone().add(b.clone()),
    }
}

/// Simplify subtraction
fn simplify_sub(a: &Expr, b: &Expr) -> Expr {
    match (a, b) {
        // x - 0 = x
        (_, Expr::Constant(0.0)) => a.clone(),
        // 0 - x = -x
        (Expr::Constant(0.0), _) => b.clone().neg(),
        // x - x = 0
        _ if a == b => Expr::c(0.0),
        // Constant folding
        (Expr::Constant(c1), Expr::Constant(c2)) => Expr::c(c1 - c2),
        // x - (-y) = x + y
        (_, Expr::Neg(y)) => a.clone().add((**y).clone()),
        _ => a.clone().sub(b.clone()),
    }
}

/// Simplify multiplication
fn simplify_mul(a: &Expr, b: &Expr) -> Expr {
    match (a, b) {
        // 0 * x = 0
        (Expr::Constant(0.0), _) | (_, Expr::Constant(0.0)) => Expr::c(0.0),
        // 1 * x = x
        (Expr::Constant(1.0), _) => b.clone(),
        (_, Expr::Constant(1.0)) => a.clone(),
        // -1 * x = -x
        (Expr::Constant(-1.0), _) => b.clone().neg(),
        (_, Expr::Constant(-1.0)) => a.clone().neg(),
        // Constant folding
        (Expr::Constant(c1), Expr::Constant(c2)) => Expr::c(c1 * c2),
        // x * x = x^2
        _ if a == b => a.clone().pow(Expr::c(2.0)),
        _ => a.clone().mul(b.clone()),
    }
}

/// Simplify division
fn simplify_div(a: &Expr, b: &Expr) -> Expr {
    match (a, b) {
        // 0 / x = 0
        (Expr::Constant(0.0), _) => Expr::c(0.0),
        // x / 1 = x
        (_, Expr::Constant(1.0)) => a.clone(),
        // x / x = 1
        _ if a == b => Expr::c(1.0),
        // x / -1 = -x
        (_, Expr::Constant(-1.0)) => a.clone().neg(),
        // Constant folding
        (Expr::Constant(c1), Expr::Constant(c2)) if *c2 != 0.0 => Expr::c(c1 / c2),
        _ => a.clone().div(b.clone()),
    }
}

/// Simplify power
fn simplify_pow(a: &Expr, b: &Expr) -> Expr {
    match (a, b) {
        // x^0 = 1
        (_, Expr::Constant(0.0)) => Expr::c(1.0),
        // x^1 = x
        (_, Expr::Constant(1.0)) => a.clone(),
        // 0^x = 0 (for x > 0)
        (Expr::Constant(0.0), Expr::Constant(c)) if *c > 0.0 => Expr::c(0.0),
        // 1^x = 1
        (Expr::Constant(1.0), _) => Expr::c(1.0),
        // Constant folding
        (Expr::Constant(c1), Expr::Constant(c2)) => Expr::c(c1.powf(*c2)),
        _ => a.clone().pow(b.clone()),
    }
}

/// Simplify negation
fn simplify_neg(a: &Expr) -> Expr {
    match a {
        // -(-x) = x
        Expr::Neg(inner) => (**inner).clone(),
        // -(c) = -c
        Expr::Constant(c) => Expr::c(-c),
        // -0 = 0
        Expr::Constant(0.0) => Expr::c(0.0),
        _ => a.clone().neg(),
    }
}

/// Expand an expression (distribute multiplication over addition)
/// 
/// # Arguments
/// * `expr` - The expression to expand
/// 
/// # Returns
/// Expanded expression
pub fn expand(expr: &Expr) -> Expr {
    match expr {
        Expr::Constant(_) | Expr::Variable(_) => expr.clone(),
        Expr::Add(a, b) => expand(a).add(expand(b)),
        Expr::Sub(a, b) => expand(a).sub(expand(b)),
        Expr::Mul(a, b) => expand_mul(a, b),
        Expr::Div(a, b) => expand(a).div(expand(b)),
        Expr::Pow(a, b) => expand(a).pow(expand(b)),
        Expr::Neg(a) => expand(a).neg(),
        Expr::Ln(a) => expand(a).ln(),
        Expr::Exp(a) => expand(a).exp(),
        Expr::Sin(a) => expand(a).sin(),
        Expr::Cos(a) => expand(a).cos(),
        Expr::Tan(a) => expand(a).tan(),
        Expr::Sqrt(a) => expand(a).sqrt(),
    }
}

/// Expand multiplication (distributive property)
fn expand_mul(a: &Expr, b: &Expr) -> Expr {
    let a_exp = expand(a);
    let b_exp = expand(b);

    match (&a_exp, &b_exp) {
        // Distribute: (x + y) * z = x*z + y*z
        (Expr::Add(a1, a2), _) => {
            expand_mul(a1, b).add(expand_mul(a2, b))
        }
        (_, Expr::Add(b1, b2)) => {
            expand_mul(&a_exp, b1).add(expand_mul(&a_exp, b2))
        }
        // Distribute: (x - y) * z = x*z - y*z
        (Expr::Sub(a1, a2), _) => {
            expand_mul(a1, b).sub(expand_mul(a2, b))
        }
        (_, Expr::Sub(b1, b2)) => {
            expand_mul(&a_exp, b1).sub(expand_mul(&a_exp, b2))
        }
        _ => a_exp.mul(b_exp),
    }
}

/// Factor an expression (reverse of expand)
/// 
/// # Arguments
/// * `expr` - The expression to factor
/// 
/// # Returns
/// Factored expression
pub fn factor(expr: &Expr) -> Expr {
    // Basic factoring - extract common factors
    match expr {
        Expr::Add(a, b) => factor_add(a, b),
        Expr::Sub(a, b) => factor_sub(a, b),
        _ => expr.clone(),
    }
}

/// Factor addition (find common factors)
fn factor_add(a: &Expr, b: &Expr) -> Expr {
    // Check if both terms have a common factor
    if let (Expr::Mul(a1, a2), Expr::Mul(b1, b2)) = (a, b) {
        if a1 == b1 {
            a1.clone().mul(a2.clone().add((**b2).clone()))
        } else if a1 == b2 {
            a1.clone().mul(a2.clone().add((**b1).clone()))
        } else if a2 == b1 {
            a2.clone().mul((**a1).clone().add((**b2).clone()))
        } else if a2 == b2 {
            a2.clone().mul((**a1).clone().add((**b1).clone()))
        } else {
            a.clone().add(b.clone())
        }
    } else {
        a.clone().add(b.clone())
    }
}

/// Factor subtraction
fn factor_sub(a: &Expr, b: &Expr) -> Expr {
    if let (Expr::Mul(a1, a2), Expr::Mul(b1, b2)) = (a, b) {
        if a1 == b1 {
            a1.clone().mul(a2.clone().sub((**b2).clone()))
        } else if a1 == b2 {
            a1.clone().mul(a2.clone().sub((**b1).clone()))
        } else if a2 == b1 {
            a2.clone().mul((**a1).clone().sub((**b2).clone()))
        } else if a2 == b2 {
            a2.clone().mul((**a1).clone().sub((**b1).clone()))
        } else {
            a.clone().sub(b.clone())
        }
    } else {
        a.clone().sub(b.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_add_zero() {
        let expr = Expr::c(0.0).add(Expr::v("x"));
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::v("x"));
    }

    #[test]
    fn test_simplify_mul_zero() {
        let expr = Expr::c(0.0).mul(Expr::v("x"));
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::c(0.0));
    }

    #[test]
    fn test_simplify_mul_one() {
        let expr = Expr::c(1.0).mul(Expr::v("x"));
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::v("x"));
    }

    #[test]
    fn test_simplify_pow_zero() {
        let expr = Expr::v("x").pow(Expr::c(0.0));
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::c(1.0));
    }

    #[test]
    fn test_simplify_neg_neg() {
        let expr = Expr::v("x").neg().neg();
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::v("x"));
    }

    #[test]
    fn test_expand() {
        let expr = Expr::v("x").add(Expr::v("y")).mul(Expr::v("z"));
        let expanded = expand(&expr);
        // (x + y) * z = x*z + y*z
        let expected = Expr::v("x").mul(Expr::v("z")).add(Expr::v("y").mul(Expr::v("z")));
        assert_eq!(expanded, expected);
    }
}
