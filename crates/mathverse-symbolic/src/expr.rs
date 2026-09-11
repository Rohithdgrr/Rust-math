//! Expression tree for symbolic computation

use std::fmt;
use std::rc::Rc;

/// Symbolic expression node
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Constant value
    Constant(f64),
    /// Variable with name
    Variable(String),
    /// Addition (a + b)
    Add(Rc<Self>, Rc<Self>),
    /// Subtraction (a - b)
    Sub(Rc<Self>, Rc<Self>),
    /// Multiplication (a * b)
    Mul(Rc<Self>, Rc<Self>),
    /// Division (a / b)
    Div(Rc<Self>, Rc<Self>),
    /// Power (a^b)
    Pow(Rc<Self>, Rc<Self>),
    /// Negation (-a)
    Neg(Rc<Self>),
    /// Natural logarithm (ln(x))
    Ln(Rc<Self>),
    /// Exponential (e^x)
    Exp(Rc<Self>),
    /// Sine (sin(x))
    Sin(Rc<Self>),
    /// Cosine (cos(x))
    Cos(Rc<Self>),
    /// Tangent (tan(x))
    Tan(Rc<Self>),
    /// Square root (sqrt(x))
    Sqrt(Rc<Self>),
}

impl Expr {
    /// Create a constant expression
    pub fn c(value: f64) -> Self {
        Self::Constant(value)
    }

    /// Create a variable expression
    pub fn v(name: &str) -> Self {
        Self::Variable(name.to_string())
    }

    /// Power
    pub fn pow(self, other: Self) -> Self {
        Self::Pow(Rc::new(self), Rc::new(other))
    }

    /// Natural logarithm
    pub fn ln(self) -> Self {
        Self::Ln(Rc::new(self))
    }

    /// Exponential
    pub fn exp(self) -> Self {
        Self::Exp(Rc::new(self))
    }

    /// Sine
    pub fn sin(self) -> Self {
        Self::Sin(Rc::new(self))
    }

    /// Cosine
    pub fn cos(self) -> Self {
        Self::Cos(Rc::new(self))
    }

    /// Tangent
    pub fn tan(self) -> Self {
        Self::Tan(Rc::new(self))
    }

    /// Square root
    pub fn sqrt(self) -> Self {
        Self::Sqrt(Rc::new(self))
    }

    /// Evaluate the expression with given variable values
    pub fn evaluate(&self, vars: &std::collections::HashMap<String, f64>) -> Result<f64, String> {
        match self {
            Self::Constant(c) => Ok(*c),
            Self::Variable(name) => vars
                .get(name)
                .copied()
                .ok_or_else(|| format!("Variable '{name}' not found")),
            Self::Add(a, b) => Ok(a.evaluate(vars)? + b.evaluate(vars)?),
            Self::Sub(a, b) => Ok(a.evaluate(vars)? - b.evaluate(vars)?),
            Self::Mul(a, b) => Ok(a.evaluate(vars)? * b.evaluate(vars)?),
            Self::Div(a, b) => {
                let denom = b.evaluate(vars)?;
                if denom == 0.0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(a.evaluate(vars)? / denom)
                }
            }
            Self::Pow(a, b) => Ok(a.evaluate(vars)?.powf(b.evaluate(vars)?)),
            Self::Neg(a) => Ok(-a.evaluate(vars)?),
            Self::Ln(a) => {
                let val = a.evaluate(vars)?;
                if val <= 0.0 {
                    Err("Logarithm of non-positive number".to_string())
                } else {
                    Ok(val.ln())
                }
            }
            Self::Exp(a) => Ok(a.evaluate(vars)?.exp()),
            Self::Sin(a) => Ok(a.evaluate(vars)?.sin()),
            Self::Cos(a) => Ok(a.evaluate(vars)?.cos()),
            Self::Tan(a) => Ok(a.evaluate(vars)?.tan()),
            Self::Sqrt(a) => {
                let val = a.evaluate(vars)?;
                if val < 0.0 {
                    Err("Square root of negative number".to_string())
                } else {
                    Ok(val.sqrt())
                }
            }
        }
    }

    /// Get all variables in the expression
    pub fn variables(&self) -> Vec<String> {
        let mut vars = std::collections::HashSet::new();
        self.collect_variables(&mut vars);
        vars.into_iter().collect()
    }

    /// Convert this expression to a SymPy-compatible string representation.
    ///
    /// The returned string can be evaluated with `sympy.sympify()` in Python.
    #[must_use]
    pub fn to_sympy_string(&self) -> String {
        match self {
            Self::Constant(c) => c.to_string(),
            Self::Variable(name) => name.clone(),
            Self::Add(a, b) => format!("({} + {})", a.to_sympy_string(), b.to_sympy_string()),
            Self::Sub(a, b) => format!("({} - {})", a.to_sympy_string(), b.to_sympy_string()),
            Self::Mul(a, b) => format!("({} * {})", a.to_sympy_string(), b.to_sympy_string()),
            Self::Div(a, b) => format!("({} / {})", a.to_sympy_string(), b.to_sympy_string()),
            Self::Pow(a, b) => format!("({} ** {})", a.to_sympy_string(), b.to_sympy_string()),
            Self::Neg(a) => format!("(-{})", a.to_sympy_string()),
            Self::Ln(a) => format!("ln({})", a.to_sympy_string()),
            Self::Exp(a) => format!("exp({})", a.to_sympy_string()),
            Self::Sin(a) => format!("sin({})", a.to_sympy_string()),
            Self::Cos(a) => format!("cos({})", a.to_sympy_string()),
            Self::Tan(a) => format!("tan({})", a.to_sympy_string()),
            Self::Sqrt(a) => format!("sqrt({})", a.to_sympy_string()),
        }
    }

    fn collect_variables(&self, vars: &mut std::collections::HashSet<String>) {
        match self {
            Self::Variable(name) => {
                vars.insert(name.clone());
            }
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Pow(a, b) => {
                a.collect_variables(vars);
                b.collect_variables(vars);
            }
            Self::Neg(a)
            | Self::Ln(a)
            | Self::Exp(a)
            | Self::Sin(a)
            | Self::Cos(a)
            | Self::Tan(a)
            | Self::Sqrt(a) => {
                a.collect_variables(vars);
            }
            Self::Constant(_) => {}
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(c) => write!(f, "{c}"),
            Self::Variable(name) => write!(f, "{name}"),
            Self::Add(a, b) => write!(f, "({a} + {b})"),
            Self::Sub(a, b) => write!(f, "({a} - {b})"),
            Self::Mul(a, b) => write!(f, "({a} * {b})"),
            Self::Div(a, b) => write!(f, "({a} / {b})"),
            Self::Pow(a, b) => write!(f, "({a} ^ {b})"),
            Self::Neg(a) => write!(f, "(-{a})"),
            Self::Ln(a) => write!(f, "ln({a})"),
            Self::Exp(a) => write!(f, "exp({a})"),
            Self::Sin(a) => write!(f, "sin({a})"),
            Self::Cos(a) => write!(f, "cos({a})"),
            Self::Tan(a) => write!(f, "tan({a})"),
            Self::Sqrt(a) => write!(f, "sqrt({a})"),
        }
    }
}

impl std::ops::Add for Expr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::Add(Rc::new(self), Rc::new(rhs))
    }
}

impl std::ops::Sub for Expr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::Sub(Rc::new(self), Rc::new(rhs))
    }
}

impl std::ops::Mul for Expr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::Mul(Rc::new(self), Rc::new(rhs))
    }
}

impl std::ops::Div for Expr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::Div(Rc::new(self), Rc::new(rhs))
    }
}

impl std::ops::Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self {
        Self::Neg(Rc::new(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let expr = Expr::c(5.0);
        assert_eq!(expr.evaluate(&std::collections::HashMap::new()), Ok(5.0));
    }

    #[test]
    fn test_variable() {
        let expr = Expr::v("x");
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 3.0);
        assert_eq!(expr.evaluate(&vars), Ok(3.0));
    }

    #[test]
    fn test_arithmetic() {
        let expr = Expr::c(2.0) + Expr::c(3.0);
        assert_eq!(expr.evaluate(&std::collections::HashMap::new()), Ok(5.0));
    }

    #[test]
    fn test_variables_extraction() {
        let expr = Expr::v("x") + Expr::v("y") * Expr::v("z");
        let vars = expr.variables();
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
        assert!(vars.contains(&"z".to_string()));
    }
}
