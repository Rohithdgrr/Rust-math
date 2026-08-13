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
    Add(Rc<Expr>, Rc<Expr>),
    /// Subtraction (a - b)
    Sub(Rc<Expr>, Rc<Expr>),
    /// Multiplication (a * b)
    Mul(Rc<Expr>, Rc<Expr>),
    /// Division (a / b)
    Div(Rc<Expr>, Rc<Expr>),
    /// Power (a^b)
    Pow(Rc<Expr>, Rc<Expr>),
    /// Negation (-a)
    Neg(Rc<Expr>),
    /// Natural logarithm (ln(x))
    Ln(Rc<Expr>),
    /// Exponential (e^x)
    Exp(Rc<Expr>),
    /// Sine (sin(x))
    Sin(Rc<Expr>),
    /// Cosine (cos(x))
    Cos(Rc<Expr>),
    /// Tangent (tan(x))
    Tan(Rc<Expr>),
    /// Square root (sqrt(x))
    Sqrt(Rc<Expr>),
}

impl Expr {
    /// Create a constant expression
    pub fn c(value: f64) -> Self {
        Expr::Constant(value)
    }

    /// Create a variable expression
    pub fn v(name: &str) -> Self {
        Expr::Variable(name.to_string())
    }

    /// Addition
    pub fn add(self, other: Expr) -> Self {
        Expr::Add(Rc::new(self), Rc::new(other))
    }

    /// Subtraction
    pub fn sub(self, other: Expr) -> Self {
        Expr::Sub(Rc::new(self), Rc::new(other))
    }

    /// Multiplication
    pub fn mul(self, other: Expr) -> Self {
        Expr::Mul(Rc::new(self), Rc::new(other))
    }

    /// Division
    pub fn div(self, other: Expr) -> Self {
        Expr::Div(Rc::new(self), Rc::new(other))
    }

    /// Power
    pub fn pow(self, other: Expr) -> Self {
        Expr::Pow(Rc::new(self), Rc::new(other))
    }

    /// Negation
    pub fn neg(self) -> Self {
        Expr::Neg(Rc::new(self))
    }

    /// Natural logarithm
    pub fn ln(self) -> Self {
        Expr::Ln(Rc::new(self))
    }

    /// Exponential
    pub fn exp(self) -> Self {
        Expr::Exp(Rc::new(self))
    }

    /// Sine
    pub fn sin(self) -> Self {
        Expr::Sin(Rc::new(self))
    }

    /// Cosine
    pub fn cos(self) -> Self {
        Expr::Cos(Rc::new(self))
    }

    /// Tangent
    pub fn tan(self) -> Self {
        Expr::Tan(Rc::new(self))
    }

    /// Square root
    pub fn sqrt(self) -> Self {
        Expr::Sqrt(Rc::new(self))
    }

    /// Evaluate the expression with given variable values
    pub fn evaluate(&self, vars: &std::collections::HashMap<String, f64>) -> Result<f64, String> {
        match self {
            Expr::Constant(c) => Ok(*c),
            Expr::Variable(name) => vars
                .get(name)
                .copied()
                .ok_or_else(|| format!("Variable '{}' not found", name)),
            Expr::Add(a, b) => Ok(a.evaluate(vars)? + b.evaluate(vars)?),
            Expr::Sub(a, b) => Ok(a.evaluate(vars)? - b.evaluate(vars)?),
            Expr::Mul(a, b) => Ok(a.evaluate(vars)? * b.evaluate(vars)?),
            Expr::Div(a, b) => {
                let denom = b.evaluate(vars)?;
                if denom == 0.0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(a.evaluate(vars)? / denom)
                }
            }
            Expr::Pow(a, b) => Ok(a.evaluate(vars)?.powf(b.evaluate(vars)?)),
            Expr::Neg(a) => Ok(-a.evaluate(vars)?),
            Expr::Ln(a) => {
                let val = a.evaluate(vars)?;
                if val <= 0.0 {
                    Err("Logarithm of non-positive number".to_string())
                } else {
                    Ok(val.ln())
                }
            }
            Expr::Exp(a) => Ok(a.evaluate(vars)?.exp()),
            Expr::Sin(a) => Ok(a.evaluate(vars)?.sin()),
            Expr::Cos(a) => Ok(a.evaluate(vars)?.cos()),
            Expr::Tan(a) => Ok(a.evaluate(vars)?.tan()),
            Expr::Sqrt(a) => {
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
            Expr::Constant(c) => c.to_string(),
            Expr::Variable(name) => name.clone(),
            Expr::Add(a, b) => format!("({} + {})", a.to_sympy_string(), b.to_sympy_string()),
            Expr::Sub(a, b) => format!("({} - {})", a.to_sympy_string(), b.to_sympy_string()),
            Expr::Mul(a, b) => format!("({} * {})", a.to_sympy_string(), b.to_sympy_string()),
            Expr::Div(a, b) => format!("({} / {})", a.to_sympy_string(), b.to_sympy_string()),
            Expr::Pow(a, b) => format!("({} ** {})", a.to_sympy_string(), b.to_sympy_string()),
            Expr::Neg(a) => format!("(-{})", a.to_sympy_string()),
            Expr::Ln(a) => format!("ln({})", a.to_sympy_string()),
            Expr::Exp(a) => format!("exp({})", a.to_sympy_string()),
            Expr::Sin(a) => format!("sin({})", a.to_sympy_string()),
            Expr::Cos(a) => format!("cos({})", a.to_sympy_string()),
            Expr::Tan(a) => format!("tan({})", a.to_sympy_string()),
            Expr::Sqrt(a) => format!("sqrt({})", a.to_sympy_string()),
        }
    }

    fn collect_variables(&self, vars: &mut std::collections::HashSet<String>) {
        match self {
            Expr::Variable(name) => {
                vars.insert(name.clone());
            }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
                a.collect_variables(vars);
                b.collect_variables(vars);
            }
            Expr::Neg(a) | Expr::Ln(a) | Expr::Exp(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a) | Expr::Sqrt(a) => {
                a.collect_variables(vars);
            }
            Expr::Constant(_) => {}
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Constant(c) => write!(f, "{}", c),
            Expr::Variable(name) => write!(f, "{}", name),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Pow(a, b) => write!(f, "({} ^ {})", a, b),
            Expr::Neg(a) => write!(f, "(-{})", a),
            Expr::Ln(a) => write!(f, "ln({})", a),
            Expr::Exp(a) => write!(f, "exp({})", a),
            Expr::Sin(a) => write!(f, "sin({})", a),
            Expr::Cos(a) => write!(f, "cos({})", a),
            Expr::Tan(a) => write!(f, "tan({})", a),
            Expr::Sqrt(a) => write!(f, "sqrt({})", a),
        }
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
        let expr = Expr::c(2.0).add(Expr::c(3.0));
        assert_eq!(expr.evaluate(&std::collections::HashMap::new()), Ok(5.0));
    }

    #[test]
    fn test_variables_extraction() {
        let expr = Expr::v("x").add(Expr::v("y").mul(Expr::v("z")));
        let vars = expr.variables();
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
        assert!(vars.contains(&"z".to_string()));
    }
}
