//! Lazy vector type that defers computation until evaluated.

use alloc::vec::Vec;
use crate::expr::Expr;

/// A lazily-computed vector that stores an expression tree.
///
/// `LazyVec` wraps an `Expr` and only computes results when `.eval()`
/// is called. Intermediate operations build the tree without allocating.
///
/// # Examples
///
/// ```
/// use mathverse_lazy::LazyVec;
///
/// let a = LazyVec::new(vec![1.0, 2.0, 3.0]);
/// let b = LazyVec::new(vec![4.0, 5.0, 6.0]);
/// let c = a.add(&b).scale(2.0);
/// assert_eq!(c.eval(), vec![10.0, 14.0, 18.0]);
/// ```
pub struct LazyVec<'a> {
    expr: Expr<'a>,
}

impl<'a> LazyVec<'a> {
    /// Create a lazy vector from owned data.
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            expr: Expr::Owned(data),
        }
    }

    /// Create a lazy vector from a borrowed slice.
    pub fn from_slice(data: &'a [f64]) -> Self {
        Self {
            expr: Expr::Slice(data),
        }
    }

    /// Element-wise addition with another lazy vector.
    pub fn add(self, other: &LazyVec<'a>) -> LazyVec<'a> {
        LazyVec {
            expr: Expr::Add(
                Box::new(self.expr),
                Box::new(other.expr.clone()),
            ),
        }
    }

    /// Element-wise subtraction.
    pub fn sub(self, other: &LazyVec<'a>) -> LazyVec<'a> {
        LazyVec {
            expr: Expr::Sub(
                Box::new(self.expr),
                Box::new(other.expr.clone()),
            ),
        }
    }

    /// Element-wise multiplication.
    pub fn mul(self, other: &LazyVec<'a>) -> LazyVec<'a> {
        LazyVec {
            expr: Expr::Mul(
                Box::new(self.expr),
                Box::new(other.expr.clone()),
            ),
        }
    }

    /// Scalar multiplication.
    pub fn scale(self, s: f64) -> LazyVec<'a> {
        LazyVec {
            expr: Expr::Scale(Box::new(self.expr), s),
        }
    }

    /// Negate all elements.
    pub fn neg(self) -> LazyVec<'a> {
        LazyVec {
            expr: Expr::Neg(Box::new(self.expr)),
        }
    }

    /// Fused multiply-add: `self * other + addend`.
    pub fn mul_add(self, other: &LazyVec<'a>, addend: &LazyVec<'a>) -> LazyVec<'a> {
        LazyVec {
            expr: Expr::MulAdd(
                Box::new(self.expr),
                Box::new(other.expr.clone()),
                Box::new(addend.expr.clone()),
            ),
        }
    }

    /// Apply a function to each element.
    pub fn map(self, f: fn(f64) -> f64) -> LazyVec<'a> {
        LazyVec {
            expr: Expr::Map(Box::new(self.expr), f),
        }
    }

    /// Evaluate the expression tree and return the result vector.
    pub fn eval(self) -> Vec<f64> {
        self.expr.eval()
    }

    /// Evaluate into a pre-allocated buffer.
    pub fn eval_into(self, out: &mut Vec<f64>) {
        self.expr.eval_into(out);
    }

    /// Get a reference to the inner expression.
    pub fn as_expr(&self) -> &Expr<'a> {
        &self.expr
    }

    /// Consume self and return the inner expression.
    pub fn into_expr(self) -> Expr<'a> {
        self.expr
    }
}

impl<'a> Clone for LazyVec<'a> {
    fn clone(&self) -> Self {
        // Reconstruct from a fresh eval (expression trees aren't cheaply cloneable)
        Self {
            expr: self.expr.clone(),
        }
    }
}

impl<'a> From<Vec<f64>> for LazyVec<'a> {
    fn from(data: Vec<f64>) -> Self {
        Self::new(data)
    }
}

impl<'a> From<&'a [f64]> for LazyVec<'a> {
    fn from(data: &'a [f64]) -> Self {
        Self::from_slice(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_lazy() {
        let a = LazyVec::new(vec![1.0, 2.0, 3.0]);
        let b = LazyVec::new(vec![4.0, 5.0, 6.0]);
        let result = a.add(&b).eval();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn fused_computation() {
        let a = LazyVec::new(vec![1.0, 2.0]);
        let b = LazyVec::new(vec![3.0, 4.0]);
        let c = LazyVec::new(vec![5.0, 6.0]);
        // a * b + c — should fuse into single pass
        let result = a.mul_add(&b, &c).eval();
        assert_eq!(result, vec![8.0, 14.0]);
    }

    #[test]
    fn chained_ops() {
        let a = LazyVec::new(vec![1.0, 2.0, 3.0]);
        let b = LazyVec::new(vec![4.0, 5.0, 6.0]);
        // (a + b) * 2 - a
        let result = a.add(&b).scale(2.0).sub(&LazyVec::new(vec![1.0, 2.0, 3.0])).eval();
        assert_eq!(result, vec![9.0, 12.0, 15.0]);
    }

    #[test]
    fn map_function() {
        let a = LazyVec::new(vec![1.0, 4.0, 9.0, 16.0]);
        let result = a.map(f64::sqrt).eval();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn from_slice() {
        let data = vec![1.0, 2.0, 3.0];
        let a = LazyVec::from_slice(&data);
        let b = LazyVec::new(vec![4.0, 5.0, 6.0]);
        let result = a.add(&b).eval();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }
}
