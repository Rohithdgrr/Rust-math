//! Expression tree nodes for lazy evaluation.

use alloc::boxed::Box;
use alloc::vec::Vec;

/// A lazily-evaluated expression over `f64` slices.
///
/// `Expr` forms a tree of deferred operations. Call `.eval()` to materialize
/// the result into a `Vec<f64>`.
#[derive(Clone)]
pub enum Expr<'a> {
    /// A reference to an existing slice (zero-copy).
    Slice(&'a [f64]),
    /// An owned vector (takes ownership).
    Owned(Vec<f64>),
    /// Element-wise addition of two sub-expressions.
    Add(Box<Expr<'a>>, Box<Expr<'a>>),
    /// Element-wise subtraction.
    Sub(Box<Expr<'a>>, Box<Expr<'a>>),
    /// Element-wise multiplication.
    Mul(Box<Expr<'a>>, Box<Expr<'a>>),
    /// Scalar multiplication.
    Scale(Box<Expr<'a>>, f64),
    /// Negation.
    Neg(Box<Expr<'a>>),
    /// Fused multiply-add: `a * b + c`.
    MulAdd(Box<Expr<'a>>, Box<Expr<'a>>, Box<Expr<'a>>),
    /// Map a function over elements.
    Map(Box<Expr<'a>>, fn(f64) -> f64),
}

impl<'a> Expr<'a> {
    /// Evaluate the expression tree, producing a `Vec<f64>`.
    ///
    /// This traverses the tree and performs all deferred computations.
    pub fn eval(self) -> Vec<f64> {
        match self {
            Self::Slice(s) => s.to_vec(),
            Self::Owned(v) => v,
            Self::Add(l, r) => {
                let lv = l.eval();
                let rv = r.eval();
                lv.iter().zip(&rv).map(|(a, b)| a + b).collect()
            }
            Self::Sub(l, r) => {
                let lv = l.eval();
                let rv = r.eval();
                lv.iter().zip(&rv).map(|(a, b)| a - b).collect()
            }
            Self::Mul(l, r) => {
                let lv = l.eval();
                let rv = r.eval();
                lv.iter().zip(&rv).map(|(a, b)| a * b).collect()
            }
            Self::Scale(inner, s) => inner.eval().iter().map(|v| v * s).collect(),
            Self::Neg(inner) => inner.eval().iter().map(|v| -v).collect(),
            Self::MulAdd(a, b, c) => {
                let av = a.eval();
                let bv = b.eval();
                let cv = c.eval();
                av.iter()
                    .zip(&bv)
                    .zip(&cv)
                    .map(|((a, b), c)| a * b + c)
                    .collect()
            }
            Self::Map(inner, f) => inner.eval().iter().map(|&v| f(v)).collect(),
        }
    }

    /// Evaluate the expression tree into a pre-allocated buffer.
    pub fn eval_into(self, out: &mut Vec<f64>) {
        let result = self.eval();
        out.clear();
        out.extend_from_slice(&result);
    }

    /// Returns the number of elements (if statically known from slices).
    pub fn len_hint(&self) -> Option<usize> {
        match self {
            Self::Slice(s) => Some(s.len()),
            Self::Owned(v) => Some(v.len()),
            _ => None,
        }
    }
}

/// Reference-counted expression for shared ownership.
pub type ExprRef<'a> = alloc::sync::Arc<Expr<'a>>;

/// Create an owned expression from a vector.
pub fn owned(data: Vec<f64>) -> Expr<'static> {
    Expr::Owned(data)
}

/// Create a borrowed expression from a slice.
pub fn borrowed<'a>(data: &'a [f64]) -> Expr<'a> {
    Expr::Slice(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_slice() {
        let e = Expr::Slice(&[1.0, 2.0, 3.0]);
        assert_eq!(e.eval(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn eval_add() {
        let a = Expr::Slice(&[1.0, 2.0]);
        let b = Expr::Slice(&[3.0, 4.0]);
        let e = Expr::Add(Box::new(a), Box::new(b));
        assert_eq!(e.eval(), vec![4.0, 6.0]);
    }

    #[test]
    fn eval_scale() {
        let a = Expr::Slice(&[1.0, 2.0, 3.0]);
        let e = Expr::Scale(Box::new(a), 2.0);
        assert_eq!(e.eval(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn eval_mul_add() {
        let a = Expr::Slice(&[1.0, 2.0]);
        let b = Expr::Slice(&[3.0, 4.0]);
        let c = Expr::Slice(&[5.0, 6.0]);
        let e = Expr::MulAdd(Box::new(a), Box::new(b), Box::new(c));
        assert_eq!(e.eval(), vec![8.0, 14.0]);
    }

    #[test]
    fn eval_map() {
        let a = Expr::Slice(&[1.0, 4.0, 9.0]);
        let e = Expr::Map(Box::new(a), f64::sqrt);
        assert_eq!(e.eval(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn nested_ops() {
        let a = Expr::Slice(&[1.0, 2.0]);
        let b = Expr::Slice(&[3.0, 4.0]);
        // (a + b) * 2
        let e = Expr::Scale(
            Box::new(Expr::Add(Box::new(a), Box::new(b))),
            2.0,
        );
        assert_eq!(e.eval(), vec![8.0, 12.0]);
    }

    #[test]
    fn len_hint() {
        assert_eq!(Expr::Slice(&[1.0, 2.0]).len_hint(), Some(2));
        assert_eq!(Expr::Owned(vec![1.0]).len_hint(), Some(1));
        let复合 = Expr::Add(
            Box::new(Expr::Slice(&[1.0])),
            Box::new(Expr::Slice(&[2.0])),
        );
        assert_eq!(复合.len_hint(), None);
    }
}
