//! Top-level lazy operation constructors.

use alloc::boxed::Box;
use crate::expr::Expr;
use crate::lazy_vec::LazyVec;

/// Lazy element-wise addition.
pub fn lazy_add<'a>(a: &LazyVec<'a>, b: &LazyVec<'a>) -> LazyVec<'a> {
    a.clone().add(b)
}

/// Lazy element-wise subtraction.
pub fn lazy_sub<'a>(a: &LazyVec<'a>, b: &LazyVec<'a>) -> LazyVec<'a> {
    a.clone().sub(b)
}

/// Lazy element-wise multiplication.
pub fn lazy_mul<'a>(a: &LazyVec<'a>, b: &LazyVec<'a>) -> LazyVec<'a> {
    a.clone().mul(b)
}

/// Lazy scalar multiplication.
pub fn lazy_scale<'a>(a: &LazyVec<'a>, s: f64) -> LazyVec<'a> {
    a.clone().scale(s)
}

/// Build an expression directly from two slices.
pub fn add_slices<'a>(a: &'a [f64], b: &'a [f64]) -> Expr<'a> {
    Expr::Add(Box::new(Expr::Slice(a)), Box::new(Expr::Slice(b)))
}

/// Build a scale expression directly from a slice.
pub fn scale_slice<'a>(a: &'a [f64], s: f64) -> Expr<'a> {
    Expr::Scale(Box::new(Expr::Slice(a)), s)
}

/// Build a fused multiply-add expression.
pub fn mul_add_slices<'a>(a: &'a [f64], b: &'a [f64], c: &'a [f64]) -> Expr<'a> {
    Expr::MulAdd(
        Box::new(Expr::Slice(a)),
        Box::new(Expr::Slice(b)),
        Box::new(Expr::Slice(c)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_ops() {
        let a = LazyVec::new(vec![1.0, 2.0]);
        let b = LazyVec::new(vec![3.0, 4.0]);
        assert_eq!(lazy_add(&a, &b).eval(), vec![4.0, 6.0]);
        assert_eq!(lazy_sub(&b, &a).eval(), vec![2.0, 2.0]);
        assert_eq!(lazy_mul(&a, &b).eval(), vec![3.0, 8.0]);
        assert_eq!(lazy_scale(&a, 3.0).eval(), vec![3.0, 6.0]);
    }

    #[test]
    fn direct_expr_builders() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(add_slices(&a, &b).eval(), vec![5.0, 7.0, 9.0]);
        assert_eq!(scale_slice(&a, 2.0).eval(), vec![2.0, 4.0, 6.0]);
        assert_eq!(
            mul_add_slices(&a, &b, &[1.0, 1.0, 1.0]).eval(),
            vec![5.0, 11.0, 19.0]
        );
    }
}
