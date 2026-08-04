//! Lazy evaluation and expression templates for the MathVerse ecosystem.
//!
//! This crate provides deferred computation primitives that build expression
//! trees instead of eagerly computing results. This enables:
//!
//! - **Fused operations**: `a + b * c` evaluates in a single pass with one allocation
//! - **Lazy evaluation**: expressions are only computed when `.eval()` is called
//! - **Expression templates**: compile-time-like optimization at runtime
//!
//! # Examples
//!
//! ```
//! use mathverse_lazy::{LazyVec, lazy_add, lazy_mul, lazy_scale};
//!
//! let a = LazyVec::new(vec![1.0, 2.0, 3.0]);
//! let b = LazyVec::new(vec![4.0, 5.0, 6.0]);
//!
//! // Build expression without allocating
//! let expr = lazy_add(&a, &b);
//! // Evaluate only when needed
//! let result = expr.eval();
//! assert_eq!(result, vec![5.0, 7.0, 9.0]);
//! ```

extern crate alloc;

pub mod expr;
pub mod fused;
pub mod lazy_vec;
pub mod ops;

pub use expr::{Expr, ExprRef};
pub use fused::{FusedAdd, FusedMul, FusedScale};
pub use lazy_vec::LazyVec;
pub use ops::{lazy_add, lazy_mul, lazy_scale, lazy_sub};
