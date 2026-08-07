//! # mathverse-parallel
//!
//! Rayon-based parallel computation for MathVerse types.
//!
//! Provides parallel versions of common operations on vectors, matrices,
//! and slices using `rayon` parallel iterators for automatic work-stealing.
//!
//! # When to use
//!
//! - Data sizes > ~10,000 elements (thread overhead is negligible)
//! - Embarrassingly parallel operations (element-wise ops, reductions)
//! - Monte Carlo simulations with independent samples
//!
//! # Examples
//!
//! ```
//! use mathverse_parallel::{par_dot_slice, par_sum_slice, par_map_slice};
//!
//! let a = vec![1.0; 100_000];
//! let b = vec![2.0; 100_000];
//! let dot = par_dot_slice(&a, &b);
//! assert!((dot - 200_000.0).abs() < 1e-6);
//! ```

extern crate alloc;

pub mod vector_par;
pub mod matrix_par;
pub mod slice_par;
pub mod monte_carlo;

pub use vector_par::*;
pub use matrix_par::*;
pub use slice_par::*;
pub use monte_carlo::*;
