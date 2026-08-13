//! # mathverse-optimization
//!
//! Optimization algorithms for the MathVerse ecosystem.
//!
//! Provides:
//! - **Gradient methods**: gradient descent, momentum, Adam, Nesterov
//! - **Constrained**: penalty method, augmented Lagrangian
//! - **Unconstrained**: Newton's method, BFGS
//! - **Convex**: convex function operations, Jensen's inequality
//! - **Linear programming**: simplex method
//! - **Combinatorial**: greedy, branch-and-bound
//! - **Line search**: backtracking, Wolfe conditions

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod gradient;
pub mod constrained;
pub mod unconstrained;
pub mod convex;
pub mod linear_programming;
pub mod combinatorial;
pub mod line_search;

pub use gradient::*;
pub use constrained::*;
pub use unconstrained::*;
pub use convex::*;
pub use linear_programming::*;
pub use combinatorial::*;
pub use line_search::*;
