//! Symbolic computation with expression trees, derivatives, and LaTeX
//! 
//! This crate provides:
//! - Expression trees for symbolic manipulation
//! - Symbolic differentiation
//! - LaTeX rendering
//! - Expression simplification

pub mod expr;
pub mod derivative;
pub mod latex;
pub mod simplify;

pub use expr::*;
pub use derivative::*;
pub use latex::*;
pub use simplify::*;
