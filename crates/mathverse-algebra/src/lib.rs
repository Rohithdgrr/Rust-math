//! Polynomials over `f64`, with evaluation, calculus, factorization,
//! equation solving, and algebraic identities.
//!
//! Module layout:
//! - [`polynomial`]: `Polynomial` type and arithmetic
//! - [`roots`]: linear/quadratic/cubic/quartic solvers, discriminants, Vieta
//! - [`factor`]: synthetic division, polynomial GCD, rational-root candidates
//! - [`identities`]: binomial theorem, Pascal's triangle, sum/difference of cubes
//! - [`rational`]: rational expressions, partial fractions
//! - [`sequences`]: arithmetic/geometric sequences and sums
//! - [`interpolate`]: Lagrange and Newton interpolation
//! - [`symmetric`]: elementary symmetric polynomials, Newton's identities
//! - [`compose`]: polynomial composition
//! - [`determinant`]: 2×2/3×3 determinants, inverses, Cramer's rule
//! - [`exponents`]: exponent/log/radical identity helpers
//! - [`systems`]: 2×2/3×3 linear systems

#![forbid(unsafe_code)]

use core::fmt;

/// Tolerance for treating a float as zero.
pub const TOL: f64 = 1e-12;

/// Error type for algebraic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgebraError {
    /// Division by a zero polynomial or zero denominator.
    DivisionByZero,
    /// The input matrix or system is singular (determinant ≈ 0).
    Singular,
    /// No real roots exist for the given polynomial.
    NoRealRoots,
    /// The polynomial degree exceeds what the solver supports.
    UnsupportedDegree(u8),
    /// Mismatched input lengths.
    DimensionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for AlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DivisionByZero => write!(f, "division by zero polynomial"),
            Self::Singular => write!(f, "matrix is singular"),
            Self::NoRealRoots => write!(f, "no real roots"),
            Self::UnsupportedDegree(d) => write!(f, "unsupported polynomial degree: {d}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "expected {expected} elements, got {actual}")
            }
        }
    }
}

impl std::error::Error for AlgebraError {}

/// Result alias for algebraic operations.
pub type Result<T> = core::result::Result<T, AlgebraError>;

pub mod compose;
pub mod determinant;
pub mod exponents;
pub mod factor;
pub mod identities;
pub mod interpolate;
pub mod polynomial;
pub mod rational;
pub mod roots;
pub mod sequences;
pub mod symmetric;
pub mod systems;

pub use polynomial::Polynomial;
pub use roots::{solve_cubic, solve_linear, solve_quadratic, solve_quartic};