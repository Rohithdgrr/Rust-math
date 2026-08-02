//! # mathverse-algebra
//!
//! A pure-Rust library for polynomial algebra: evaluation, calculus,
//! factorization, equation solving, and algebraic identities.
//!
//! ## Quick Start
//!
//! ```rust
//! use mathverse_algebra::Polynomial;
//!
//! // Create a polynomial: x^2 - 5x + 6
//! let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]);
//!
//! // Evaluate at x = 3
//! assert_eq!(p.eval(3.0), 0.0);
//!
//! // Find roots
//! let roots = p.roots();
//! assert_eq!(roots.len(), 2);
//! ```
//!
//! ## Modules
//!
//! | Module | Description |
//! |---|---|
//! | [`polynomial`] | `Polynomial` type with arithmetic, evaluation, calculus |
//! | [`roots`] | Linear, quadratic, cubic, quartic equation solvers |
//! | [`factor`] | Synthetic division, polynomial GCD, rational-root theorem |
//! | [`determinant`] | 2×2/3×3 determinants, inverses, Cramer's rule, rank, trace |
//! | [`systems`] | 2×2/3×3 linear system solvers |
//! | [`rational`] | Rational expressions, partial-fraction decomposition |
//! | [`identities`] | Binomial theorem, Pascal's triangle, sum/difference of cubes |
//! | [`sequences`] | Arithmetic/geometric sequences and series |
//! | [`interpolate`] | Lagrange and Newton polynomial interpolation |
//! | [`symmetric`] | Elementary symmetric polynomials, Newton's identities |
//! | [`compose`] | Polynomial composition `f(g(x))` |
//! | [`exponents`] | Exponent, logarithm, and radical identity verifiers |
//! | [`solvability`] | Solvability-by-radicals classification (Galois flavored) |
//! | [`latex`] | LaTeX rendering of polynomials and solutions |
//!
//! ## Error Handling
//!
//! All fallible operations return [`Result<T>`] with [`AlgebraError`]:
//!
//! - [`AlgebraError::DivisionByZero`] — division by zero polynomial
//! - [`AlgebraError::Singular`] — singular matrix or system
//! - [`AlgebraError::NoRealRoots`] — no real roots exist
//! - [`AlgebraError::UnsupportedDegree`] — polynomial degree not supported
//! - [`AlgebraError::DimensionMismatch`] — input length mismatch
//!
//! ## Conventions
//!
//! - Polynomials use **lowest-degree-first** coefficient order: `coeffs[i]` is the coefficient of `x^i`
//! - A global tolerance of `TOL = 1e-12` is used for float comparisons
//! - All functions are `#[must_use]` — unused return values produce a warning
//! - `#![forbid(unsafe_code)]` — no unsafe code in the entire crate

#![forbid(unsafe_code)]

use core::fmt;

/// Tolerance for treating a float as zero.
pub const TOL: f64 = 1e-12;

/// Error type for algebraic operations.
///
/// All fallible operations in this crate return `Result<T, AlgebraError>`.
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
pub mod latex;
pub mod polynomial;
pub mod rational;
pub mod roots;
pub mod sequences;
pub mod solvability;
pub mod symmetric;
pub mod systems;

pub use latex::{equation_solution_latex, factors_latex, polynomial_latex, roots_latex};
pub use polynomial::Polynomial;
pub use roots::{solve_cubic, solve_linear, solve_quadratic, solve_quartic};
pub use solvability::{
    degree, divide_by_linear, eisenstein_irreducible, integer_root, is_antipalindromic,
    is_binomial, is_cyclotomic, is_palindromic, reduce_reciprocal, solvable_by_radicals,
    Solvability,
};
