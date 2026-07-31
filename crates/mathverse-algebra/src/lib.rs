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
