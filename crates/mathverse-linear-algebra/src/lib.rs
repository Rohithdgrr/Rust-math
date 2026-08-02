//! # mathverse-linear-algebra
//!
//! Linear algebra operations for the MathVerse ecosystem.
//!
//! Provides:
//! - **Decompositions**: LU, QR, Cholesky, eigenvalue (2×2 and power iteration)
//! - **Solvers**: QR-based, Gaussian elimination, least-squares
//! - **Norms**: L1, L∞, Frobenius, L2 (spectral), condition number
//! - **Matrix properties**: rank, inverse, singular values
//!
//! All matrices are represented as `Vec<Vec<f64>>` in row-major order.

pub mod decomposition;
pub mod solve;
pub mod norm;
pub mod eigen;
pub mod inverse;
pub mod rank;
pub mod least_squares;

// Selective re-exports to avoid ambiguity
pub use decomposition::{lu_decompose, qr_decompose, cholesky, eigenvalue_2x2, power_iteration, Complex};
pub use solve::{solve_qr, solve_2x2, solve_3x3, solve_gauss, ls_solve, residual_norm};
pub use norm::{norm_1, norm_inf, norm_frobenius, norm_2, singular_values, condition_number, matrix_norm};
