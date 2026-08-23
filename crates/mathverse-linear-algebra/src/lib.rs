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
//! All matrices use [`mathverse_matrix::Matrix`] with flat row-major storage
//! (`data[row * cols + col]`).

/// Matrix decompositions: LU, QR, Cholesky, eigenvalue algorithms.
pub mod decomposition;
/// Linear system solvers and least-squares.
pub mod solve;
/// Vector and matrix norms, condition numbers.
pub mod norm;
/// Eigenvalue computation.
pub mod eigen;
/// Matrix inversion.
pub mod inverse;
/// Matrix rank.
pub mod rank;
/// Least-squares fitting.
pub mod least_squares;

// Selective re-exports to avoid ambiguity
pub use decomposition::{lu_decompose, qr_decompose, cholesky, eigenvalue_2x2, power_iteration, Complex};
pub use solve::{solve_qr, solve_2x2, solve_3x3, solve_gauss, ls_solve, residual_norm};
pub use norm::{norm_1, norm_inf, norm_frobenius, norm_2, singular_values, condition_number, matrix_norm};
pub use inverse::matrix_inverse;
pub use rank::matrix_rank;
