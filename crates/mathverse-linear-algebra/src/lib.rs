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
/// Eigenvalue computation.
pub mod eigen;
/// Matrix inversion.
pub mod inverse;
/// Least-squares fitting.
pub mod least_squares;
/// Vector and matrix norms, condition numbers.
pub mod norm;
/// Matrix rank.
pub mod rank;
/// Linear system solvers and least-squares.
pub mod solve;

// Selective re-exports to avoid ambiguity
pub use decomposition::{
    cholesky, eigenvalue_2x2, lu_decompose, power_iteration, qr_decompose, Complex,
};
pub use inverse::matrix_inverse;
pub use norm::{
    condition_number, matrix_norm, norm_1, norm_2, norm_frobenius, norm_inf, singular_values,
};
pub use rank::matrix_rank;
pub use solve::{ls_solve, residual_norm, solve_2x2, solve_3x3, solve_gauss, solve_qr};
