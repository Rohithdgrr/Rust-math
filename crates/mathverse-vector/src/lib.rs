#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

//! # mathverse-vector
//!
//! Comprehensive vector operations for the `MathVerse` ecosystem.
//!
//! Provides a dense `f64` [`Vector`] type with support for:
//! - **Element-wise operations**: addition, subtraction, scalar multiply/divide
//! - **Norms**: L1, L2 (Euclidean), L∞, and p-norms
//! - **Geometry**: dot product, cross product, angle between vectors, projection
//! - **Linear algebra**: matrix-vector products, linear independence, basis
//! - **Statistics**: mean, variance, standard deviation, covariance, correlation
//! - **Distance metrics**: Euclidean, Manhattan, Chebyshev, cosine distance
//! - **Utilities**: normalization, clamping, interpolation, concatenation
//!
//! # Optional acceleration features
//!
//! - `simd`: accelerates the O(n) reductions with safe 128-bit SIMD lanes
//!   (via the `wide` crate; SSE2 on x86-64, NEON on `AArch64`).
//! - `parallel`: accelerates the same reductions with `rayon` parallel
//!   iterators when inputs are large enough to amortize thread-pool overhead.
//!
//! Both features are opt-in and additive; the default build is fully scalar.

/// Distance metrics between vectors.
pub mod distance;
/// Geometric operations: dot, cross, angle, projection.
pub mod geometry;
/// Matrix-vector products and linear independence helpers.
pub mod linear_algebra;
/// L1/L2/L∞/p norms.
pub mod norms;
/// Element-wise arithmetic operations.
pub mod operations;
/// Statistical reductions over vectors.
pub mod statistics;
/// Utility functions: normalization, clamping, interpolation.
pub mod utils;
/// Core [`Vector`] type and constructors.
pub mod vector;

/// Optional SIMD acceleration (feature `simd`).
#[cfg(feature = "simd")]
pub mod simd;

/// Optional rayon-based parallel reductions (feature `parallel`).
#[cfg(feature = "parallel")]
pub mod parallel;

pub use distance::*;
pub use geometry::*;
pub use linear_algebra::*;
pub use norms::*;
pub use operations::*;
pub use statistics::*;
pub use utils::*;
pub use vector::Vector;
