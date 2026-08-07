#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

//! # mathverse-vector
//!
//! Comprehensive vector operations for the MathVerse ecosystem.
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
//!   (via the `wide` crate; SSE2 on x86-64, NEON on AArch64).
//! - `parallel`: accelerates the same reductions with `rayon` parallel
//!   iterators when inputs are large enough to amortize thread-pool overhead.
//!
//! Both features are opt-in and additive; the default build is fully scalar.

pub mod vector;
pub mod operations;
pub mod norms;
pub mod geometry;
pub mod linear_algebra;
pub mod statistics;
pub mod distance;
pub mod utils;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "parallel")]
pub mod parallel;

pub use vector::Vector;
pub use operations::*;
pub use norms::*;
pub use geometry::*;
pub use linear_algebra::*;
pub use statistics::*;
pub use distance::*;
pub use utils::*;
