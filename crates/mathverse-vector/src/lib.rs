//! # mathverse-vector
//!
//! Comprehensive vector operations for the MathVerse ecosystem.
//!
//! Provides a generic [`Vector`] type with support for:
//! - **Element-wise operations**: addition, subtraction, scalar multiply/divide
//! - **Norms**: L1, L2 (Euclidean), L∞, and p-norms
//! - **Geometry**: dot product, cross product, angle between vectors, projection
//! - **Linear algebra**: matrix-vector products, linear independence, basis
//! - **Statistics**: mean, variance, standard deviation, covariance, correlation
//! - **Distance metrics**: Euclidean, Manhattan, Chebyshev, cosine distance
//! - **Utilities**: normalization, clamping, interpolation, concatenation

pub mod vector;
pub mod operations;
pub mod norms;
pub mod geometry;
pub mod linear_algebra;
pub mod statistics;
pub mod distance;
pub mod utils;

pub use vector::Vector;
pub use operations::*;
pub use norms::*;
pub use geometry::*;
pub use linear_algebra::*;
pub use statistics::*;
pub use distance::*;
pub use utils::*;
