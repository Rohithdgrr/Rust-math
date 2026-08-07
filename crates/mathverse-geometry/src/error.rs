//! Centralized error types for `mathverse-geometry`.

use thiserror::Error;

/// Errors that can occur in geometric operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum GeometryError {
    /// A negative radius or dimension was provided.
    #[error("invalid dimension: expected non-negative, got {value}")]
    InvalidDimension { value: f64 },

    /// A polygon has too few vertices.
    #[error("polygon requires at least 3 vertices, got {count}")]
    TooFewVertices { count: usize },

    /// A mesh contains zero triangles.
    #[error("mesh contains no triangles")]
    EmptyMesh,

    /// Intersection computation failed.
    #[error("intersection computation failed: {reason}")]
    IntersectionFailed { reason: String },

    /// Distance computation failed (e.g., degenerate segment).
    #[error("distance computation failed: {reason}")]
    DistanceFailed { reason: String },
}

impl GeometryError {
    /// Returns a new invalid-dimension error.
    pub fn invalid_dim(value: f64) -> Self {
        Self::InvalidDimension { value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = GeometryError::TooFewVertices { count: 1 };
        assert!(err.to_string().contains("at least 3"));
    }
}
