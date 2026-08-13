//! Shared input-validation helpers used across the machine-learning crate.
//!
//! Every public fitting/prediction entry point validates its inputs up front
//! and returns a descriptive [`MathError`] instead of panicking or silently
//! producing garbage.

use mathverse_core::error::{MathError, MathResult};

/// Validates a feature matrix and target vector, returning the number of
/// features on success.
///
/// Checks:
/// - the matrix is non-empty and has at least one feature;
/// - every row has the same number of features;
/// - the target has the same length as the matrix.
pub(crate) fn validate_xy(x: &[Vec<f64>], y: &[f64]) -> MathResult<usize> {
    if x.is_empty() {
        return Err(MathError::InvalidArgument("feature matrix is empty"));
    }
    let p = x[0].len();
    if p == 0 {
        return Err(MathError::InvalidArgument(
            "feature matrix has zero features",
        ));
    }
    if x.len() != y.len() {
        return Err(MathError::DimensionMismatch);
    }
    for (i, row) in x.iter().enumerate() {
        if row.len() != p {
            return Err(MathError::InvalidArgument(
                "feature matrix rows have inconsistent lengths",
            ));
        }
        for &v in row {
            if v.is_nan() || v.is_infinite() {
                return Err(MathError::InvalidArgument(
                    "feature matrix contains NaN or infinity",
                ));
            }
        }
        debug_assert!(i < y.len(), "lengths already validated");
    }
    for &v in y {
        if v.is_nan() || v.is_infinite() {
            return Err(MathError::InvalidArgument(
                "target vector contains NaN or infinity",
            ));
        }
    }
    Ok(p)
}

/// Validates a feature matrix for prediction/inference (no target).
pub(crate) fn validate_x(x: &[Vec<f64>], n_features: usize) -> MathResult<()> {
    if x.is_empty() {
        return Err(MathError::InvalidArgument("feature matrix is empty"));
    }
    for row in x {
        if row.len() != n_features {
            return Err(MathError::DimensionMismatch);
        }
        for &v in row {
            if v.is_nan() || v.is_infinite() {
                return Err(MathError::InvalidArgument(
                    "feature matrix contains NaN or infinity",
                ));
            }
        }
    }
    Ok(())
}
