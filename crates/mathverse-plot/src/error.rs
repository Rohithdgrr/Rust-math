//! Plot error taxonomy.

use thiserror::Error;

/// Errors produced by plot construction and rendering.
#[derive(Error, Debug)]
pub enum PlotError {
    /// Degenerate or malformed input data.
    #[error("invalid data: {0}")]
    InvalidData(String),

    /// Errors surfaced from the mathverse ecosystem.
    #[error("math error: {0}")]
    Math(#[from] mathverse_core::error::MathError),

    /// IO failures (writing output files, etc.).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenient result alias for fallible plot operations.
pub type PlotResult<T> = Result<T, PlotError>;
