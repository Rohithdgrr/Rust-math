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

    /// XML generation / serialization failure.
    #[error("xml generation failed: {0}")]
    Xml(String),

    /// The requested output format is not supported by this backend.
    #[error("unsupported output format: {0}")]
    UnsupportedFormat(String),

    /// Font metrics / text layout failure.
    #[error("font error: {0}")]
    Font(String),

    /// Backend-specific rendering failure.
    #[error("backend error: {0}")]
    Backend(String),

    /// Downsampling failed (e.g. degenerate series).
    #[error("downsampling failed: {0}")]
    Downsample(String),
}

/// Convenient result alias for fallible plot operations.
pub type PlotResult<T> = Result<T, PlotError>;
