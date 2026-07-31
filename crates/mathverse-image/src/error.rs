//! Error types for image processing operations.

use thiserror::Error;

/// Errors that can occur during image processing.
#[derive(Error, Debug)]
pub enum ImageError {
    /// Invalid dimensions provided.
    #[error("invalid dimensions: width {width} and height {height} must be greater than 0")]
    InvalidDimensions { width: usize, height: usize },

    /// Data length mismatch for given dimensions.
    #[error("data length {data_len} does not match expected length {expected_len} for {width}x{height} image")]
    DataLengthMismatch {
        data_len: usize,
        expected_len: usize,
        width: usize,
        height: usize,
    },

    /// Pixel coordinates out of bounds.
    #[error("pixel coordinates ({x}, {y}) out of bounds for {width}x{height} image")]
    OutOfBounds { x: usize, y: usize, width: usize, height: usize },

    /// Invalid pixel value (must be in [0, 1]).
    #[error("pixel value {value} is invalid, must be in [0, 1]")]
    InvalidPixelValue { value: f64 },

    /// I/O error when loading or saving images.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Image encoding/decoding error.
    #[error("image encoding/decoding error: {0}")]
    ImageError(#[from] image::ImageError),
}

/// Result type alias for image operations.
pub type Result<T> = std::result::Result<T, ImageError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ImageError::InvalidDimensions { width: 0, height: 10 };
        assert!(err.to_string().contains("invalid dimensions"));
    }

    #[test]
    fn test_data_length_mismatch() {
        let err = ImageError::DataLengthMismatch {
            data_len: 100,
            expected_len: 64,
            width: 8,
            height: 8,
        };
        assert!(err.to_string().contains("does not match expected length"));
    }
}
