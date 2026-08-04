//! JSON serialization for `Matrix`.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use mathverse_core::error::{MathError, MathResult};
use mathverse_matrix::Matrix;
use serde::{Deserialize, Serialize};

/// Serializable matrix representation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MatrixJson {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major data.
    pub data: Vec<f64>,
}

impl MatrixJson {
    /// Serialize a `Matrix` to a JSON string.
    pub fn to_json(m: &Matrix) -> MathResult<String> {
        let repr = Self {
            rows: m.rows,
            cols: m.cols,
            data: m.data.clone(),
        };
        serde_json::to_string(&repr).map_err(|_| MathError::Io)
    }

    /// Deserialize a `Matrix` from a JSON string.
    pub fn from_json(s: &str) -> MathResult<Matrix> {
        let repr: Self = serde_json::from_str(s).map_err(|_| MathError::Parse)?;
        if repr.rows * repr.cols != repr.data.len() {
            return Err(MathError::DimensionMismatch);
        }
        Ok(Matrix {
            rows: repr.rows,
            cols: repr.cols,
            data: repr.data,
        })
    }

    /// Serialize to a JSON byte vector.
    pub fn to_json_bytes(m: &Matrix) -> MathResult<Vec<u8>> {
        let repr = Self {
            rows: m.rows,
            cols: m.cols,
            data: m.data.clone(),
        };
        serde_json::to_vec(&repr).map_err(|_| MathError::Io)
    }

    /// Deserialize from JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> MathResult<Matrix> {
        let repr: Self = serde_json::from_slice(bytes).map_err(|_| MathError::Parse)?;
        if repr.rows * repr.cols != repr.data.len() {
            return Err(MathError::DimensionMismatch);
        }
        Ok(Matrix {
            rows: repr.rows,
            cols: repr.cols,
            data: repr.data,
        })
    }

    /// Pretty-print JSON.
    pub fn to_json_pretty(m: &Matrix) -> MathResult<String> {
        let repr = Self {
            rows: m.rows,
            cols: m.cols,
            data: m.data.clone(),
        };
        serde_json::to_string_pretty(&repr).map_err(|_| MathError::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_json() {
        let m = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();
        let json = MatrixJson::to_json(&m).unwrap();
        let m2 = MatrixJson::from_json(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn pretty_json() {
        let m = Matrix::identity(2);
        let json = MatrixJson::to_json_pretty(&m).unwrap();
        assert!(json.contains("rows"));
        assert!(json.contains("data"));
    }

    #[test]
    fn invalid_json() {
        assert!(MatrixJson::from_json("not json").is_err());
    }

    #[test]
    fn dimension_mismatch() {
        let json = r#"{"rows":2,"cols":2,"data":[1.0,2.0,3.0]}"#;
        assert!(MatrixJson::from_json(json).is_err());
    }
}
