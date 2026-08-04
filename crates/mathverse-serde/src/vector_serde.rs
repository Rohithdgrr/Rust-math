//! JSON serialization for `Vector`.

use alloc::string::String;
use alloc::vec::Vec;
use mathverse_core::error::{MathError, MathResult};
use mathverse_vector::Vector;
use serde::{Deserialize, Serialize};

/// Serializable vector representation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorJson {
    /// The vector data.
    pub data: Vec<f64>,
}

impl VectorJson {
    /// Serialize a `Vector` to a JSON string.
    pub fn to_json(v: &Vector) -> MathResult<String> {
        let repr = Self {
            data: v.data.clone(),
        };
        serde_json::to_string(&repr).map_err(|_| MathError::Io)
    }

    /// Deserialize a `Vector` from a JSON string.
    pub fn from_json(s: &str) -> MathResult<Vector> {
        let repr: Self = serde_json::from_str(s).map_err(|_| MathError::Parse)?;
        Ok(Vector::new(repr.data))
    }

    /// Serialize to JSON bytes.
    pub fn to_json_bytes(v: &Vector) -> MathResult<Vec<u8>> {
        let repr = Self {
            data: v.data.clone(),
        };
        serde_json::to_vec(&repr).map_err(|_| MathError::Io)
    }

    /// Deserialize from JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> MathResult<Vector> {
        let repr: Self = serde_json::from_slice(bytes).map_err(|_| MathError::Parse)?;
        Ok(Vector::new(repr.data))
    }

    /// Serialize a slice of f64 to JSON.
    pub fn slice_to_json(data: &[f64]) -> MathResult<String> {
        let repr = Self {
            data: data.to_vec(),
        };
        serde_json::to_string(&repr).map_err(|_| MathError::Io)
    }

    /// Deserialize a slice of f64 from JSON.
    pub fn slice_from_json(s: &str) -> MathResult<Vec<f64>> {
        let repr: Self = serde_json::from_str(s).map_err(|_| MathError::Parse)?;
        Ok(repr.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_json() {
        let v = Vector::new(vec![1.0, 2.0, 3.0, 4.0]);
        let json = VectorJson::to_json(&v).unwrap();
        let v2 = VectorJson::from_json(&json).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn empty_vector() {
        let v = Vector::new(vec![]);
        let json = VectorJson::to_json(&v).unwrap();
        let v2 = VectorJson::from_json(&json).unwrap();
        assert!(v2.is_empty());
    }

    #[test]
    fn invalid_json() {
        assert!(VectorJson::from_json("bad").is_err());
    }

    #[test]
    fn slice_roundtrip() {
        let data = vec![1.5, 2.5, 3.5];
        let json = VectorJson::slice_to_json(&data).unwrap();
        let back = VectorJson::slice_from_json(&json).unwrap();
        assert_eq!(data, back);
    }
}
