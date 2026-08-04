//! Safetensors format support for efficient tensor storage.
//!
//! The safetensors format (used by Hugging Face) stores tensors in a
//! memory-mapped-friendly format with metadata headers. This module
//! provides conversion between MathVerse types and safetensors tensors.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use mathverse_core::error::{MathError, MathResult};
use mathverse_matrix::Matrix;
use mathverse_vector::Vector;

/// Metadata header for a safetensors file.
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    tensors: BTreeMap<String, TensorInfo>,
}

/// Information about a single tensor.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorInfo {
    pub dtype: DType,
    pub offsets: (usize, usize),
    pub shape: Vec<usize>,
}

/// Supported data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    fn byte_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

impl SafetensorsHeader {
    /// Create a new header.
    pub fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
        }
    }

    /// Add a matrix entry.
    pub fn add_matrix(&mut self, name: &str, m: &Matrix, dtype: DType) {
        let numel = m.rows * m.cols;
        let byte_size = dtype.byte_size();
        self.tensors.insert(
            name.to_string(),
            TensorInfo {
                dtype,
                offsets: (0, numel * byte_size),
                shape: vec![m.rows, m.cols],
            },
        );
    }

    /// Add a vector entry.
    pub fn add_vector(&mut self, name: &str, v: &Vector, dtype: DType) {
        let numel = v.len();
        let byte_size = dtype.byte_size();
        self.tensors.insert(
            name.to_string(),
            TensorInfo {
                dtype,
                offsets: (0, numel * byte_size),
                shape: vec![numel],
            },
        );
    }

    /// Serialize the header to JSON bytes.
    pub fn to_json_bytes(&self) -> MathResult<Vec<u8>> {
        let map: BTreeMap<String, &TensorInfo> = self
            .tensors
            .iter()
            .map(|(k, v)| (k.clone(), v))
            .collect();
        serde_json::to_vec(&map).map_err(|_| MathError::Io)
    }

    /// Get the list of tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor info by name.
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the header is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl Default for SafetensorsHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Serialize a matrix to safetensors-compatible bytes (header + data).
pub fn serialize_matrix(m: &Matrix, name: &str) -> MathResult<Vec<u8>> {
    let mut header = SafetensorsHeader::new();
    header.add_matrix(name, m, DType::F64);

    let header_json = header.to_json_bytes()?;
    let header_len = header_json.len();

    // safetensors format: 8-byte little-endian header length + header + data
    let mut output = Vec::with_capacity(8 + header_len + m.data.len() * 8);
    output.extend_from_slice(&(header_len as u64).to_le_bytes());
    output.extend_from_slice(&header_json);

    // Write f64 data as little-endian bytes
    for &val in &m.data {
        output.extend_from_slice(&val.to_le_bytes());
    }

    Ok(output)
}

/// Serialize a vector to safetensors-compatible bytes.
pub fn serialize_vector(v: &Vector, name: &str) -> MathResult<Vec<u8>> {
    let mut header = SafetensorsHeader::new();
    header.add_vector(name, v, DType::F64);

    let header_json = header.to_json_bytes()?;
    let header_len = header_json.len();

    let mut output = Vec::with_capacity(8 + header_len + v.len() * 8);
    output.extend_from_slice(&(header_len as u64).to_le_bytes());
    output.extend_from_slice(&header_json);

    for &val in &v.data {
        output.extend_from_slice(&val.to_le_bytes());
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let mut header = SafetensorsHeader::new();
        header.add_matrix("w", &Matrix::identity(2), DType::F64);
        header.add_vector("b", &Vector::new(vec![1.0, 2.0]), DType::F64);

        assert_eq!(header.len(), 2);
        assert!(header.get("w").is_some());
        assert!(header.get("b").is_some());
        assert!(!header.is_empty());
    }

    #[test]
    fn serialize_matrix_safetensors() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let bytes = serialize_matrix(&m, "test").unwrap();
        // Should contain header length + header + 4 * 8 bytes of data
        assert!(bytes.len() > 8 + 4 * 8);
    }

    #[test]
    fn serialize_vector_safetensors() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        let bytes = serialize_vector(&v, "test").unwrap();
        assert!(bytes.len() > 8 + 3 * 8);
    }

    #[test]
    fn tensor_names() {
        let mut header = SafetensorsHeader::new();
        header.add_matrix("alpha", &Matrix::zeros(1, 1), DType::F64);
        header.add_matrix("beta", &Matrix::zeros(1, 1), DType::F64);
        let names = header.tensor_names();
        assert_eq!(names, vec!["alpha", "beta"]);
    }
}
