//! Checkpoint save/load for model state and computation state.

use alloc::string::String;
use alloc::vec::Vec;
use mathverse_core::error::{MathError, MathResult};
use mathverse_matrix::Matrix;
use mathverse_vector::Vector;
use serde::{Deserialize, Serialize};

/// A checkpoint containing named matrices and vectors.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Checkpoint {
    /// Named matrices.
    pub matrices: Vec<(String, MatrixRepr)>,
    /// Named vectors.
    pub vectors: Vec<(String, VectorRepr)>,
    /// Arbitrary metadata.
    pub metadata: Vec<(String, String)>,
}

/// Serializable matrix representation inside checkpoints.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MatrixRepr {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

/// Serializable vector representation inside checkpoints.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorRepr {
    pub data: Vec<f64>,
}

impl Checkpoint {
    /// Create an empty checkpoint.
    pub fn new() -> Self {
        Self {
            matrices: Vec::new(),
            vectors: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Add a named matrix.
    pub fn add_matrix(&mut self, name: impl Into<String>, m: &Matrix) {
        self.matrices.push((
            name.into(),
            MatrixRepr {
                rows: m.rows,
                cols: m.cols,
                data: m.data.clone(),
            },
        ));
    }

    /// Add a named vector.
    pub fn add_vector(&mut self, name: impl Into<String>, v: &Vector) {
        self.vectors.push((
            name.into(),
            VectorRepr {
                data: v.data.clone(),
            },
        ));
    }

    /// Add metadata.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.push((key.into(), value.into()));
    }

    /// Get a matrix by name.
    pub fn get_matrix(&self, name: &str) -> MathResult<Matrix> {
        self.matrices
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| Matrix {
                rows: r.rows,
                cols: r.cols,
                data: r.data.clone(),
            })
            .ok_or(MathError::InvalidArgument("matrix not found in checkpoint"))
    }

    /// Get a vector by name.
    pub fn get_vector(&self, name: &str) -> MathResult<Vector> {
        self.vectors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| Vector::new(r.data.clone()))
            .ok_or(MathError::InvalidArgument("vector not found in checkpoint"))
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> MathResult<String> {
        serde_json::to_string(self).map_err(|_| MathError::Io)
    }

    /// Deserialize from JSON.
    pub fn from_json(s: &str) -> MathResult<Self> {
        serde_json::from_str(s).map_err(|_| MathError::Parse)
    }

    /// Serialize to bincode.
    #[cfg(feature = "bincode")]
    pub fn to_bincode(&self) -> MathResult<Vec<u8>> {
        bincode::serialize(self).map_err(|_| MathError::Io)
    }

    /// Deserialize from bincode.
    #[cfg(feature = "bincode")]
    pub fn from_bincode(bytes: &[u8]) -> MathResult<Self> {
        bincode::deserialize(bytes).map_err(|_| MathError::Parse)
    }
}

impl Default for Checkpoint {
    fn default() -> Self {
        Self::new()
    }
}

/// Save a checkpoint to a JSON string.
pub fn save_checkpoint(ckpt: &Checkpoint) -> MathResult<String> {
    ckpt.to_json()
}

/// Load a checkpoint from a JSON string.
pub fn load_checkpoint(json: &str) -> MathResult<Checkpoint> {
    Checkpoint::from_json(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_roundtrip() {
        let mut ckpt = Checkpoint::new();
        ckpt.add_matrix("weights", &Matrix::identity(3));
        ckpt.add_vector("bias", &Vector::new(vec![1.0, 2.0, 3.0]));
        ckpt.add_metadata("epoch", "42");

        let json = ckpt.to_json().unwrap();
        let loaded = Checkpoint::from_json(&json).unwrap();

        assert_eq!(loaded.get_matrix("weights").unwrap(), Matrix::identity(3));
        assert_eq!(
            loaded.get_vector("bias").unwrap(),
            Vector::new(vec![1.0, 2.0, 3.0])
        );
        assert_eq!(
            loaded.metadata.iter().find(|(k, _)| k == "epoch").unwrap().1,
            "42"
        );
    }

    #[test]
    fn missing_key() {
        let ckpt = Checkpoint::new();
        assert!(ckpt.get_matrix("missing").is_err());
        assert!(ckpt.get_vector("missing").is_err());
    }
}
