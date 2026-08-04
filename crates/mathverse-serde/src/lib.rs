//! # mathverse-serde
//!
//! Serde-based serialization for MathVerse types.
//!
//! Save and load matrices, vectors, and other math types to/from JSON,
//! safetensors, and bincode formats.
//!
//! # Features
//!
//! - `json` (default): JSON serialization via `serde_json`
//! - `safetensors`: Hugging Face safetensors format for efficient tensor storage
//! - `bincode`: Compact binary format via `bincode`
//!
//! # Examples
//!
//! ```
//! use mathverse_serde::{MatrixJson, VectorJson};
//! use mathverse_matrix::Matrix;
//! use mathverse_vector::Vector;
//!
//! let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
//! let json = MatrixJson::to_json(&m).unwrap();
//! let m2 = MatrixJson::from_json(&json).unwrap();
//! assert_eq!(m, m2);
//! ```

extern crate alloc;

pub mod matrix_serde;
pub mod vector_serde;
pub mod checkpoint;
pub mod safetensors_io;

pub use matrix_serde::MatrixJson;
pub use vector_serde::VectorJson;
pub use checkpoint::{save_checkpoint, load_checkpoint};
