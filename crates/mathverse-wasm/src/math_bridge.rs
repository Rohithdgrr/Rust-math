//! Bridge between WASM and core MathVerse types.

use alloc::vec::Vec;
use mathverse_core::error::{MathError, MathResult};
use mathverse_matrix::Matrix;
use mathverse_vector::Vector;

/// Convert a `Matrix` to flat data suitable for WASM.
pub fn matrix_to_wasm(m: &Matrix) -> (Vec<f64>, usize, usize) {
    (m.data.clone(), m.rows, m.cols)
}

/// Convert WASM data back to a `Matrix`.
pub fn wasm_to_matrix(data: Vec<f64>, rows: usize, cols: usize) -> MathResult<Matrix> {
    if data.len() != rows * cols {
        return Err(MathError::DimensionMismatch);
    }
    Ok(Matrix { rows, cols, data })
}

/// Convert a `Vector` to flat data suitable for WASM.
pub fn vector_to_wasm(v: &Vector) -> Vec<f64> {
    v.data.clone()
}

/// Convert WASM data back to a `Vector`.
pub fn wasm_to_vector(data: Vec<f64>) -> Vector {
    Vector::new(data)
}

/// FFI-safe matrix metadata.
#[repr(C)]
pub struct MatrixMeta {
    pub rows: u32,
    pub cols: u32,
    pub data_ptr: *const f64,
}

/// FFI-safe vector metadata.
#[repr(C)]
pub struct VectorMeta {
    pub len: u32,
    pub data_ptr: *const f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_matrix() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let (data, rows, cols) = matrix_to_wasm(&m);
        let m2 = wasm_to_matrix(data, rows, cols).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn roundtrip_vector() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        let data = vector_to_wasm(&v);
        let v2 = wasm_to_vector(data);
        assert_eq!(v, v2);
    }

    #[test]
    fn dimension_mismatch() {
        let result = wasm_to_matrix(vec![1.0, 2.0], 2, 2);
        assert!(result.is_err());
    }
}
