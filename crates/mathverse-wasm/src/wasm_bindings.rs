//! WASM-bindgen bindings for JavaScript interop.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use alloc::vec::Vec;

/// WASM-compatible matrix wrapper.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmMatrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmMatrix {
    /// Create a new matrix from row-major data.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        Self { data, rows, cols }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self {
            data,
            rows: n,
            cols: n,
        }
    }

    /// Create a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Get element at (row, col).
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Set element at (row, col).
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Get number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Get number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Get raw data as a flat vector.
    pub fn to_flat(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Matrix multiplication.
    pub fn multiply(&self, other: &WasmMatrix) -> WasmMatrix {
        assert_eq!(self.cols, other.rows, "dimension mismatch");
        let mut out = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                for j in 0..other.cols {
                    out[i * other.cols + j] += a * other.get(k, j);
                }
            }
        }
        WasmMatrix {
            data: out,
            rows: self.rows,
            cols: other.cols,
        }
    }

    /// Element-wise addition.
    pub fn add(&self, other: &WasmMatrix) -> WasmMatrix {
        assert!(self.rows == other.rows && self.cols == other.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        WasmMatrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> WasmMatrix {
        let data: Vec<f64> = self.data.iter().map(|v| v * s).collect();
        WasmMatrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Transpose.
    pub fn transpose(&self) -> WasmMatrix {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.get(i, j);
            }
        }
        WasmMatrix {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Trace (sum of diagonal).
    pub fn trace(&self) -> f64 {
        assert_eq!(self.rows, self.cols, "must be square");
        (0..self.rows).map(|i| self.get(i, i)).sum()
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

/// WASM-compatible vector wrapper.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmVector {
    data: Vec<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmVector {
    /// Create a new vector.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    /// Create a zero vector.
    pub fn zeros(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    /// Get length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get element at index.
    pub fn get(&self, i: usize) -> f64 {
        self.data[i]
    }

    /// Set element at index.
    pub fn set(&mut self, i: usize, v: f64) {
        self.data[i] = v;
    }

    /// Get raw data.
    pub fn to_flat(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Dot product.
    pub fn dot(&self, other: &WasmVector) -> f64 {
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Element-wise addition.
    pub fn add(&self, other: &WasmVector) -> WasmVector {
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        WasmVector { data }
    }

    /// Scalar multiply.
    pub fn scale(&self, s: f64) -> WasmVector {
        let data: Vec<f64> = self.data.iter().map(|v| v * s).collect();
        WasmVector { data }
    }

    /// L2 norm.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Mean.
    pub fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    /// Normalize to unit vector.
    pub fn normalized(&self) -> WasmVector {
        let n = self.norm();
        let data: Vec<f64> = self.data.iter().map(|v| v / n).collect();
        WasmVector { data }
    }

    /// Sum of elements.
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_matrix_basic() {
        let m = WasmMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 4.0);
    }

    #[test]
    fn wasm_matrix_multiply() {
        let a = WasmMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = WasmMatrix::identity(2);
        let c = a.multiply(&b);
        assert_eq!(c.get(0, 0), 1.0);
        assert_eq!(c.get(1, 1), 4.0);
    }

    #[test]
    fn wasm_vector_basic() {
        let v = WasmVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(1), 2.0);
        assert!((v.norm() - (14.0_f64).sqrt()).abs() < 1e-12);
    }
}
