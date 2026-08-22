//! # mathverse-ndarray-interop
//!
//! Zero-copy conversions between MathVerse and `ndarray` types.
//!
//! This crate bridges the MathVerse ecosystem with the `ndarray` crate,
//! enabling seamless interop between `mathverse_matrix::Matrix` and
//! `ndarray::Array2<f64>`, and between `mathverse_vector::Vector` and
//! `ndarray::Array1<f64>`.
//!
//! # Conversions
//!
//! | MathVerse | ndarray | Direction |
//! |-----------|---------|-----------|
//! | `Vector` | `Array1<f64>` | Bidirectional (copy) |
//! | `Matrix` | `Array2<f64>` | Bidirectional (copy) |
//! | `&[f64]` | `ArrayView1<f64>` | Zero-copy borrow |
//! | `&mut [f64]` | `ArrayViewMut1<f64>` | Zero-copy borrow |
//! | `&Matrix` | `ArrayView2<f64>` | Zero-copy borrow |

use mathverse_matrix::Matrix;
use mathverse_vector::Vector;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, ShapeError};

// ─── Vector ↔ Array1 ──────────────────────────────────────────────

/// Convert a `Vector` into an `Array1<f64>`.
pub fn vector_to_array1(v: Vector) -> Array1<f64> {
    Array1::from_vec(v.data)
}

/// Convert an `Array1<f64>` into a `Vector`.
pub fn array1_to_vector(arr: Array1<f64>) -> Vector {
    Vector::new(arr.to_vec())
}

/// Convert a `&Vector` into an `ArrayView1<f64>`.
pub fn vector_to_view1<'a>(v: &'a Vector) -> ArrayView1<'a, f64> {
    ArrayView1::from_shape(v.len(), &v.data).expect("valid shape")
}

/// Convert a `&mut Vector` into an `ArrayViewMut1<f64>`.
pub fn vector_to_view1_mut<'a>(v: &'a mut Vector) -> ArrayViewMut1<'a, f64> {
    let len = v.len();
    ArrayViewMut1::from_shape(len, &mut v.data).expect("valid shape")
}

/// Convert an `ArrayView1<f64>` into a `Vector`.
pub fn view1_to_vector<'a>(arr: ArrayView1<'a, f64>) -> Vector {
    Vector::new(arr.to_vec())
}

// ─── Matrix ↔ Array2 ──────────────────────────────────────────────

/// Convert a `Matrix` into an `Array2<f64>`.
pub fn matrix_to_array2(m: Matrix) -> Array2<f64> {
    let (rows, cols) = m.shape();
    Array2::from_shape_vec((rows, cols), m.into_data())
        .expect("Matrix shape should be valid")
}

/// Convert an `Array2<f64>` into a `Matrix`.
pub fn array2_to_matrix(arr: Array2<f64>) -> Result<Matrix, ShapeError> {
    let (rows, cols) = arr.dim();
    // A valid `Array2` always has `rows * cols` elements, so this cannot fail.
    Ok(Matrix::new(rows, cols, arr.into_raw_vec())
        .expect("Array2 shape should be valid"))
}

/// Convert a `&Matrix` into an `ArrayView2<f64>`.
pub fn matrix_to_view2<'a>(m: &'a Matrix) -> ArrayView2<'a, f64> {
    ArrayView2::from_shape(m.shape(), m.as_slice()).expect("valid shape")
}

/// Convert a `&mut Matrix` into an `ArrayViewMut2<f64>`.
pub fn matrix_to_view2_mut<'a>(m: &'a mut Matrix) -> ArrayViewMut2<'a, f64> {
    let (rows, cols) = m.shape();
    ArrayViewMut2::from_shape((rows, cols), m.data_mut()).expect("valid shape")
}

/// Convert an `ArrayView2<f64>` into a `Matrix`.
pub fn view2_to_matrix<'a>(arr: ArrayView2<'a, f64>) -> Matrix {
    let (rows, cols) = arr.dim();
    Matrix::new(rows, cols, arr.to_owned().into_raw_vec())
        .expect("ArrayView2 shape should be valid")
}

// ─── Slice ↔ ArrayView ────────────────────────────────────────────

/// Convert a slice to an ndarray view.
pub fn slice_to_view<'a>(data: &'a [f64]) -> ArrayView1<'a, f64> {
    ArrayView1::from(data)
}

/// Convert a 2D slice to an ndarray view.
pub fn slice2d_to_view<'a>(data: &'a [f64], rows: usize, cols: usize) -> ArrayView2<'a, f64> {
    ArrayView2::from_shape((rows, cols), data).expect("valid shape")
}

/// Convert an ndarray 1D view back to a slice.
pub fn view_to_slice<'a>(arr: &'a ArrayView1<'a, f64>) -> &'a [f64] {
    arr.as_slice().expect("ArrayView should be contiguous")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_roundtrip() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        let arr = vector_to_array1(v.clone());
        let v2 = view1_to_vector(vector_to_view1(&v));
        assert_eq!(v, v2);
    }

    #[test]
    fn matrix_roundtrip() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let arr = matrix_to_array2(m.clone());
        let m2 = array2_to_matrix(arr).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn test_vector_to_array1() {
        let v = Vector::new(vec![10.0, 20.0, 30.0]);
        let arr = vector_to_array1(v);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], 10.0);
        assert_eq!(arr[2], 30.0);
    }

    #[test]
    fn test_matrix_to_array2() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let arr = matrix_to_array2(m);
        assert_eq!(arr.dim(), (2, 2));
        assert_eq!((arr[[0, 0]], arr[[1, 1]]), (1.0, 4.0));
    }

    #[test]
    fn slice_interop() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let view = slice_to_view(&data);
        assert_eq!(view.len(), 4);
        let back = view_to_slice(&view);
        assert_eq!(back, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn slice2d_interop() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = slice2d_to_view(&data, 2, 3);
        assert_eq!(view.dim(), (2, 3));
        assert_eq!((view[[0, 0]], view[[1, 2]]), (1.0, 6.0));
    }
}