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

impl From<Vector> for Array1<f64> {
    fn from(v: Vector) -> Self {
        Array1::from_vec(v.data)
    }
}

impl From<Array1<f64>> for Vector {
    fn from(arr: Array1<f64>) -> Self {
        Vector::new(arr.to_vec())
    }
}

impl<'a> From<&'a Vector> for ArrayView1<'a, f64> {
    fn from(v: &'a Vector) -> Self {
        ArrayView1::from_shape(v.len(), &v.data).expect("valid shape")
    }
}

impl<'a> From<&'a mut Vector> for ArrayViewMut1<'a, f64> {
    fn from(v: &'a mut Vector) -> Self {
        let len = v.len();
        ArrayViewMut1::from_shape(len, &mut v.data).expect("valid shape")
    }
}

impl<'a> From<ArrayView1<'a, f64>> for Vector {
    fn from(arr: ArrayView1<'a, f64>) -> Self {
        Vector::new(arr.to_vec())
    }
}

// ─── Matrix ↔ Array2 ──────────────────────────────────────────────

impl From<Matrix> for Array2<f64> {
    fn from(m: Matrix) -> Self {
        Array2::from_shape_vec((m.rows, m.cols), m.data)
            .expect("Matrix shape should be valid")
    }
}

impl TryFrom<Array2<f64>> for Matrix {
    type Error = ShapeError;

    fn try_from(arr: Array2<f64>) -> Result<Self, Self::Error> {
        let (rows, cols) = arr.dim();
        Ok(Matrix {
            rows,
            cols,
            data: arr.into_raw_vec(),
        })
    }
}

impl<'a> From<&'a Matrix> for ArrayView2<'a, f64> {
    fn from(m: &'a Matrix) -> Self {
        ArrayView2::from_shape((m.rows, m.cols), &m.data).expect("valid shape")
    }
}

impl<'a> From<&'a mut Matrix> for ArrayViewMut2<'a, f64> {
    fn from(m: &'a mut Matrix) -> Self {
        let (rows, cols) = (m.rows, m.cols);
        ArrayViewMut2::from_shape((rows, cols), &mut m.data).expect("valid shape")
    }
}

impl<'a> From<ArrayView2<'a, f64>> for Matrix {
    fn from(arr: ArrayView2<'a, f64>) -> Self {
        let (rows, cols) = arr.dim();
        Matrix {
            rows,
            cols,
            data: arr.to_owned().into_raw_vec(),
        }
    }
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
pub fn view_to_slice<'a>(arr: &ArrayView1<'a, f64>) -> &'a [f64] {
    arr.as_slice().expect("ArrayView should be contiguous")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_roundtrip() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        let arr: Array1<f64> = Array1::from(v.clone());
        let v2: Vector = Vector::from(arr);
        assert_eq!(v, v2);
    }

    #[test]
    fn matrix_roundtrip() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let arr: Array2<f64> = Array2::from(m.clone());
        let m2: Matrix = Matrix::try_from(arr).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn vector_to_array1() {
        let v = Vector::new(vec![10.0, 20.0, 30.0]);
        let arr: Array1<f64> = v.into();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], 10.0);
        assert_eq!(arr[2], 30.0);
    }

    #[test]
    fn matrix_to_array2() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let arr: Array2<f64> = m.into();
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
