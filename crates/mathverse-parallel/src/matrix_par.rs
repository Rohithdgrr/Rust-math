//! Parallel operations on `mathverse_matrix::Matrix`.

use mathverse_matrix::Matrix;
use mathverse_vector::Vector;
use rayon::prelude::*;

/// Parallel matrix-vector product.
pub fn par_mat_vec(m: &Matrix, v: &Vector) -> Vector {
    let data: Vec<f64> = (0..m.rows)
        .into_par_iter()
        .map(|i| {
            (0..m.cols)
                .map(|j| m.get(i, j) * v.get(j))
                .sum()
        })
        .collect();
    Vector::new(data)
}

/// Parallel matrix-matrix product.
pub fn par_mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(a.rows, b.cols);
    // Parallelize over rows of the result
    out.data
        .par_chunks_mut(b.cols)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..b.cols {
                row[j] = (0..a.cols).map(|k| a.get(i, k) * b.get(k, j)).sum();
            }
        });
    out
}

/// Parallel element-wise matrix addition.
pub fn par_mat_add(a: &Matrix, b: &Matrix) -> Matrix {
    let data: Vec<f64> = a
        .data
        .par_iter()
        .zip(b.data.par_iter())
        .map(|(x, y)| x + y)
        .collect();
    Matrix {
        rows: a.rows,
        cols: a.cols,
        data,
    }
}

/// Parallel matrix trace.
pub fn par_trace(m: &Matrix) -> f64 {
    (0..m.rows).into_par_iter().map(|i| m.get(i, i)).sum()
}

/// Parallel Frobenius norm of a matrix.
pub fn par_frobenius_norm(m: &Matrix) -> f64 {
    m.data.par_iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Parallel column means.
pub fn par_col_means(m: &Matrix) -> Vec<f64> {
    (0..m.cols)
        .into_par_iter()
        .map(|j| {
            let sum: f64 = (0..m.rows).map(|i| m.get(i, j)).sum();
            sum / m.rows as f64
        })
        .collect()
}

/// Parallel row means.
pub fn par_row_means(m: &Matrix) -> Vec<f64> {
    (0..m.rows)
        .into_par_iter()
        .map(|i| {
            let sum: f64 = (0..m.cols).map(|j| m.get(i, j)).sum();
            sum / m.cols as f64
        })
        .collect()
}

/// Parallel element-wise matrix multiplication (Hadamard product).
pub fn par_hadamard(a: &Matrix, b: &Matrix) -> Matrix {
    let data: Vec<f64> = a
        .data
        .par_iter()
        .zip(b.data.par_iter())
        .map(|(x, y)| x * y)
        .collect();
    Matrix {
        rows: a.rows,
        cols: a.cols,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_mat_vec() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let v = Vector::new(vec![1.0, 1.0]);
        let result = par_mat_vec(&m, &v);
        assert_eq!(result, Vector::new(vec![3.0, 7.0]));
    }

    #[test]
    fn test_par_mat_mul() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let b = Matrix::identity(2);
        let result = par_mat_mul(&a, &b);
        assert_eq!(result, a);
    }

    #[test]
    fn test_par_trace() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        assert!((par_trace(&m) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_par_frobenius() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!((par_frobenius_norm(&m) - expected).abs() < 1e-12);
    }
}
