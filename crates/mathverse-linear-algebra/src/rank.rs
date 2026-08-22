//! Matrix rank computation via Gaussian elimination.

use mathverse_matrix::Matrix;

/// Compute the rank of a matrix using Gaussian elimination.
///
/// The rank is the number of non-zero rows in the row-echelon form.
///
/// # Examples
///
/// ```rust
/// use mathverse_linear_algebra::rank::matrix_rank;
/// use mathverse_matrix::Matrix;
///
/// // Full-rank 2x2
/// let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
/// assert_eq!(matrix_rank(&a), 2);
///
/// // Rank-1 matrix
/// let b = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]).unwrap();
/// assert_eq!(matrix_rank(&b), 1);
/// ```
#[allow(clippy::needless_range_loop)] // index arithmetic clearer in elimination loops
pub fn matrix_rank(a: &Matrix) -> usize {
    let rows = a.rows();
    let cols = a.cols();
    if rows == 0 || cols == 0 {
        return 0;
    }
    // Copy into a mutable working matrix
    let mut m = a.clone();
    let mut rank = 0;
    let tol = 1e-12;
    for col in 0..cols {
        // Find pivot row
        let pivot_row = (rank..rows).find(|&r| m.get(r, col).abs() > tol);
        if let Some(pivot) = pivot_row {
            swap_rows(&mut m, rank, pivot);
            let lv = m.get(rank, col);
            for r in (rank + 1)..rows {
                let factor = m.get(r, col) / lv;
                for c in col..cols {
                    m.set(r, c, m.get(r, c) - factor * m.get(rank, c));
                }
            }
            rank += 1;
        }
    }
    rank
}

fn swap_rows(m: &mut Matrix, i: usize, k: usize) {
    for c in 0..m.cols() {
        let temp = m.get(i, c);
        m.set(i, c, m.get(k, c));
        m.set(k, c, temp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_and_low_rank() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        assert_eq!(matrix_rank(&a), 2);

        let b = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]).unwrap();
        assert_eq!(matrix_rank(&b), 1);
    }

    #[test]
    fn rectangular_rank() {
        let a = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]).unwrap();
        assert_eq!(matrix_rank(&a), 2);
    }
}
