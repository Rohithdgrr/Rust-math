//! Matrix rank computation via Gaussian elimination.

/// Compute the rank of a matrix using Gaussian elimination.
///
/// The rank is the number of non-zero rows in the row-echelon form.
///
/// # Examples
///
/// ```rust
/// use mathverse_linear_algebra::rank::matrix_rank;
///
/// // Full-rank 2x2
/// let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// assert_eq!(matrix_rank(&a), 2);
///
/// // Rank-1 matrix
/// let b = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
/// assert_eq!(matrix_rank(&b), 1);
/// ```
pub fn matrix_rank(a: &[Vec<f64>]) -> usize {
    if a.is_empty() || a[0].is_empty() {
        return 0;
    }
    let rows = a.len();
    let cols = a[0].len();
    // Copy into a mutable working matrix
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut rank = 0;
    let tol = 1e-12;
    for col in 0..cols {
        // Find pivot row
        let pivot_row = (rank..rows).find(|&r| m[r][col].abs() > tol);
        if let Some(pivot) = pivot_row {
            m.swap(rank, pivot);
            let lv = m[rank][col];
            for r in (rank + 1)..rows {
                let factor = m[r][col] / lv;
                for c in col..cols {
                    m[r][c] -= factor * m[rank][c];
                }
            }
            rank += 1;
        }
    }
    rank
}