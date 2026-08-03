//! Linear programming via the simplex method.

/// Solves `max c^T x` subject to `Ax ≤ b, x ≥ 0`.
pub fn simplex(c: &[f64], a: &[Vec<f64>], b: &[f64]) -> Option<(f64, Vec<f64>)> {
    let m = b.len();
    let n = c.len();
    let mut tableau = vec![vec![0.0; n + m + 1]; m + 1];
    for j in 0..n { tableau[0][j] = -c[j]; }
    for i in 0..m {
        for j in 0..n { tableau[i + 1][j] = a[i][j]; }
        tableau[i + 1][n + i] = 1.0;
        tableau[i + 1][n + m] = b[i];
    }
    loop {
        let mut pivot_col = None;
        let mut min_val = 0.0;
        for j in 0..n + m {
            if tableau[0][j] < min_val - 1e-15 { min_val = tableau[0][j]; pivot_col = Some(j); }
        }
        let col = match pivot_col { Some(c) => c, None => break, };
        let mut pivot_row = None;
        let mut min_ratio = f64::INFINITY;
        for i in 1..=m {
            if tableau[i][col] > 1e-15 {
                let ratio = tableau[i][n + m] / tableau[i][col];
                if ratio < min_ratio { min_ratio = ratio; pivot_row = Some(i); }
            }
        }
        let row = match pivot_row { Some(r) => r, None => return None, };
        let pivot = tableau[row][col];
        for j in 0..=n + m { tableau[row][j] /= pivot; }
        for i in 0..=m {
            if i != row {
                let factor = tableau[i][col];
                for j in 0..=n + m { tableau[i][j] -= factor * tableau[row][j]; }
            }
        }
    }
    let obj = tableau[0][n + m];
    let mut x = vec![0.0; n];
    for j in 0..n {
        let mut row = None;
        let mut one_count = 0;
        for i in 1..=m {
            if (tableau[i][j] - 1.0).abs() < 1e-10 { row = Some(i); one_count += 1; }
            else if tableau[i][j].abs() > 1e-10 { one_count += 2; }
        }
        if one_count == 1 { x[j] = tableau[row.unwrap()][n + m]; }
    }
    Some((-obj, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_test() {
        let c = vec![3.0, 2.0];
        let a = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![4.0, 3.0, 2.0];
        let (obj, x) = simplex(&c, &a, &b).unwrap();
        assert!((obj - 11.0).abs() < 1e-6);
    }
}
