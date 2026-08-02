//! Linear system solvers: 2×2, 3×3, Gaussian row reduction.

/// Solves a 2×2 system `Ax = b` using Cramer's rule.
pub fn solve_2x2(a: [[f64; 2]; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-15 { return None; }
    Some([
        (b[0] * a[1][1] - b[1] * a[0][1]) / det,
        (a[0][0] * b[1] - a[1][0] * b[0]) / det,
    ])
}

pub fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-15 { return None; }
    let inv = [
        [a[1][1]*a[2][2]-a[1][2]*a[2][1], a[0][2]*a[2][1]-a[0][1]*a[2][2], a[0][1]*a[1][2]-a[0][2]*a[1][1]],
        [a[1][2]*a[2][0]-a[1][0]*a[2][2], a[0][0]*a[2][2]-a[0][2]*a[2][0], a[0][2]*a[1][0]-a[0][0]*a[1][2]],
        [a[1][0]*a[2][1]-a[1][1]*a[2][0], a[0][1]*a[2][0]-a[0][0]*a[2][1], a[0][0]*a[1][1]-a[0][1]*a[1][0]],
    ];
    Some([
        inv[0][0]*b[0]+inv[0][1]*b[1]+inv[0][2]*b[2],
        inv[1][0]*b[0]+inv[1][1]*b[1]+inv[1][2]*b[2],
        inv[2][0]*b[0]+inv[2][1]*b[1]+inv[2][2]*b[2],
    ].map(|v| v / det))
}

pub fn row_reduce(matrix: &mut Vec<Vec<f64>>) -> bool {
    let n = matrix.len();
    if n == 0 { return true; }
    let m = matrix[0].len();
    let mut row = 0;
    for col in 0..m-1 {
        if row >= n { break; }
        let mut max_row = row;
        for r in row+1..n {
            if matrix[r][col].abs() > matrix[max_row][col].abs() { max_row = r; }
        }
        matrix.swap(row, max_row);
        if matrix[row][col].abs() < 1e-15 { continue; }
        for r in row+1..n {
            let f = matrix[r][col] / matrix[row][col];
            for c in col..m { matrix[r][c] -= f * matrix[row][c]; }
        }
        row += 1;
    }
    row == n.min(m - 1)
}

#[deprecated(note = "misleading name — only does row reduction, does not solve. Use `row_reduce` or `solve_gauss`")]
pub fn gaussian_elimination(matrix: &mut Vec<Vec<f64>>) -> bool {
    row_reduce(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_by_two() {
        let a = [[2.0, 1.0], [1.0, 3.0]];
        let b = [5.0, 7.0];
        let x = solve_2x2(a, b).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn three_by_three() {
        let a = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]];
        let b = [8.0, -11.0, -3.0];
        let x = solve_3x3(a, b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
        assert!((x[2] - -1.0).abs() < 1e-10);
    }

    #[test]
    fn gaussian() {
        let mut m = vec![
            vec![2.0, 1.0, -1.0, 8.0],
            vec![-3.0, -1.0, 2.0, -11.0],
            vec![-2.0, 1.0, 2.0, -3.0],
        ];
        assert!(row_reduce(&mut m));
    }
}
