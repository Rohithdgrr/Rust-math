pub fn solve_gauss(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n { return None; }
    let mut aug: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut row = a[i].clone();
        row.push(b[i]);
        row
    }).collect();
    for col in 0..n {
        let mut max_row = col;
        for r in col+1..n {
            if aug[r][col].abs() > aug[max_row][col].abs() { max_row = r; }
        }
        aug.swap(col, max_row);
        if aug[col][col].abs() < 1e-15 { return None; }
        for r in col+1..n {
            let f = aug[r][col] / aug[col][col];
            for c in col..=n { aug[r][c] -= f * aug[col][c]; }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (aug[i][n] - (i+1..n).map(|j| aug[i][j] * x[j]).sum::<f64>()) / aug[i][i];
    }
    Some(x)
}

pub fn matrix_inverse(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 || a.iter().any(|r| r.len() != n) { return None; }
    let mut aug: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut row = a[i].clone();
        row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
        row
    }).collect();
    for col in 0..n {
        let mut max_row = col;
        for r in col+1..n {
            if aug[r][col].abs() > aug[max_row][col].abs() { max_row = r; }
        }
        aug.swap(col, max_row);
        if aug[col][col].abs() < 1e-15 { return None; }
        let pivot = aug[col][col];
        for c in col..2*n { aug[col][c] /= pivot; }
        for r in 0..n {
            if r != col {
                let f = aug[r][col];
                for c in col..2*n { aug[r][c] -= f * aug[col][c]; }
            }
        }
    }
    Some((0..n).map(|i| aug[i][n..2*n].to_vec()).collect())
}

pub fn determinant(a: &[Vec<f64>]) -> Option<f64> {
    let n = a.len();
    if n == 0 || a.iter().any(|r| r.len() != n) { return None; }
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut det = 1.0;
    let mut swaps = 0usize;
    for col in 0..n {
        let mut max_row = col;
        for r in col+1..n {
            if m[r][col].abs() > m[max_row][col].abs() { max_row = r; }
        }
        if max_row != col {
            m.swap(col, max_row);
            swaps += 1;
        }
        if m[col][col].abs() < 1e-15 { return Some(0.0); }
        det *= m[col][col];
        for r in col+1..n {
            let f = m[r][col] / m[col][col];
            for c in col..n { m[r][c] -= f * m[col][c]; }
        }
    }
    if swaps % 2 != 0 { det = -det; }
    Some(det)
}

pub fn rank(a: &[Vec<f64>]) -> usize {
    let mut m: Vec<Vec<f64>> = a.to_vec();
    if m.is_empty() { return 0; }
    let (rows, cols) = (m.len(), m[0].len());
    let mut rank = 0;
    for col in 0..cols {
        let mut max_row = rank;
        for r in rank+1..rows {
            if m[r][col].abs() > m[max_row][col].abs() { max_row = r; }
        }
        if max_row < rows && m[max_row][col].abs() > 1e-15 {
            m.swap(rank, max_row);
            for r in rank+1..rows {
                let f = m[r][col] / m[rank][col];
                for c in col..cols { m[r][c] -= f * m[rank][c]; }
            }
            rank += 1;
        }
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = solve_gauss(&a, &b).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn inv() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let inv = matrix_inverse(&a).unwrap();
        assert!((inv[0][0] - 0.6).abs() < 1e-10);
        assert!((inv[0][1] - -0.2).abs() < 1e-10);
    }

    #[test]
    fn det() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((determinant(&a).unwrap() - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn det_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!((determinant(&a).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn det_swap_rows() {
        let a = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        assert!((determinant(&a).unwrap() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn det_3x3_identity() {
        let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        assert!((determinant(&a).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rank_full() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(rank(&a), 2);
    }

    #[test]
    fn rank_deficient() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert_eq!(rank(&a), 1);
    }
}
