pub fn mat_vec_mul(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| row.iter().zip(v).map(|(a,b)| a*b).sum()).collect()
}
pub fn det2x2(m: &[[f64; 2]; 2]) -> f64 { m[0][0]*m[1][1] - m[0][1]*m[1][0] }
pub fn det3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
    - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
    + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
}
pub fn rank(vectors: &[Vec<f64>]) -> usize {
    let mut mat: Vec<Vec<f64>> = vectors.to_vec();
    let mut rank = 0;
    let n_rows = mat.len();
    if n_rows == 0 { return 0; }
    let n_cols = mat[0].len();
    for col in 0..n_cols {
        let mut pivot_row = None;
        for row in rank..n_rows {
            if mat[row][col].abs() > 1e-10 { pivot_row = Some(row); break; }
        }
        if let Some(pr) = pivot_row {
            mat.swap(rank, pr);
            let pivot = mat[rank][col];
            for j in 0..n_cols { mat[rank][j] /= pivot; }
            for row in 0..n_rows {
                if row != rank && mat[row][col].abs() > 1e-10 {
                    let factor = mat[row][col];
                    for j in 0..n_cols { mat[row][j] -= factor * mat[rank][j]; }
                }
            }
            rank += 1;
        }
    }
    rank
}
pub fn is_orthogonal(vectors: &[Vec<f64>], tol: f64) -> bool {
    for i in 0..vectors.len() {
        for j in i+1..vectors.len() {
            let dot: f64 = vectors[i].iter().zip(&vectors[j]).map(|(a,b)| a*b).sum();
            if dot.abs() > tol { return false; }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn mat_vec_test() {
        assert_eq!(mat_vec_mul(&[vec![1.0,2.0], vec![3.0,4.0]], &[5.0,6.0]), vec![17.0,39.0]);
    }
    #[test] fn det3_test() {
        let m = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,0.0]];
        let det = det3x3(&m);
        assert!((det - 27.0).abs() < 1e-10);
    }
}
