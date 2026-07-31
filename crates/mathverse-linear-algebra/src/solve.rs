pub fn solve_lu(l: &[Vec<f64>], u: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0; n];
    for i in 0..n { y[i] = b[i] - (0..i).map(|j| l[i][j] * y[j]).sum::<f64>(); }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() { x[i] = (y[i] - (i+1..n).map(|j| u[i][j] * x[j]).sum::<f64>()) / u[i][i]; }
    x
}

pub fn solve_qr(q: &[Vec<f64>], r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut qt_b = vec![0.0; n];
    for i in 0..n { qt_b[i] = (0..n).map(|j| q[j][i] * b[j]).sum(); }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() { x[i] = (qt_b[i] - (i+1..n).map(|j| r[i][j] * x[j]).sum::<f64>()) / r[i][i]; }
    x
}

pub fn solve_2x2(a: [[f64; 2]; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-15 { return None; }
    Some([(b[0]*a[1][1]-b[1]*a[0][1])/det, (a[0][0]*b[1]-a[1][0]*b[0])/det])
}

pub fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1]) - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0]) + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
    if det.abs() < 1e-15 { return None; }
    Some([
        (b[0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1]) - a[0][1]*(b[1]*a[2][2]-a[1][2]*b[2]) + a[0][2]*(b[1]*a[2][1]-a[1][1]*b[2])) / det,
        (a[0][0]*(b[1]*a[2][2]-a[1][2]*b[2]) - b[0]*(a[1][0]*a[2][2]-a[1][2]*a[2][0]) + a[0][2]*(a[1][0]*b[2]-b[1]*a[2][0])) / det,
        (a[0][0]*(a[1][1]*b[2]-b[1]*a[2][1]) - a[0][1]*(a[1][0]*b[2]-b[1]*a[2][0]) + b[0]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])) / det,
    ])
}

pub fn solve_gauss(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    let mut aug: Vec<Vec<f64>> = (0..n).map(|i| { let mut r = a[i].clone(); r.push(b[i]); r }).collect();
    for col in 0..n {
        let mut max_row = col;
        for r in col+1..n { if aug[r][col].abs() > aug[max_row][col].abs() { max_row = r; } }
        aug.swap(col, max_row);
        if aug[col][col].abs() < 1e-15 { return None; }
        for r in col+1..n {
            let f = aug[r][col] / aug[col][col];
            for c in col..=n { aug[r][c] -= f * aug[col][c]; }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() { x[i] = (aug[i][n] - (i+1..n).map(|j| aug[i][j]*x[j]).sum::<f64>()) / aug[i][i]; }
    Some(x)
}

pub fn ls_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let (m, n) = (a.len(), a[0].len());
    let mut ata = vec![vec![0.0; n]; n];
    let mut atb = vec![0.0; n];
    for i in 0..n { for j in 0..n { ata[i][j] = (0..m).map(|k| a[k][i] * a[k][j]).sum(); } }
    for i in 0..n { atb[i] = (0..m).map(|k| a[k][i] * b[k]).sum(); }
    solve_gauss(&ata, &atb)
}

pub fn residual_norm(a: &[Vec<f64>], b: &[f64], x: &[f64]) -> f64 {
    let m = a.len();
    let mut sum = 0.0;
    for i in 0..m {
        let r = b[i] - (0..x.len()).map(|j| a[i][j] * x[j]).sum::<f64>();
        sum += r * r;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solve() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let (l, u) = crate::decomposition::lu_decompose(&a).unwrap();
        let x = solve_lu(&l, &u, &[5.0, 7.0]);
        assert!((x[0] - 1.6).abs() < 1e-10);
    }

    #[test]
    fn gauss_test() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let x = solve_gauss(&a, &[5.0, 7.0]).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10);
    }

    #[test]
    fn least_squares() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let b = vec![1.0, 2.0, 3.5];
        let x = ls_solve(&a, &b).unwrap();
        assert!((x[0] - 1.083).abs() < 0.01);
    }
}
