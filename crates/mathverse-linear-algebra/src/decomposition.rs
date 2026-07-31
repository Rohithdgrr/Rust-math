pub fn lu_decompose(a: &[Vec<f64>]) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let n = a.len();
    if n == 0 || a.iter().any(|r| r.len() != n) { return None; }
    let mut l = vec![vec![0.0; n]; n];
    let mut u = a.to_vec();
    for i in 0..n {
        for k in i..n {
            let mut sum = 0.0;
            for j in 0..i { sum += l[i][j] * u[j][k]; }
            u[i][k] = a[i][k] - sum;
        }
        if u[i][i].abs() < 1e-15 { return None; }
        l[i][i] = 1.0;
        for k in i+1..n {
            let mut sum = 0.0;
            for j in 0..i { sum += l[k][j] * u[j][i]; }
            l[k][i] = (a[k][i] - sum) / u[i][i];
        }
    }
    Some((l, u))
}

pub fn qr_decompose(a: &[Vec<f64>]) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let (m, n) = (a.len(), a[0].len());
    let mut q = vec![vec![0.0; m]; m];
    let mut r = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut v: Vec<f64> = (0..m).map(|i| a[i][j]).collect();
        for i in 0..j {
            let dot: f64 = (0..m).map(|k| q[k][i] * v[k]).sum();
            r[i][j] = dot;
            for k in 0..m { v[k] -= dot * q[k][i]; }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 { return None; }
        r[j][j] = norm;
        for k in 0..m { q[k][j] = v[k] / norm; }
    }
    Some((q, r))
}

pub fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..i { sum += l[i][k] * l[i][k]; }
        let diag = a[i][i] - sum;
        if diag <= 1e-15 { return None; }
        l[i][i] = diag.sqrt();
        for j in i+1..n {
            let mut sum = 0.0;
            for k in 0..i { sum += l[j][k] * l[i][k]; }
            l[j][i] = (a[j][i] - sum) / l[i][i];
        }
    }
    Some(l)
}

pub fn solve_lu(l: &[Vec<f64>], u: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0; n];
    for i in 0..n { y[i] = b[i] - (0..i).map(|j| l[i][j] * y[j]).sum::<f64>(); }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() { x[i] = (y[i] - (i+1..n).map(|j| u[i][j] * x[j]).sum::<f64>()) / u[i][i]; }
    x
}

pub fn eigenvalue_2x2(a: [[f64; 2]; 2]) -> Vec<f64> {
    let trace = a[0][0] + a[1][1];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let disc = trace * trace - 4.0 * det;
    if disc < 0.0 { return vec![trace / 2.0]; }
    let sqrt_disc = disc.sqrt();
    vec![(trace + sqrt_disc) / 2.0, (trace - sqrt_disc) / 2.0]
}

pub fn power_iteration(a: &[Vec<f64>], max_iter: usize, tol: f64) -> Option<(Vec<f64>, f64)> {
    let n = a.len();
    if n == 0 { return None; }
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda = 0.0;
    for _ in 0..max_iter {
        let mut w = vec![0.0; n];
        for i in 0..n { for j in 0..n { w[i] += a[i][j] * v[j]; } }
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 { break; }
        for i in 0..n { v[i] = w[i] / norm; }
        let mut new_lambda = 0.0;
        for i in 0..n { new_lambda += v[i] * (0..n).map(|j| a[i][j] * v[j]).sum::<f64>(); }
        if (new_lambda - lambda).abs() < tol { return Some((v, new_lambda)); }
        lambda = new_lambda;
    }
    Some((v, lambda))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_test() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let (l, u) = lu_decompose(&a).unwrap();
        let x = solve_lu(&l, &u, &[5.0, 7.0]);
        assert!((x[0] - 1.6).abs() < 1e-10);
    }

    #[test]
    fn qr_test() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (q, r) = qr_decompose(&a).unwrap();
        assert!((q[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cholesky_test() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&a).unwrap();
        assert!((l[0][0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn eigen_2x2() {
        let eigs = eigenvalue_2x2([[1.0, 0.0], [0.0, 2.0]]);
        assert_eq!(eigs.len(), 2);
    }
}
