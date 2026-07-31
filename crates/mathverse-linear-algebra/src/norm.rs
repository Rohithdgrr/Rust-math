pub fn norm_1(a: &[Vec<f64>]) -> f64 {
    let (m, n) = (a.len(), a[0].len());
    (0..n).map(|j| (0..m).map(|i| a[i][j].abs()).sum::<f64>()).fold(0.0f64, f64::max)
}

pub fn norm_inf(a: &[Vec<f64>]) -> f64 {
    a.iter().map(|r| r.iter().map(|v| v.abs()).sum::<f64>()).fold(0.0f64, f64::max)
}

pub fn norm_frobenius(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|r| r.iter()).map(|v| v * v).sum::<f64>().sqrt()
}

pub fn norm_2(a: &[Vec<f64>]) -> f64 {
    let singular = singular_values(a);
    singular.first().copied().unwrap_or(0.0)
}

pub fn singular_values(a: &[Vec<f64>]) -> Vec<f64> {
    let ata: Vec<Vec<f64>> = {
        let n = a[0].len();
        let m = a.len();
        (0..n).map(|i| (0..n).map(|j| (0..m).map(|k| a[k][i]*a[k][j]).sum()).collect()).collect()
    };
    let mut vals = Vec::new();
    let mut temp = ata;
    for _ in 0..30 {
        let n = temp.len();
        if n == 0 { break; }
        if n == 1 { vals.push(temp[0][0].max(0.0).sqrt()); break; }
        let eigen = crate::decomposition::power_iteration(&temp, 100, 1e-10);
        if let Some((_, lambda)) = eigen {
            vals.push(lambda.max(0.0).sqrt());
            for i in 0..n { for j in 0..n { temp[i][j] -= lambda * if i == j { 1.0 } else { 0.0 }; } }
        } else { break; }
    }
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    vals
}

pub fn condition_number(a: &[Vec<f64>]) -> f64 {
    let sv = singular_values(a);
    if sv.is_empty() { return f64::INFINITY; }
    sv[0] / sv.last().unwrap_or(&1.0)
}

pub fn matrix_norm(a: &[Vec<f64>], p: f64) -> f64 {
    if p == 1.0 { norm_1(a) }
    else if p == f64::INFINITY { norm_inf(a) }
    else if p == 2.0 { norm_2(a) }
    else if p == 0.0 { norm_frobenius(a) }
    else { norm_frobenius(a) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norms() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((norm_1(&a) - 6.0).abs() < 1e-10);
        assert!((norm_inf(&a) - 7.0).abs() < 1e-10);
        assert!((norm_frobenius(&a) - (1.0+4.0+9.0+16.0).sqrt()).abs() < 1e-10);
    }
}
