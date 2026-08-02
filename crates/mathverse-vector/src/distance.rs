/// Euclidean (L2) distance between two points.
pub fn euclidean(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt() }

/// Manhattan (L1) distance between two points.
pub fn manhattan(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| (x-y).abs()).sum() }

/// Chebyshev (L∞) distance between two points.
pub fn chebyshev(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0, f64::max) }

/// Cosine distance: `1 - cos(θ)`. Returns 0.0 if either vector is zero.
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x,y)| x*y).sum();
    let ma = a.iter().map(|x| x*x).sum::<f64>().sqrt();
    let mb = b.iter().map(|x| x*x).sum::<f64>().sqrt();
    if ma == 0.0 || mb == 0.0 { 0.0 } else { 1.0 - dot / (ma * mb) }
}
/// Mahalanobis distance using the inverse covariance matrix.
pub fn mahalanobis(a: &[f64], b: &[f64], cov_inv: &[Vec<f64>]) -> f64 {
    let diff: Vec<f64> = a.iter().zip(b).map(|(x,y)| x-y).collect();
    let md = diff.iter().enumerate().map(|(i, &d)| d * cov_inv[i].iter().zip(&diff).map(|(c, &x)| c*x).sum::<f64>()).sum::<f64>();
    md.sqrt()
}
/// Minkowski distance: `(sum(|a_i - b_i|^p))^(1/p)`.
pub fn minkowski(a: &[f64], b: &[f64], p: f64) -> f64 { a.iter().zip(b).map(|(x,y)| (x-y).abs().powf(p)).sum::<f64>().powf(1.0/p) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn euclid_test() { assert!((euclidean(&[0.0,0.0], &[3.0,4.0]) - 5.0).abs() < 1e-10); }
    #[test] fn manhattan_test() { assert!((manhattan(&[1.0,2.0], &[4.0,6.0]) - 7.0).abs() < 1e-10); }
    #[test] fn cosine_test() { assert!((cosine(&[1.0,0.0], &[1.0,0.0])).abs() < 1e-10); }
}
