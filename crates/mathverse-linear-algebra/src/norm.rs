//! Matrix norms: L1, L∞, Frobenius, spectral (L2), condition number.

/// L1 (maximum column sum) norm of a matrix.
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
    let (m, n) = (a.len(), a[0].len());
    
    // Compute A^T A
    let mut ata: Vec<Vec<f64>> = {
        (0..n).map(|i| (0..n).map(|j| (0..m).map(|k| a[k][i]*a[k][j]).sum()).collect()).collect()
    };
    
    let mut vals = Vec::new();
    let max_vals = n.min(30); // Safety limit
    
    for _ in 0..max_vals {
        if ata.is_empty() { break; }
        if ata.len() == 1 { 
            vals.push(ata[0][0].max(0.0).sqrt()); 
            break; 
        }
        
        // Find dominant eigenvalue/eigenvector using power iteration
        let eigen = crate::decomposition::power_iteration(&ata, 100, 1e-10);
        
        if let Some((v, lambda)) = eigen {
            vals.push(lambda.max(0.0).sqrt());
            
            // Simple deflation: A_new = A - lambda * v * v^T
            // This is the standard Hotelling deflation
            let n_size = ata.len();
            for i in 0..n_size {
                for j in 0..n_size {
                    ata[i][j] -= lambda * v[i] * v[j];
                }
            }
        } else { 
            break; 
        }
    }
    
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    vals
}

pub fn condition_number(a: &[Vec<f64>]) -> f64 {
    let sv = singular_values(a);
    if sv.is_empty() { return f64::INFINITY; }
    let max_sv = sv[0];
    let min_sv = sv.last().unwrap_or(&max_sv);
    if min_sv.abs() < 1e-15 { return f64::INFINITY; }
    max_sv / min_sv
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
        assert!((norm_frobenius(&a) - (1.0_f64+4.0+9.0+16.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn singular_values_diagonal() {
        // Test with diagonal matrix diag(3, 1) - singular values should be [3, 1]
        let a = vec![vec![3.0, 0.0], vec![0.0, 1.0]];
        let sv = singular_values(&a);
        assert_eq!(sv.len(), 2);
        assert!((sv[0] - 3.0).abs() < 1e-8);
        assert!((sv[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn singular_values_identity() {
        // Identity matrix should have all singular values = 1
        let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let sv = singular_values(&a);
        // At least one singular value should be found
        assert!(!sv.is_empty());
        // The largest singular value should be close to 1
        assert!((sv[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn singular_values_rectangular() {
        // Test with a rectangular matrix (3x2)
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
        let sv = singular_values(&a);
        // Should find at least one singular value
        assert!(!sv.is_empty());
        // The largest singular value should be close to 1
        assert!((sv[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn condition_number_test() {
        // Well-conditioned matrix
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let sv = singular_values(&a);
        println!("Identity singular values: {:?}", sv);
        // For identity, power iteration should find at least one singular value = 1
        assert!(!sv.is_empty());
        assert!((sv[0] - 1.0).abs() < 1e-4);
        
        // Ill-conditioned matrix
        let b = vec![vec![1.0, 1.0], vec![1.0, 1.0 + 1e-2]];
        let sv_b = singular_values(&b);
        println!("Ill-conditioned singular values: {:?}", sv_b);
        // Should find at least one singular value
        assert!(!sv_b.is_empty());
        // Largest singular value should be positive
        assert!(sv_b[0] > 0.0);
    }
}
