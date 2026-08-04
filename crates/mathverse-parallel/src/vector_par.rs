//! Parallel operations on `mathverse_vector::Vector`.

use mathverse_vector::Vector;
use rayon::prelude::*;

/// Parallel dot product.
pub fn par_dot(a: &Vector, b: &Vector) -> f64 {
    a.data
        .par_iter()
        .zip(b.data.par_iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Parallel element-wise addition.
pub fn par_add(a: &Vector, b: &Vector) -> Vector {
    let data: Vec<f64> = a
        .data
        .par_iter()
        .zip(b.data.par_iter())
        .map(|(x, y)| x + y)
        .collect();
    Vector::new(data)
}

/// Parallel element-wise subtraction.
pub fn par_sub(a: &Vector, b: &Vector) -> Vector {
    let data: Vec<f64> = a
        .data
        .par_iter()
        .zip(b.data.par_iter())
        .map(|(x, y)| x - y)
        .collect();
    Vector::new(data)
}

/// Parallel scalar multiply.
pub fn par_scale(v: &Vector, s: f64) -> Vector {
    let data: Vec<f64> = v.data.par_iter().map(|x| x * s).collect();
    Vector::new(data)
}

/// Parallel sum of elements.
pub fn par_sum(v: &Vector) -> f64 {
    v.data.par_iter().sum()
}

/// Parallel L2 norm.
pub fn par_norm(v: &Vector) -> f64 {
    v.data.par_iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Parallel mean.
pub fn par_mean(v: &Vector) -> f64 {
    par_sum(v) / v.len() as f64
}

/// Parallel variance (population).
pub fn par_variance(v: &Vector) -> f64 {
    let mean = par_mean(v);
    v.data.par_iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64
}

/// Parallel element-wise map.
pub fn par_map(v: &Vector, f: impl Fn(f64) -> f64 + Send + Sync) -> Vector {
    let data: Vec<f64> = v.data.par_iter().map(|&x| f(x)).collect();
    Vector::new(data)
}

/// Parallel element-wise multiply.
pub fn par_mul(a: &Vector, b: &Vector) -> Vector {
    let data: Vec<f64> = a
        .data
        .par_iter()
        .zip(b.data.par_iter())
        .map(|(x, y)| x * y)
        .collect();
    Vector::new(data)
}

/// Parallel cosine similarity.
pub fn par_cosine_similarity(a: &Vector, b: &Vector) -> f64 {
    let dot = par_dot(a, b);
    let norm_a = par_norm(a);
    let norm_b = par_norm(b);
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_dot() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        assert!((par_dot(&a, &b) - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_par_add() {
        let a = Vector::new(vec![1.0, 2.0]);
        let b = Vector::new(vec![3.0, 4.0]);
        assert_eq!(par_add(&a, &b), Vector::new(vec![4.0, 6.0]));
    }

    #[test]
    fn test_par_sum() {
        let v = Vector::new((0..1000).map(|i| i as f64).collect());
        assert!((par_sum(&v) - 499_500.0).abs() < 1e-10);
    }

    #[test]
    fn test_par_norm() {
        let v = Vector::new(vec![3.0, 4.0]);
        assert!((par_norm(&v) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_par_variance() {
        let v = Vector::new(vec![1.0, 1.0, 1.0]);
        assert!((par_variance(&v)).abs() < 1e-12);
    }

    #[test]
    fn test_par_cosine_similarity() {
        let a = Vector::new(vec![1.0, 0.0]);
        let b = Vector::new(vec![0.0, 1.0]);
        assert!((par_cosine_similarity(&a, &b)).abs() < 1e-12);
    }
}
