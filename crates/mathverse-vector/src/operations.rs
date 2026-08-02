/// Element-wise addition of two slices.
pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() }

/// Element-wise subtraction of two slices.
pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x - y).collect() }

/// Scalar multiplication of a slice.
pub fn scale(v: &[f64], s: f64) -> Vec<f64> { v.iter().map(|x| x * s).collect() }

/// Dot product of two slices.
pub fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

/// Cross product of two 3D vectors. Returns an empty vector if inputs are not length 3.
pub fn cross(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() != 3 || b.len() != 3 { return Vec::new(); }
    vec![a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

/// L2 (Euclidean) magnitude of a vector.
pub fn magnitude(v: &[f64]) -> f64 { v.iter().map(|x| x*x).sum::<f64>().sqrt() }

/// Returns a unit vector in the same direction. Returns the input unchanged if magnitude is zero.
pub fn normalize(v: &[f64]) -> Vec<f64> { let m = magnitude(v); if m == 0.0 { v.to_vec() } else { v.iter().map(|x| x/m).collect() } }

/// Element-wise (Hadamard) product of two slices.
pub fn hadamard(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x * y).collect() }

/// Outer product of two vectors, producing a matrix.
pub fn outer(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> { a.iter().map(|ai| b.iter().map(|bi| ai * bi).collect()).collect() }

/// Negates every element.
pub fn negate(v: &[f64]) -> Vec<f64> { v.iter().map(|x| -x).collect() }

/// Adds a scalar to every element.
pub fn add_scalar(v: &[f64], s: f64) -> Vec<f64> { v.iter().map(|x| x + s).collect() }

/// Linear interpolation between two vectors: `a + t * (b - a)`.
pub fn lerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + t * (y - x)).collect() }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn add_test() { assert_eq!(add(&[1.0,2.0], &[3.0,4.0]), vec![4.0,6.0]); }
    #[test] fn dot_test() { assert!((dot(&[1.0,2.0,3.0], &[4.0,5.0,6.0]) - 32.0).abs() < 1e-10); }
    #[test] fn cross_test() {
        let c = cross(&[1.0,0.0,0.0], &[0.0,1.0,0.0]);
        assert!((c[0]).abs() < 1e-10 && (c[1]).abs() < 1e-10 && (c[2] - 1.0).abs() < 1e-10);
    }
    #[test] fn normalize_test() {
        let n = normalize(&[3.0,4.0]);
        assert!((n[0] - 0.6).abs() < 1e-10 && (n[1] - 0.8).abs() < 1e-10);
    }
}
