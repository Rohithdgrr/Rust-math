//! Advanced interpolation: spline, Hermite, barycentric, and multidimensional methods.

use mathverse_core::error::{MathError, MathResult};

/// Cubic spline interpolation with natural boundary conditions.
pub struct CubicSpline {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub coeffs: Vec<(f64, f64, f64, f64)>, // (a, b, c, d) for each interval
}

impl CubicSpline {
    /// Create cubic spline from data points.
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> MathResult<Self> {
        if xs.len() != ys.len() || xs.len() < 2 {
            return Err(MathError::InvalidArgument("invalid data points"));
        }
        
        let n = xs.len();
        
        // Solve tridiagonal system for second derivatives
        let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
        let alpha: Vec<f64> = (1..n - 1)
            .map(|i| 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]))
            .collect();
        
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];
        
        for i in 1..n - 1 {
            l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i];
        }
        
        l[n - 1] = 1.0;
        z[n - 1] = 0.0;
        
        let mut c = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut d = vec![0.0; n];
        
        for j in (1..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }
        
        for j in 0..n - 1 {
            b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (2.0 * c[j] + c[j + 1]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }
        
        let coeffs: Vec<(f64, f64, f64, f64)> = (0..n - 1)
            .map(|i| (ys[i], b[i], c[i], d[i]))
            .collect();
        
        Ok(CubicSpline { xs, ys, coeffs })
    }

    /// Evaluate spline at x.
    pub fn evaluate(&self, x: f64) -> f64 {
        if x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= self.xs[self.xs.len() - 1] {
            return self.ys[self.ys.len() - 1];
        }
        
        // Find interval
        let i = self.xs.partition_point(|&v| v <= x) - 1;
        let dx = x - self.xs[i];
        let (a, b, c, d) = self.coeffs[i];
        
        a + b * dx + c * dx * dx + d * dx * dx * dx
    }

    /// Get derivative at x.
    pub fn derivative(&self, x: f64) -> f64 {
        if x <= self.xs[0] {
            let (_, b, _, _) = self.coeffs[0];
            return b;
        }
        if x >= self.xs[self.xs.len() - 1] {
            let n = self.coeffs.len();
            let (_, b, c, d) = self.coeffs[n - 1];
            let dx = self.xs[self.xs.len() - 1] - self.xs[self.xs.len() - 2];
            return b + 2.0 * c * dx + 3.0 * d * dx * dx;
        }
        
        let i = self.xs.partition_point(|&v| v <= x) - 1;
        let dx = x - self.xs[i];
        let (_, b, c, d) = self.coeffs[i];
        
        b + 2.0 * c * dx + 3.0 * d * dx * dx
    }
}

/// Hermite interpolation with specified derivatives.
pub struct HermiteInterpolation {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub dys: Vec<f64>,
}

impl HermiteInterpolation {
    /// Create Hermite interpolator with derivatives.
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, dys: Vec<f64>) -> MathResult<Self> {
        if xs.len() != ys.len() || xs.len() != dys.len() || xs.len() < 2 {
            return Err(MathError::InvalidArgument("invalid data points"));
        }
        Ok(HermiteInterpolation { xs, ys, dys })
    }

    /// Evaluate Hermite polynomial at x.
    pub fn evaluate(&self, x: f64) -> f64 {
        let n = self.xs.len();
        
        // Build divided differences table with derivatives
        let mut z = Vec::new();
        let mut q = vec![vec![0.0; 2 * n]; 2 * n];
        
        for i in 0..n {
            z.push(self.xs[i]);
            z.push(self.xs[i]);
            q[2 * i][0] = self.ys[i];
            q[2 * i + 1][0] = self.ys[i];
            q[2 * i + 1][1] = self.dys[i];
            
            if i != 0 {
                q[2 * i][1] = (q[2 * i][0] - q[2 * i - 1][0]) / (z[2 * i] - z[2 * i - 1]);
            }
        }
        
        for j in 2..2 * n {
            for i in j..2 * n {
                q[i][j] = (q[i][j - 1] - q[i - 1][j - 1]) / (z[i] - z[i - j]);
            }
        }
        
        // Evaluate polynomial
        let mut result = q[0][0];
        let mut product = 1.0;
        
        for j in 1..2 * n {
            product *= x - z[j - 1];
            result += q[j][j] * product;
        }
        
        result
    }
}

/// Barycentric interpolation (numerically stable).
pub struct BarycentricInterpolation {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub weights: Vec<f64>,
}

impl BarycentricInterpolation {
    /// Create barycentric interpolator.
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> MathResult<Self> {
        if xs.len() != ys.len() || xs.len() < 2 {
            return Err(MathError::InvalidArgument("invalid data points"));
        }
        
        let n = xs.len();
        let mut weights = vec![1.0; n];
        
        for j in 0..n {
            for k in 0..n {
                if j != k {
                    weights[j] *= xs[j] - xs[k];
                }
            }
            weights[j] = 1.0 / weights[j];
        }
        
        Ok(BarycentricInterpolation { xs, ys, weights })
    }

    /// Evaluate at x.
    pub fn evaluate(&self, x: f64) -> f64 {
        if self.xs.is_empty() {
            return 0.0;
        }
        
        // Check for exact match
        for i in 0..self.xs.len() {
            if (x - self.xs[i]).abs() < 1e-15 {
                return self.ys[i];
            }
        }
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..self.xs.len() {
            let w = self.weights[i] / (x - self.xs[i]);
            numerator += w * self.ys[i];
            denominator += w;
        }
        
        numerator / denominator
    }
}

/// Radial basis function interpolation.
pub struct RBFInterpolation {
    pub xs: Vec<Vec<f64>>,
    pub ys: Vec<f64>,
    pub centers: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
    pub epsilon: f64,
}

impl RBFInterpolation {
    /// Create RBF interpolator using multiquadric basis.
    pub fn new(xs: Vec<Vec<f64>>, ys: Vec<f64>, epsilon: f64) -> MathResult<Self> {
        if xs.len() != ys.len() || xs.is_empty() {
            return Err(MathError::InvalidArgument("invalid data points"));
        }
        
        let n = xs.len();
        let _dim = xs[0].len();
        
        // Build interpolation matrix
        let mut a = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                let dist_sq: f64 = xs[i].iter()
                    .zip(&xs[j])
                    .map(|(&xi, &xj)| (xi - xj).powi(2))
                    .sum();
                a[i][j] = (1.0 + (epsilon * dist_sq).sqrt()).exp();
            }
        }
        
        // Solve linear system (simplified Gaussian elimination)
        let weights = Self::solve_linear(&a, &ys)?;
        
        Ok(RBFInterpolation {
            xs: xs.clone(),
            ys,
            centers: xs,
            weights,
            epsilon,
        })
    }

    fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> MathResult<Vec<f64>> {
        let n = a.len();
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        
        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut pivot = i;
            for j in (i + 1)..n {
                if a[j][i].abs() > a[pivot][i].abs() {
                    pivot = j;
                }
            }
            
            if a[pivot][i].abs() < 1e-15 {
                return Err(MathError::InvalidArgument("singular matrix"));
            }
            
            a.swap(i, pivot);
            b.swap(i, pivot);
            
            for j in (i + 1)..n {
                let factor = a[j][i] / a[i][i];
                for k in i..n {
                    a[j][k] -= factor * a[i][k];
                }
                b[j] -= factor * b[i];
            }
        }
        
        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= a[i][j] * x[j];
            }
            x[i] = sum / a[i][i];
        }
        
        Ok(x)
    }

    /// Evaluate RBF at x.
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        let mut result = 0.0;
        
        for i in 0..self.centers.len() {
            let dist_sq: f64 = x.iter()
                .zip(&self.centers[i])
                .map(|(&xi, &ci)| (xi - ci).powi(2))
                .sum();
            let phi = (1.0 + (self.epsilon * dist_sq).sqrt()).exp();
            result += self.weights[i] * phi;
        }
        
        result
    }
}

/// Multidimensional linear interpolation on regular grid.
pub struct MultilinearInterpolation {
    pub grid_dims: Vec<usize>,
    pub grid_min: Vec<f64>,
    pub grid_max: Vec<f64>,
    pub values: Vec<f64>,
}

impl MultilinearInterpolation {
    /// Create multilinear interpolator on regular grid.
    pub fn new(grid_dims: Vec<usize>, grid_min: Vec<f64>, grid_max: Vec<f64>, values: Vec<f64>) -> MathResult<Self> {
        let total_points: usize = grid_dims.iter().product();
        if values.len() != total_points {
            return Err(MathError::InvalidArgument("values length doesn't match grid dimensions"));
        }
        
        Ok(MultilinearInterpolation {
            grid_dims,
            grid_min,
            grid_max,
            values,
        })
    }

    /// Convert coordinates to grid indices.
    fn coords_to_indices(&self, x: &[f64]) -> Vec<usize> {
        x.iter()
            .zip(&self.grid_dims)
            .zip(&self.grid_min)
            .zip(&self.grid_max)
            .map(|((( &xi, &n), &min), &max)| {
                let normalized = (xi - min) / (max - min);
                let idx = (normalized * (n - 1) as f64).floor() as usize;
                idx.min(n - 2)
            })
            .collect()
    }

    /// Get value at grid indices.
    fn get_value(&self, indices: &[usize]) -> f64 {
        let mut index = 0;
        let mut stride = 1;
        
        for (i, &idx) in indices.iter().enumerate() {
            index += idx * stride;
            stride *= self.grid_dims[i];
        }
        
        self.values[index]
    }

    /// Evaluate at x.
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        let indices = self.coords_to_indices(x);
        let dim = x.len();
        
        if dim == 1 {
            let i = indices[0];
            let x0 = self.grid_min[0];
            let x1 = self.grid_max[0];
            let n = self.grid_dims[0];
            let t = (x[0] - x0) / (x1 - x0) * (n - 1) as f64 - i as f64;
            
            let v0 = self.get_value(&[i]);
            let v1 = self.get_value(&[i + 1]);
            
            v0 + t * (v1 - v0)
        } else if dim == 2 {
            let i = indices[0];
            let j = indices[1];
            
            let t = indices[0] as f64;
            let s = indices[1] as f64;
            
            let v00 = self.get_value(&[i, j]);
            let v01 = self.get_value(&[i, j + 1]);
            let v10 = self.get_value(&[i + 1, j]);
            let v11 = self.get_value(&[i + 1, j + 1]);
            
            let v0 = v00 + s * (v01 - v00);
            let v1 = v10 + s * (v11 - v10);
            
            v0 + t * (v1 - v0)
        } else {
            // Fallback to nearest neighbor for higher dimensions
            self.get_value(&indices)
        }
    }
}

/// Chebyshev interpolation on Chebyshev nodes.
pub struct ChebyshevInterpolation {
    pub coeffs: Vec<f64>,
    pub a: f64,
    pub b: f64,
}

impl ChebyshevInterpolation {
    /// Create Chebyshev interpolator from function.
    pub fn from_function(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> Self {
        let mut coeffs = vec![0.0; n];
        
        for i in 0..n {
            let theta = core::f64::consts::PI * (i as f64 + 0.5) / n as f64;
            let x = (a + b) / 2.0 + (b - a) / 2.0 * theta.cos();
            let y = f(x);
            
            for j in 0..n {
                coeffs[j] += y * (core::f64::consts::PI * j as f64 * (i as f64 + 0.5) / n as f64).cos();
            }
        }
        
        for coeff in &mut coeffs {
            *coeff /= n as f64;
        }
        
        coeffs[0] /= 2.0;
        
        ChebyshevInterpolation { coeffs, a, b }
    }

    /// Evaluate at x.
    pub fn evaluate(&self, x: f64) -> f64 {
        // Map x to [-1, 1]
        let z = 2.0 * (x - self.a) / (self.b - self.a) - 1.0;
        
        let mut result = 0.0;
        
        for (j, &coeff) in self.coeffs.iter().enumerate() {
            result += coeff * (j as f64 * z.acos()).cos();
        }
        
        result
    }
}

/// Nearest neighbor interpolation.
pub struct NearestNeighbor {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

impl NearestNeighbor {
    /// Create nearest neighbor interpolator.
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> MathResult<Self> {
        if xs.len() != ys.len() || xs.is_empty() {
            return Err(MathError::InvalidArgument("invalid data points"));
        }
        Ok(NearestNeighbor { xs, ys })
    }

    /// Evaluate at x.
    pub fn evaluate(&self, x: f64) -> f64 {
        let idx = self.xs.partition_point(|&v| v <= x);
        
        if idx == 0 {
            return self.ys[0];
        }
        if idx >= self.xs.len() {
            return self.ys[self.ys.len() - 1];
        }
        
        // Find nearest
        let dist_left = (x - self.xs[idx - 1]).abs();
        let dist_right = (self.xs[idx] - x).abs();
        
        if dist_left < dist_right {
            self.ys[idx - 1]
        } else {
            self.ys[idx]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_spline() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 4.0, 9.0]; // y = x²
        
        let spline = CubicSpline::new(xs, ys).unwrap();
        let result = spline.evaluate(1.5);
        
        assert!((result - 2.25).abs() < 0.1);
    }

    #[test]
    fn test_hermite() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let dys = vec![0.0, 2.0]; // y = x²
        
        let hermite = HermiteInterpolation::new(xs, ys, dys).unwrap();
        let result = hermite.evaluate(0.5);
        
        assert!((result - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_barycentric() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 4.0, 9.0];
        
        let bary = BarycentricInterpolation::new(xs, ys).unwrap();
        let result = bary.evaluate(1.5);
        
        assert!((result - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev() {
        let f = |x: f64| x * x;
        let cheb = ChebyshevInterpolation::from_function(&f, 0.0, 1.0, 10);
        
        let result = cheb.evaluate(0.5);
        assert!((result - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_nearest_neighbor() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 10.0, 20.0, 30.0];
        
        let nn = NearestNeighbor::new(xs, ys).unwrap();
        assert_eq!(nn.evaluate(0.4), 0.0);
        assert_eq!(nn.evaluate(0.6), 10.0);
    }

    #[test]
    fn test_multilinear() {
        let grid_dims = vec![3, 3];
        let grid_min = vec![0.0, 0.0];
        let grid_max = vec![2.0, 2.0];
        let values = vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0]; // x + y
        
        let interp = MultilinearInterpolation::new(grid_dims, grid_min, grid_max, values).unwrap();
        let result = interp.evaluate(&[0.5, 0.5]);
        
        assert!((result - 1.0).abs() < 0.1);
    }
}
