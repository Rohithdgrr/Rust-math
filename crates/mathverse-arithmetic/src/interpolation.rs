//! Interpolation methods: linear, polynomial, spline, and other interpolation techniques.

use mathverse_core::error::{MathError, MathResult};

/// Linear interpolation.
pub struct LinearInterpolation;

impl LinearInterpolation {
    /// Interpolate between two points.
    pub fn interpolate(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
        if x1 == x0 {
            return y0;
        }
        
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }

    /// Interpolate from a set of points.
    pub fn interpolate_points(points: &[(f64, f64)], x: f64) -> MathResult<f64> {
        if points.is_empty() {
            return Err(MathError::InvalidArgument("empty points"));
        }
        
        if points.len() == 1 {
            return Ok(points[0].1);
        }
        
        // Find the interval containing x
        let mut idx = 0;
        for i in 0..points.len() - 1 {
            if x >= points[i].0 && x <= points[i + 1].0 {
                idx = i;
                break;
            }
        }
        
        Ok(Self::interpolate(
            points[idx].0,
            points[idx].1,
            points[idx + 1].0,
            points[idx + 1].1,
            x,
        ))
    }

    /// Extrapolate beyond the data range.
    pub fn extrapolate(points: &[(f64, f64)], x: f64) -> MathResult<f64> {
        if points.len() < 2 {
            return Err(MathError::InvalidArgument("need at least 2 points"));
        }
        
        if x < points[0].0 {
            // Extrapolate using first two points
            Ok(Self::interpolate(
                points[0].0,
                points[0].1,
                points[1].0,
                points[1].1,
                x,
            ))
        } else if x > points[points.len() - 1].0 {
            // Extrapolate using last two points
            let n = points.len();
            Ok(Self::interpolate(
                points[n - 2].0,
                points[n - 2].1,
                points[n - 1].0,
                points[n - 1].1,
                x,
            ))
        } else {
            Self::interpolate_points(points, x)
        }
    }
}

/// Polynomial interpolation.
pub struct PolynomialInterpolation;

impl PolynomialInterpolation {
    /// Lagrange interpolation.
    pub fn lagrange(points: &[(f64, f64)], x: f64) -> f64 {
        let mut result = 0.0;
        
        for (i, &(xi, yi)) in points.iter().enumerate() {
            let mut term = yi;
            
            for (j, &(xj, _)) in points.iter().enumerate() {
                if i != j {
                    term *= (x - xj) / (xi - xj);
                }
            }
            
            result += term;
        }
        
        result
    }

    /// Newton's divided differences interpolation.
    pub fn newton(points: &[(f64, f64)], x: f64) -> f64 {
        let n = points.len();
        if n == 0 {
            return 0.0;
        }
        
        // Build divided differences table
        let mut dd = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            dd[i][0] = points[i].1;
        }
        
        for j in 1..n {
            for i in 0..n - j {
                dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / (points[i + j].0 - points[i].0);
            }
        }
        
        // Evaluate polynomial
        let mut result = dd[0][0];
        let mut product = 1.0;
        
        for j in 1..n {
            product *= (x - points[j - 1].0);
            result += dd[0][j] * product;
        }
        
        result
    }

    /// Hermite interpolation (with derivatives).
    pub fn hermite(points: &[(f64, f64, f64)], x: f64) -> f64 {
        // points: (x, y, y')
        let n = points.len();
        if n == 0 {
            return 0.0;
        }
        
        // Build divided differences with derivatives
        let mut dd = vec![vec![0.0; 2 * n]; 2 * n];
        let mut x_vals = Vec::new();
        
        for i in 0..n {
            x_vals.push(points[i].0);
            x_vals.push(points[i].0);
            dd[2 * i][0] = points[i].1;
            dd[2 * i + 1][0] = points[i].1;
        }
        
        // First divided differences
        for i in 0..n {
            dd[2 * i][1] = points[i].2;
            if i < n - 1 {
                dd[2 * i + 1][1] = (points[i + 1].1 - points[i].1) / (points[i + 1].0 - points[i].0);
            }
        }
        
        // Higher order divided differences
        for j in 2..2 * n {
            for i in 0..2 * n - j {
                if x_vals[i] == x_vals[i + j] {
                    // Repeated x value, use derivative
                    dd[i][j] = points[i / 2].2 / (j as f64).powi((j - 1) as i32);
                } else {
                    dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / (x_vals[i + j] - x_vals[i]);
                }
            }
        }
        
        // Evaluate polynomial
        let mut result = dd[0][0];
        let mut product = 1.0;
        
        for j in 1..2 * n {
            product *= (x - x_vals[j - 1]);
            result += dd[0][j] * product;
        }
        
        result
    }
}

/// Spline interpolation.
pub struct SplineInterpolation;

impl SplineInterpolation {
    /// Cubic spline interpolation (natural spline).
    pub fn cubic_natural(points: &[(f64, f64)], x: f64) -> MathResult<f64> {
        let n = points.len();
        if n < 2 {
            return Err(MathError::InvalidArgument("need at least 2 points"));
        }
        
        if n == 2 {
            return Ok(LinearInterpolation::interpolate(
                points[0].0, points[0].1,
                points[1].0, points[1].1,
                x,
            ));
        }
        
        // Solve tridiagonal system for second derivatives
        let h: Vec<f64> = points.windows(2).map(|w| w[1].0 - w[0].0).collect();
        let alpha: Vec<f64> = points.windows(2)
            .map(|w| 3.0 * ((w[1].1 - w[1].0) / h[1] - (w[1].0 - w[0].0) / h[0]))
            .collect();
        
        let n_minus_1 = n - 1;
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];
        
        for i in 1..n_minus_1 {
            l[i] = 2.0 * (points[i + 1].0 - points[i - 1].0) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i];
        }
        
        l[n_minus_1] = 1.0;
        z[n_minus_1] = 0.0;
        
        let mut c = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut d = vec![0.0; n];
        
        for j in (1..n_minus_1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (points[j + 1].1 - points[j].1) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }
        
        for j in 0..n_minus_1 {
            b[j] = (points[j + 1].1 - points[j].1) / h[j] - h[j] * (2.0 * c[j] + c[j + 1]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }
        
        // Find interval
        let mut idx = 0;
        for i in 0..n - 1 {
            if x >= points[i].0 && x <= points[i + 1].0 {
                idx = i;
                break;
            }
        }
        
        let dx = x - points[idx].0;
        Ok(points[idx].1 + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx)
    }

    /// Linear spline (piecewise linear).
    pub fn linear(points: &[(f64, f64)], x: f64) -> MathResult<f64> {
        LinearInterpolation::interpolate_points(points, x)
    }

    /// Quadratic spline.
    pub fn quadratic(points: &[(f64, f64)], x: f64) -> MathResult<f64> {
        let n = points.len();
        if n < 3 {
            return Err(MathError::InvalidArgument("need at least 3 points for quadratic spline"));
        }
        
        // Find interval
        let mut idx = 0;
        for i in 0..n - 1 {
            if x >= points[i].0 && x <= points[i + 1].0 {
                idx = i;
                break;
            }
        }
        
        // Use three points for quadratic interpolation
        let start = if idx == 0 { 0 } else { idx - 1 };
        let end = (start + 3).min(n);
        
        let subset = &points[start..end];
        Ok(PolynomialInterpolation::lagrange(subset, x))
    }
}

/// Barycentric interpolation (numerically stable).
pub struct BarycentricInterpolation;

impl BarycentricInterpolation {
    /// Compute barycentric weights.
    pub fn compute_weights(points: &[(f64, f64)]) -> Vec<f64> {
        let n = points.len();
        let mut weights = vec![1.0; n];
        
        for j in 0..n {
            for k in 0..n {
                if j != k {
                    weights[j] *= points[j].0 - points[k].0;
                }
            }
            weights[j] = 1.0 / weights[j];
        }
        
        weights
    }

    /// Interpolate using barycentric formula.
    pub fn interpolate(points: &[(f64, f64)], weights: &[f64], x: f64) -> f64 {
        if points.is_empty() {
            return 0.0;
        }
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..points.len() {
            if (x - points[i].0).abs() < 1e-15 {
                return points[i].1;
            }
            
            let w = weights[i] / (x - points[i].0);
            numerator += w * points[i].1;
            denominator += w;
        }
        
        numerator / denominator
    }

    /// Fast barycentric interpolation with precomputed weights.
    pub fn interpolate_fast(points: &[(f64, f64)], x: f64) -> f64 {
        let weights = Self::compute_weights(points);
        Self::interpolate(points, &weights, x)
    }
}

/// Multidimensional interpolation.
pub struct MultidimensionalInterpolation;

impl MultidimensionalInterpolation {
    /// Bilinear interpolation on a 2D grid.
    pub fn bilinear(grid: &[Vec<f64>], x: f64, y: f64) -> MathResult<f64> {
        if grid.is_empty() || grid[0].is_empty() {
            return Err(MathError::InvalidArgument("empty grid"));
        }
        
        let rows = grid.len();
        let cols = grid[0].len();
        
        let x_idx = (x * (cols - 1) as f64).floor() as usize;
        let y_idx = (y * (rows - 1) as f64).floor() as usize;
        
        let x_frac = x * (cols - 1) as f64 - x_idx as f64;
        let y_frac = y * (rows - 1) as f64 - y_idx as f64;
        
        let x1 = x_idx.min(cols - 2);
        let x2 = x1 + 1;
        let y1 = y_idx.min(rows - 2);
        let y2 = y1 + 1;
        
        let top = LinearInterpolation::interpolate(
            x1 as f64, grid[y1][x1],
            x2 as f64, grid[y1][x2],
            x,
        );
        let bottom = LinearInterpolation::interpolate(
            x1 as f64, grid[y2][x1],
            x2 as f64, grid[y2][x2],
            x,
        );
        
        Ok(LinearInterpolation::interpolate(
            y1 as f64, top,
            y2 as f64, bottom,
            y,
        ))
    }

    /// Bicubic interpolation (simplified).
    pub fn bicubic(grid: &[Vec<f64>], x: f64, y: f64) -> MathResult<f64> {
        // Simplified: use bilinear for now
        Self::bilinear(grid, x, y)
    }

    /// Trilinear interpolation on a 3D grid.
    pub fn trilinear(grid: &[Vec<Vec<f64>>], x: f64, y: f64, z: f64) -> MathResult<f64> {
        if grid.is_empty() || grid[0].is_empty() || grid[0][0].is_empty() {
            return Err(MathError::InvalidArgument("empty grid"));
        }
        
        let depth = grid.len();
        let rows = grid[0].len();
        let cols = grid[0][0].len();
        
        let x_idx = (x * (cols - 1) as f64).floor() as usize;
        let y_idx = (y * (rows - 1) as f64).floor() as usize;
        let z_idx = (z * (depth - 1) as f64).floor() as usize;
        
        let x1 = x_idx.min(cols - 2);
        let x2 = x1 + 1;
        let y1 = y_idx.min(rows - 2);
        let y2 = y1 + 1;
        let z1 = z_idx.min(depth - 2);
        let z2 = z1 + 1;
        
        // Interpolate along x
        let c000 = grid[z1][y1][x1];
        let c001 = grid[z1][y1][x2];
        let c010 = grid[z1][y2][x1];
        let c011 = grid[z1][y2][x2];
        let c100 = grid[z2][y1][x1];
        let c101 = grid[z2][y1][x2];
        let c110 = grid[z2][y2][x1];
        let c111 = grid[z2][y2][x2];
        
        let c00 = LinearInterpolation::interpolate(x1 as f64, c000, x2 as f64, c001, x);
        let c01 = LinearInterpolation::interpolate(x1 as f64, c010, x2 as f64, c011, x);
        let c10 = LinearInterpolation::interpolate(x1 as f64, c100, x2 as f64, c101, x);
        let c11 = LinearInterpolation::interpolate(x1 as f64, c110, x2 as f64, c111, x);
        
        // Interpolate along y
        let c0 = LinearInterpolation::interpolate(y1 as f64, c00, y2 as f64, c01, y);
        let c1 = LinearInterpolation::interpolate(y1 as f64, c10, y2 as f64, c11, y);
        
        // Interpolate along z
        Ok(LinearInterpolation::interpolate(z1 as f64, c0, z2 as f64, c1, z))
    }
}

/// Interpolation utilities.
pub struct InterpolationUtils;

impl InterpolationUtils {
    /// Check if points are monotonic in x.
    pub fn is_monotonic(points: &[(f64, f64)]) -> bool {
        points.windows(2).all(|w| w[0].0 < w[1].0)
    }

    /// Sort points by x coordinate.
    pub fn sort_points(points: &mut [(f64, f64)]) {
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    /// Remove duplicate x values.
    pub fn remove_duplicates(points: &mut Vec<(f64, f64)>) {
        points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-15);
    }

    /// Estimate interpolation error.
    pub fn error_estimate(points: &[(f64, f64)], f: impl Fn(f64) -> f64, x: f64) -> f64 {
        let interpolated = PolynomialInterpolation::lagrange(points, x);
        (f(x) - interpolated).abs()
    }

    /// Adaptive interpolation (refine where error is large).
    pub fn adaptive(
        f: impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        tolerance: f64,
        max_points: usize,
    ) -> Vec<(f64, f64)> {
        let mut points = vec![(a, f(a)), (b, f(b))];
        
        while points.len() < max_points {
            let mut max_error = 0.0;
            let mut worst_idx = 0;
            
            for i in 0..points.len() - 1 {
                let mid = (points[i].0 + points[i + 1].0) / 2.0;
                let actual = f(mid);
                let interpolated = LinearInterpolation::interpolate(
                    points[i].0, points[i].1,
                    points[i + 1].0, points[i + 1].1,
                    mid,
                );
                let error = (actual - interpolated).abs();
                
                if error > max_error {
                    max_error = error;
                    worst_idx = i;
                }
            }
            
            if max_error < tolerance {
                break;
            }
            
            let mid = (points[worst_idx].0 + points[worst_idx + 1].0) / 2.0;
            points.insert(worst_idx + 1, (mid, f(mid)));
        }
        
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let result = LinearInterpolation::interpolate(0.0, 0.0, 1.0, 1.0, 0.5);
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_lagrange() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
        let result = PolynomialInterpolation::lagrange(&points, 1.5);
        
        // Should interpolate x^2
        assert!((result - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_newton() {
        let points = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)];
        let result = PolynomialInterpolation::newton(&points, 1.5);
        
        assert!(result > 2.0 && result < 4.0);
    }

    #[test]
    fn test_cubic_spline() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)];
        let result = SplineInterpolation::cubic_natural(&points, 0.5).unwrap();
        
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_barycentric() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
        let weights = BarycentricInterpolation::compute_weights(&points);
        let result = BarycentricInterpolation::interpolate(&points, &weights, 1.5);
        
        assert!((result - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_bilinear() {
        let grid = vec![
            vec![0.0, 1.0],
            vec![1.0, 2.0],
        ];
        
        let result = MultidimensionalInterpolation::bilinear(&grid, 0.5, 0.5).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_monotonic() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        assert!(InterpolationUtils::is_monotonic(&points));
        
        let points2 = vec![(1.0, 0.0), (0.0, 1.0), (2.0, 2.0)];
        assert!(!InterpolationUtils::is_monotonic(&points2));
    }
}
