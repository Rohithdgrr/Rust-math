//! Numerical integration: Gaussian quadrature, Romberg, adaptive methods.

use mathverse_core::error::{MathError, MathResult};

/// Gaussian quadrature with Legendre polynomials.
pub struct GaussianQuadrature;

impl GaussianQuadrature {
    /// Integrate using Gaussian-Legendre quadrature.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let (nodes, weights) = Self::legendre_nodes_weights(n);
        
        let mut sum = 0.0;
        for (&x, &w) in nodes.iter().zip(weights.iter()) {
            let transformed = 0.5 * (b - a) * x + 0.5 * (a + b);
            sum += w * f(transformed);
        }
        
        0.5 * (b - a) * sum
    }

    /// Get Legendre polynomial nodes and weights.
    fn legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
        match n {
            1 => (vec![0.0], vec![2.0]),
            2 => (vec![-0.5773502691896257, 0.5773502691896257], vec![1.0, 1.0]),
            3 => (vec![-0.7745966692414834, 0.0, 0.7745966692414834],
                vec![0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
            4 => (vec![-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
                vec![0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539]),
            5 => (vec![-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
                vec![0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]),
            6 => (vec![-0.9324695142031521, -0.6612093864662645, -0.2386191860831970, 0.2386191860831970, 0.6612093864662645, 0.9324695142031521],
                vec![0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 0.4679139345726910, 0.3607615730481386, 0.1713244923791704]),
            7 => (vec![-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585],
                vec![0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.2797053914892766, 0.1294849661688697]),
            8 => (vec![-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498, 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363],
                vec![0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763]),
            _ => {
                // Fallback to trapezoidal for higher n
                let nodes: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64).collect();
                let weights = vec![2.0 / n as f64; n];
                (nodes, weights)
            }
        }
    }

    /// 2D Gaussian quadrature.
    pub fn integrate_2d(f: &dyn Fn(f64, f64) -> f64, a: f64, b: f64, c: f64, d: f64, n: usize) -> f64 {
        let (nodes, weights) = Self::legendre_nodes_weights(n);
        
        let mut sum = 0.0;
        for (&xi, &wi) in nodes.iter().zip(weights.iter()) {
            let x = 0.5 * (b - a) * xi + 0.5 * (a + b);
            
            for (&yj, &wj) in nodes.iter().zip(weights.iter()) {
                let y = 0.5 * (d - c) * yj + 0.5 * (c + d);
                sum += wi * wj * f(x, y);
            }
        }
        
        0.25 * (b - a) * (d - c) * sum
    }
}

/// Romberg integration (Richardson extrapolation).
pub struct RombergIntegration;

impl RombergIntegration {
    /// Integrate using Romberg method.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, max_levels: usize) -> f64 {
        let mut r = vec![vec![0.0; max_levels]; max_levels];
        
        // First column: trapezoidal rule with increasing subdivisions
        for k in 0..max_levels {
            let n = 2_usize.pow(k as u32);
            r[k][0] = Self::trapezoidal(f, a, b, n);
        }
        
        // Richardson extrapolation
        for j in 1..max_levels {
            for k in j..max_levels {
                let factor = 4.0_f64.powi(j as i32);
                r[k][j] = (factor * r[k][j - 1] - r[k - 1][j - 1]) / (factor - 1.0);
            }
        }
        
        r[max_levels - 1][max_levels - 1]
    }

    fn trapezoidal(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        
        for i in 1..n {
            let x = a + i as f64 * h;
            sum += f(x);
        }
        
        sum * h
    }
}

/// Adaptive Simpson's rule.
pub struct AdaptiveSimpson;

impl AdaptiveSimpson {
    /// Integrate with adaptive Simpson's rule.
    pub fn integrate(
        f: &dyn Fn(f64) -> f64,
        a: f64,
        b: f64,
        tolerance: f64,
        max_depth: usize,
    ) -> f64 {
        Self::adaptive_simpson_recursive(f, a, b, tolerance, max_depth, f(a), f(b), f((a + b) / 2.0))
    }

    fn adaptive_simpson_recursive(
        f: &dyn Fn(f64) -> f64,
        a: f64,
        b: f64,
        tolerance: f64,
        depth: usize,
        fa: f64,
        fb: f64,
        fm: f64,
    ) -> f64 {
        let h = b - a;
        let m = (a + b) / 2.0;
        
        let whole = h / 6.0 * (fa + 4.0 * fm + fb);
        
        let fl = f((a + m) / 2.0);
        let fr = f((m + b) / 2.0);
        
        let left = h / 12.0 * (fa + 4.0 * fl + fm);
        let right = h / 12.0 * (fm + 4.0 * fr + fb);
        
        if depth == 0 || (left + right - whole).abs() < 15.0 * tolerance {
            left + right + (left + right - whole) / 15.0
        } else {
            Self::adaptive_simpson_recursive(f, a, m, tolerance / 2.0, depth - 1, fa, fm, fl)
                + Self::adaptive_simpson_recursive(f, m, b, tolerance / 2.0, depth - 1, fm, fb, fr)
        }
    }
}

/// Monte Carlo integration.
pub struct MonteCarloIntegration;

impl MonteCarloIntegration {
    /// Integrate using Monte Carlo method.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, samples: usize) -> (f64, f64) {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..samples {
            let x = a + (b - a) * rand::random::<f64>();
            let y = f(x);
            sum += y;
            sum_sq += y * y;
        }
        
        let mean = sum / samples as f64;
        let variance = (sum_sq / samples as f64 - mean * mean).max(0.0);
        let error = (variance / samples as f64).sqrt() * (b - a);
        
        (mean * (b - a), error)
    }

    /// 2D Monte Carlo integration.
    pub fn integrate_2d(f: &dyn Fn(f64, f64) -> f64, a: f64, b: f64, c: f64, d: f64, samples: usize) -> (f64, f64) {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..samples {
            let x = a + (b - a) * rand::random::<f64>();
            let y = c + (d - c) * rand::random::<f64>();
            let z = f(x, y);
            sum += z;
            sum_sq += z * z;
        }
        
        let mean = sum / samples as f64;
        let variance = (sum_sq / samples as f64 - mean * mean).max(0.0);
        let error = (variance / samples as f64).sqrt() * (b - a) * (d - c);
        
        (mean * (b - a) * (d - c), error)
    }
}

/// Simpson's rule with fixed number of intervals.
pub struct SimpsonRule;

impl SimpsonRule {
    /// Integrate using Simpson's rule.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        if n % 2 != 0 {
            return Self::trapezoidal(f, a, b, n);
        }
        
        let h = (b - a) / n as f64;
        let mut sum = f(a) + f(b);
        
        for i in 1..n {
            let x = a + i as f64 * h;
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += weight * f(x);
        }
        
        sum * h / 3.0
    }

    fn trapezoidal(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        
        for i in 1..n {
            let x = a + i as f64 * h;
            sum += f(x);
        }
        
        sum * h
    }
}

/// Midpoint rule.
pub struct MidpointRule;

impl MidpointRule {
    /// Integrate using midpoint rule.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.0;
        
        for i in 0..n {
            let x = a + (i as f64 + 0.5) * h;
            sum += f(x);
        }
        
        sum * h
    }
}

/// Boole's rule (4th order Newton-Cotes).
pub struct BooleRule;

impl BooleRule {
    /// Integrate using Boole's rule.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        if n % 4 != 0 {
            return SimpsonRule::integrate(f, a, b, n);
        }
        
        let h = (b - a) / n as f64;
        let mut sum = 0.0;
        
        for i in 0..n / 4 {
            let x0 = a + (4 * i) as f64 * h;
            let x1 = a + (4 * i + 1) as f64 * h;
            let x2 = a + (4 * i + 2) as f64 * h;
            let x3 = a + (4 * i + 3) as f64 * h;
            let x4 = a + (4 * i + 4) as f64 * h;
            
            sum += 7.0 * f(x0) + 32.0 * f(x1) + 12.0 * f(x2) + 32.0 * f(x3) + 7.0 * f(x4);
        }
        
        sum * 2.0 * h / 45.0
    }
}

/// Clenshaw-Curtis quadrature.
pub struct ClenshawCurtis;

impl ClenshawCurtis {
    /// Integrate using Clenshaw-Curtis quadrature.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let mut weights = vec![0.0; n];
        let mut nodes = vec![0.0; n];
        
        for i in 0..n {
            nodes[i] = (core::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
        }
        
        // Clenshaw-Curtis weights
        for k in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if k == 0 || k == n - 1 {
                    sum += 2.0 / (n - 1) as f64;
                } else {
                    sum += 2.0 * (2.0 * core::f64::consts::PI * j as f64 / (n - 1) as f64).cos()
                        * (core::f64::consts::PI * k as f64 * j as f64 / (n - 1) as f64).cos()
                        / (n - 1) as f64;
                }
            }
            weights[k] = sum / n as f64;
        }
        
        let mut sum = 0.0;
        for i in 0..n {
            let x = 0.5 * (b - a) * nodes[i] + 0.5 * (a + b);
            sum += weights[i] * f(x);
        }
        
        0.5 * (b - a) * sum
    }
}

/// Double exponential integration (tanh-sinh quadrature).
pub struct DoubleExponential;

impl DoubleExponential {
    /// Integrate using double exponential quadrature.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = 1.0 / n as f64;
        let mut sum = 0.0;
        
        for k in -(n as isize)..=(n as isize) {
            let t = k as f64 * h;
            let phi = core::f64::consts::PI / 2.0 * t.sinh();
            let x = (b + a) / 2.0 + (b - a) / 2.0 * phi.tanh();
            
            let weight = core::f64::consts::PI / 2.0 * phi.cosh();
            let jacobian = (b - a) / 2.0 * weight;
            
            sum += jacobian * f(x);
        }
        
        sum * h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_quadrature() {
        let result = GaussianQuadrature::integrate(&|x| x * x, -1.0, 1.0, 5);
        assert!((result - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_2d() {
        let result = GaussianQuadrature::integrate_2d(&|x, y| x * y, 0.0, 1.0, 0.0, 1.0, 5);
        assert!((result - 0.25).abs() < 1e-8);
    }

    #[test]
    fn test_romberg() {
        let result = RombergIntegration::integrate(&|x| x * x, 0.0, 1.0, 8);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_simpson() {
        let result = AdaptiveSimpson::integrate(&|x| x * x, 0.0, 1.0, 1e-10, 20);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_monte_carlo() {
        let (result, error) = MonteCarloIntegration::integrate(&|x| x * x, 0.0, 1.0, 10000);
        assert!((result - 1.0 / 3.0).abs() < 0.01);
        assert!(error < 0.01);
    }

    #[test]
    fn test_simpson_rule() {
        let result = SimpsonRule::integrate(&|x| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint_rule() {
        let result = MidpointRule::integrate(&|x| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_boole_rule() {
        let result = BooleRule::integrate(&|x| x * x, 0.0, 1.0, 8);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_clenshaw_curtis() {
        let result = ClenshawCurtis::integrate(&|x| x * x, -1.0, 1.0, 10);
        assert!((result - 2.0 / 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_double_exponential() {
        let result = DoubleExponential::integrate(&|x| x * x, 0.0, 1.0, 10);
        assert!((result - 1.0 / 3.0).abs() < 1e-6);
    }
}
