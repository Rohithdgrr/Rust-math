//! Complex analysis: residues, contour integration, analytic continuation.

use crate::Complex;

/// Complex analysis operations.
pub struct ComplexAnalysis;

impl ComplexAnalysis {
    /// Compute residue of f(z) at z0 using Laurent series approximation.
    /// For simple pole: lim_{z->z0} (z - z0) * f(z)
    pub fn residue_simple_pole(
        f: &dyn Fn(Complex) -> Complex,
        z0: Complex,
        h: f64,
    ) -> Complex {
        let z = z0 + Complex::new(h, 0.0);
        (z - z0) * f(z)
    }

    /// Compute residue using derivative formula for higher order poles.
    /// For pole of order n: (1/(n-1)!) * lim_{z->z0} d^{n-1}/dz^{n-1} [(z-z0)^n * f(z)]
    pub fn residue_pole_order_n(
        f: &dyn Fn(Complex) -> Complex,
        z0: Complex,
        order: usize,
        h: f64,
    ) -> Complex {
        if order == 1 {
            return Self::residue_simple_pole(f, z0, h);
        }

        let g = |z: Complex| (z - z0).powf(order as f64) * f(z);
        let derivative = Self::nth_derivative(&g, z0, order - 1, h);
        
        // Divide by factorial
        let factorial: f64 = (1..=order - 1).map(|x| x as f64).product();
        derivative / Complex::real(factorial)
    }

    /// Numerical derivative of complex function.
    pub fn derivative(f: &dyn Fn(Complex) -> Complex, z: Complex, h: f64) -> Complex {
        let z_plus = z + Complex::new(h, 0.0);
        let z_minus = z - Complex::new(h, 0.0);
        (f(z_plus) - f(z_minus)) / Complex::new(2.0 * h, 0.0)
    }

    /// Nth derivative of complex function.
    pub fn nth_derivative(
        f: &dyn Fn(Complex) -> Complex,
        z: Complex,
        n: usize,
        h: f64,
    ) -> Complex {
        if n == 0 {
            return f(z);
        }
        if n == 1 {
            return Self::derivative(f, z, h);
        }

        // Use finite differences for higher derivatives
        let mut result = Complex::zero();
        
        for k in 0..=n {
            let coeff = Self::binomial_coefficient(n, k) as f64;
            let sign: f64 = if (n - k) % 2 == 0 { 1.0 } else { -1.0 };
            let z_k = z + Complex::new((k as f64 - n as f64 / 2.0) * h, 0.0);
            let term = f(z_k) * Complex::real(coeff * sign);
            result = result + term;
        }
        
        result / Complex::new(h.powi(n as i32), 0.0)
    }

    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        
        let mut result = 1;
        for i in 0..k.min(n - k) {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Contour integral using trapezoidal rule on a circle.
    /// ∮_C f(z) dz where C is circle centered at z0 with radius r.
    pub fn contour_integral_circle(
        f: &dyn Fn(Complex) -> Complex,
        z0: Complex,
        radius: f64,
        n: usize,
    ) -> Complex {
        let mut result = Complex::zero();
        let h = 2.0 * std::f64::consts::PI / n as f64;
        
        for k in 0..n {
            let theta = k as f64 * h;
            let z = z0 + Complex::polar(radius, theta);
            // dz = i·r·e^(iθ)·dθ, i.e. angle is θ + π/2 (same θ as z)
            let dz = Complex::polar(radius * h, theta + std::f64::consts::FRAC_PI_2);
            result = result + f(z) * dz;
        }
        
        result
    }

    /// Cauchy integral formula: f(z0) = (1/(2πi)) ∮_C f(z)/(z-z0) dz
    pub fn cauchy_integral_formula(
        f: &dyn Fn(Complex) -> Complex,
        z0: Complex,
        radius: f64,
        n: usize,
    ) -> Complex {
        let integrand = |z: Complex| f(z) / (z - z0);
        let integral = Self::contour_integral_circle(&integrand, z0, radius, n);
        integral / Complex::new(0.0, 2.0 * std::f64::consts::PI)
    }

    /// Cauchy's derivative formula: f^(n)(z0) = (n!/(2πi)) ∮_C f(z)/(z-z0)^(n+1) dz
    pub fn cauchy_derivative_formula(
        f: &dyn Fn(Complex) -> Complex,
        z0: Complex,
        n: usize,
        radius: f64,
        contour_n: usize,
    ) -> Complex {
        let integrand = |z: Complex| f(z) / (z - z0).powf((n + 1) as f64);
        let integral = Self::contour_integral_circle(&integrand, z0, radius, contour_n);
        
        let factorial: f64 = (1..=n).map(|x| x as f64).product();
        integral * Complex::real(factorial) / Complex::new(0.0, 2.0 * std::f64::consts::PI)
    }

    /// Check if function is analytic at point (using Cauchy-Riemann equations).
    pub fn is_analytic(f: &dyn Fn(Complex) -> Complex, z: Complex, h: f64) -> bool {
        let z_plus = z + Complex::new(h, 0.0);
        let z_minus = z - Complex::new(h, 0.0);
        let z_i_plus = z + Complex::new(0.0, h);
        let z_i_minus = z - Complex::new(0.0, h);
        
        let f_z_plus = f(z_plus);
        let f_z_minus = f(z_minus);
        let f_z_i_plus = f(z_i_plus);
        let f_z_i_minus = f(z_i_minus);
        
        // Partial derivatives
        let u_x = (f_z_plus.re - f_z_minus.re) / (2.0 * h);
        let u_y = (f_z_i_plus.re - f_z_i_minus.re) / (2.0 * h);
        let v_x = (f_z_plus.im - f_z_minus.im) / (2.0 * h);
        let v_y = (f_z_i_plus.im - f_z_i_minus.im) / (2.0 * h);
        
        // Cauchy-Riemann equations: u_x = v_y and u_y = -v_x
        let cr1 = (u_x - v_y).abs() < 1e-6;
        let cr2 = (u_y + v_x).abs() < 1e-6;
        
        cr1 && cr2
    }

    /// Compute Laurent series coefficients around z0.
    /// Returns (a_n for n >= 0, a_n for n < 0)
    pub fn laurent_series_coefficients(
        f: &dyn Fn(Complex) -> Complex,
        z0: Complex,
        max_positive: usize,
        max_negative: usize,
        radius: f64,
        n_points: usize,
    ) -> (Vec<Complex>, Vec<Complex>) {
        let mut positive = Vec::new();
        let mut negative = Vec::new();
        
        // Positive coefficients using Cauchy integral formula
        for n in 0..=max_positive {
            let coeff = Self::cauchy_derivative_formula(f, z0, n, radius, n_points);
            positive.push(coeff);
        }
        
        // Negative coefficients using residue formula
        for n in 1..=max_negative {
            let integrand = |z: Complex| f(z) * (z - z0).powf(n as f64 - 1.0);
            let integral = Self::contour_integral_circle(&integrand, z0, radius, n_points);
            let coeff = integral / Complex::new(0.0, 2.0 * std::f64::consts::PI);
            negative.push(coeff);
        }
        
        (positive, negative)
    }

    /// Conformal mapping: Möbius transformation.
    /// f(z) = (az + b) / (cz + d)
    pub fn mobius_transform(z: Complex, a: Complex, b: Complex, c: Complex, d: Complex) -> Complex {
        (a * z + b) / (c * z + d)
    }

    /// Inverse Möbius transformation.
    pub fn mobius_inverse(w: Complex, a: Complex, b: Complex, c: Complex, d: Complex) -> Complex {
        (d * w - b) / (a - c * w)
    }

    /// Schwarz-Christoffel mapping (simplified for polygon).
    /// Maps upper half-plane to polygon interior.
    pub fn schwarz_christoffel(
        z: Complex,
        vertices: &[Complex],
        angles: &[f64],
        pre_factor: Complex,
    ) -> Complex {
        let mut integral = Complex::zero();
        let n = vertices.len();
        
        for i in 0..n {
            let alpha = angles[i];
            let vertex = vertices[i];
            let term = (z - vertex).powf(alpha - 1.0);
            integral = integral + term;
        }
        
        pre_factor * integral
    }

    /// Argument principle: (1/(2πi)) ∮ f'(z)/f(z) dz = Z - P
    /// where Z is number of zeros and P is number of poles inside contour.
    pub fn argument_principle(
        f: &dyn Fn(Complex) -> Complex,
        z0: Complex,
        radius: f64,
        n: usize,
    ) -> f64 {
        let f_prime = |z: Complex| Self::derivative(f, z, 1e-6);
        let integrand = |z: Complex| f_prime(z) / f(z);
        let integral = Self::contour_integral_circle(&integrand, z0, radius, n);
        (integral.im / (2.0 * std::f64::consts::PI)).round()
    }

    /// Rouché's theorem test: if |f(z)| > |g(z)| for every z on contour C,
    /// then f and f+g have same number of zeros inside C.
    pub fn rouches_theorem(
        f: &dyn Fn(Complex) -> Complex,
        g: &dyn Fn(Complex) -> Complex,
        z0: Complex,
        radius: f64,
        n: usize,
    ) -> bool {
        // Sufficient sampled check: min |f| > max |g| on the contour
        let mut min_f: f64 = f64::INFINITY;
        let mut max_g: f64 = 0.0;
        
        for k in 0..n {
            let theta = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            let z = z0 + Complex::polar(radius, theta);
            
            let f_val = f(z).norm();
            let g_val = g(z).norm();
            
            min_f = min_f.min(f_val);
            max_g = max_g.max(g_val);
        }
        
        min_f > max_g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_simple_pole() {
        // f(z) = 1/(z-1), residue at z=1 should be 1
        let f = |z: Complex| Complex::one() / (z - Complex::real(1.0));
        let residue = ComplexAnalysis::residue_simple_pole(&f, Complex::real(1.0), 0.001);
        
        assert!((residue.re - 1.0).abs() < 0.1);
        assert!(residue.im.abs() < 0.1);
    }

    #[test]
    fn test_derivative() {
        let f = |z: Complex| z * z;
        let z = Complex::new(2.0, 0.0);
        let deriv = ComplexAnalysis::derivative(&f, z, 1e-6);
        
        // derivative of z^2 is 2z = 4
        assert!((deriv.re - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_mobius_transform() {
        let z = Complex::real(1.0);
        let a = Complex::real(1.0);
        let b = Complex::zero();
        let c = Complex::zero();
        let d = Complex::real(1.0);
        
        let result = ComplexAnalysis::mobius_transform(z, a, b, c, d);
        assert_eq!(result, z);
    }

    #[test]
    fn test_cauchy_integral_formula() {
        // f(z) = z, evaluate at z0 = 1
        let f = |z: Complex| z;
        let z0 = Complex::real(1.0);
        let result = ComplexAnalysis::cauchy_integral_formula(&f, z0, 0.5, 100);
        
        assert!((result.re - 1.0).abs() < 0.1);
        assert!(result.im.abs() < 0.1);
    }

    #[test]
    fn test_is_analytic() {
        // f(z) = z^2 is analytic everywhere
        let f = |z: Complex| z * z;
        let z = Complex::new(1.0, 1.0);
        
        assert!(ComplexAnalysis::is_analytic(&f, z, 1e-6));
    }

    #[test]
    fn test_contour_integral_z_dz_is_zero() {
        // ∮ z dz around unit circle = 0 (regression: dz was missing the i·e^(iθ) factor)
        let f = |z: Complex| z;
        let result = ComplexAnalysis::contour_integral_circle(&f, Complex::zero(), 1.0, 1000);
        assert!(result.norm() < 1e-10);
    }

    #[test]
    fn test_contour_integral_1_over_z() {
        // ∮ 1/z dz around unit circle = 2πi
        let f = |z: Complex| Complex::one() / z;
        let result = ComplexAnalysis::contour_integral_circle(&f, Complex::zero(), 1.0, 1000);
        let expected = Complex::new(0.0, 2.0 * std::f64::consts::PI);
        assert!((result - expected).norm() < 1e-8);
    }

    #[test]
    fn test_cauchy_integral_formula_quadratic() {
        // f(z) = z² at z0 = 1 should give 1
        let f = |z: Complex| z * z;
        let result = ComplexAnalysis::cauchy_integral_formula(&f, Complex::real(1.0), 0.5, 200);
        assert!((result.re - 1.0).abs() < 1e-6);
        assert!(result.im.abs() < 1e-6);
    }

    #[test]
    fn test_nth_derivative_second() {
        // d²/dz² z³ = 6z; at z=1 that's 6 (regression: term signs were wrong)
        let f = |z: Complex| z * z * z;
        let deriv = ComplexAnalysis::nth_derivative(&f, Complex::real(1.0), 2, 0.01);
        assert!((deriv.re - 6.0).abs() < 1e-6);
        assert!(deriv.im.abs() < 1e-6);
    }

    #[test]
    fn test_mobius_inverse_round_trip() {
        // Non-unit determinant: (2z+1)/(3z+5), det = 2·5 - 1·3 = 7
        let a = Complex::real(2.0);
        let b = Complex::real(1.0);
        let c = Complex::real(3.0);
        let d = Complex::real(5.0);

        let z = Complex::new(1.0, 1.0);
        let w = ComplexAnalysis::mobius_transform(z, a, b, c, d);
        let back = ComplexAnalysis::mobius_inverse(w, a, b, c, d);
        assert!((back - z).norm() < 1e-12);
    }

    #[test]
    fn test_rouches_theorem() {
        // |z| > 0.5 on unit circle: true
        let f = |z: Complex| z;
        let g = |_z: Complex| Complex::real(0.5);
        assert!(ComplexAnalysis::rouches_theorem(&f, &g, Complex::zero(), 1.0, 100));

        // |z| < 2 on unit circle: false (regression: max-vs-max check gave a false positive)
        let h = |_z: Complex| Complex::real(2.0);
        assert!(!ComplexAnalysis::rouches_theorem(&f, &h, Complex::zero(), 1.0, 100));
    }
}
