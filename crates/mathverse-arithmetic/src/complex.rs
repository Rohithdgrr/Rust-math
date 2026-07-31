//! Complex numbers: arithmetic, polar form, complex functions.

use mathverse_core::error::{MathError, MathResult};

/// Complex number representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    /// Create new complex number.
    pub fn new(real: f64, imag: f64) -> Self {
        Complex { real, imag }
    }

    /// Create from polar coordinates.
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Complex {
            real: r * theta.cos(),
            imag: r * theta.sin(),
        }
    }

    /// Zero complex number.
    pub fn zero() -> Self {
        Complex { real: 0.0, imag: 0.0 }
    }

    /// One complex number.
    pub fn one() -> Self {
        Complex { real: 1.0, imag: 0.0 }
    }

    /// Imaginary unit i.
    pub fn i() -> Self {
        Complex { real: 0.0, imag: 1.0 }
    }

    /// Get magnitude (modulus).
    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    /// Get magnitude squared.
    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    /// Get argument (angle).
    pub fn argument(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    /// Get complex conjugate.
    pub fn conjugate(&self) -> Self {
        Complex {
            real: self.real,
            imag: -self.imag,
        }
    }

    /// Convert to polar form (r, theta).
    pub fn to_polar(&self) -> (f64, f64) {
        (self.magnitude(), self.argument())
    }

    /// Add two complex numbers.
    pub fn add(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }

    /// Subtract two complex numbers.
    pub fn sub(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }

    /// Multiply two complex numbers.
    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }

    /// Divide two complex numbers.
    pub fn div(&self, other: &Complex) -> MathResult<Complex> {
        let denom = other.magnitude_squared();
        if denom == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        let conj = other.conjugate();
        let num = self.mul(&conj);
        
        Ok(Complex {
            real: num.real / denom,
            imag: num.imag / denom,
        })
    }

    /// Scale by real number.
    pub fn scale(&self, scalar: f64) -> Complex {
        Complex {
            real: self.real * scalar,
            imag: self.imag * scalar,
        }
    }

    /// Complex power: z^n for integer n.
    pub fn pow(&self, n: i32) -> Complex {
        if n == 0 {
            return Complex::one();
        }
        
        let mut result = Complex::one();
        let mut base = *self;
        let mut exp = n.abs();
        
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }
        
        if n < 0 {
            result = Complex::one().div(&result).unwrap();
        }
        
        result
    }

    /// Complex exponential: e^z.
    pub fn exp(&self) -> Complex {
        let e_real = self.real.exp();
        Complex {
            real: e_real * self.imag.cos(),
            imag: e_real * self.imag.sin(),
        }
    }

    /// Natural logarithm: ln(z).
    pub fn ln(&self) -> MathResult<Complex> {
        if self.real == 0.0 && self.imag == 0.0 {
            return Err(MathError::InvalidArgument("logarithm of zero"));
        }
        
        let (r, theta) = self.to_polar();
        Ok(Complex {
            real: r.ln(),
            imag: theta,
        })
    }

    /// Logarithm with base b.
    pub fn log(&self, base: f64) -> MathResult<Complex> {
        let ln_z = self.ln()?;
        let ln_b = base.ln();
        Ok(ln_z.scale(1.0 / ln_b))
    }

    /// Complex square root.
    pub fn sqrt(&self) -> Complex {
        let (r, theta) = self.to_polar();
        let sqrt_r = r.sqrt();
        Complex::from_polar(sqrt_r, theta / 2.0)
    }

    /// Complex sine.
    pub fn sin(&self) -> Complex {
        Complex {
            real: self.real.sin() * self.imag.cosh(),
            imag: self.real.cos() * self.imag.sinh(),
        }
    }

    /// Complex cosine.
    pub fn cos(&self) -> Complex {
        Complex {
            real: self.real.cos() * self.imag.cosh(),
            imag: -self.real.sin() * self.imag.sinh(),
        }
    }

    /// Complex tangent.
    pub fn tan(&self) -> Complex {
        let sin_z = self.sin();
        let cos_z = self.cos();
        sin_z.div(&cos_z).unwrap()
    }

    /// Complex hyperbolic sine.
    pub fn sinh(&self) -> Complex {
        Complex {
            real: self.real.sinh() * self.imag.cos(),
            imag: self.real.cosh() * self.imag.sin(),
        }
    }

    /// Complex hyperbolic cosine.
    pub fn cosh(&self) -> Complex {
        Complex {
            real: self.real.cosh() * self.imag.cos(),
            imag: self.real.sinh() * self.imag.sin(),
        }
    }

    /// Complex hyperbolic tangent.
    pub fn tanh(&self) -> Complex {
        let sinh_z = self.sinh();
        let cosh_z = self.cosh();
        sinh_z.div(&cosh_z).unwrap()
    }

    /// Check if real.
    pub fn is_real(&self) -> bool {
        self.imag.abs() < 1e-15
    }

    /// Check if imaginary.
    pub fn is_imaginary(&self) -> bool {
        self.real.abs() < 1e-15
    }

    /// Check if pure real (imaginary part exactly zero).
    pub fn is_pure_real(&self) -> bool {
        self.imag == 0.0
    }

    /// Check if pure imaginary (real part exactly zero).
    pub fn is_pure_imaginary(&self) -> bool {
        self.real == 0.0
    }
}

/// Complex number operations.
pub struct ComplexOps;

impl ComplexOps {
    /// Compute nth roots of unity.
    pub fn roots_of_unity(n: u32) -> Vec<Complex> {
        (0..n).map(|k| {
            let theta = 2.0 * core::f64::consts::PI * k as f64 / n as f64;
            Complex::from_polar(1.0, theta)
        }).collect()
    }

    /// Compute all nth roots of a complex number.
    pub fn nth_roots(z: &Complex, n: u32) -> Vec<Complex> {
        let (r, theta) = z.to_polar();
        let r_root = r.powf(1.0 / n as f64);
        
        (0..n).map(|k| {
            let theta_k = (theta + 2.0 * core::f64::consts::PI * k as f64) / n as f64;
            Complex::from_polar(r_root, theta_k)
        }).collect()
    }

    /// De Moivre's theorem: (r(cos θ + i sin θ))^n = r^n(cos nθ + i sin nθ).
    pub fn de_moivre(r: f64, theta: f64, n: i32) -> Complex {
        Complex::from_polar(r.powi(n), theta * n as f64)
    }

    /// Euler's formula: e^(iθ) = cos θ + i sin θ.
    pub fn euler(theta: f64) -> Complex {
        Complex {
            real: theta.cos(),
            imag: theta.sin(),
        }
    }

    /// Check if two complex numbers are approximately equal.
    pub fn almost_equal(a: &Complex, b: &Complex, tolerance: f64) -> bool {
        (a.real - b.real).abs() < tolerance && (a.imag - b.imag).abs() < tolerance
    }

    /// Compute distance between two complex numbers.
    pub fn distance(a: &Complex, b: &Complex) -> f64 {
        a.sub(b).magnitude()
    }

    /// Complex dot product (real part of a * conjugate(b)).
    pub fn dot(a: &Complex, b: &Complex) -> f64 {
        a.mul(&b.conjugate()).real
    }

    /// Complex cross product (imaginary part of a * conjugate(b)).
    pub fn cross(a: &Complex, b: &Complex) -> f64 {
        a.mul(&b.conjugate()).imag
    }
}

/// Complex analysis functions.
pub struct ComplexAnalysis;

impl ComplexAnalysis {
    /// Complex derivative of f(z) = z^n.
    pub fn derivative_power(n: i32, z: &Complex) -> Complex {
        z.pow(n - 1).scale(n as f64)
    }

    /// Residue theorem (simplified for simple poles).
    pub fn residue_simple_pole(f: impl Fn(&Complex) -> Complex, z0: &Complex, h: f64) -> Complex {
        let z1 = Complex::new(z0.real + h, z0.imag);
        let z2 = Complex::new(z0.real - h, z0.imag);
        
        let f1 = f(&z1);
        let f2 = f(&z2);
        
        f1.sub(&f2).scale(1.0 / (2.0 * h))
    }

    /// Cauchy integral formula (simplified numerical approximation).
    pub fn cauchy_integral(
        f: impl Fn(&Complex) -> Complex,
        z0: &Complex,
        radius: f64,
        n_points: usize,
    ) -> Complex {
        let mut sum = Complex::zero();
        
        for k in 0..n_points {
            let theta = 2.0 * core::f64::consts::PI * k as f64 / n_points as f64;
            let z = Complex::from_polar(radius, theta).add(z0);
            let dz = Complex::from_polar(radius, theta + core::f64::consts::PI / n_points as f64)
                .add(z0)
                .sub(&z);
            
            let f_z = f(&z);
            let term = f_z.mul(&dz);
            sum = sum.add(&term);
        }
        
        sum.scale(1.0 / (2.0 * core::f64::consts::PI * radius))
    }

    /// Analytic continuation (simplified).
    pub fn analytic_continuation(
        f: impl Fn(&Complex) -> Complex,
        z_start: &Complex,
        direction: &Complex,
        steps: usize,
        step_size: f64,
    ) -> Complex {
        let mut z = *z_start;
        
        for _ in 0..steps {
            z = z.add(&direction.scale(step_size));
        }
        
        f(&z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_creation() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.real, 3.0);
        assert_eq!(z.imag, 4.0);
    }

    #[test]
    fn test_complex_polar() {
        let z = Complex::from_polar(5.0, 0.927295218); // 3 + 4i
        assert!((z.real - 3.0).abs() < 1e-6);
        assert!((z.imag - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_magnitude() {
        let z = Complex::new(3.0, 4.0);
        assert!((z.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_conjugate() {
        let z = Complex::new(3.0, 4.0);
        let conj = z.conjugate();
        assert_eq!(conj.real, 3.0);
        assert_eq!(conj.imag, -4.0);
    }

    #[test]
    fn test_complex_arithmetic() {
        let z1 = Complex::new(1.0, 2.0);
        let z2 = Complex::new(3.0, 4.0);
        
        let sum = z1.add(&z2);
        assert_eq!(sum.real, 4.0);
        assert_eq!(sum.imag, 6.0);
        
        let product = z1.mul(&z2);
        assert_eq!(product.real, -5.0); // 1*3 - 2*4
        assert_eq!(product.imag, 10.0); // 1*4 + 2*3
    }

    #[test]
    fn test_complex_division() {
        let z1 = Complex::new(1.0, 2.0);
        let z2 = Complex::new(3.0, 4.0);
        
        let quotient = z1.div(&z2).unwrap();
        // (1+2i)/(3+4i) = (1+2i)(3-4i)/(3^2+4^2) = (11+2i)/25
        assert!((quotient.real - 0.44).abs() < 1e-10);
        assert!((quotient.imag - 0.08).abs() < 1e-10);
    }

    #[test]
    fn test_complex_exp() {
        let z = Complex::new(0.0, core::f64::consts::PI);
        let result = z.exp();
        assert!((result.real + 1.0).abs() < 1e-10);
        assert!(result.imag.abs() < 1e-10);
    }

    #[test]
    fn test_complex_sqrt() {
        let z = Complex::new(3.0, 4.0);
        let sqrt_z = z.sqrt();
        let squared = sqrt_z.mul(&sqrt_z);
        assert!(ComplexOps::almost_equal(&z, &squared, 1e-10));
    }

    #[test]
    fn test_roots_of_unity() {
        let roots = ComplexOps::roots_of_unity(4);
        assert_eq!(roots.len(), 4);
        
        // Check that each root raised to 4 equals 1
        for root in &roots {
            let power = root.pow(4);
            assert!(ComplexOps::almost_equal(&power, &Complex::one(), 1e-10));
        }
    }

    #[test]
    fn test_de_moivre() {
        let result = ComplexOps::de_moivre(1.0, core::f64::consts::PI / 4.0, 2);
        let expected = Complex::new(0.0, 1.0); // (cos π/4 + i sin π/4)^2 = i
        assert!(ComplexOps::almost_equal(&result, &expected, 1e-10));
    }

    #[test]
    fn test_euler() {
        let result = ComplexOps::euler(core::f64::consts::PI);
        assert!((result.real + 1.0).abs() < 1e-10);
        assert!(result.imag.abs() < 1e-10);
    }
}
