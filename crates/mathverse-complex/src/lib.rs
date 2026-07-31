//! Complex numbers over `f64`.
//!
//! Division by zero and `ln(0)` etc. yield `NaN`/`inf` components, mirroring
//! std float semantics (no panics).

use core::ops::{Add, Div, Mul, Neg, Sub};

pub mod analysis;
pub mod special_functions;
pub mod matrix;

pub use analysis::ComplexAnalysis;
pub use special_functions::ComplexSpecialFunctions;
pub use matrix::ComplexMatrix;

/// Complex number `re + im·i`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }
    pub fn real(r: f64) -> Complex {
        Complex { re: r, im: 0.0 }
    }
    pub fn i() -> Complex {
        Complex { re: 0.0, im: 1.0 }
    }
    /// Zero constant (0 + 0i).
    pub fn zero() -> Complex {
        Complex { re: 0.0, im: 0.0 }
    }
    /// One constant (1 + 0i).
    pub fn one() -> Complex {
        Complex { re: 1.0, im: 0.0 }
    }
    /// From polar form `r·e^(iθ)`.
    pub fn polar(r: f64, theta: f64) -> Complex {
        Complex::new(r * theta.cos(), r * theta.sin())
    }
    pub fn conjugate(&self) -> Complex {
        Complex::new(self.re, -self.im)
    }
    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }
    /// Principal argument in `(-π, π]`.
    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }
    /// `(r, θ)` such that `self == r·e^(iθ)`.
    pub fn to_polar(&self) -> (f64, f64) {
        (self.norm(), self.arg())
    }
    pub fn is_zero(&self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
    /// Principal square root.
    ///
    /// ```
    /// use mathverse_complex::Complex;
    /// let s = Complex::new(-4.0, 0.0).sqrt();
    /// assert!((s.im - 2.0).abs() < 1e-12 && s.re.abs() < 1e-12);
    /// ```
    pub fn sqrt(&self) -> Complex {
        let (r, theta) = self.to_polar();
        Complex::polar(r.sqrt(), theta / 2.0)
    }
    /// `e^self`.
    ///
    /// ```
    /// use mathverse_complex::Complex;
    /// let e = Complex::new(0.0, core::f64::consts::PI).exp();
    /// assert!((e.re + 1.0).abs() < 1e-12 && e.im.abs() < 1e-12);
    /// ```
    pub fn exp(&self) -> Complex {
        let e = self.re.exp();
        Complex::new(e * self.im.cos(), e * self.im.sin())
    }
    /// Principal natural logarithm.
    pub fn ln(&self) -> Complex {
        Complex::new(self.norm().ln(), self.arg())
    }
    /// `self^p = e^(p·ln self)`, principal branch.
    pub fn pow(&self, p: Complex) -> Complex {
        (p * self.ln()).exp()
    }
    /// `self^p` for real exponent.
    pub fn powf(&self, p: f64) -> Complex {
        self.pow(Complex::real(p))
    }
    pub fn sin(&self) -> Complex {
        Complex::new(self.re.sin() * self.im.cosh(), self.re.cos() * self.im.sinh())
    }
    pub fn cos(&self) -> Complex {
        Complex::new(self.re.cos() * self.im.cosh(), -self.re.sin() * self.im.sinh())
    }
    pub fn tan(&self) -> Complex {
        self.sin() / self.cos()
    }
    /// Hyperbolic sine.
    pub fn sinh(&self) -> Complex {
        Complex::new(self.re.sinh() * self.im.cos(), self.re.cosh() * self.im.sin())
    }
    /// Hyperbolic cosine.
    pub fn cosh(&self) -> Complex {
        Complex::new(self.re.cosh() * self.im.cos(), self.re.sinh() * self.im.sin())
    }
    /// Hyperbolic tangent.
    pub fn tanh(&self) -> Complex {
        self.sinh() / self.cosh()
    }
    /// Inverse hyperbolic sine.
    pub fn asinh(&self) -> Complex {
        let s = *self;
        (s + (s * s + Complex::real(1.0)).sqrt()).ln()
    }
    /// Inverse hyperbolic cosine.
    pub fn acosh(&self) -> Complex {
        let s = *self;
        (s + (s + Complex::real(1.0)).sqrt() * (s - Complex::real(1.0)).sqrt()).ln()
    }
    /// Inverse hyperbolic tangent.
    pub fn atanh(&self) -> Complex {
        let s = *self;
        ((Complex::real(1.0) + s) / (Complex::real(1.0) - s)).ln() / Complex::real(2.0)
    }
    /// Inverse sine.
    pub fn asin(&self) -> Complex {
        let s = *self;
        -Complex::i() * (Complex::i() * s + (Complex::real(1.0) - s * s).sqrt()).ln()
    }
    /// Inverse cosine.
    pub fn acos(&self) -> Complex {
        let s = *self;
        Complex::i() * (s + Complex::i() * (Complex::real(1.0) - s * s).sqrt()).ln()
    }
    /// Inverse tangent.
    pub fn atan(&self) -> Complex {
        let s = *self;
        let i = Complex::i();
        (i * (Complex::real(1.0) - i * s) / (Complex::real(1.0) + i * s)).ln() / (Complex::real(2.0) * i)
    }
    /// Base-10 logarithm.
    pub fn log10(&self) -> Complex {
        self.ln() / Complex::real(10.0_f64.ln())
    }
    /// Base-2 logarithm.
    pub fn log2(&self) -> Complex {
        self.ln() / Complex::real(2.0_f64.ln())
    }
    /// Reciprocal (1/self).
    pub fn recip(&self) -> Complex {
        let d = self.norm_sq();
        Complex::new(self.re / d, -self.im / d)
    }
    /// Returns true if either component is NaN.
    pub fn is_nan(&self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    /// Returns true if either component is infinite.
    pub fn is_infinite(&self) -> bool {
        self.re.is_infinite() || self.im.is_infinite()
    }
    /// Returns true if both components are finite.
    pub fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    /// Signum (sign) function: returns a complex number with same direction but unit magnitude.
    /// Returns zero if self is zero.
    pub fn signum(&self) -> Complex {
        if self.is_zero() {
            Complex::new(0.0, 0.0)
        } else {
            let s = *self;
            s / s.norm()
        }
    }
    /// Cube root.
    pub fn cbrt(&self) -> Complex {
        self.powf(1.0 / 3.0)
    }
    /// Nth root (principal branch).
    pub fn root(&self, n: f64) -> Complex {
        self.powf(1.0 / n)
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for Complex {
    type Output = Complex;
    fn mul(self, o: Complex) -> Complex {
        Complex::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl Div for Complex {
    type Output = Complex;
    fn div(self, o: Complex) -> Complex {
        let d = o.norm_sq();
        Complex::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
}
impl Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

impl From<f64> for Complex {
    fn from(r: f64) -> Complex {
        Complex::real(r)
    }
}

impl From<(f64, f64)> for Complex {
    fn from((re, im): (f64, f64)) -> Complex {
        Complex::new(re, im)
    }
}

impl<'a, 'b> Add<&'b Complex> for &'a Complex {
    type Output = Complex;
    fn add(self, o: &'b Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}
impl<'a, 'b> Sub<&'b Complex> for &'a Complex {
    type Output = Complex;
    fn sub(self, o: &'b Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}
impl<'a, 'b> Mul<&'b Complex> for &'a Complex {
    type Output = Complex;
    fn mul(self, o: &'b Complex) -> Complex {
        Complex::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl<'a, 'b> Div<&'b Complex> for &'a Complex {
    type Output = Complex;
    fn div(self, o: &'b Complex) -> Complex {
        let d = o.norm_sq();
        Complex::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
}
impl<'a> Neg for &'a Complex {
    type Output = Complex;
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}
impl<'a> Add<f64> for &'a Complex {
    type Output = Complex;
    fn add(self, o: f64) -> Complex {
        Complex::new(self.re + o, self.im)
    }
}
impl<'a> Sub<f64> for &'a Complex {
    type Output = Complex;
    fn sub(self, o: f64) -> Complex {
        Complex::new(self.re - o, self.im)
    }
}
impl<'a> Mul<f64> for &'a Complex {
    type Output = Complex;
    fn mul(self, o: f64) -> Complex {
        Complex::new(self.re * o, self.im * o)
    }
}
impl<'a> Div<f64> for &'a Complex {
    type Output = Complex;
    fn div(self, o: f64) -> Complex {
        Complex::new(self.re / o, self.im / o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        let a = Complex::new(3.0, 4.0);
        assert_eq!(a * a.conjugate(), Complex::real(25.0));
        assert_eq!(a + Complex::i(), Complex::new(3.0, 5.0));
        let q = Complex::new(1.0, 1.0) / Complex::new(1.0, -1.0);
        assert!((q.re - 0.0).abs() < 1e-12 && (q.im - 1.0).abs() < 1e-12);
        assert_eq!(-a, Complex::new(-3.0, -4.0));
    }

    #[test]
    fn polar_and_functions() {
        let z = Complex::polar(2.0, core::f64::consts::FRAC_PI_2);
        assert!((z.re - 0.0).abs() < 1e-12 && (z.im - 2.0).abs() < 1e-12);
        assert!((Complex::new(1.0, 1.0).arg() - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((Complex::new(1.0, 1.0).norm() - 2.0f64.sqrt()).abs() < 1e-12);
        // ln/e round trip
        let w = Complex::new(1.0, 2.0);
        assert!((w.ln().exp() - w).norm() < 1e-12);
        assert!((Complex::i().powf(2.0).re + 1.0).abs() < 1e-12);
        assert!((Complex::new(0.5, 0.0).sin().re - 0.5f64.sin()).abs() < 1e-12);
        assert!((Complex::i().ln().im - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn hyperbolic_functions() {
        let z = Complex::new(1.0, 0.5);
        let sinh = z.sinh();
        let cosh = z.cosh();
        // Check identity: cosh² - sinh² = 1
        let diff = cosh * cosh - sinh * sinh;
        assert!((diff - Complex::one()).norm() < 1e-10);
        
        // Test atanh
        let z2 = Complex::new(0.5, 0.0);
        let atanh = z2.atanh();
        assert!((atanh.re - 0.5_f64.atanh()).abs() < 1e-10);
    }

    #[test]
    fn inverse_trigonometric() {
        let z = Complex::new(0.5, 0.0);
        let asin = z.asin();
        assert!((asin.re - 0.5_f64.asin()).abs() < 1e-10);
        
        let acos = z.acos();
        assert!((acos.re - 0.5_f64.acos()).abs() < 1e-10);
        
        let atan = z.atan();
        assert!((atan.re - 0.5_f64.atan()).abs() < 1e-10);
    }

    #[test]
    fn logarithms() {
        let z = Complex::new(10.0, 0.0);
        let log10 = z.log10();
        assert!((log10.re - 1.0).abs() < 1e-10);
        
        let z2 = Complex::new(2.0, 0.0);
        let log2 = z2.log2();
        assert!((log2.re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn reciprocal() {
        let z = Complex::new(2.0, 3.0);
        let recip = z.recip();
        let product = z * recip;
        assert!((product - Complex::one()).norm() < 1e-10);
    }

    #[test]
    fn special_values() {
        assert!(Complex::zero().is_zero());
        assert_eq!(Complex::zero(), Complex::new(0.0, 0.0));
        assert_eq!(Complex::one(), Complex::new(1.0, 0.0));
        
        let nan_z = Complex::new(f64::NAN, 0.0);
        assert!(nan_z.is_nan());
        
        let inf_z = Complex::new(f64::INFINITY, 0.0);
        assert!(inf_z.is_infinite());
        
        let finite_z = Complex::new(1.0, 2.0);
        assert!(finite_z.is_finite());
    }

    #[test]
    fn signum() {
        let z = Complex::new(3.0, 4.0);
        let s = z.signum();
        assert!((s.norm() - 1.0).abs() < 1e-10);
        assert!((s * z.norm() - z).norm() < 1e-10);
        
        assert_eq!(Complex::zero().signum(), Complex::zero());
    }

    #[test]
    fn roots() {
        let z = Complex::new(8.0, 0.0);
        let cbrt = z.cbrt();
        assert!((cbrt.re - 2.0).abs() < 1e-8);
        
        let z2 = Complex::new(16.0, 0.0);
        let fourth = z2.root(4.0);
        assert!((fourth.re - 2.0).abs() < 1e-8);
    }

    #[test]
    fn from_implementations() {
        let z1: Complex = 5.0.into();
        assert_eq!(z1, Complex::real(5.0));
        
        let z2: Complex = (3.0, 4.0).into();
        assert_eq!(z2, Complex::new(3.0, 4.0));
    }
}
