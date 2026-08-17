//! Complex numbers over any [`RealFull`] type (`f64` by default).
//!
//! Division by zero and `ln(0)` etc. yield `NaN`/`inf` components, mirroring
//! std float semantics (no panics).
//!
//! # Generic precision
//!
//! [`Complex`] defaults to `f64` components (`Complex<T = f64>`), so existing
//! code that writes `Complex` keeps working unchanged. [`C32`] and [`C64`] are
//! convenience aliases. The whole arithmetic suite is generic over
//! `T: RealFull` (conversion, powers, transcendental, trigonometric,
//! hyperbolic and float-classification operations), so `Complex<f32>`,
//! `Complex<f64>`, and any future `RealFull` type share the same code.
//!
//! Several method names follow the `cmath`/`numpy` conventions for Python
//! parity: [`Complex::phase`], [`Complex::to_polar`], [`Complex::rect`] and
//! [`Complex::is_close`].
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::needless_range_loop
)]

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};
use mathverse_core::traits::RealFull;

pub mod analysis;
pub mod batch;
pub mod activations;
pub mod convolution;
pub mod correlation;
pub mod distributions;
pub mod fft;
pub mod matrix;
pub mod pca;
pub mod polar;
pub mod polynomial;
pub mod regression;
pub mod special_functions;
pub mod wavelets;

pub use analysis::ComplexAnalysis;
pub use fft::{fft, fft_in_place, ifft};
pub use matrix::ComplexMatrix;
pub use polynomial::{eval_polynomial, polynomial_roots};
pub use special_functions::ComplexSpecialFunctions;

/// Single-precision complex number (`f32` components).
pub type C32 = Complex<f32>;
/// Double-precision complex number (`f64` components).
pub type C64 = Complex<f64>;

/// Complex number `re + im·i` over any real type `T` (defaults to `f64`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct Complex<T: RealFull = f64> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T: RealFull + fmt::Display> fmt::Display for Complex<T> {
    /// Formats as `a+bi` (e.g. `3+4i`, `1-2i`), honoring precision like
    /// `{:.6}` for the component digits.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match f.precision() {
            Some(p) => write!(f, "{:.p$}{:+.p$}i", self.re, self.im, p = p),
            None => write!(f, "{}{:+}i", self.re, self.im),
        }
    }
}

impl<T: RealFull> Complex<T> {
    /// Create a complex number from its real and imaginary parts.
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }
    /// The real number `r` as a complex number.
    pub fn real(r: T) -> Complex<T> {
        Complex {
            re: r,
            im: T::zero(),
        }
    }
    /// The imaginary unit `i` (0 + 1i).
    pub fn i() -> Complex<T> {
        Complex {
            re: T::zero(),
            im: T::one(),
        }
    }
    /// Zero constant (0 + 0i).
    pub fn zero() -> Complex<T> {
        Complex {
            re: T::zero(),
            im: T::zero(),
        }
    }
    /// One constant (1 + 0i).
    pub fn one() -> Complex<T> {
        Complex {
            re: T::one(),
            im: T::zero(),
        }
    }
    /// From polar form `r·e^(iθ)` (equiv. `cmath.rect(r, θ)`).
    pub fn polar(r: T, theta: T) -> Complex<T> {
        Complex::new(r * theta.cos(), r * theta.sin())
    }
    /// Complex conjugate `re − im·i`.
    pub fn conjugate(&self) -> Complex<T> {
        Complex::new(self.re, -self.im)
    }
    /// Squared modulus `re² + im²`.
    pub fn norm_sq(&self) -> T {
        self.re * self.re + self.im * self.im
    }
    /// Modulus `|z| = √(re² + im²)`.
    pub fn norm(&self) -> T {
        self.norm_sq().sqrt()
    }
    /// Principal argument in `(-π, π]` (equiv. `cmath.phase(z)`).
    pub fn arg(&self) -> T {
        let theta = self.im.atan2(self.re);
        // atan2(0, -x) = π but atan2(-0, -x) = -π; normalize to principal
        // branch (-π, π] so that negative reals always get arg = π.
        if self.re < T::zero() && self.im == T::zero() {
            T::from_f64(core::f64::consts::PI)
        } else {
            theta
        }
    }
    /// `(r, θ)` such that `self == r·e^(iθ)` (equiv. `cmath.polar(z)`).
    pub fn to_polar(&self) -> (T, T) {
        (self.norm(), self.arg())
    }
    /// Alias for [`arg`](Self::arg), matching `numpy.angle`/`cmath.phase`.
    pub fn phase(&self) -> T {
        self.arg()
    }
    /// Build from polar form, matching `cmath.rect(r, φ)`.
    pub fn rect(r: T, phi: T) -> Complex<T> {
        Complex::polar(r, phi)
    }
    /// `cmath.isclose` equivalent: true if `|a − b| ≤ abs_tol` or
    /// `|a − b| ≤ rel_tol · max(|a|, |b|)`.
    pub fn is_close(&self, other: &Complex<T>, rel_tol: T, abs_tol: T) -> bool {
        let diff = (*self - *other).norm();
        let scale = self.norm().max(other.norm());
        diff <= abs_tol || diff <= rel_tol * scale
    }
    /// Returns `true` if both components are exactly zero.
    pub fn is_zero(&self) -> bool {
        self.re == T::zero() && self.im == T::zero()
    }
    /// Principal square root.
    ///
    /// Uses the algebraically stable formula (Numerical Recipes §5.4) instead
    /// of the polar form, avoiding the two-square-root + `atan2` round trip
    /// and keeping full precision for `z` near the real axis.
    ///
    /// The branch cut is the negative real axis: `sqrt(-4 + 0⁻i) = -2i`.
    ///
    /// ```
    /// use mathverse_complex::Complex;
    /// let s: Complex = Complex::new(-4.0, 0.0).sqrt();
    /// assert!((s.im - 2.0).abs() < 1e-12 && s.re.abs() < 1e-12);
    /// ```
    pub fn sqrt(&self) -> Complex<T> {
        if self.is_zero() {
            return Complex::<T>::zero();
        }
        let n = self.norm();
        if self.re >= T::zero() {
            let r = ((n + self.re) / T::from_f64(2.0)).sqrt();
            Complex::<T>::new(r, self.im / (T::from_f64(2.0) * r))
        } else {
            let i = self.im.signum() * ((n - self.re) / T::from_f64(2.0)).sqrt();
            let r = if i == T::zero() {
                T::zero()
            } else {
                self.im.abs() / (T::from_f64(2.0) * i.abs())
            };
            Complex::<T>::new(r, i)
        }
    }
    /// `e^self`.
    ///
    /// ```
    /// use mathverse_complex::Complex;
    /// let e = Complex::new(0.0, core::f64::consts::PI).exp();
    /// assert!((e.re + 1.0).abs() < 1e-12 && e.im.abs() < 1e-12);
    /// ```
    pub fn exp(&self) -> Complex<T> {
        let e = self.re.exp();
        Complex::<T>::new(e * self.im.cos(), e * self.im.sin())
    }
    /// Principal natural logarithm.
    pub fn ln(&self) -> Complex<T> {
        Complex::<T>::new(self.norm().ln(), self.arg())
    }
    /// `self^p = e^(p·ln self)`, principal branch.
    ///
    /// For `0^p` the standard limiting convention is used: `0` when
    /// `Re(p) > 0`, `+∞` when `Re(p) < 0`, `1` for `p = 0` (combinatorial
    /// convention), and `NaN` for purely imaginary exponents where the limit
    /// does not converge.
    pub fn pow(&self, p: Complex<T>) -> Complex<T> {
        if self.is_zero() {
            return if p.re > T::zero() {
                Complex::<T>::zero()
            } else if p.re < T::zero() {
                Complex::<T>::new(T::from_f64(f64::INFINITY), T::zero())
            } else if p.im == T::zero() {
                // 0^0 = 1 (combinatorial convention)
                Complex::<T>::one()
            } else {
                // 0^(i·y) is undefined — limit does not converge
                Complex::<T>::new(T::from_f64(f64::NAN), T::from_f64(f64::NAN))
            };
        }
        (p * self.ln()).exp()
    }
    /// `self^p` for a real exponent.
    pub fn powf(&self, p: T) -> Complex<T> {
        self.pow(Complex::<T>::real(p))
    }
    /// Sine: `sin(re)·cosh(im) + i·cos(re)·sinh(im)`.
    pub fn sin(&self) -> Complex<T> {
        Complex::<T>::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }
    /// Cosine: `cos(re)·cosh(im) − i·sin(re)·sinh(im)`.
    pub fn cos(&self) -> Complex<T> {
        Complex::<T>::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }
    /// Tangent: `sin(z) / cos(z)`.
    pub fn tan(&self) -> Complex<T> {
        self.sin() / self.cos()
    }
    /// Hyperbolic sine.
    pub fn sinh(&self) -> Complex<T> {
        Complex::<T>::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin(),
        )
    }
    /// Hyperbolic cosine.
    pub fn cosh(&self) -> Complex<T> {
        Complex::<T>::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin(),
        )
    }
    /// Hyperbolic tangent.
    pub fn tanh(&self) -> Complex<T> {
        self.sinh() / self.cosh()
    }
    /// Inverse hyperbolic sine.
    pub fn asinh(&self) -> Complex<T> {
        let s = *self;
        (s + (s * s + Complex::<T>::real(T::one())).sqrt()).ln()
    }
    /// Inverse hyperbolic cosine.
    ///
    /// `acosh(z) = ln(z + sqrt(z² − 1))` with branch cut `(-∞, 1]`.
    /// The single-square-root form avoids the double `sqrt` round trip of the
    /// equivalent `sqrt(z+1)·sqrt(z-1)` formulation and matches `numpy`'s
    /// `np.arccosh`. Result satisfies `Re(acosh(z)) ≥ 0`.
    pub fn acosh(&self) -> Complex<T> {
        let s = *self;
        (s + (s * s - Complex::<T>::one()).sqrt()).ln()
    }
    /// Inverse hyperbolic tangent.
    ///
    /// `atanh(z) = ½·ln((1+z)/(1−z))` with branch cuts `(-∞, -1]` and
    /// `[1, ∞)`. On the real axis: for real `x > 1` the principal value has
    /// `Im = +π/2`; for `x < -1`, `Im = -π/2` (matches `cmath.atanh`).
    /// The sign follows the sign of the imaginary part of the input, so
    /// `atanh(x ± 0i)` sits on the matching side of the cut.
    pub fn atanh(&self) -> Complex<T> {
        let s = *self;
        let one = Complex::<T>::one();
        // On the real-axis branch cuts the intermediate division `(1+z)/(1−z)`
        // can contaminate the imaginary sign (e.g. produce −0.0), flipping the
        // principal value to the wrong side of the cut. Fix it explicitly:
        // the upper edge (im = +0) gives Im = +π/2 on the right cut [1, ∞)
        // but −π/2 on the left cut (−∞, −1]; the lower edge flips both.
        if s.im == T::zero() && (s.re > one.re || s.re < -one.re) {
            let mag = ((one.re + s.re) / (s.re - one.re)).ln() / T::from_f64(2.0);
            let side = if s.im.is_sign_negative() {
                -T::one()
            } else {
                T::one()
            };
            let cut = if s.re > one.re { T::one() } else { -T::one() };
            let im = side * cut * T::from_f64(core::f64::consts::FRAC_PI_2);
            return Complex::<T>::new(mag, im);
        }
        ((one + s) / (one - s)).ln() / Complex::<T>::real(T::from_f64(2.0))
    }
    /// Inverse sine.
    pub fn asin(&self) -> Complex<T> {
        let s = *self;
        -Complex::<T>::i() * (Complex::<T>::i() * s + (Complex::<T>::one() - s * s).sqrt()).ln()
    }
    /// Inverse cosine.
    pub fn acos(&self) -> Complex<T> {
        let s = *self;
        -Complex::<T>::i() * (s + Complex::<T>::i() * (Complex::<T>::one() - s * s).sqrt()).ln()
    }
    /// Inverse tangent.
    pub fn atan(&self) -> Complex<T> {
        let s = *self;
        let i = Complex::<T>::i();
        (i / Complex::<T>::real(T::from_f64(2.0)))
            * ((Complex::<T>::one() - i * s) / (Complex::<T>::one() + i * s)).ln()
    }
    /// Base-10 logarithm.
    pub fn log10(&self) -> Complex<T> {
        self.ln() / Complex::<T>::real(T::from_f64(10.0_f64.ln()))
    }
    /// Base-2 logarithm.
    pub fn log2(&self) -> Complex<T> {
        self.ln() / Complex::<T>::real(T::from_f64(2.0_f64.ln()))
    }
    /// Reciprocal (1/self).
    ///
    /// Uses Smith's algorithm: scale by the larger component before the
    /// squaring, so `re² + im²` cannot overflow (e.g. `(1e200+0i).recip()`
    /// is `1e-200`, not `NaN`).
    pub fn recip(&self) -> Complex<T> {
        if self.re.abs() >= self.im.abs() {
            let r = self.im / self.re;
            let d = self.re + self.im * r;
            Complex::<T>::new(T::one() / d, -r / d)
        } else {
            let r = self.re / self.im;
            let d = self.re * r + self.im;
            Complex::<T>::new(r / d, -T::one() / d)
        }
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
    /// Signum (sign) function: same direction, unit magnitude. Returns zero
    /// if `self` is zero.
    pub fn signum(&self) -> Complex<T> {
        if self.is_zero() {
            Complex::<T>::new(T::zero(), T::zero())
        } else {
            let s = *self;
            s / Complex::<T>::real(s.norm())
        }
    }
    /// Cube root.
    pub fn cbrt(&self) -> Complex<T> {
        self.powf(T::from_f64(1.0 / 3.0))
    }
    /// Nth root (principal branch).
    pub fn root(&self, n: T) -> Complex<T> {
        self.powf(T::one() / n)
    }
}

impl<T: RealFull> Add for Complex<T> {
    type Output = Complex<T>;
    fn add(self, o: Complex<T>) -> Complex<T> {
        Complex::<T>::new(self.re + o.re, self.im + o.im)
    }
}
impl<T: RealFull> Sub for Complex<T> {
    type Output = Complex<T>;
    fn sub(self, o: Complex<T>) -> Complex<T> {
        Complex::<T>::new(self.re - o.re, self.im - o.im)
    }
}
impl<T: RealFull> Mul for Complex<T> {
    type Output = Complex<T>;
    fn mul(self, o: Complex<T>) -> Complex<T> {
        Complex::<T>::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
impl<T: RealFull> Div for Complex<T> {
    type Output = Complex<T>;
    // self · (1/o) with an overflow-safe reciprocal (Smith's algorithm)
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, o: Complex<T>) -> Complex<T> {
        self * o.recip()
    }
}
impl<T: RealFull> Neg for Complex<T> {
    type Output = Complex<T>;
    fn neg(self) -> Complex<T> {
        Complex::<T>::new(-self.re, -self.im)
    }
}

// Scalar arithmetic (numpy-style broadcasting of a real onto a complex).
impl<T: RealFull> Add<T> for Complex<T> {
    type Output = Complex<T>;
    fn add(self, o: T) -> Complex<T> {
        Complex::<T>::new(self.re + o, self.im)
    }
}
impl<T: RealFull> Sub<T> for Complex<T> {
    type Output = Complex<T>;
    fn sub(self, o: T) -> Complex<T> {
        Complex::<T>::new(self.re - o, self.im)
    }
}
impl<T: RealFull> Mul<T> for Complex<T> {
    type Output = Complex<T>;
    fn mul(self, o: T) -> Complex<T> {
        Complex::<T>::new(self.re * o, self.im * o)
    }
}
impl<T: RealFull> Div<T> for Complex<T> {
    type Output = Complex<T>;
    fn div(self, o: T) -> Complex<T> {
        Complex::<T>::new(self.re / o, self.im / o)
    }
}

impl<'b, T: RealFull> Add<&'b Complex<T>> for &Complex<T> {
    type Output = Complex<T>;
    fn add(self, o: &'b Complex<T>) -> Complex<T> {
        Complex::<T>::new(self.re + o.re, self.im + o.im)
    }
}
impl<'b, T: RealFull> Sub<&'b Complex<T>> for &Complex<T> {
    type Output = Complex<T>;
    fn sub(self, o: &'b Complex<T>) -> Complex<T> {
        Complex::<T>::new(self.re - o.re, self.im - o.im)
    }
}
impl<'b, T: RealFull> Mul<&'b Complex<T>> for &Complex<T> {
    type Output = Complex<T>;
    fn mul(self, o: &'b Complex<T>) -> Complex<T> {
        Complex::<T>::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
impl<'b, T: RealFull> Div<&'b Complex<T>> for &Complex<T> {
    type Output = Complex<T>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, o: &'b Complex<T>) -> Complex<T> {
        *self * o.recip()
    }
}
impl<T: RealFull> Neg for &Complex<T> {
    type Output = Complex<T>;
    fn neg(self) -> Complex<T> {
        Complex::<T>::new(-self.re, -self.im)
    }
}
impl<T: RealFull> Add<T> for &Complex<T> {
    type Output = Complex<T>;
    fn add(self, o: T) -> Complex<T> {
        Complex::<T>::new(self.re + o, self.im)
    }
}
impl<T: RealFull> Sub<T> for &Complex<T> {
    type Output = Complex<T>;
    fn sub(self, o: T) -> Complex<T> {
        Complex::<T>::new(self.re - o, self.im)
    }
}
impl<T: RealFull> Mul<T> for &Complex<T> {
    type Output = Complex<T>;
    fn mul(self, o: T) -> Complex<T> {
        Complex::<T>::new(self.re * o, self.im * o)
    }
}
impl<T: RealFull> Div<T> for &Complex<T> {
    type Output = Complex<T>;
    fn div(self, o: T) -> Complex<T> {
        Complex::<T>::new(self.re / o, self.im / o)
    }
}

impl<T: RealFull> From<T> for Complex<T> {
    fn from(r: T) -> Complex<T> {
        Complex::<T>::real(r)
    }
}

impl<T: RealFull> From<(T, T)> for Complex<T> {
    fn from((re, im): (T, T)) -> Complex<T> {
        Complex::<T>::new(re, im)
    }
}

#[cfg(feature = "rand")]
impl rand::distributions::Distribution<Complex<f64>> for rand::distributions::Standard {
    /// Sample a complex number with independent standard-normal real and
    /// imaginary parts (circular Gaussian).
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Complex<f64> {
        let re: f64 = rng.gen();
        let im: f64 = rng.gen();
        Complex::new(re, im)
    }
}

/// Sample a complex number uniformly distributed on the unit disk.
#[cfg(feature = "rand")]
pub fn complex_uniform_disk<R: rand::Rng + ?Sized>(rng: &mut R) -> Complex<f64> {
    loop {
        let re: f64 = rng.gen();
        let im: f64 = rng.gen();
        let z = Complex::new(re, im);
        if z.norm_sq() <= 1.0 {
            return z;
        }
    }
}

/// Sample a complex number from a circular Gaussian with standard deviation
/// `sigma` (i.e., `Re` and `Im` are independent `N(0, sigma²)`).
#[cfg(feature = "rand")]
pub fn complex_gaussian<R: rand::Rng + ?Sized>(rng: &mut R, sigma: f64) -> Complex<f64> {
    let re: f64 = rng.gen::<f64>() * sigma;
    let im: f64 = rng.gen::<f64>() * sigma;
    Complex::new(re, im)
}

/// Iterate `z → z² + c` until escape or `max_iterations`.
/// Returns the iteration count at which `|z|² > escape_radius²`
/// (`max_iterations` if the orbit never escapes).
pub fn mandelbrot_iterate<T: RealFull>(
    c: Complex<T>,
    max_iterations: usize,
    escape_radius: T,
) -> usize {
    let r2 = escape_radius * escape_radius;
    let mut z = Complex::<T>::zero();
    for i in 0..max_iterations {
        if z.norm_sq() > r2 {
            return i;
        }
        z = z * z + c;
    }
    max_iterations
}

/// Smooth iteration count for Mandelbrot coloring:
/// `n + 1 − log₂(log|z|)` at escape (fractional for smooth gradients).
pub fn mandelbrot_smooth<T: RealFull>(
    c: Complex<T>,
    max_iterations: usize,
    escape_radius: T,
) -> f64 {
    let r2 = escape_radius * escape_radius;
    let mut z = Complex::<T>::zero();
    for i in 0..max_iterations {
        let norm_sq = z.norm_sq();
        if norm_sq > r2 {
            let lg = norm_sq.to_f64().ln().ln();
            if lg.is_finite() {
                return i as f64 + 1.0 - lg / core::f64::consts::LN_2;
            }
            return i as f64 + 1.0;
        }
        z = z * z + c;
    }
    max_iterations as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        let a = Complex::new(3.0, 4.0);
        assert_eq!(a * a.conjugate(), Complex::real(25.0));
        assert_eq!(a + Complex::i(), Complex::new(3.0, 5.0));
        let q: Complex = Complex::new(1.0, 1.0) / Complex::new(1.0, -1.0);
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
        assert!((Complex::<f64>::i().powf(2.0).re + 1.0).abs() < 1e-12);
        assert!((Complex::new(0.5, 0.0).sin().re - 0.5f64.sin()).abs() < 1e-12);
        assert!((Complex::<f64>::i().ln().im - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
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
    fn inverse_trig_special_values() {
        // acos(0) = π/2, atan(1) = π/4, asin(1) = π/2 (regression: sign/branch bugs)
        let acos0 = Complex::<f64>::zero().acos();
        assert!((acos0.re - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!(acos0.im.abs() < 1e-12);

        let atan1 = Complex::<f64>::one().atan();
        assert!((atan1.re - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!(atan1.im.abs() < 1e-12);

        let asin1 = Complex::<f64>::one().asin();
        assert!((asin1.re - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!(asin1.im.abs() < 1e-12);

        // acos(cos z) = z round trip off the real axis
        let z = Complex::new(0.7, 0.4);
        let round = z.cos().acos();
        assert!((round - z).norm() < 1e-10);

        // atan(tan z) = z round trip
        let round2 = z.tan().atan();
        assert!((round2 - z).norm() < 1e-10);
    }

    #[test]
    fn logarithms() {
        let z: Complex = Complex::new(10.0, 0.0);
        let log10 = z.log10();
        assert!((log10.re - 1.0).abs() < 1e-10);

        let z2: Complex = Complex::new(2.0, 0.0);
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
    fn division_no_overflow() {
        // Regression: the naive (a·d + b·c)/|d|² formula overflows to
        // inf/inf = NaN when |d|² ~ 1e422 (e.g. Bessel-series terms at
        // n ≈ 73). Smith's algorithm must give exact quotients instead.
        let big: Complex = Complex::new(1e300, 1e300);
        let one = big / big;
        assert!((one.re - 1.0).abs() < 1e-12 && one.im.abs() < 1e-12);

        // Numerator of the naive formula would overflow: (1e200, 0)/(1e200, 1e200)
        // computes a·c = 1e400 = inf before the denominator scales it down.
        let q: Complex = Complex::new(1e200, 0.0) / Complex::new(1e200, 1e200);
        assert!((q.re - 0.5).abs() < 1e-12 && (q.im + 0.5).abs() < 1e-12);

        // Mixed magnitudes: large / small-magnitude imaginary
        let mixed: Complex = Complex::new(1e150, 0.0) / Complex::new(0.0, 1e-150);
        assert!(mixed.re.abs() < 1e-12 && mixed.im.abs() > 1e290);

        // Small / large stays finite (exact value ~5e-601 underflows to 0)
        let tiny: Complex = Complex::new(1e-300, 0.0) / Complex::new(1e300, 1e300);
        assert!(tiny.is_finite() && tiny == Complex::zero());

        // Round-trip with reciprocal
        let z: Complex = Complex::new(1e200, -1e200);
        assert!((z * z.recip() - Complex::one()).norm() < 1e-12);
    }

    #[test]
    fn special_values() {
        assert!(Complex::<f64>::zero().is_zero());
        assert_eq!(Complex::<f64>::zero(), Complex::new(0.0, 0.0));
        assert_eq!(Complex::<f64>::one(), Complex::new(1.0, 0.0));

        let nan_z = Complex::new(f64::NAN, 0.0);
        assert!(nan_z.is_nan());

        let inf_z = Complex::new(f64::INFINITY, 0.0);
        assert!(inf_z.is_infinite());

        let finite_z = Complex::new(1.0, 2.0);
        assert!(finite_z.is_finite());
    }

    #[test]
    fn signum() {
        let z: Complex = Complex::new(3.0, 4.0);
        let s = z.signum();
        assert!((s.norm() - 1.0).abs() < 1e-10);
        assert!((s * Complex::real(z.norm()) - z).norm() < 1e-10);

        assert_eq!(Complex::<f64>::zero().signum(), Complex::<f64>::zero());
    }

    #[test]
    fn roots() {
        let z: Complex = Complex::new(8.0, 0.0);
        let cbrt = z.cbrt();
        assert!((cbrt.re - 2.0).abs() < 1e-8);

        let z2: Complex = Complex::new(16.0, 0.0);
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

    // ---- Audit-driven regression tests -------------------------------------

    #[test]
    fn c32_generic_operates() {
        let z = C32::new(1.0f32, 2.0);
        let w = z * z;
        assert!((w.re - (-3.0f32)).abs() < 1e-6);
        assert!((w.im - 4.0f32).abs() < 1e-6);
        let s = C32::new(9.0f32, 0.0).sqrt();
        assert!((s.re - 3.0f32).abs() < 1e-5);
        assert!((s.im - 0.0f32).abs() < 1e-5);
        let e = C32::new(0.0f32, core::f32::consts::PI).exp();
        assert!((e.re + 1.0f32).abs() < 1e-5);
    }

    #[test]
    fn sqrt_precision_near_negative_axis() {
        // z = -4 + 1e-12i must not lose precision through atan2/cos round trips
        let s: Complex = Complex::new(-4.0, 1e-12).sqrt();
        assert!((s.im - 2.0).abs() < 1e-12);
        assert!(s.re.abs() < 1e-12);

        // Branch cut: sqrt(-4 - 0i) = -2i
        let sm: Complex = Complex::new(-4.0, -0.0).sqrt();
        assert!((sm.im + 2.0).abs() < 1e-12);

        // sqrt(z)·sqrt(z) = z
        let z: Complex = Complex::new(0.7, -2.3);
        let back = z.sqrt() * z.sqrt();
        assert!((back - z).norm() < 1e-12);
    }

    #[test]
    fn zero_to_zero_is_one() {
        assert_eq!(
            Complex::<f64>::zero().pow(Complex::<f64>::zero()),
            Complex::<f64>::one()
        );
        assert_eq!(Complex::<f64>::zero().powf(0.0), Complex::<f64>::one());
        // 0^p still behaves for positive/negative real exponents
        assert_eq!(
            Complex::<f64>::zero().pow(Complex::real(2.0)),
            Complex::<f64>::zero()
        );
        assert!(Complex::<f64>::zero()
            .pow(Complex::real(-1.0))
            .is_infinite());
    }

    #[test]
    fn acosh_edge_values() {
        // acosh(1) = 0; acosh(-1) = iπ; acosh(0.5) = i·acos(0.5)
        let a1 = Complex::<f64>::one().acosh();
        assert!(a1.norm() < 1e-12);
        let am1 = Complex::real(-1.0).acosh();
        assert!((am1.im - core::f64::consts::PI).abs() < 1e-12);
        let a05 = Complex::real(0.5).acosh();
        assert!((a05.im - core::f64::consts::FRAC_PI_3).abs() < 1e-12);
        // round trip: cosh(acosh(z)) = z
        let z = Complex::new(1.5, 0.6);
        let round = z.acosh().cosh();
        assert!((round - z).norm() < 1e-10);
    }

    #[test]
    fn atanh_branch_values() {
        // atanh(2): principal value has Im = +π/2 (branch cut [1, ∞))
        let a2 = Complex::real(2.0).atanh();
        assert!((a2.re - 0.5 * 3.0f64.ln()).abs() < 1e-12);
        assert!((a2.im - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        // atanh(-2): Im = -π/2
        let am2 = Complex::real(-2.0).atanh();
        assert!((am2.re + 0.5 * 3.0f64.ln()).abs() < 1e-12);
        assert!((am2.im + core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        // round trip: tanh(atanh(z)) = z
        let z = Complex::new(0.3, 0.4);
        let round = z.atanh().tanh();
        assert!((round - z).norm() < 1e-10);
    }

    #[test]
    fn cmath_parity_roundtrips() {
        let z: Complex = Complex::new(3.0, 4.0);
        assert!((z.phase() - z.arg()).abs() < 1e-15);
        let (r, p) = z.to_polar();
        assert!((r - 5.0).abs() < 1e-12);
        assert!((p - z.arg()).abs() < 1e-15);
        let back = Complex::rect(r, p);
        assert!((back - z).norm() < 1e-12);
        // rect(1, π/2) = i (cos(π/2) is 6e-17, not exactly 0)
        let ri: Complex = Complex::rect(1.0, core::f64::consts::FRAC_PI_2);
        assert!(ri.re.abs() < 1e-15 && (ri.im - 1.0).abs() < 1e-15);
    }

    #[test]
    fn is_close_semantics() {
        let a: Complex = Complex::new(1.0, 2.0);
        assert!(a.is_close(&Complex::new(1.0, 2.0 + 1e-10), 1e-8, 1e-12));
        assert!(!a.is_close(&Complex::new(1.1, 2.0), 1e-8, 1e-12));
        // abs_tol alone covers near-zero comparisons
        assert!(Complex::<f64>::zero().is_close(&Complex::new(1e-13, 1e-13), 1e-9, 1e-10));
        assert!(!Complex::<f64>::zero().is_close(&Complex::new(1e-5, 0.0), 1e-9, 1e-12));
    }

    #[test]
    fn scalar_ops_broadcast() {
        let z: Complex = Complex::new(1.0, 2.0);
        assert_eq!(z + 1.0, Complex::new(2.0, 2.0));
        assert_eq!(z - 1.0, Complex::new(0.0, 2.0));
        assert_eq!(z * 2.0, Complex::new(2.0, 4.0));
        assert_eq!(z / 2.0, Complex::new(0.5, 1.0));
    }

    #[test]
    fn mandelbrot_iteration() {
        // c = -0.5 is inside the set: never escapes
        assert_eq!(mandelbrot_iterate(Complex::new(-0.5, 0.0), 100, 2.0), 100);
        // c = 1 escapes quickly
        assert_eq!(mandelbrot_iterate(Complex::new(1.0, 0.0), 100, 2.0), 3);
        assert_eq!(mandelbrot_iterate(Complex::new(0.0, 0.0), 100, 2.0), 100);
        // smooth count is continuous with the integer count for in-set points
        assert!((mandelbrot_smooth(Complex::new(-0.5, 0.0), 100, 2.0) - 100.0).abs() < 1e-12);
        let s = mandelbrot_smooth(Complex::new(1.0, 0.0), 100, 2.0);
        assert!(s > 0.0 && s < 10.0);
    }

    #[test]
    fn property_conjugate_norm() {
        let z: Complex = Complex::new(1.3, -0.7);
        assert!((z * z.conjugate() - Complex::real(z.norm_sq())).norm() < 1e-12);
        let w: Complex = Complex::new(0.8, 0.6);
        assert!((w.ln().exp() - w).norm() < 1e-12);
    }
}
