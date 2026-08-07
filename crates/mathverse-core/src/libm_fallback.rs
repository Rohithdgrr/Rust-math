//! `libm`-backed transcendental functions for `no_std` builds.
//!
//! When the `std` feature is disabled and the `libm` feature is enabled,
//! these functions provide the same API as `std`'s float methods but
//! backed by the `libm` crate (software floating-point math).
//!
//! On `std` builds, the standard library's hardware-accelerated versions
//! are used instead.

/// Square root.
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// Cube root.
pub fn cbrt(x: f64) -> f64 {
    libm::cbrt(x)
}

/// Raise to a floating-point power.
pub fn powf(x: f64, e: f64) -> f64 {
    libm::pow(x, e)
}

/// Raise to an integer power.
pub fn powi(x: f64, e: i32) -> f64 {
    libm::pow(x, e as f64)
}

/// `e^x`.
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// `e^x - 1`, accurate for small `x`.
pub fn exp_m1(x: f64) -> f64 {
    libm::expm1(x)
}

/// Natural logarithm.
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// `ln(1 + x)`, accurate for small `x`.
pub fn ln_1p(x: f64) -> f64 {
    libm::log1p(x)
}

/// Logarithm with arbitrary base.
pub fn log(x: f64, base: f64) -> f64 {
    libm::log(x) / libm::log(base)
}

/// Log base 10.
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}

/// Log base 2.
pub fn log2(x: f64) -> f64 {
    libm::log2(x)
}

/// Sine (radians).
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// Cosine (radians).
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// Tangent (radians).
pub fn tan(x: f64) -> f64 {
    libm::tan(x)
}

/// Arcsine.
pub fn asin(x: f64) -> f64 {
    libm::asin(x)
}

/// Arccosine.
pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}

/// Arctangent.
pub fn atan(x: f64) -> f64 {
    libm::atan(x)
}

/// Two-argument arctangent: `atan(y / x)`, with correct quadrant.
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

/// Hyperbolic sine.
pub fn sinh(x: f64) -> f64 {
    libm::sinh(x)
}

/// Hyperbolic cosine.
pub fn cosh(x: f64) -> f64 {
    libm::cosh(x)
}

/// Hyperbolic tangent.
pub fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}

/// Inverse hyperbolic sine.
pub fn asinh(x: f64) -> f64 {
    libm::asinh(x)
}

/// Inverse hyperbolic cosine.
pub fn acosh(x: f64) -> f64 {
    libm::acosh(x)
}

/// Inverse hyperbolic tangent.
pub fn atanh(x: f64) -> f64 {
    libm::atanh(x)
}

/// Sine and cosine simultaneously.
pub fn sin_cos(x: f64) -> (f64, f64) {
    libm::sincos(x)
}

/// Hypotenuse: `sqrt(x^2 + y^2)` without intermediate overflow.
pub fn hypot(x: f64, y: f64) -> f64 {
    libm::hypot(x, y)
}

/// Floor: largest integer <= x.
pub fn floor(x: f64) -> f64 {
    libm::floor(x)
}

/// Ceiling: smallest integer >= x.
pub fn ceil(x: f64) -> f64 {
    libm::ceil(x)
}

/// Round to nearest integer.
pub fn round(x: f64) -> f64 {
    libm::round(x)
}

/// Truncate toward zero.
pub fn trunc(x: f64) -> f64 {
    libm::trunc(x)
}

/// f32 variants for no_std builds.
pub mod f32 {
    /// Square root.
    pub fn sqrt(x: f32) -> f32 {
        libm::sqrtf(x)
    }

    /// Cube root.
    pub fn cbrt(x: f32) -> f32 {
        libm::cbrtf(x)
    }

    /// Raise to a floating-point power.
    pub fn powf(x: f32, e: f32) -> f32 {
        libm::powf(x, e)
    }

    /// Raise to an integer power.
    pub fn powi(x: f32, e: i32) -> f32 {
        libm::powf(x, e as f32)
    }

    /// `e^x`.
    pub fn exp(x: f32) -> f32 {
        libm::expf(x)
    }

    /// `e^x - 1`, accurate for small `x`.
    pub fn exp_m1(x: f32) -> f32 {
        libm::expm1f(x)
    }

    /// Natural logarithm.
    pub fn ln(x: f32) -> f32 {
        libm::logf(x)
    }

    /// `ln(1 + x)`, accurate for small `x`.
    pub fn ln_1p(x: f32) -> f32 {
        libm::log1pf(x)
    }

    /// Logarithm with arbitrary base.
    pub fn log(x: f32, base: f32) -> f32 {
        libm::logf(x) / libm::logf(base)
    }

    /// Log base 10.
    pub fn log10(x: f32) -> f32 {
        libm::log10f(x)
    }

    /// Log base 2.
    pub fn log2(x: f32) -> f32 {
        libm::log2f(x)
    }

    /// Sine (radians).
    pub fn sin(x: f32) -> f32 {
        libm::sinf(x)
    }

    /// Cosine (radians).
    pub fn cos(x: f32) -> f32 {
        libm::cosf(x)
    }

    /// Tangent (radians).
    pub fn tan(x: f32) -> f32 {
        libm::tanf(x)
    }

    /// Arcsine.
    pub fn asin(x: f32) -> f32 {
        libm::asinf(x)
    }

    /// Arccosine.
    pub fn acos(x: f32) -> f32 {
        libm::acosf(x)
    }

    /// Arctangent.
    pub fn atan(x: f32) -> f32 {
        libm::atanf(x)
    }

    /// Two-argument arctangent.
    pub fn atan2(y: f32, x: f32) -> f32 {
        libm::atan2f(y, x)
    }

    /// Hyperbolic sine.
    pub fn sinh(x: f32) -> f32 {
        libm::sinhf(x)
    }

    /// Hyperbolic cosine.
    pub fn cosh(x: f32) -> f32 {
        libm::coshf(x)
    }

    /// Hyperbolic tangent.
    pub fn tanh(x: f32) -> f32 {
        libm::tanhf(x)
    }

    /// Inverse hyperbolic sine.
    pub fn asinh(x: f32) -> f32 {
        libm::asinhf(x)
    }

    /// Inverse hyperbolic cosine.
    pub fn acosh(x: f32) -> f32 {
        libm::acoshf(x)
    }

    /// Inverse hyperbolic tangent.
    pub fn atanh(x: f32) -> f32 {
        libm::atanhf(x)
    }

    /// Sine and cosine simultaneously.
    pub fn sin_cos(x: f32) -> (f32, f32) {
        libm::sincosf(x)
    }

    /// Hypotenuse.
    pub fn hypot(x: f32, y: f32) -> f32 {
        libm::hypotf(x, y)
    }

    /// Floor.
    pub fn floor(x: f32) -> f32 {
        libm::floorf(x)
    }

    /// Ceiling.
    pub fn ceil(x: f32) -> f32 {
        libm::ceilf(x)
    }

    /// Round to nearest integer.
    pub fn round(x: f32) -> f32 {
        libm::roundf(x)
    }

    /// Truncate toward zero.
    pub fn trunc(x: f32) -> f32 {
        libm::truncf(x)
    }
}
