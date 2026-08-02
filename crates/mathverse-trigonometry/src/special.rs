//! Special trigonometric functions: sinc, haversine, Gudermannian, Chebyshev, versine, exsecant.

use mathverse_core::traits::Real;
use crate::util::map_real as f;

// ---------------------------------------------------------------------------
// Sinc
// ---------------------------------------------------------------------------

/// Normalized sinc: sin(πx) / (πx). Returns 1 at x=0.
pub fn sinc<T: Real>(x: T) -> T {
    if x.abs() < T::from_f64(1e-15) {
        T::one()
    } else {
        let px = x * T::from_f64(core::f64::consts::PI);
        f(px, f64::sin) / px
    }
}

/// Unnormalized sinc: sin(x) / x. Returns 1 at x=0.
pub fn sinc_unnorm<T: Real>(x: T) -> T {
    if x.abs() < T::from_f64(1e-15) {
        T::one()
    } else {
        f(x, f64::sin) / x
    }
}

// ---------------------------------------------------------------------------
// Versine family
// ---------------------------------------------------------------------------

/// Versine(x) = 1 - cos(x).
pub fn versine<T: Real>(x: T) -> T {
    T::one() - f(x, f64::cos)
}

/// Coversine(x) = 1 - sin(x).
pub fn coversine<T: Real>(x: T) -> T {
    T::one() - f(x, f64::sin)
}

/// Verco versed sine: versin(π - x) = 1 + cos(x).
pub fn vercosine<T: Real>(x: T) -> T {
    T::one() + f(x, f64::cos)
}

/// Covercosine: cos versed = 1 + sin(x).
pub fn covercosine<T: Real>(x: T) -> T {
    T::one() + f(x, f64::sin)
}

/// Haversine: versin(x)/2 = (1 - cos(x))/2.
pub fn haversine<T: Real>(x: T) -> T {
    versine(x) / T::from_f64(2.0)
}

/// Havercosine: vercosin(x)/2 = (1 + cos(x))/2.
pub fn havercosine<T: Real>(x: T) -> T {
    vercosine(x) / T::from_f64(2.0)
}

/// Hacoversine: coversin(x)/2 = (1 - sin(x))/2.
pub fn hacoversine<T: Real>(x: T) -> T {
    coversine(x) / T::from_f64(2.0)
}

/// Hacovercosine: covercos(x)/2 = (1 + sin(x))/2.
pub fn hacovercosine<T: Real>(x: T) -> T {
    covercosine(x) / T::from_f64(2.0)
}

/// Exsecant: sec(x) - 1.
pub fn exsecant<T: Real>(x: T) -> T {
    T::from_f64(1.0) / f(x, f64::cos) - T::one()
}

/// Excosecant: csc(x) - 1.
pub fn excosecant<T: Real>(x: T) -> T {
    T::from_f64(1.0) / f(x, f64::sin) - T::one()
}

// ---------------------------------------------------------------------------
// Gudermannian
// ---------------------------------------------------------------------------

/// Gudermannian: gd(x) = 2·arctan(eˣ) - π/2.
/// Relates circular and hyperbolic functions.
pub fn gudermannian<T: Real>(x: T) -> T {
    let ex = f(x, f64::exp);
    T::from_f64(2.0) * f(ex, f64::atan) - T::from_f64(core::f64::consts::FRAC_PI_2)
}

/// Inverse Gudermannian: gd⁻¹(x) = ln(tan(x/2 + π/4)).
pub fn gudermannian_inv<T: Real>(x: T) -> T {
    let half = x / T::from_f64(2.0) + T::from_f64(core::f64::consts::FRAC_PI_4);
    f(half, f64::tan).ln()
}

/// gd(x) also equals: 2·atan(tanh(x/2)).
pub fn gudermannian_alt<T: Real>(x: T) -> T {
    let half = x / T::from_f64(2.0);
    T::from_f64(2.0) * f(half, f64::tanh).atan()
}

// ---------------------------------------------------------------------------
// Chebyshev polynomials
// ---------------------------------------------------------------------------

/// Chebyshev polynomial of the first kind Tₙ(x) = cos(n·acos(x)).
pub fn chebyshev_first<T: Real>(n: u32, x: T) -> T {
    if n == 0 {
        T::one()
    } else if n == 1 {
        x
    } else {
        // Recurrence: Tₙ(x) = 2x·Tₙ₋₁(x) - Tₙ₋₂(x)
        let mut t_prev2 = T::one();
        let mut t_prev1 = x;
        for _ in 2..=n {
            let t = T::from_f64(2.0) * x * t_prev1 - t_prev2;
            t_prev2 = t_prev1;
            t_prev1 = t;
        }
        t_prev1
    }
}

/// Chebyshev polynomial of the second kind Uₙ(x) = sin((n+1)·acos(x)) / sin(acos(x)).
pub fn chebyshev_second<T: Real>(n: u32, x: T) -> T {
    if n == 0 {
        T::one()
    } else if n == 1 {
        T::from_f64(2.0) * x
    } else {
        let mut u_prev2 = T::one();
        let mut u_prev1 = T::from_f64(2.0) * x;
        for _ in 2..=n {
            let u = T::from_f64(2.0) * x * u_prev1 - u_prev2;
            u_prev2 = u_prev1;
            u_prev1 = u;
        }
        u_prev1
    }
}

// ---------------------------------------------------------------------------
// Trig power via Chebyshev
// ---------------------------------------------------------------------------

/// sinⁿ(x) expressed via Chebyshev for integer n.
pub fn sin_power<T: Real>(n: u32, x: T) -> T {
    if n == 0 {
        T::one()
    } else if n == 1 {
        f(x, f64::sin)
    } else if n % 2 == 0 {
        // sin²ⁿ(x) = ((1 - cos(2x))/2)ⁿ
        let half = (T::one() - f(T::from_f64(2.0) * x, f64::cos)) / T::from_f64(2.0);
        pow(half, n / 2)
    } else {
        // sin²ⁿ⁺¹(x) = sin(x) · sin²ⁿ(x)
        let half = (T::one() - f(T::from_f64(2.0) * x, f64::cos)) / T::from_f64(2.0);
        f(x, f64::sin) * pow(half, n / 2)
    }
}

/// cosⁿ(x) expressed via Chebyshev for integer n.
pub fn cos_power<T: Real>(n: u32, x: T) -> T {
    if n == 0 {
        T::one()
    } else if n == 1 {
        f(x, f64::cos)
    } else if n % 2 == 0 {
        let half = (T::one() + f(T::from_f64(2.0) * x, f64::cos)) / T::from_f64(2.0);
        pow(half, n / 2)
    } else {
        let half = (T::one() + f(T::from_f64(2.0) * x, f64::cos)) / T::from_f64(2.0);
        f(x, f64::cos) * pow(half, n / 2)
    }
}

fn pow<T: Real>(base: T, exp: u32) -> T {
    let mut result = T::one();
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result * b;
        }
        b = b * b;
        e >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_2, PI};

    const EPS: f64 = 1e-12;

    #[test]
    fn sinc_test() {
        assert!((sinc(0.0f64) - 1.0).abs() < EPS);
        assert!((sinc(1.0f64)).abs() < EPS);
        assert!((sinc(0.5f64) - 2.0 / PI).abs() < EPS);
    }

    #[test]
    fn versine_test() {
        assert!((versine(0.0f64)).abs() < EPS);
        assert!((versine(PI) - 2.0).abs() < EPS);
        assert!((haversine(PI) - 1.0).abs() < EPS);
        assert!((exsecant(0.0f64)).abs() < EPS);
    }

    #[test]
    fn gudermannian_test() {
        // gd(0) = 0
        assert!((gudermannian(0.0f64)).abs() < EPS);
        // gd(∞) ≈ π/2
        assert!((gudermannian(10.0f64) - FRAC_PI_2).abs() < 1e-4);
        // gd⁻¹(0) = 0
        assert!((gudermannian_inv(0.0f64)).abs() < EPS);
        // Roundtrip
        for x in [-2.0f64, -0.5, 0.0, 0.5, 2.0] {
            assert!((gudermannian_inv(gudermannian(x)) - x).abs() < 1e-10);
        }
    }

    #[test]
    fn chebyshev_test() {
        // T₀(x) = 1
        assert!((chebyshev_first(0, 0.5f64) - 1.0).abs() < EPS);
        // T₁(x) = x
        assert!((chebyshev_first(1, 0.5f64) - 0.5).abs() < EPS);
        // T₂(x) = 2x² - 1
        assert!((chebyshev_first(2, 0.5f64) - (-0.5)).abs() < EPS);
        // U₀(x) = 1
        assert!((chebyshev_second(0, 0.5f64) - 1.0).abs() < EPS);
        // U₁(x) = 2x
        assert!((chebyshev_second(1, 0.5f64) - 1.0).abs() < EPS);
    }

    #[test]
    fn chebyshev_cos_relation() {
        // Tₙ(cos θ) = cos(nθ)
        for n in 1..=5 {
            for theta in [0.1, 0.5, 1.0, 2.0] {
                let x = theta.cos();
                let expected = (n as f64 * theta).cos();
                assert!((chebyshev_first(n, x) - expected).abs() < 1e-10, "T_{n}({theta})");
            }
        }
    }

    #[test]
    fn sin_power_test() {
        for x in [0.0f64, 0.5, 1.0, 2.0] {
            assert!((sin_power(2, x) - x.sin().powi(2)).abs() < 1e-10, "sin²({x})");
            assert!((sin_power(3, x) - x.sin().powi(3)).abs() < 1e-10, "sin³({x})");
            assert!((cos_power(2, x) - x.cos().powi(2)).abs() < 1e-10, "cos²({x})");
            assert!((cos_power(3, x) - x.cos().powi(3)).abs() < 1e-10, "cos³({x})");
        }
    }
}
