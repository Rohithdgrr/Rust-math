//! Exact values for special angles.
//!
//! Returns closed-form sine/cosine/tangent values for angles that are
//! multiples of 30° or 45° (i.e. multiples of `π/6` or `π/4`), e.g.
//! `sin(30°) = 1/2`, `cos(45°) = √2/2`. All other angles yield `None`.
//!
//! Values are represented exactly by [`ExactValue`] (integers, halves, and
//! `c·√r / d`) rather than floats, so they can be rendered symbolically or
//! converted with [`ExactValue::to_f64`].

use core::fmt::{self, Display};
use mathverse_core::traits::Real;

/// An exact trigonometric value of the form `n`, `n/2`, or `c·√r / d`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExactValue {
    /// An integer value, e.g. `sin(0°) = 0`.
    Integer(i32),
    /// A half-integer, e.g. `sin(30°) = 1/2`.
    Half(i32),
    /// `coeff · √radicand / denom`, e.g. `cos(30°) = √3/2`.
    Root {
        /// Integer coefficient.
        coeff: i32,
        /// Perfect-free radicand (2 or 3 in this crate).
        radicand: u32,
        /// Denominator (1, 2, or 3).
        denom: u32,
    },
}

impl ExactValue {
    /// Exact value as a float.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        match self {
            ExactValue::Integer(n) => f64::from(n),
            ExactValue::Half(n) => f64::from(n) / 2.0,
            ExactValue::Root {
                coeff,
                radicand,
                denom,
            } => f64::from(coeff) * f64::from(radicand).sqrt() / f64::from(denom),
        }
    }

    /// Negated value.
    #[must_use]
    pub fn negate(self) -> Self {
        match self {
            ExactValue::Integer(n) => ExactValue::Integer(-n),
            ExactValue::Half(n) => ExactValue::Half(-n),
            ExactValue::Root {
                coeff,
                radicand,
                denom,
            } => ExactValue::Root {
                coeff: -coeff,
                radicand,
                denom,
            },
        }
    }
}

impl Display for ExactValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ExactValue::Integer(n) => write!(f, "{n}"),
            ExactValue::Half(n) => write!(f, "{n}/2"),
            ExactValue::Root {
                coeff,
                radicand,
                denom,
            } => {
                match coeff {
                    1 => {}
                    -1 => write!(f, "-")?,
                    c => write!(f, "{c}")?,
                }
                write!(f, "√{radicand}")?;
                if denom != 1 {
                    write!(f, "/{denom}")?;
                }
                Ok(())
            }
        }
    }
}

fn base_sin(rem: i32) -> Option<ExactValue> {
    match rem {
        0 => Some(ExactValue::Integer(0)),
        30 => Some(ExactValue::Half(1)),
        45 => Some(ExactValue::Root {
            coeff: 1,
            radicand: 2,
            denom: 2,
        }),
        60 => Some(ExactValue::Root {
            coeff: 1,
            radicand: 3,
            denom: 2,
        }),
        90 => Some(ExactValue::Integer(1)),
        _ => None,
    }
}

fn base_cos(rem: i32) -> Option<ExactValue> {
    match rem {
        0 => Some(ExactValue::Integer(1)),
        30 => Some(ExactValue::Root {
            coeff: 1,
            radicand: 3,
            denom: 2,
        }),
        45 => Some(ExactValue::Root {
            coeff: 1,
            radicand: 2,
            denom: 2,
        }),
        60 => Some(ExactValue::Half(1)),
        90 => Some(ExactValue::Integer(0)),
        _ => None,
    }
}

fn base_tan(rem: i32) -> Option<ExactValue> {
    match rem {
        0 => Some(ExactValue::Integer(0)),
        30 => Some(ExactValue::Root {
            coeff: 1,
            radicand: 3,
            denom: 3,
        }),
        45 => Some(ExactValue::Integer(1)),
        60 => Some(ExactValue::Root {
            coeff: 1,
            radicand: 3,
            denom: 1,
        }),
        _ => None, // tan(90°) is undefined
    }
}

/// (quadrant, reference angle in degrees) for `n` in `[0, 360)`.
fn quadrant_ref(n: i32) -> (i32, i32) {
    match n {
        0..90 => (0, n),
        90..180 => (1, 180 - n),
        180..270 => (2, n - 180),
        _ => (3, 360 - n),
    }
}

/// Exact sine of an angle in degrees, for multiples of 30° or 45°.
///
/// # Examples
///
/// ```rust
/// use mathverse_trigonometry::sin_exact_deg;
///
/// assert_eq!(sin_exact_deg(30), Some(mathverse_trigonometry::ExactValue::Half(1)));
/// assert_eq!(sin_exact_deg(15), None);
/// ```
#[must_use]
pub fn sin_exact_deg(deg: i32) -> Option<ExactValue> {
    let n = deg.rem_euclid(360);
    if n % 90 == 0 {
        return Some(match n {
            0 | 180 => ExactValue::Integer(0),
            90 => ExactValue::Integer(1),
            _ => ExactValue::Integer(-1), // 270
        });
    }
    let (q, ref_angle) = quadrant_ref(n);
    let base = base_sin(ref_angle)?;
    let positive = q == 0 || q == 1;
    Some(if positive { base } else { base.negate() })
}

/// Exact cosine of an angle in degrees, for multiples of 30° or 45°.
///
/// # Examples
///
/// ```rust
/// use mathverse_trigonometry::cos_exact_deg;
///
/// assert_eq!(cos_exact_deg(60), Some(mathverse_trigonometry::ExactValue::Half(1)));
/// assert_eq!(cos_exact_deg(180), Some(mathverse_trigonometry::ExactValue::Integer(-1)));
/// ```
#[must_use]
pub fn cos_exact_deg(deg: i32) -> Option<ExactValue> {
    let n = deg.rem_euclid(360);
    if n % 90 == 0 {
        return Some(match n {
            0 => ExactValue::Integer(1),
            180 => ExactValue::Integer(-1),
            _ => ExactValue::Integer(0), // 90, 270
        });
    }
    let (q, ref_angle) = quadrant_ref(n);
    let base = base_cos(ref_angle)?;
    let positive = q == 0 || q == 3;
    Some(if positive { base } else { base.negate() })
}

/// Exact tangent of an angle in degrees, for multiples of 30° or 45°.
///
/// Returns `None` for the undefined `tan(90°)` and `tan(270°)`.
///
/// # Examples
///
/// ```rust
/// use mathverse_trigonometry::tan_exact_deg;
///
/// assert_eq!(tan_exact_deg(45), Some(mathverse_trigonometry::ExactValue::Integer(1)));
/// assert_eq!(tan_exact_deg(90), None);
/// ```
#[must_use]
pub fn tan_exact_deg(deg: i32) -> Option<ExactValue> {
    let n = deg.rem_euclid(360);
    if n % 90 == 0 {
        return if n % 180 == 0 {
            Some(ExactValue::Integer(0)) // 0°, 180°
        } else {
            None // 90°, 270°
        };
    }
    let (q, ref_angle) = quadrant_ref(n);
    let base = base_tan(ref_angle)?;
    let positive = q == 0 || q == 2;
    Some(if positive { base } else { base.negate() })
}

/// Exact sine of an angle in radians, for angles that are multiples of
/// `π/6` or `π/4`.
///
/// # Examples
///
/// ```rust
/// use mathverse_trigonometry::sin_exact_radians;
///
/// assert_eq!(sin_exact_radians(core::f64::consts::FRAC_PI_6), Some(mathverse_trigonometry::ExactValue::Half(1)));
/// ```
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#[must_use]
pub fn sin_exact_radians<T: Real>(radians: T) -> Option<ExactValue> {
    let deg = radians.to_f64().to_degrees().round().rem_euclid(360.0) as i32;
    sin_exact_deg(deg)
}

/// Exact cosine of an angle in radians, for angles that are multiples of
/// `π/6` or `π/4`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#[must_use]
pub fn cos_exact_radians<T: Real>(radians: T) -> Option<ExactValue> {
    let deg = radians.to_f64().to_degrees().round().rem_euclid(360.0) as i32;
    cos_exact_deg(deg)
}

/// Exact tangent of an angle in radians, for angles that are multiples of
/// `π/6` or `π/4`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#[must_use]
pub fn tan_exact_radians<T: Real>(radians: T) -> Option<ExactValue> {
    let deg = radians.to_f64().to_degrees().round().rem_euclid(360.0) as i32;
    tan_exact_deg(deg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_6, FRAC_PI_4, PI};

    #[test]
    fn sine_table() {
        assert_eq!(sin_exact_deg(0), Some(ExactValue::Integer(0)));
        assert_eq!(sin_exact_deg(30), Some(ExactValue::Half(1)));
        assert_eq!(sin_exact_deg(90), Some(ExactValue::Integer(1)));
        assert_eq!(sin_exact_deg(120), Some(ExactValue::Root { coeff: 1, radicand: 3, denom: 2 }));
        assert_eq!(sin_exact_deg(150), Some(ExactValue::Half(1)));
        assert_eq!(sin_exact_deg(180), Some(ExactValue::Integer(0)));
        assert_eq!(sin_exact_deg(210), Some(ExactValue::Half(-1)));
        assert_eq!(sin_exact_deg(270), Some(ExactValue::Integer(-1)));
        assert_eq!(sin_exact_deg(330), Some(ExactValue::Half(-1)));
        assert_eq!(sin_exact_deg(360), Some(ExactValue::Integer(0)));
        assert_eq!(sin_exact_deg(-30), Some(ExactValue::Half(-1)));
        assert_eq!(sin_exact_deg(720), Some(ExactValue::Integer(0)));
    }

    #[test]
    fn cosine_table() {
        assert_eq!(cos_exact_deg(0), Some(ExactValue::Integer(1)));
        assert_eq!(cos_exact_deg(60), Some(ExactValue::Half(1)));
        assert_eq!(cos_exact_deg(120), Some(ExactValue::Half(-1)));
        assert_eq!(cos_exact_deg(180), Some(ExactValue::Integer(-1)));
        assert_eq!(cos_exact_deg(300), Some(ExactValue::Half(1)));
        assert_eq!(cos_exact_deg(90), Some(ExactValue::Integer(0)));
    }

    #[test]
    fn tangent_table() {
        assert_eq!(tan_exact_deg(0), Some(ExactValue::Integer(0)));
        assert_eq!(tan_exact_deg(45), Some(ExactValue::Integer(1)));
        assert_eq!(tan_exact_deg(135), Some(ExactValue::Integer(-1)));
        assert_eq!(tan_exact_deg(180), Some(ExactValue::Integer(0)));
        assert_eq!(tan_exact_deg(90), None);
        assert_eq!(tan_exact_deg(270), None);
    }

    #[test]
    fn to_float_roundtrip() {
        let cases = [0.0, 0.5, 2.0_f64.sqrt() / 2.0, 3.0_f64.sqrt() / 2.0, 1.0, -1.0];
        for deg in [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360] {
            if let Some(v) = sin_exact_deg(deg) {
                let expected = (deg as f64).to_radians().sin();
                assert!((v.to_f64() - expected).abs() < 1e-12, "sin({deg})");
            }
        }
        for deg in [0, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360] {
            if let Some(v) = tan_exact_deg(deg) {
                let expected = (deg as f64).to_radians().tan();
                assert!((v.to_f64() - expected).abs() < 1e-12, "tan({deg})");
            }
        }
        let _ = cases;
    }

    #[test]
    fn radians_variants() {
        assert_eq!(sin_exact_radians(FRAC_PI_6), Some(ExactValue::Half(1)));
        assert_eq!(cos_exact_radians(FRAC_PI_4), Some(ExactValue::Root { coeff: 1, radicand: 2, denom: 2 }));
        assert_eq!(sin_exact_radians(PI), Some(ExactValue::Integer(0)));
        assert_eq!(sin_exact_radians(0.3f64), None);
    }

    #[test]
    fn display_rendering() {
        assert_eq!(sin_exact_deg(30).unwrap().to_string(), "1/2");
        assert_eq!(sin_exact_deg(45).unwrap().to_string(), "√2/2");
        assert_eq!(tan_exact_deg(60).unwrap().to_string(), "√3");
        assert_eq!(sin_exact_deg(210).unwrap().to_string(), "-1/2");
    }
}
