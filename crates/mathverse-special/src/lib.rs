//! # mathverse-special
//!
//! Real-valued special functions for the MathVerse ecosystem.
//!
//! Provides:
//! - **Gamma family**: [`gamma`], [`log_gamma`], [`digamma`], [`beta`],
//!   incomplete gamma [`gamma_p`] / [`gamma_q`]
//! - **Error function**: [`erf`], [`erfc`]
//! - **Bessel functions**: [`bessel_j0`], [`bessel_j1`], [`bessel_jn`],
//!   [`bessel_y0`], [`bessel_y1`], [`bessel_i0`], [`bessel_i1`],
//!   [`bessel_k0`], [`bessel_k1`]
//! - **Riemann zeta**: [`zeta`]
//!
//! Complex-domain counterparts of several of these live in
//! `mathverse-complex::special_functions`.

pub mod bessel;
pub mod erf;
pub mod gamma;
pub mod zeta;

pub use bessel::*;
pub use erf::*;
pub use gamma::*;
pub use zeta::*;
