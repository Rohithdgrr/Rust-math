#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::approx_constant)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::float_cmp)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::many_single_char_names)]


//! Shared substrate for the `MathVerse` ecosystem.
//!
//! Everything here is generic over the [`traits::Num`] hierarchy, works under
//! `no_std`, and has zero dependencies beyond `core`/`alloc`.

extern crate alloc;

/// Compile-time error: `no_std` builds require the `libm` feature.
///
/// `mathverse-core` needs software floating-point math (via `libm`) when
/// the standard library is not available. Enable the `libm` feature:
///
/// ```toml
/// mathverse-core = { version = "0.1", default-features = false, features = ["libm"] }
/// ```
#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!(
    "mathverse-core: either the `std` or `libm` feature must be enabled. \
     no_std builds require `libm` for transcendental function support."
);

pub mod algorithms;
pub mod arrays;
pub mod constants;
pub mod error;
pub mod float;
pub mod integer;
pub mod ops;
pub mod precision;
pub mod prelude;
pub mod traits;

/// `libm`-backed transcendental functions for `no_std` builds.
///
/// Only available when the `libm` feature is enabled and `std` is disabled.
#[cfg(all(not(feature = "std"), feature = "libm"))]
pub mod libm_fallback;

#[cfg(feature = "pyo3")]
#[pymodule]
fn mathverse_core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::*;

    #[pyfunction]
    fn add(a: f64, b: f64) -> PyResult<f64> {
        Ok(a + b)
    }

    #[pyfunction]
    fn pi() -> PyResult<f64> {
        Ok(core::f64::consts::PI)
    }

    m.add_wrapped(wrap_pyfunction!(add)?)?;
    m.add_wrapped(wrap_pyfunction!(pi)?)?;
    Ok(())
}


