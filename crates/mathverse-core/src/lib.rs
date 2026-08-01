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

pub mod algorithms;
pub mod constants;
pub mod error;
pub mod integer;
pub mod ops;
pub mod precision;
pub mod prelude;
pub mod traits;
