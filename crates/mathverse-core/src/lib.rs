#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![allow(clippy::approx_constant)] // math crate: pi-like literals are intentional (consts + test inputs)

//! Shared substrate for the MathVerse ecosystem.
//!
//! Everything here is generic over the [`traits::Num`] hierarchy, works under
//! `no_std`, and has zero dependencies beyond `core`/`alloc`.

extern crate alloc;

pub mod algorithms;
pub mod constants;
pub mod error;
pub mod ops;
pub mod precision;
pub mod prelude;
pub mod traits;
