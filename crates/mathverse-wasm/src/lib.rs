//! # mathverse-wasm
//!
//! WebAssembly-compatible and `no_std` math operations.
//!
//! This crate provides:
//! - Re-exports of core MathVerse types for WASM targets
//! - WASM-bindgen bindings for JavaScript interop
//! - `no_std` compatible versions of common operations
//!
//! # Features
//!
//! - `std` (default): Standard library support
//! - `wasm`: WebAssembly-bindgen bindings for JS interop
//!
//! # Examples
//!
//! ```no_run
//! // WASM usage
//! use mathverse_wasm::{WasmMatrix, WasmVector};
//!
//! let m = WasmMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//! let result = m.multiply(&WasmMatrix::identity(2));
//! ```
//!
//! ```ignore
//! // no_std usage (with libm feature in core)
//! use mathverse_core::ops::lerp;
//! let v = lerp(0.0f64, 10.0, 0.5);
//! assert_eq!(v, 5.0);
//! ```

extern crate alloc;

pub mod wasm_bindings;
pub mod no_std_ops;
pub mod math_bridge;

pub use wasm_bindings::*;
pub use no_std_ops::*;
pub use math_bridge::*;
