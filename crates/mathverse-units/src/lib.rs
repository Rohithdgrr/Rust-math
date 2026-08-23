//! Compile-time dimensional analysis and units
//!
//! This crate provides type-safe dimensional analysis using Rust's type system:
//! - Compile-time dimension checking
//! - Common SI units
//! - Unit conversions
//! - Dimension-aware arithmetic

pub mod conversions;
pub mod dimensions;
pub mod quantity;
pub mod si;

pub use conversions::*;
pub use dimensions::*;
pub use quantity::*;
pub use si::*;
