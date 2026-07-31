//! Compile-time dimensional analysis and units
//! 
//! This crate provides type-safe dimensional analysis using Rust's type system:
//! - Compile-time dimension checking
//! - Common SI units
//! - Unit conversions
//! - Dimension-aware arithmetic

pub mod dimensions;
pub mod si;
pub mod quantity;
pub mod conversions;

pub use dimensions::*;
pub use si::*;
pub use quantity::*;
pub use conversions::*;
