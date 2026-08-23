//! Finance domain applications for MathVerse
//!
//! This crate provides financial calculations including:
//! - Time value of money
//! - Investment analysis
//! - Risk management
//! - Options pricing
//! - Portfolio management

pub mod investment;
pub mod options;
pub mod portfolio;
pub mod risk;
pub mod tvm;

pub use investment::*;
pub use options::*;
pub use portfolio::*;
pub use risk::*;
pub use tvm::*;
