//! Finance domain applications for MathVerse
//! 
//! This crate provides financial calculations including:
//! - Time value of money
//! - Investment analysis
//! - Risk management
//! - Options pricing
//! - Portfolio management

pub mod tvm;
pub mod investment;
pub mod risk;
pub mod options;
pub mod portfolio;

pub use tvm::*;
pub use investment::*;
pub use risk::*;
pub use options::*;
pub use portfolio::*;
