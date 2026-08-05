//! Physics domain applications for `MathVerse`
//! 
//! This crate provides physics-specific calculations and constants including:
//! - Classical mechanics (kinematics, dynamics)
//! - Thermodynamics
//! - Electromagnetism
//! - Waves and optics
//! - Physical constants

pub mod mechanics;
pub mod thermodynamics;
pub mod electromagnetism;
pub mod waves;
pub mod constants;

pub use mechanics::*;
pub use thermodynamics::*;
pub use electromagnetism::*;
pub use waves::*;
pub use constants::*;
